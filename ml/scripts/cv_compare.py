"""Compare model versions with 5-fold cross-validation on TRAIN + VAL (the test set is never touched).

Why: the val set has only ~75 nights (9 severe), so one or two nights change the numbers a lot.
CV gives every one of the ~430 nights a prediction from a model that did not see it.

    python scripts/cv_compare.py --data ../data/sgr/shhs          # ~20 minutes

Writes ../data/results/cv_compare.json (aggregate numbers) and cv_predictions.csv
(one row per night: id, true AHI, predicted AHI). Both stay on this laptop.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sleepguard_ml.baseline import BaselineModel, analyze_baseline, fit_calibration
from sleepguard_ml.io import load_parquet
from sleepguard_ml.m1 import analyze_m1, train_m1
from sleepguard_ml.metrics import ahi_agreement, bootstrap_ci, screening, severity_metrics, to_severity

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--split", default="../data/splits/shhs_split.json")
p.add_argument("--folds", type=int, default=5)
p.add_argument("--results", default="../data/results")
a = p.parse_args()

CONFIGS = {"v1_features": ["odi25", "hb"], "v2_features": []}  # night columns to drop

t0 = time.time()
split = json.loads(Path(a.split).read_text())
use = set(split["train"]) | set(split["val"])
recs = [r for r in (load_parquet(f) for f in sorted(Path(a.data).glob("*.parquet")))
        if r.recording_id in use and r.labels is not None and r.labels.ahi is not None]
ahi = np.array([r.labels.ahi for r in recs])
# folds stratified by severity
sev = np.array(to_severity(ahi))
rng = np.random.default_rng(7)
fold = np.zeros(len(recs), int)
for s in np.unique(sev):
    idx = np.where(sev == s)[0]
    fold[idx[rng.permutation(len(idx))]] = np.arange(len(idx)) % a.folds
print(f"{len(recs)} nights (train + val), {a.folds} folds, test set untouched")

pred = {"M0_odi3_calibrated": np.zeros(len(recs))}
for c in CONFIGS:
    pred[f"M1_{c}_raw"] = np.zeros(len(recs))
    pred[f"M1_{c}_cal"] = np.zeros(len(recs))

for k in range(a.folds):
    tr = [recs[i] for i in range(len(recs)) if fold[i] != k]
    te = [i for i in range(len(recs)) if fold[i] == k]
    m0 = fit_calibration(tr, BaselineModel(drop=3.0))
    for i in te:
        pred["M0_odi3_calibrated"][i] = analyze_baseline(recs[i], m0).ahi_est or 0.0
    for c, drop in CONFIGS.items():
        m = train_m1(tr, verbose=False, drop_night_cols=drop)
        for i in te:
            pred[f"M1_{c}_raw"][i] = analyze_m1(recs[i], m, calibrated=False).ahi_est or 0.0
            pred[f"M1_{c}_cal"][i] = analyze_m1(recs[i], m).ahi_est or 0.0
    print(f"  fold {k + 1}/{a.folds} done ({time.time() - t0:.0f} s)")


def sens30(t, q):
    return screening(t, q, 30)["sensitivity"]


def sens15(t, q):
    return screening(t, q, 15)["sensitivity"]


n15, n30 = int((ahi >= 15).sum()), int((ahi >= 30).sum())
print(f"\nnights with AHI >= 15: {n15}, AHI >= 30: {n30}")
print(f"{'model':<26}{'MAE':>6}{'ICC':>6}{'kappa':>7}{'acc4':>6}{'sens15':>8}{'spec15':>8}{'sens30':>8}{'spec30':>8}")
report = {"n": len(recs), "n_ahi15": n15, "n_ahi30": n30, "models": {}}
for name, q in pred.items():
    ag, sv = ahi_agreement(ahi, q), severity_metrics(ahi, q)
    s15, s30 = screening(ahi, q, 15), screening(ahi, q, 30)
    report["models"][name] = {"ahi": ag, "severity": sv, "screen_15": s15, "screen_30": s30,
                              "sens15_ci": bootstrap_ci(sens15, ahi, q), "sens30_ci": bootstrap_ci(sens30, ahi, q)}
    print(f"{name:<26}{ag['mae']:6.1f}{ag['icc']:6.2f}{sv['kappa_linear']:7.2f}{sv['accuracy']:6.2f}"
          f"{s15['sensitivity']:8.2f}{s15['specificity']:8.2f}{s30['sensitivity']:8.2f}{s30['specificity']:8.2f}")

out = Path(a.results)
out.mkdir(parents=True, exist_ok=True)
(out / "cv_compare.json").write_text(json.dumps(report, indent=2))
pd.DataFrame({"id": [r.recording_id for r in recs], "true_ahi": ahi, **pred}).round(2).to_csv(
    out / "cv_predictions.csv", index=False)
print(f"\nsaved {out / 'cv_compare.json'} and cv_predictions.csv  ({time.time() - t0:.0f} s)")
