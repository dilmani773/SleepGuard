"""Train M0 (calibrated ODI) and M1 (XGBoost) on the TRAIN split and compare them.

    python scripts/train_eval.py --data ../data/sgr/shhs              # report on VAL (use this while developing)
    python scripts/train_eval.py --data ../data/sgr/shhs --eval test  # ONCE, at the end, for the paper

Splits are saved in ../data/splits/ and never change. Outputs stay in ../data/ (git-ignored).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from sleepguard_ml.baseline import BaselineModel, analyze_baseline, fit_calibration
from sleepguard_ml.io import load_parquet
from sleepguard_ml.m1 import analyze_m1, train_m1
from sleepguard_ml.metrics import ahi_agreement, bootstrap_ci, screening, severity_metrics
from sleepguard_ml.splits import load_or_make

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--split", default="../data/splits/shhs_split.json")
p.add_argument("--eval", choices=["val", "test"], default="val")
p.add_argument("--out", default="../data/models/m1")
p.add_argument("--results", default="../data/results")
p.add_argument("--tag", default="", help="added to the results file name, e.g. _v2")
a = p.parse_args()

t0 = time.time()
recs = [r for r in (load_parquet(f) for f in sorted(Path(a.data).glob("*.parquet")))
        if r.labels is not None and r.labels.ahi is not None]
by_id = {r.recording_id: r for r in recs}
split = load_or_make(a.split, list(by_id), [r.labels.ahi for r in recs])
train = [by_id[i] for i in split["train"] if i in by_id]
test = [by_id[i] for i in split[a.eval] if i in by_id]
print(f"{len(recs)} nights | train {len(train)} | {a.eval} {len(test)}")
if a.eval == "test":
    print("!!! Evaluating on the SEALED TEST set. Do this only once, for final results. !!!")

m0 = fit_calibration(train, BaselineModel(drop=3.0))
print(f"M0 calibration: AHI = {m0.slope:.2f} * ODI3 + {m0.intercept:.2f}")
print("training M1 ...")
m1 = train_m1(train)
m1.save(a.out)

true = np.array([r.labels.ahi for r in test])
preds = {
    "M0_odi3_calibrated": np.array([analyze_baseline(r, m0).ahi_est or 0.0 for r in test]),
    "M1_raw": np.array([analyze_m1(r, m1, calibrated=False).ahi_est or 0.0 for r in test]),
    "M1_calibrated": np.array([analyze_m1(r, m1).ahi_est or 0.0 for r in test]),
}

def sens(t, q):
    return screening(t, q, 15)["sensitivity"]

def spec(t, q):
    return screening(t, q, 15)["specificity"]

report = {"eval_set": a.eval, "n_train": len(train), "n_eval": len(test), "models": {}}
print(f"\n{'model':<22}{'MAE':>6}{'ICC':>6}{'kappa':>7}{'sens15':>8}{'spec15':>8}{'AUC15':>7}{'sens30':>8}{'spec30':>8}")
for name, q in preds.items():
    ag, sv = ahi_agreement(true, q), severity_metrics(true, q)
    s15, s30 = screening(true, q, 15), screening(true, q, 30)
    report["models"][name] = {
        "ahi": ag, "severity": sv, "screen_15": s15, "screen_30": s30, "screen_5": screening(true, q, 5),
        "sens15_ci": bootstrap_ci(sens, true, q), "spec15_ci": bootstrap_ci(spec, true, q),
    }
    print(f"{name:<22}{ag['mae']:6.1f}{ag['icc']:6.2f}{sv['kappa_linear']:7.2f}"
          f"{s15['sensitivity']:8.2f}{s15['specificity']:8.2f}{s15['auc']:7.2f}"
          f"{s30['sensitivity']:8.2f}{s30['specificity']:8.2f}")

Path(a.results).mkdir(parents=True, exist_ok=True)
out = Path(a.results) / f"m0_vs_m1_{a.eval}{a.tag}.json"
out.write_text(json.dumps(report, indent=2))
print(f"\nsaved {out}  ({time.time() - t0:.0f} s)")
