"""Fix the noise problem without touching the test sets.

5-fold cross-validation on SHHS train + val (428 nights). Each candidate model is trained on 4 folds
and tested on the 5th fold under: clean data and three cheap-sensor conditions (same as robustness.py).

    python scripts/cv_robust.py                     # all configs, about 3 hours, resumable
    python scripts/cv_robust.py --configs jitter    # only some

Choice rule (fixed BEFORE seeing results):
  keep configs whose clean MAE is at most 0.3 worse than "current" and clean referral sensitivity >= 0.93;
  among them pick the highest mean referral specificity over the three noisy conditions.

Writes ../data/results/cv_robust.json (aggregate numbers). Re-running skips finished configs.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from sleepguard_ml.augment import CONDITIONS, perturb
from sleepguard_ml.io import load_parquet
from sleepguard_ml.m1 import analyze_m1, train_m1
from sleepguard_ml.metrics import ahi_agreement, to_severity
from sleepguard_ml.summary import REFER_CUTOFF

CONFIGS = {
    "current": dict(drop_window_cols=["jitter"], drop_night_cols=["spo2_jitter"]),
    "jitter": dict(),
    "median5": dict(prefilter=5),
    "aug1": dict(augment=1),
    "aug1_median5": dict(augment=1, prefilter=5),
}

p = argparse.ArgumentParser()
p.add_argument("--data", default="../data/sgr/shhs")
p.add_argument("--split", default="../data/splits/shhs_split.json")
p.add_argument("--folds", type=int, default=5)
p.add_argument("--results", default="../data/results")
p.add_argument("--configs", nargs="*", default=list(CONFIGS))
a = p.parse_args()

out = Path(a.results) / "cv_robust.json"
out.parent.mkdir(parents=True, exist_ok=True)
report = json.loads(out.read_text()) if out.exists() else {}

split = json.loads(Path(a.split).read_text())
use = set(split["train"]) | set(split["val"])
recs = [r for r in (load_parquet(f) for f in sorted(Path(a.data).glob("*.parquet")))
        if r.recording_id in use and r.labels is not None and r.labels.ahi is not None]
ahi = np.array([r.labels.ahi for r in recs])
sev = np.array(to_severity(ahi))
rng = np.random.default_rng(7)  # same folds as cv_compare.py
fold = np.zeros(len(recs), int)
for s in np.unique(sev):
    idx = np.where(sev == s)[0]
    fold[idx[rng.permutation(len(idx))]] = np.arange(len(idx)) % a.folds
print(f"{len(recs)} nights, {a.folds} folds, test sets untouched")


def summarize(q):
    t, ref = ahi >= 15, q >= REFER_CUTOFF
    return {"mae": ahi_agreement(ahi, q)["mae"], "refer_sens": float(ref[t].mean()),
            "refer_spec": float((~ref[~t]).mean())}


t0 = time.time()
for cname in a.configs:
    if cname in report:
        print(f"{cname}: already done, skipped")
        continue
    preds = {c: np.zeros(len(recs)) for c in CONDITIONS}
    for k in range(a.folds):
        tr = [recs[i] for i in range(len(recs)) if fold[i] != k]
        m = train_m1(tr, verbose=False, **CONFIGS[cname])
        for i in np.where(fold == k)[0]:
            for ci, (c, fn) in enumerate(CONDITIONS.items()):
                r = recs[i] if c == "clean" else perturb(recs[i], fn, 1000 * ci + int(i))
                preds[c][i] = analyze_m1(r, m).ahi_est or 0.0
        print(f"  {cname}: fold {k + 1}/{a.folds} ({time.time() - t0:.0f} s)")
    report[cname] = {c: summarize(q) for c, q in preds.items()}
    out.write_text(json.dumps(report, indent=2))  # save after every config

noisy = [c for c in CONDITIONS if c != "clean"]
print(f"\n{'config':<16}{'clean MAE':>10}{'clean sens':>11}{'clean spec':>11}"
      f"{'noisy MAE':>10}{'noisy sens':>11}{'noisy spec':>11}")
for cname, r in report.items():
    print(f"{cname:<16}{r['clean']['mae']:10.1f}{r['clean']['refer_sens']:11.2f}{r['clean']['refer_spec']:11.2f}"
          f"{np.mean([r[c]['mae'] for c in noisy]):10.1f}{np.mean([r[c]['refer_sens'] for c in noisy]):11.2f}"
          f"{np.mean([r[c]['refer_spec'] for c in noisy]):11.2f}")
print("\nper condition (referral spec):")
for cname, r in report.items():
    print(f"  {cname:<16}" + "  ".join(f"{c}: {r[c]['refer_spec']:.2f}" for c in noisy))

if "current" in report:
    base = report["current"]["clean"]["mae"]
    ok = [c for c, r in report.items()
          if r["clean"]["mae"] <= base + 0.3 and r["clean"]["refer_sens"] >= 0.93]
    if ok:
        best = max(ok, key=lambda c: np.mean([report[c][n]["refer_spec"] for n in noisy]))
        print(f"\nchoice rule -> {best}")
print(f"saved {out}  ({time.time() - t0:.0f} s)")
