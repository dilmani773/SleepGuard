"""FINAL evaluation of the noise-robust model (M1-robust = "aug1" from cv_robust.py). Run ONCE.

Chosen on cross-validation only. Note for the paper: the pre-set rule in cv_robust.py picked "jitter"
because aug1's clean MAE (3.6) was 0.1 above the allowed limit (3.5). We chose aug1 anyway because it
raised specificity on noisy data from 0.39 to 0.87. This deviation is reported, and the test sets were
still never used for any choice.

Trains on SHHS train + val, tests on the SHHS test split and MESA, clean and under the same
cheap-sensor conditions and seeds as robustness.py, so the numbers compare directly.

    python scripts/final_eval_robust.py --confirm
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from sleepguard_ml.augment import combo, drop, perturb, smooth
from sleepguard_ml.io import load_parquet
from sleepguard_ml.m1 import analyze_m1, train_m1
from sleepguard_ml.metrics import ahi_agreement, bootstrap_ci, screening, severity_metrics
from sleepguard_ml.summary import REFER_CUTOFF

# same order and functions as robustness.py (seed = 1000 * condition index + night index)
CONDITIONS = {
    "clean (reference)":        lambda s, h, r: (s, h),
    "raw noise SD 1% (worst)":  lambda s, h, r: (s + r.normal(0, 1, len(s)), h + r.normal(0, 2, len(h))),
    "noise SD 1% + 4 s avg":    lambda s, h, r: (smooth(s + r.normal(0, 1, len(s)), 4), smooth(h + r.normal(0, 2, len(h)), 4)),
    "noise SD 2% + 4 s avg":    lambda s, h, r: (smooth(s + r.normal(0, 2, len(s)), 4), smooth(h + r.normal(0, 4, len(h)), 4)),
    "bias -2%":                 lambda s, h, r: (s - 2, h),
    "bias +2%":                 lambda s, h, r: (s + 2, h),
    "dropouts 5%":              lambda s, h, r: drop(s, h, 0.05, r),
    "dropouts 15%":             lambda s, h, r: drop(s, h, 0.15, r),
    "4 s averaging":            lambda s, h, r: (smooth(s, 4), smooth(h, 4)),
    "realistic MAX30102 combo": combo,
}

p = argparse.ArgumentParser()
p.add_argument("--shhs", default="../data/sgr/shhs")
p.add_argument("--mesa", default="../data/sgr/mesa")
p.add_argument("--split", default="../data/splits/shhs_split.json")
p.add_argument("--results", default="../data/results")
p.add_argument("--models", default="../data/models/m1_robust")
p.add_argument("--confirm", action="store_true")
p.add_argument("--force", action="store_true")
a = p.parse_args()

out = Path(a.results) / "final_eval_robust.json"
if not a.confirm:
    raise SystemExit("This uses the sealed test sets. Run again with --confirm.")
if out.exists() and not a.force:
    raise SystemExit(f"{out} already exists. The final evaluation is done once only.")


def load(folder):
    return [r for r in (load_parquet(f) for f in sorted(Path(folder).glob("*.parquet")))
            if r.labels is not None and r.labels.ahi is not None]


t0 = time.time()
split = json.loads(Path(a.split).read_text())
shhs = {r.recording_id: r for r in load(a.shhs)}
train = [shhs[i] for i in split["train"] + split["val"] if i in shhs]
tests = {"shhs_test": [shhs[i] for i in split["test"] if i in shhs], "mesa_external": load(a.mesa)}
print(f"train: {len(train)} | " + " | ".join(f"{k}: {len(v)}" for k, v in tests.items()))

m = train_m1(train, augment=1)
m.save(a.models)
print(f"saved M1-robust to {a.models}  ({time.time() - t0:.0f} s)")


def s15(t, q):
    return screening(t, q, 15)["sensitivity"]


def sp15(t, q):
    return screening(t, q, 15)["specificity"]


report = {"model": "M1-robust (aug1)", "refer_cutoff": REFER_CUTOFF, "sets": {}}
for sname, recs in tests.items():
    ahi = np.array([r.labels.ahi for r in recs])
    t = ahi >= 15
    print(f"\n== {sname}: {len(recs)} nights")
    print(f"{'condition':<28}{'MAE':>6}{'ICC':>6}{'kappa':>7}{'sens15':>8}{'spec15':>8}{'refer_sens':>12}{'refer_spec':>12}")
    report["sets"][sname] = {}
    for ci, (cname, fn) in enumerate(CONDITIONS.items()):
        q = np.array([analyze_m1(recs[k] if ci == 0 else perturb(recs[k], fn, 1000 * ci + k), m).ahi_est or 0.0
                      for k in range(len(recs))])
        ag, sv, sc = ahi_agreement(ahi, q), severity_metrics(ahi, q), screening(ahi, q, 15)
        ref = q >= REFER_CUTOFF
        row = {"ahi": ag, "kappa": sv["kappa_linear"], "accuracy4": sv["accuracy"], "screen_15": sc,
               "screen_30": screening(ahi, q, 30),
               "refer_sens": float(ref[t].mean()), "refer_spec": float((~ref[~t]).mean())}
        if ci == 0:
            row["sens15_ci"], row["spec15_ci"] = bootstrap_ci(s15, ahi, q), bootstrap_ci(sp15, ahi, q)
        report["sets"][sname][cname] = row
        print(f"{cname:<28}{ag['mae']:6.1f}{ag['icc']:6.2f}{sv['kappa_linear']:7.2f}{sc['sensitivity']:8.2f}"
              f"{sc['specificity']:8.2f}{row['refer_sens']:12.2f}{row['refer_spec']:12.2f}")

out.write_text(json.dumps(report, indent=2))
print(f"\nsaved {out}  ({time.time() - t0:.0f} s)")
