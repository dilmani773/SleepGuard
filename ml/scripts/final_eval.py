"""FINAL evaluation: run this ONCE, after every model choice is frozen.

Frozen choices (decided on cross-validation, never on test data):
  * M1 = night features v2 (incl. HB, ODI2.5) with out-of-fold linear calibration
  * referral rule: predicted AHI >= 12  (REFER_CUTOFF in summary.py)

Trains M0 and M1 on SHHS train + val, then tests on
  1. the sealed SHHS test split (internal test)
  2. all converted MESA nights (external test: other people, other hospital, other equipment)

    python scripts/final_eval.py --confirm

Writes ../data/results/final_eval.json and refuses to run twice (use --force only if
something crashed before the results were written).
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
from sleepguard_ml.summary import REFER_CUTOFF

p = argparse.ArgumentParser()
p.add_argument("--shhs", default="../data/sgr/shhs")
p.add_argument("--mesa", default="../data/sgr/mesa")
p.add_argument("--split", default="../data/splits/shhs_split.json")
p.add_argument("--results", default="../data/results")
p.add_argument("--models", default="../data/models/m1_final")
p.add_argument("--confirm", action="store_true", help="required: you know this uses the test sets")
p.add_argument("--force", action="store_true")
a = p.parse_args()

out = Path(a.results) / "final_eval.json"
if not a.confirm:
    raise SystemExit("This uses the sealed test sets. Run again with --confirm when every choice is frozen.")
if out.exists() and not a.force:
    raise SystemExit(f"{out} already exists. The final evaluation is done once only.")


def load(folder):
    recs = [load_parquet(f) for f in sorted(Path(folder).glob("*.parquet"))]
    return [r for r in recs if r.labels is not None and r.labels.ahi is not None]


t0 = time.time()
split = json.loads(Path(a.split).read_text())
shhs = {r.recording_id: r for r in load(a.shhs)}
train = [shhs[i] for i in split["train"] + split["val"] if i in shhs]
tests = {"shhs_test": [shhs[i] for i in split["test"] if i in shhs], "mesa_external": load(a.mesa)}
print(f"train (SHHS train+val): {len(train)} | " + " | ".join(f"{k}: {len(v)}" for k, v in tests.items()))

m0 = fit_calibration(train, BaselineModel(drop=3.0))
m1 = train_m1(train)
m1.save(a.models)
print(f"saved final M1 to {a.models}")


def s15(t, q):
    return screening(t, q, 15)["sensitivity"]


def sp15(t, q):
    return screening(t, q, 15)["specificity"]


def s30(t, q):
    return screening(t, q, 30)["sensitivity"]


def refer(t, q):
    """sens/spec for AHI >= 15 when we refer at predicted AHI >= REFER_CUTOFF"""
    t, q = np.asarray(t) >= 15, np.asarray(q) >= REFER_CUTOFF
    return {"sensitivity": float(q[t].mean()), "specificity": float((~q[~t]).mean()),
            "tp": int((t & q).sum()), "fn": int((t & ~q).sum()),
            "fp": int((~t & q).sum()), "tn": int((~t & ~q).sum())}


report = {"refer_cutoff": REFER_CUTOFF, "n_train": len(train), "sets": {}}
for name, recs in tests.items():
    if not recs:
        print(f"\n{name}: no nights, skipped")
        continue
    ahi = np.array([r.labels.ahi for r in recs])
    preds = {"M0_odi3_calibrated": np.array([analyze_baseline(r, m0).ahi_est or 0.0 for r in recs]),
             "M1_final": np.array([analyze_m1(r, m1).ahi_est or 0.0 for r in recs])}
    print(f"\n== {name}: {len(recs)} nights, AHI>=15: {int((ahi >= 15).sum())}, AHI>=30: {int((ahi >= 30).sum())}")
    print(f"{'model':<22}{'MAE':>6}{'ICC':>6}{'kappa':>7}{'acc4':>6}{'sens15':>8}{'spec15':>8}{'sens30':>8}"
          f"{'refer_sens':>12}{'refer_spec':>12}")
    report["sets"][name] = {"n": len(recs), "models": {}}
    for mname, q in preds.items():
        ag, sv, r12 = ahi_agreement(ahi, q), severity_metrics(ahi, q), refer(ahi, q)
        sc15, sc30 = screening(ahi, q, 15), screening(ahi, q, 30)
        report["sets"][name]["models"][mname] = {
            "ahi": ag, "severity": sv, "screen_15": sc15, "screen_30": sc30, f"refer_{REFER_CUTOFF:g}": r12,
            "sens15_ci": bootstrap_ci(s15, ahi, q), "spec15_ci": bootstrap_ci(sp15, ahi, q),
            "sens30_ci": bootstrap_ci(s30, ahi, q)}
        print(f"{mname:<22}{ag['mae']:6.1f}{ag['icc']:6.2f}{sv['kappa_linear']:7.2f}{sv['accuracy']:6.2f}"
              f"{sc15['sensitivity']:8.2f}{sc15['specificity']:8.2f}{sc30['sensitivity']:8.2f}"
              f"{r12['sensitivity']:12.2f}{r12['specificity']:12.2f}")

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2))
print(f"\nsaved {out}  ({time.time() - t0:.0f} s)")
