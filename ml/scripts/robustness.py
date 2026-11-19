"""Stress test: how much does the FROZEN final model degrade on cheap-sensor problems?

Adds the problems a MAX30102 + ESP32 clip is likely to have to the SHHS test and MESA nights,
then re-runs the frozen model (no retraining, nothing is tuned on the result).

    python scripts/robustness.py           # a few minutes

Writes ../data/results/robustness.json (aggregate numbers only).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from sleepguard_ml.io import from_arrays, load_parquet, to_arrays
from sleepguard_ml.m1 import M1Model, analyze_m1
from sleepguard_ml.metrics import ahi_agreement, screening
from sleepguard_ml.summary import REFER_CUTOFF

p = argparse.ArgumentParser()
p.add_argument("--shhs", default="../data/sgr/shhs")
p.add_argument("--mesa", default="../data/sgr/mesa")
p.add_argument("--split", default="../data/splits/shhs_split.json")
p.add_argument("--model", default="../data/models/m1_final")
p.add_argument("--results", default="../data/results")
a = p.parse_args()


def gaps(n, frac, rng, lo=30, hi=120):
    """Random signal dropouts (finger moved, clip slipped) covering about `frac` of the night."""
    m = np.zeros(n, bool)
    while m.mean() < frac:
        L = int(rng.integers(lo, hi))
        s = int(rng.integers(0, max(n - L, 1)))
        m[s:s + L] = True
    return m


def smooth(x, k):
    """Moving average, like a cheap sensor's internal averaging (keeps NaN gaps)."""
    ok = np.isfinite(x)
    num = np.convolve(np.where(ok, x, 0.0), np.ones(k), "same")
    den = np.convolve(ok.astype(float), np.ones(k), "same")
    out = num / np.maximum(den, 1)
    out[~ok] = np.nan
    return out


# name -> function(spo2, hr, rng) -> (spo2, hr). Values are rounded to whole numbers afterwards,
# like the sensor output (SGR stores integers).
CONDITIONS = {
    "clean (reference)":        lambda s, h, r: (s, h),
    # raw per-second white noise is a worst case: real oximeters average over several seconds
    "raw noise SD 1% (worst)":  lambda s, h, r: (s + r.normal(0, 1, len(s)), h + r.normal(0, 2, len(h))),
    "noise SD 1% + 4 s avg":    lambda s, h, r: (smooth(s + r.normal(0, 1, len(s)), 4), smooth(h + r.normal(0, 2, len(h)), 4)),
    "noise SD 2% + 4 s avg":    lambda s, h, r: (smooth(s + r.normal(0, 2, len(s)), 4), smooth(h + r.normal(0, 4, len(h)), 4)),
    "bias -2%":                 lambda s, h, r: (s - 2, h),
    "bias +2%":                 lambda s, h, r: (s + 2, h),
    "dropouts 5%":              lambda s, h, r: _drop(s, h, 0.05, r),
    "dropouts 15%":             lambda s, h, r: _drop(s, h, 0.15, r),
    "4 s averaging":            lambda s, h, r: (smooth(s, 4), smooth(h, 4)),
    "realistic MAX30102 combo": lambda s, h, r: _combo(s, h, r),
}


def _drop(s, h, frac, r):
    m = gaps(len(s), frac, r)
    s, h = s.copy(), h.copy()
    s[m] = np.nan
    h[m] = np.nan
    return s, h


def _combo(s, h, r):
    """per-night bias in [-2, 2], SD 1% noise, 4 s averaging, 5% dropouts"""
    s = smooth(s + r.uniform(-2, 2) + r.normal(0, 1, len(s)), 4)
    h = smooth(h + r.normal(0, 2, len(h)), 4)
    return _drop(s, h, 0.05, r)


def perturb(rec, fn, seed):
    s, h = to_arrays(rec)
    s, h = fn(s, h, np.random.default_rng(seed))
    s = np.clip(s, 0, 100)
    return from_arrays(rec.recording_id, rec.source, s, h, labels=rec.labels)


def load(folder):
    return [r for r in (load_parquet(f) for f in sorted(Path(folder).glob("*.parquet")))
            if r.labels is not None and r.labels.ahi is not None]


t0 = time.time()
model = M1Model.load(a.model)
split = json.loads(Path(a.split).read_text())
shhs = {r.recording_id: r for r in load(a.shhs)}
sets = {"shhs_test": [shhs[i] for i in split["test"] if i in shhs], "mesa_external": load(a.mesa)}

report = {"refer_cutoff": REFER_CUTOFF, "sets": {}}
for sname, recs in sets.items():
    ahi = np.array([r.labels.ahi for r in recs])
    print(f"\n== {sname}: {len(recs)} nights")
    print(f"{'condition':<28}{'MAE':>6}{'ICC':>6}{'sens15':>8}{'spec15':>8}{'refer_sens':>12}{'refer_spec':>12}")
    report["sets"][sname] = {}
    for ci, (cname, fn) in enumerate(CONDITIONS.items()):
        q = np.array([analyze_m1(perturb(r, fn, 1000 * ci + k), model).ahi_est or 0.0
                      for k, r in enumerate(recs)])
        ag, sc = ahi_agreement(ahi, q), screening(ahi, q, 15)
        t, ref = ahi >= 15, q >= REFER_CUTOFF
        rs, rp = float(ref[t].mean()), float((~ref[~t]).mean())
        report["sets"][sname][cname] = {"mae": ag["mae"], "icc": ag["icc"], "bias": ag["bias"],
                                        "sens15": sc["sensitivity"], "spec15": sc["specificity"],
                                        "refer_sens": rs, "refer_spec": rp}
        print(f"{cname:<28}{ag['mae']:6.1f}{ag['icc']:6.2f}{sc['sensitivity']:8.2f}{sc['specificity']:8.2f}"
              f"{rs:12.2f}{rp:12.2f}")

out = Path(a.results) / "robustness.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2))
print(f"\nsaved {out}  ({time.time() - t0:.0f} s)")
