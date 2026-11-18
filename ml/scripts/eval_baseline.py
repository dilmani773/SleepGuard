"""Evaluate M0 (ODI baseline) on a folder of labelled SGR Parquet files.

    python scripts/eval_baseline.py --data ../data/synthetic
    python scripts/eval_baseline.py --data ../data/sgr/ucd --cv 5

With --cv, the ODI to AHI calibration is fitted on the other folds (no leakage).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from sleepguard_ml.baseline import BaselineModel, analyze_baseline, fit_calibration
from sleepguard_ml.io import load_parquet
from sleepguard_ml.metrics import ahi_agreement, match_events, screening, severity_metrics

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--cv", type=int, default=0)
p.add_argument("--drop", type=float, default=3.0)
a = p.parse_args()

recs = [r for r in (load_parquet(f) for f in sorted(Path(a.data).glob("*.parquet")))
        if r.labels is not None and r.labels.ahi is not None]
print(f"{len(recs)} labelled recordings")

pred = np.zeros(len(recs))
models = []
if a.cv:
    folds = np.arange(len(recs)) % a.cv
    for k in range(a.cv):
        m = fit_calibration([r for r, f in zip(recs, folds) if f != k], BaselineModel(drop=a.drop))
        models.append(m)
        for i in np.where(folds == k)[0]:
            pred[i] = analyze_baseline(recs[i], m).ahi_est or 0.0
else:
    m = BaselineModel(drop=a.drop)  # raw ODI, no calibration
    for i, r in enumerate(recs):
        pred[i] = analyze_baseline(r, m).ahi_est or 0.0

true = np.array([r.labels.ahi for r in recs])
ev = [match_events([(e.start, e.end) for e in r.labels.events],
                   [(e.start, e.end) for e in analyze_baseline(r, BaselineModel(drop=a.drop)).events])
      for r in recs]
report = {
    "n": len(recs),
    "event_f1_mean": float(np.mean([e["f1"] for e in ev])),
    "ahi": ahi_agreement(true, pred),
    "severity": severity_metrics(true, pred),
    "screen_15": screening(true, pred, 15),
    "screen_30": screening(true, pred, 30),
}
print(json.dumps(report, indent=2))
