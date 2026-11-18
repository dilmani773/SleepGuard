"""M0: rule-based oxygen desaturation index (ODI) baseline.

Event = SpO2 at least `drop` points below the recent baseline for at least
`min_dur` seconds. Baseline = 90th percentile of the previous `window` seconds,
so earlier dips do not pull it down much.
AHI estimate = slope * ODI + intercept, fitted on the training set.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

from .io import to_arrays
from .preprocess import clean, valid_hours
from .schemas import AnalysisResult, DetectedEvent, Quality, Recording
from .summary import action_from_severity, gap_warnings, severity_from_ahi, spo2_stats


@dataclass
class BaselineModel:
    drop: float = 3.0
    min_dur: int = 10
    max_dur: int = 180
    window: int = 120
    merge_gap: int = 3
    slope: float = 1.0
    intercept: float = 0.0
    version: str = "sg-m0-0.1.0"

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "BaselineModel":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def rolling_baseline(spo2: np.ndarray, window: int = 120) -> np.ndarray:
    s = pd.Series(spo2)
    base = s.rolling(window, min_periods=window // 4).quantile(0.9).shift(1)
    return base.to_numpy()


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """[start, end) index pairs of True runs."""
    if mask.size == 0:
        return []
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def detect_desaturations(spo2: np.ndarray, model: BaselineModel | None = None) -> List[Tuple[int, int]]:
    m = model or BaselineModel()
    base = rolling_baseline(spo2, m.window)
    below = np.isfinite(spo2) & np.isfinite(base) & (spo2 <= base - m.drop)
    runs = _runs(below)
    merged: List[List[int]] = []
    for s, e in runs:
        if merged and s - merged[-1][1] <= m.merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged if m.min_dur <= e - s <= m.max_dur]


def odi(spo2: np.ndarray, model: BaselineModel | None = None) -> float | None:
    vh = valid_hours(spo2)
    if vh == 0:
        return None
    return len(detect_desaturations(spo2, model)) / vh


def fit_calibration(recordings: Iterable[Recording], model: BaselineModel | None = None) -> BaselineModel:
    """Least squares fit of AHI = slope * ODI + intercept on labelled recordings."""
    m = model or BaselineModel()
    xs, ys = [], []
    for rec in recordings:
        if rec.labels is None or rec.labels.ahi is None:
            continue
        spo2, hr = clean(*to_arrays(rec))
        o = odi(spo2, m)
        if o is not None:
            xs.append(o)
            ys.append(rec.labels.ahi)
    if len(xs) < 2:
        raise ValueError("need at least 2 labelled recordings to calibrate")
    slope, intercept = np.polyfit(np.array(xs), np.array(ys), 1)
    m.slope, m.intercept = float(slope), float(intercept)
    return m


def analyze_baseline(rec: Recording, model: BaselineModel | None = None) -> AnalysisResult:
    m = model or BaselineModel()
    spo2, hr = clean(*to_arrays(rec))
    vh = valid_hours(spo2)
    events = detect_desaturations(spo2, m)
    odi3 = len(events) / vh if vh > 0 else None
    ahi = None if odi3 is None else max(0.0, m.slope * odi3 + m.intercept)
    sev = severity_from_ahi(ahi, vh)
    return AnalysisResult(
        recording_id=rec.recording_id,
        model_version=m.version,
        valid_hours=round(vh, 2),
        ahi_est=None if ahi is None else round(ahi, 1),
        severity=sev,
        action=action_from_severity(sev),
        odi3=None if odi3 is None else round(odi3, 1),
        events=[DetectedEvent(start=int(s), end=int(e), prob=1.0) for s, e in events],
        quality=Quality(valid_fraction=round(float(np.isfinite(spo2).mean()) if spo2.size else 0.0, 3),
                        warnings=gap_warnings(spo2, rec.start_time)),
        **spo2_stats(spo2),
    )
