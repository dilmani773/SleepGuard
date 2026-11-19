"""Window and night features from 1 Hz SpO2 + pulse rate (used by M1)."""
from __future__ import annotations

import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd

from .baseline import BaselineModel, detect_desaturations, rolling_baseline

WIN, STEP = 60, 30  # seconds


def _nanstat(f, x, default=np.nan):
    x = x[np.isfinite(x)]
    return f(x) if x.size else default


def window_starts(n: int, win: int = WIN, step: int = STEP) -> np.ndarray:
    return np.arange(0, max(n - win + 1, 1), step)


def window_features(spo2: np.ndarray, hr: np.ndarray, win: int = WIN, step: int = STEP) -> pd.DataFrame:
    """One row per window. Context: the baseline uses the 2 minutes before each sample."""
    n = len(spo2)
    base = rolling_baseline(spo2, 120)
    drop = base - spo2  # positive = below recent baseline
    m3 = BaselineModel(drop=3.0)
    m4 = BaselineModel(drop=4.0)
    d3 = np.zeros(n, bool)
    d4 = np.zeros(n, bool)
    for s, e in detect_desaturations(spo2, m3):
        d3[s:e] = True
    for s, e in detect_desaturations(spo2, m4):
        d4[s:e] = True
    hr_base = pd.Series(hr).rolling(120, min_periods=30).median().shift(1).to_numpy()
    hr_rise = hr - hr_base

    rows = []
    for st in window_starts(n, win, step):
        sl = slice(st, st + win)
        s, d, h, hrr = spo2[sl], drop[sl], hr[sl], hr_rise[sl]
        valid = np.isfinite(s).mean()
        rows.append({
            "start": int(st),
            "valid": valid,
            "spo2_mean": _nanstat(np.mean, s),
            "spo2_min": _nanstat(np.min, s),
            "spo2_range": _nanstat(np.ptp, s),
            "spo2_std": _nanstat(np.std, s),
            "slope_min": _nanstat(np.min, np.diff(s)) if len(s) > 1 else np.nan,
            "slope_max": _nanstat(np.max, np.diff(s)) if len(s) > 1 else np.nan,
            "drop_max": _nanstat(np.max, d),
            "drop_mean": _nanstat(np.mean, d),
            "below_3": np.nanmean(d >= 3) if np.isfinite(d).any() else np.nan,
            "frac_d3": d3[sl].mean(),
            "frac_d4": d4[sl].mean(),
            "t90": np.nanmean(s < 90) if np.isfinite(s).any() else np.nan,
            "hr_mean": _nanstat(np.mean, h),
            "hr_std": _nanstat(np.std, h),
            "hr_rise_max": _nanstat(np.max, hrr),
        })
    df = pd.DataFrame(rows)
    # neighbour context: SpO2 dips come 10 to 30 s after the breathing event
    for col in ["drop_max", "frac_d3", "hr_rise_max"]:
        df[f"{col}_next"] = df[col].shift(-1)
        df[f"{col}_prev"] = df[col].shift(1)
    return df


def window_labels(starts: np.ndarray, n: int, events: List[Tuple[int, int]],
                  win: int = WIN, min_overlap: int = 10) -> np.ndarray:
    mask = np.zeros(n + win, bool)
    for s, e in events:
        mask[max(s, 0):max(e, 0)] = True
    cum = np.concatenate([[0], np.cumsum(mask)])
    overlap = cum[np.minimum(starts + win, len(mask))] - cum[starts]
    return (overlap >= min_overlap).astype(int)


def delta_index(spo2: np.ndarray, block: int = 12) -> float:
    m = len(spo2) // block
    if m < 2:
        return np.nan
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        b = np.nanmean(spo2[: m * block].reshape(m, block), axis=1)
        return float(np.nanmean(np.abs(np.diff(b))))


def hypoxic_burden(spo2: np.ndarray, valid_h: float, drop: float = 3.0, recovery: int = 30) -> float:
    """Desaturation area per hour (%·min/h), after Azarbarzin 2019 / Qian 2026 (simplified):
    for each >= `drop`% desaturation, sum (baseline - SpO2) from event start to `recovery` s after it."""
    base = rolling_baseline(spo2, 120)
    deficit = np.clip(base - spo2, 0, None)
    deficit[~np.isfinite(deficit)] = 0.0
    area = 0.0
    last = 0
    for s, e in detect_desaturations(spo2, BaselineModel(drop=drop)):
        s = max(s, last)  # no double counting when events are close
        e = min(e + recovery, len(spo2))
        area += float(deficit[s:e].sum())
        last = e
    return area / 60.0 / max(valid_h, 1e-6)


def night_features(spo2: np.ndarray, hr: np.ndarray, valid_h: float) -> dict:
    v = spo2[np.isfinite(spo2)]
    h = hr[np.isfinite(hr)]
    vh = max(valid_h, 1e-6)
    return {
        "valid_h": valid_h,
        "odi3": len(detect_desaturations(spo2, BaselineModel(drop=3.0))) / vh,
        "odi4": len(detect_desaturations(spo2, BaselineModel(drop=4.0))) / vh,
        "odi25": len(detect_desaturations(spo2, BaselineModel(drop=2.5, min_dur=5))) / vh,  # Wu 2026
        "hb": hypoxic_burden(spo2, valid_h),  # Qian 2026
        "t90_frac": float((v < 90).mean()) if v.size else np.nan,
        "spo2_mean": float(v.mean()) if v.size else np.nan,
        "spo2_p5": float(np.percentile(v, 5)) if v.size else np.nan,
        "delta_index": delta_index(spo2),
        "hr_mean": float(h.mean()) if h.size else np.nan,
        "hr_std": float(h.std()) if h.size else np.nan,
    }
