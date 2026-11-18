"""Severity, action, summary statistics and quality warnings."""
from __future__ import annotations

from datetime import timedelta
from typing import List, Optional

import numpy as np

MIN_VALID_HOURS = 4.0
CUTOFFS = (5.0, 15.0, 30.0)  # mild, moderate, severe
GAP_WARNING_SECONDS = 600


def severity_from_ahi(ahi: Optional[float], valid_hours: float) -> str:
    if ahi is None or valid_hours < MIN_VALID_HOURS:
        return "insufficient_data"
    if ahi < CUTOFFS[0]:
        return "normal"
    if ahi < CUTOFFS[1]:
        return "mild"
    if ahi < CUTOFFS[2]:
        return "moderate"
    return "severe"


def action_from_severity(severity: str) -> str:
    return {
        "insufficient_data": "repeat_recording",
        "normal": "no_referral",
        "mild": "consider_referral",
        "moderate": "refer_sleep_lab",
        "severe": "refer_sleep_lab",
    }[severity]


def spo2_stats(spo2: np.ndarray) -> dict:
    v = spo2[np.isfinite(spo2)]
    if v.size == 0:
        return {"spo2_mean": None, "spo2_min": None, "t90_min": None}
    return {
        "spo2_mean": round(float(v.mean()), 1),
        "spo2_min": float(v.min()),
        "t90_min": round(float((v < 90).sum()) / 60.0, 1),
    }


def _fmt(sec: int, start_time) -> str:
    if start_time is not None:
        return (start_time + timedelta(seconds=int(sec))).strftime("%H:%M")
    h, m = divmod(int(sec) // 60, 60)
    return f"+{h:02d}:{m:02d}"


def gap_warnings(spo2: np.ndarray, start_time=None) -> List[str]:
    warnings, invalid, n, i = [], ~np.isfinite(spo2), len(spo2), 0
    while i < n:
        if invalid[i]:
            j = i
            while j + 1 < n and invalid[j + 1]:
                j += 1
            if j - i + 1 >= GAP_WARNING_SECONDS:
                warnings.append(f"low signal {_fmt(i, start_time)} to {_fmt(j + 1, start_time)}")
            i = j + 1
        else:
            i += 1
    return warnings
