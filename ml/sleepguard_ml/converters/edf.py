"""Shared EDF helpers (needs: pip install pyedflib)."""
from __future__ import annotations

from datetime import datetime
from typing import Iterable, Optional, Tuple

import numpy as np

from ..preprocess import resample_to_1hz

SPO2_ALIASES = ["sao2", "spo2", "osat", "sat", "oxygen saturation"]
HR_ALIASES = ["h.r.", "hr", "pr", "pulse", "pulse rate", "heart rate"]


def _find(labels: list, aliases: Iterable[str]) -> Optional[int]:
    norm = [l.strip().lower() for l in labels]
    for a in aliases:  # exact match first
        if a in norm:
            return norm.index(a)
    for a in aliases:  # then contains
        for i, l in enumerate(norm):
            if a in l:
                return i
    return None


def read_spo2_hr(path: str) -> Tuple[np.ndarray, np.ndarray, Optional[datetime], dict]:
    """Returns 1 Hz SpO2, 1 Hz HR (NaN if no HR channel), start time, info."""
    import pyedflib

    f = pyedflib.EdfReader(str(path))
    try:
        labels = f.getSignalLabels()
        i_spo2 = _find(labels, SPO2_ALIASES)
        if i_spo2 is None:
            raise ValueError(f"no SpO2 channel in {path}. Channels: {labels}")
        i_hr = _find(labels, HR_ALIASES)
        fs_spo2 = f.getSampleFrequency(i_spo2)
        spo2 = resample_to_1hz(f.readSignal(i_spo2), fs_spo2)
        if i_hr is not None:
            hr = resample_to_1hz(f.readSignal(i_hr), f.getSampleFrequency(i_hr))
        else:
            hr = np.full_like(spo2, np.nan)
        n = min(len(spo2), len(hr))
        info = {"spo2_label": labels[i_spo2], "spo2_fs": fs_spo2,
                "hr_label": None if i_hr is None else labels[i_hr]}
        return spo2[:n], hr[:n], f.getStartdatetime(), info
    finally:
        f.close()
