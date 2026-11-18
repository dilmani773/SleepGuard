"""Signal cleaning shared by every model and every data source."""
from __future__ import annotations

import numpy as np

SPO2_MIN, SPO2_MAX = 50, 100
HR_MIN, HR_MAX = 30, 200
FLAT_SECONDS = 10


def resample_to_1hz(signal: np.ndarray, fs: float) -> np.ndarray:
    """Median of each 1 second block. Works for fs >= 1 (integer or not)."""
    signal = np.asarray(signal, dtype=float)
    if fs == 1:
        return signal.copy()
    n_sec = int(len(signal) // fs)
    out = np.full(n_sec, np.nan)
    for s in range(n_sec):
        block = signal[int(round(s * fs)):int(round((s + 1) * fs))]
        block = block[np.isfinite(block)]
        if block.size:
            out[s] = np.median(block)
    return out


def _mask_flat(x: np.ndarray, min_len: int) -> np.ndarray:
    """Mark runs of identical values longer than min_len (sensor stuck / detached)."""
    x = x.copy()
    n = len(x)
    i = 0
    while i < n:
        if not np.isfinite(x[i]):
            i += 1
            continue
        j = i
        while j + 1 < n and x[j + 1] == x[i]:
            j += 1
        if j - i + 1 > min_len:
            x[i:j + 1] = np.nan
        i = j + 1
    return x


def clean(spo2: np.ndarray, hr: np.ndarray, flat_check: bool = False):
    """Out-of-range -> NaN. Flat-line check is optional: healthy SpO2 is often flat."""
    spo2 = np.asarray(spo2, dtype=float).copy()
    hr = np.asarray(hr, dtype=float).copy()
    spo2[(spo2 < SPO2_MIN) | (spo2 > SPO2_MAX)] = np.nan
    hr[(hr < HR_MIN) | (hr > HR_MAX)] = np.nan
    if flat_check:
        hr = _mask_flat(hr, 60)  # HR rarely stays exactly constant for 60 s
    return spo2, hr


def valid_mask(spo2: np.ndarray) -> np.ndarray:
    return np.isfinite(spo2)


def valid_hours(spo2: np.ndarray) -> float:
    return float(np.isfinite(spo2).sum()) / 3600.0
