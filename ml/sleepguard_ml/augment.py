"""Cheap-sensor problems, used for (1) robustness tests and (2) noise-augmented training.

The conditions match scripts/robustness.py exactly, so CV results and the final stress test agree.
"""
from __future__ import annotations

import numpy as np

from .io import from_arrays, to_arrays
from .schemas import Recording


def gaps(n, frac, rng, lo=30, hi=120):
    m = np.zeros(n, bool)
    while m.mean() < frac:
        L = int(rng.integers(lo, hi))
        s = int(rng.integers(0, max(n - L, 1)))
        m[s:s + L] = True
    return m


def smooth(x, k):
    ok = np.isfinite(x)
    num = np.convolve(np.where(ok, x, 0.0), np.ones(k), "same")
    den = np.convolve(ok.astype(float), np.ones(k), "same")
    out = num / np.maximum(den, 1)
    out[~ok] = np.nan
    return out


def drop(s, h, frac, r):
    m = gaps(len(s), frac, r)
    s, h = s.copy(), h.copy()
    s[m] = np.nan
    h[m] = np.nan
    return s, h


def combo(s, h, r):
    """per-night bias in [-2, 2], SD 1% noise, 4 s averaging, 5% dropouts"""
    s = smooth(s + r.uniform(-2, 2) + r.normal(0, 1, len(s)), 4)
    h = smooth(h + r.normal(0, 2, len(h)), 4)
    return drop(s, h, 0.05, r)


CONDITIONS = {
    "clean": lambda s, h, r: (s, h),
    "raw noise SD 1%": lambda s, h, r: (s + r.normal(0, 1, len(s)), h + r.normal(0, 2, len(h))),
    "noise SD 1% + 4 s avg": lambda s, h, r: (smooth(s + r.normal(0, 1, len(s)), 4),
                                              smooth(h + r.normal(0, 2, len(h)), 4)),
    "realistic MAX30102 combo": combo,
}


def random_cheap_sensor(s, h, r):
    """A random cheap-sensor version for training: noise SD 0.5 to 2%, sometimes averaged,
    bias -2..2%, 0 to 10% dropouts."""
    sd = r.choice([0.5, 1.0, 1.5, 2.0])
    s = s + r.uniform(-2, 2) + r.normal(0, sd, len(s))
    h = h + r.normal(0, 2 * sd, len(h))
    if r.random() < 0.5:
        s, h = smooth(s, int(r.choice([3, 4, 6]))), smooth(h, 4)
    return drop(s, h, float(r.uniform(0, 0.10)), r)


def perturb(rec: Recording, fn, seed: int) -> Recording:
    s, h = to_arrays(rec)
    s, h = fn(s, h, np.random.default_rng(seed))
    return from_arrays(rec.recording_id, rec.source, np.clip(s, 0, 100), h, labels=rec.labels,
                       start_time=rec.start_time)
