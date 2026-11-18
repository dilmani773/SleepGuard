"""Synthetic nights with known labels.

Used to develop and test the app, backend, dashboard and pipeline before real
data arrives. NOT for training the final model or for any reported result.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import numpy as np

from .io import from_arrays
from .schemas import Event, Labels, Source

AHI_RANGES = {"normal": (0, 4), "mild": (5, 14), "moderate": (15, 29), "severe": (30, 60)}
SL_TZ = timezone(timedelta(hours=5, minutes=30))


def _add_dip(spo2, hr, start, end, depth, rng):
    """SpO2 falls after a lag, reaches its lowest point after the event ends, then recovers."""
    n = len(spo2)
    lag = int(rng.integers(8, 15))
    fall_start, nadir = start + lag, end + int(rng.integers(5, 12))
    recover = nadir + int(rng.integers(8, 16))
    for t in range(fall_start, min(recover, n)):
        if t <= nadir:
            f = (t - fall_start) / max(nadir - fall_start, 1)
        else:
            f = 1 - (t - nadir) / max(recover - nadir, 1)
        spo2[t] -= depth * f
    # pulse rises when breathing restarts
    rise = rng.uniform(5, 15)
    for t in range(end, min(end + 30, n)):
        k = t - end
        hr[t] += rise * (k / 8 if k < 8 else np.exp(-(k - 8) / 8))


def make_night(severity: str = "moderate", hours: float = 7.0, seed: int | None = None,
               ahi: float | None = None, dropouts: bool = True):
    rng = np.random.default_rng(seed)
    n = int(hours * 3600)
    t = np.arange(n)

    base = rng.uniform(95, 98)
    spo2 = base + 0.4 * np.sin(2 * np.pi * t / rng.uniform(1800, 5400))
    hr = rng.uniform(55, 75) + 2 * np.sin(2 * np.pi * t / rng.uniform(1200, 3600))

    if ahi is None:
        lo, hi = AHI_RANGES[severity]
        ahi = rng.uniform(lo, hi)
    n_events = int(round(ahi * hours))

    events, cursor = [], int(rng.integers(300, 900))
    mean_gap = (n - 600 - cursor) / max(n_events, 1)
    for _ in range(n_events):
        dur = int(rng.integers(10, 40))
        if cursor + dur + 60 >= n:
            break
        kind = rng.choice(["obstructive_apnea", "hypopnea", "central_apnea"], p=[0.5, 0.45, 0.05])
        depth = rng.uniform(3, 5) if kind == "hypopnea" else rng.uniform(4, 10)
        _add_dip(spo2, hr, cursor, cursor + dur, depth, rng)
        events.append(Event(start=cursor, end=cursor + dur, type=str(kind)))
        cursor += dur + max(45, int(rng.exponential(max(mean_gap - dur, 15))))

    spo2 = np.minimum(spo2 + rng.normal(0, 0.3, n), 100)
    hr = hr + rng.normal(0, 1.0, n)

    if dropouts:
        for _ in range(int(rng.integers(2, 8))):
            s = int(rng.integers(0, n - 60))
            spo2[s:s + int(rng.integers(5, 60))] = np.nan
        if rng.random() < 0.2:  # one long gap, as when the clip falls off
            s = int(rng.integers(0, n - 900))
            spo2[s:s + 720] = np.nan
    hr[~np.isfinite(spo2)] = np.nan

    true_ahi = len(events) / hours
    start = datetime(2026, 11, 20, 22, 30, tzinfo=SL_TZ)
    return from_arrays(
        recording_id=f"synthetic-{severity}-{uuid.uuid4().hex[:8]}",
        source=Source(type="synthetic"),
        spo2=spo2, hr=hr, start_time=start,
        labels=Labels(events=events, ahi=round(true_ahi, 1), ahi_definition="synthetic"),
    )
