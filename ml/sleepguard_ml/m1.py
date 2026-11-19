"""M1: XGBoost on SpO2 / pulse-rate features.

Stage 1 (window model): is there a breathing event in this 60 s window?
Stage 2 (night model): AHI from night features + stage-1 outputs. This stage can learn
things a fixed rule cannot, for example that long wake periods lower the true AHI.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd

from .features import WIN, night_features, window_features, window_labels
from .io import to_arrays
from .preprocess import clean, valid_hours
from .schemas import AnalysisResult, DetectedEvent, Quality, Recording
from .summary import action_from_severity, gap_warnings, severity_from_ahi, spo2_stats

WINDOW_PARAMS = dict(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                     colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1)
NIGHT_PARAMS = dict(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.8, objective="reg:squarederror", n_jobs=-1)


def prepare(rec: Recording, prefilter: int = 0):
    """Clean signals; optional centred rolling median over `prefilter` seconds to remove noise spikes."""
    spo2, hr = clean(*to_arrays(rec))
    if prefilter and prefilter > 1:
        gap = ~np.isfinite(spo2)
        spo2 = pd.Series(spo2).rolling(prefilter, center=True, min_periods=1).median().to_numpy(copy=True)
        spo2[gap] = np.nan  # keep gaps as gaps
    return spo2, hr


def recording_tables(rec: Recording, prefilter: int = 0):
    spo2, hr = prepare(rec, prefilter)
    wf = window_features(spo2, hr)
    events = [(e.start, e.end) for e in rec.labels.events] if rec.labels else []
    y = window_labels(wf["start"].to_numpy(), len(spo2), events)
    return spo2, hr, wf, y


def merged_events(starts: np.ndarray, prob: np.ndarray, thr: float) -> List[tuple]:
    out = []
    for s, p in zip(starts, prob):
        if p < thr:
            continue
        s, e = int(s), int(s) + WIN
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], e, max(out[-1][2], float(p)))
        else:
            out.append((s, e, float(p)))
    return out


def stage2_row(spo2, hr, wf: pd.DataFrame, prob: np.ndarray, thr: float) -> dict:
    vh = valid_hours(spo2)
    row = night_features(spo2, hr, vh)
    ok = wf["valid"].to_numpy() >= 0.5
    row["prob_mean"] = float(prob[ok].mean()) if ok.any() else 0.0
    row["prob_sum_h"] = float(prob[ok].sum() / max(vh, 1e-6))
    row["frac_pos"] = float((prob[ok] >= thr).mean()) if ok.any() else 0.0
    row["ev_rate"] = len(merged_events(wf["start"].to_numpy(), prob, thr)) / max(vh, 1e-6)
    return row


@dataclass
class M1Model:
    window_model: object = None
    night_model: object = None
    window_cols: List[str] = field(default_factory=list)
    night_cols: List[str] = field(default_factory=list)
    threshold: float = 0.5
    version: str = "sg-m1-0.2.0"
    calib: tuple = (1.0, 0.0)  # AHI = a * raw + b, fitted on out-of-fold night predictions
    prefilter: int = 0  # seconds of rolling median applied to SpO2 before features (0 = none)

    def save(self, folder) -> None:
        f = Path(folder)
        f.mkdir(parents=True, exist_ok=True)
        self.window_model.save_model(str(f / "window.json"))
        self.night_model.save_model(str(f / "night.json"))
        (f / "meta.json").write_text(json.dumps({
            "window_cols": self.window_cols, "night_cols": self.night_cols,
            "threshold": self.threshold, "version": self.version,
            "calib": list(self.calib), "prefilter": self.prefilter}, indent=2))

    @classmethod
    def load(cls, folder) -> "M1Model":
        from xgboost import XGBClassifier, XGBRegressor
        f = Path(folder)
        meta = json.loads((f / "meta.json").read_text())
        wm, nm = XGBClassifier(), XGBRegressor()
        wm.load_model(str(f / "window.json"))
        nm.load_model(str(f / "night.json"))
        return cls(wm, nm, meta["window_cols"], meta["night_cols"], meta["threshold"], meta["version"],
                   tuple(meta.get("calib", (1.0, 0.0))), int(meta.get("prefilter", 0)))


def train_m1(recs: Sequence[Recording], n_folds: int = 5, seed: int = 0, verbose: bool = True,
             drop_night_cols: Sequence[str] = (), augment: int = 0, prefilter: int = 0,
             drop_window_cols: Sequence[str] = ()) -> M1Model:
    """Window model on all train windows; night model on out-of-fold window probabilities
    (so stage 2 never sees probabilities from a model that trained on the same night).

    augment=k adds k cheap-sensor copies of every night (noise, bias, averaging, dropouts).
    Copies of a night always sit in the same fold as the original, so nothing leaks."""
    from xgboost import XGBClassifier, XGBRegressor

    from .augment import perturb, random_cheap_sensor

    n0 = len(recs)
    group = list(range(n0))
    if augment:
        extra = [perturb(r, random_cheap_sensor, seed * 100003 + 1000 * j + i)
                 for j in range(augment) for i, r in enumerate(recs)]
        recs = list(recs) + extra
        group += [i for _ in range(augment) for i in range(n0)]
        if verbose:
            print(f"  augmented: {n0} nights + {len(extra)} cheap-sensor copies")
    group = np.array(group)

    tabs = [recording_tables(r, prefilter) for r in recs]
    cols = [c for c in tabs[0][2].columns if c != "start" and c not in set(drop_window_cols)]
    rng = np.random.default_rng(seed)
    folds = (rng.permutation(n0) % n_folds)[group]

    oof = [None] * len(recs)
    for k in range(n_folds):
        tr = [i for i in range(len(recs)) if folds[i] != k]
        X = pd.concat([tabs[i][2][cols] for i in tr])
        y = np.concatenate([tabs[i][3] for i in tr])
        m = XGBClassifier(**WINDOW_PARAMS, random_state=seed)
        m.fit(X, y)
        for i in range(len(recs)):
            if folds[i] == k:
                oof[i] = m.predict_proba(tabs[i][2][cols])[:, 1]
        if verbose:
            print(f"  window model fold {k + 1}/{n_folds} done")

    X = pd.concat([t[2][cols] for t in tabs])
    y = np.concatenate([t[3] for t in tabs])
    wm = XGBClassifier(**WINDOW_PARAMS, random_state=seed)
    wm.fit(X, y)

    thr = 0.5
    night = pd.DataFrame([stage2_row(t[0], t[1], t[2], p, thr) for t, p in zip(tabs, oof)])
    ahi = np.array([r.labels.ahi for r in recs], float)
    ncols = [c for c in night.columns if c not in set(drop_night_cols)]
    # Calibration: a model trained on log(AHI) shrinks predictions toward the middle, so it
    # under-predicts severe nights. Fit a straight line from out-of-fold night predictions to true AHI.
    raw_oof = np.zeros(len(recs))
    for k in range(n_folds):
        tr, te = folds != k, folds == k
        m = XGBRegressor(**NIGHT_PARAMS, random_state=seed)
        m.fit(night.loc[tr, ncols], np.log1p(ahi[tr]))
        raw_oof[te] = np.expm1(m.predict(night.loc[te, ncols]))
    a_, b_ = np.polyfit(raw_oof, ahi, 1)
    if verbose:
        print(f"  calibration: AHI = {a_:.2f} * raw + {b_:.2f}")

    nm = XGBRegressor(**NIGHT_PARAMS, random_state=seed)
    nm.fit(night[ncols], np.log1p(ahi))  # log scale: AHI is very skewed
    return M1Model(wm, nm, cols, ncols, thr, version="sg-m1-0.3.0" if (augment or prefilter) else "sg-m1-0.2.0",
                   calib=(float(a_), float(b_)), prefilter=prefilter)


def analyze_m1(rec: Recording, model: M1Model, calibrated: bool = True) -> AnalysisResult:
    spo2, hr = prepare(rec, model.prefilter)
    vh = valid_hours(spo2)
    wf = window_features(spo2, hr)
    prob = model.window_model.predict_proba(wf[model.window_cols])[:, 1]
    row = pd.DataFrame([stage2_row(spo2, hr, wf, prob, model.threshold)])[model.night_cols]
    ahi = None
    if vh > 0:
        ahi = float(np.expm1(model.night_model.predict(row)[0]))
        if calibrated:
            ahi = model.calib[0] * ahi + model.calib[1]
        ahi = max(ahi, 0.0)
    sev = severity_from_ahi(ahi, vh)
    ev = merged_events(wf["start"].to_numpy(), prob, model.threshold)
    return AnalysisResult(
        recording_id=rec.recording_id, model_version=model.version,
        valid_hours=round(vh, 2), ahi_est=None if ahi is None else round(ahi, 1),
        severity=sev, action=action_from_severity(sev, ahi),
        odi3=round(float(row["odi3"].iloc[0]), 1) if "odi3" in row else None,
        events=[DetectedEvent(start=s, end=e, prob=round(p, 3)) for s, e, p in ev],
        quality=Quality(valid_fraction=round(float(np.isfinite(spo2).mean()) if spo2.size else 0.0, 3),
                        warnings=gap_warnings(spo2, rec.start_time)),
        **spo2_stats(spo2),
    )
