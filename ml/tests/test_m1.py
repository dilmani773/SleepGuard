import numpy as np
import pytest

from sleepguard_ml import analyze, load_model
from sleepguard_ml.features import window_labels
from sleepguard_ml.splits import make_split
from sleepguard_ml.synthetic import make_night

xgboost = pytest.importorskip("xgboost")
from sleepguard_ml.m1 import train_m1  # noqa: E402


def test_window_labels():
    y = window_labels(np.array([0, 30, 60, 90]), 200, [(70, 85)])
    assert y.tolist() == [0, 1, 1, 0]


def test_split_is_stratified_and_disjoint():
    ids = [f"r{i}" for i in range(100)]
    ahi = [2, 8, 20, 40] * 25
    s = make_split(ids, ahi)
    assert not (set(s["train"]) & set(s["test"])) and not (set(s["val"]) & set(s["test"]))
    assert len(s["train"]) + len(s["val"]) + len(s["test"]) == 100


def test_m1_train_save_load(tmp_path):
    recs = [make_night(sv, hours=5, seed=i) for i, sv in enumerate(["normal", "mild", "moderate", "severe"] * 3)]
    m = train_m1(recs, n_folds=3, verbose=False)
    m.save(tmp_path / "m1")
    m2 = load_model(str(tmp_path / "m1"))
    res = analyze(make_night("severe", hours=5, seed=99), m2)
    low = analyze(make_night("normal", hours=5, seed=98), m2)
    assert res.model_version.startswith("sg-m1") and res.ahi_est > low.ahi_est


def test_hb_and_odi25_rise_with_severity():
    from sleepguard_ml.features import night_features
    from sleepguard_ml.io import to_arrays
    from sleepguard_ml.preprocess import clean, valid_hours

    def feats(sv):
        s, h = clean(*to_arrays(make_night(sv, hours=5, seed=1)))
        return night_features(s, h, valid_hours(s))

    lo, hi = feats("normal"), feats("severe")
    assert hi["hb"] > lo["hb"] >= 0 and hi["odi25"] > lo["odi25"]


def test_calibration_saved(tmp_path):
    recs = [make_night(sv, hours=4, seed=i) for i, sv in enumerate(["normal", "mild", "moderate", "severe"] * 3)]
    m = train_m1(recs, n_folds=3, verbose=False)
    m.save(tmp_path / "m1")
    assert load_model(str(tmp_path / "m1")).calib == m.calib
