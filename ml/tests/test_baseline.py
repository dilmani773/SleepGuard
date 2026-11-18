import numpy as np

from sleepguard_ml import analyze, analyze_baseline, load_model
from sleepguard_ml.baseline import BaselineModel, detect_desaturations, fit_calibration
from sleepguard_ml.synthetic import make_night


def test_single_dip_detected():
    spo2 = np.full(1200, 97.0)
    spo2[600:630] = 92.0
    ev = detect_desaturations(spo2)
    assert len(ev) == 1 and 595 <= ev[0][0] <= 605


def test_short_dip_ignored():
    spo2 = np.full(1200, 97.0)
    spo2[600:605] = 92.0
    assert detect_desaturations(spo2) == []


def test_severity_order():
    ahis = [analyze_baseline(make_night(s, hours=6, seed=7, dropouts=False)).ahi_est
            for s in ["normal", "mild", "moderate", "severe"]]
    assert ahis == sorted(ahis)


def test_insufficient_data():
    rec = make_night("moderate", hours=2, seed=3)
    res = analyze_baseline(rec)
    assert res.severity == "insufficient_data" and res.action == "repeat_recording"


def test_calibration_and_save(tmp_path):
    recs = [make_night(s, hours=5, seed=i) for i, s in enumerate(["normal", "mild", "moderate", "severe"] * 2)]
    m = fit_calibration(recs)
    m.save(tmp_path / "m0.json")
    m2 = load_model(str(tmp_path / "m0.json"))
    assert isinstance(m2, BaselineModel) and abs(m2.slope - m.slope) < 1e-9
    res = analyze(recs[0], m2)
    assert res.model_version.startswith("sg-m0")
