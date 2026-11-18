import numpy as np

from sleepguard_ml.metrics import ahi_agreement, match_events, screening, severity_metrics


def test_match_events():
    r = match_events([(100, 120), (500, 520)], [(115, 140), (900, 920)])
    assert r["tp"] == 1 and r["fp"] == 1 and r["fn"] == 1


def test_perfect_agreement():
    t = np.array([2, 8, 20, 40.0])
    a = ahi_agreement(t, t)
    assert a["mae"] == 0 and abs(a["icc"] - 1) < 1e-9
    assert severity_metrics(t, t)["accuracy"] == 1.0
    s = screening(t, t, 15)
    assert s["sensitivity"] == 1.0 and s["specificity"] == 1.0
