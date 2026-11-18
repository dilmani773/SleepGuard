import pytest

from sleepguard_ml import Recording
from sleepguard_ml.io import load_json, load_parquet, save_json, save_parquet
from sleepguard_ml.synthetic import make_night


def test_json_roundtrip(tmp_path):
    rec = make_night("mild", hours=1, seed=1)
    save_json(rec, tmp_path / "a.json")
    back = load_json(tmp_path / "a.json")
    assert back.spo2 == rec.spo2 and back.labels.ahi == rec.labels.ahi


def test_parquet_roundtrip(tmp_path):
    rec = make_night("severe", hours=1, seed=2)
    save_parquet(rec, tmp_path / "a.parquet")
    back = load_parquet(tmp_path / "a.parquet")
    assert back.spo2 == rec.spo2 and back.hr == rec.hr
    assert len(back.labels.events) == len(rec.labels.events)


def test_length_mismatch_rejected():
    with pytest.raises(ValueError):
        Recording(recording_id="x", source={"type": "synthetic"}, spo2=[97, 97], hr=[60])


def test_fs_must_be_1():
    with pytest.raises(ValueError):
        Recording(recording_id="x", source={"type": "synthetic"}, fs=8, spo2=[97], hr=[60])
