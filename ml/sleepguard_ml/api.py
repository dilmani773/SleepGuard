"""Interface I4: what the backend calls."""
from __future__ import annotations

from pathlib import Path
from typing import Union

from .baseline import BaselineModel, analyze_baseline
from .m1 import M1Model, analyze_m1
from .schemas import AnalysisResult, Recording

Model = Union[BaselineModel, M1Model]


def load_model(path: str | None = None) -> Model:
    """None -> default M0. *.json -> saved M0 config. Folder with meta.json -> M1. *.onnx -> M2 (later)."""
    if path is None:
        return BaselineModel()
    p = Path(path)
    if p.is_dir() and (p / "meta.json").exists():
        return M1Model.load(p)
    if p.suffix == ".json":
        return BaselineModel.load(p)
    if p.suffix == ".onnx":
        raise NotImplementedError("M2 ONNX loader comes in Week 4")
    raise ValueError(f"unknown model file: {p.name}")


def analyze(rec: Recording, model: Model | None = None) -> AnalysisResult:
    model = model or BaselineModel()
    if isinstance(model, BaselineModel):
        return analyze_baseline(rec, model)
    if isinstance(model, M1Model):
        return analyze_m1(rec, model)
    raise TypeError(f"unsupported model: {type(model).__name__}")
