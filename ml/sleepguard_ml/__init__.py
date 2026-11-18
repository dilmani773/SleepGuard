"""SleepGuard ML package.

Public API (interface I4):
    from sleepguard_ml import load_model, analyze, analyze_baseline
"""
from .schemas import Recording, Labels, Event, AnalysisResult, DetectedEvent, Quality
from .baseline import analyze_baseline, BaselineModel
from .api import load_model, analyze

__version__ = "0.1.0"

__all__ = [
    "Recording", "Labels", "Event", "AnalysisResult", "DetectedEvent", "Quality",
    "analyze_baseline", "BaselineModel", "load_model", "analyze",
]
