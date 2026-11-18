"""Shared data models (interfaces I2 and I3).

Recording      = one night in SGR v1 format (1 Hz SpO2 + pulse rate).
AnalysisResult = what the backend stores and the app / dashboard show.
The backend imports these same classes, so keep changes in sync with docs/interfaces/.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

SourceType = Literal["sleepguard_clip", "import_oximeter", "shhs", "mesa", "ucd", "synthetic"]
EventType = Literal["obstructive_apnea", "central_apnea", "mixed_apnea", "hypopnea", "other"]
Severity = Literal["normal", "mild", "moderate", "severe", "insufficient_data"]
Action = Literal["no_referral", "consider_referral", "refer_sleep_lab", "repeat_recording"]


class Source(BaseModel):
    type: SourceType
    device_id: Optional[str] = None
    fw: Optional[str] = None
    original_id: Optional[str] = None  # dataset record name; keep out of public outputs


class Event(BaseModel):
    start: int = Field(ge=0, description="seconds from recording start")
    end: int = Field(ge=0)
    type: EventType = "other"

    @model_validator(mode="after")
    def _check(self):
        if self.end < self.start:
            raise ValueError("event end < start")
        return self


class Labels(BaseModel):
    events: List[Event] = []
    ahi: Optional[float] = None
    ahi_definition: Optional[str] = None


class Recording(BaseModel):
    format: Literal["SGR"] = "SGR"
    version: int = 1
    recording_id: str
    source: Source
    start_time: Optional[datetime] = None
    fs: int = 1
    spo2: List[Optional[int]]
    hr: List[Optional[int]]
    quality: Optional[List[Optional[int]]] = None
    labels: Optional[Labels] = None

    @model_validator(mode="after")
    def _check(self):
        if self.fs != 1:
            raise ValueError("SGR v1 requires fs = 1")
        n = len(self.spo2)
        if len(self.hr) != n:
            raise ValueError("spo2 and hr must have the same length")
        if self.quality is not None and len(self.quality) != n:
            raise ValueError("quality must have the same length as spo2")
        return self

    @property
    def n_seconds(self) -> int:
        return len(self.spo2)


class DetectedEvent(BaseModel):
    start: int
    end: int
    prob: float = 1.0


class Quality(BaseModel):
    valid_fraction: float
    warnings: List[str] = []


class AnalysisResult(BaseModel):
    recording_id: str
    model_version: str
    valid_hours: float
    ahi_est: Optional[float]
    severity: Severity
    action: Action
    odi3: Optional[float] = None
    spo2_mean: Optional[float] = None
    spo2_min: Optional[float] = None
    t90_min: Optional[float] = None
    events: List[DetectedEvent] = []
    quality: Quality
