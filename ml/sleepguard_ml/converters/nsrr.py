"""SHHS / MESA (NSRR): EDF + NSRR XML annotations -> SGR.

Run ONLY on the approved data user's secure machine. Output stays in data/ (git-ignored).
"""
from __future__ import annotations

import csv
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

from ..io import from_arrays
from ..preprocess import clean
from ..schemas import Event, Labels, Recording, Source
from .edf import read_spo2_hr


def map_concept(concept: str) -> Optional[str]:
    c = concept.lower()
    if "obstructive apnea" in c:
        return "obstructive_apnea"
    if "central apnea" in c:
        return "central_apnea"
    if "mixed apnea" in c:
        return "mixed_apnea"
    if "hypopnea" in c:
        return "hypopnea"
    return None  # desaturations, arousals, limb movements, "unsure" ... ignored


def _desaturations(root, min_drop: float) -> List[float]:
    """Start times of scored SpO2 desaturations with baseline - nadir >= min_drop."""
    starts = []
    for ev in root.iter("ScoredEvent"):
        if "desaturation" not in (ev.findtext("EventConcept") or "").lower():
            continue
        try:
            drop = float(ev.findtext("SpO2Baseline")) - float(ev.findtext("SpO2Nadir"))
        except (TypeError, ValueError):
            continue
        if drop >= min_drop:
            starts.append(float(ev.findtext("Start") or 0))
    return sorted(starts)


def parse_nsrr_xml(xml_path_or_text: str, n_seconds: Optional[int] = None,
                   hypopnea_desat: Optional[float] = 3.0, desat_window: float = 30.0) -> List[Event]:
    """Respiratory events from an NSRR XML file.

    Apneas are always kept. A hypopnea is kept only if a scored desaturation of at least
    `hypopnea_desat` % starts between the hypopnea start and `desat_window` s after its end.
    This matches the ahi_a0h3 definition (3%) or ahi_a0h4 (4%). Use None to keep all hypopneas.
    """
    text = xml_path_or_text
    if not text.lstrip().startswith("<"):
        text = Path(xml_path_or_text).read_text(encoding="utf-8", errors="ignore")
    root = ET.fromstring(text)
    desats = _desaturations(root, hypopnea_desat) if hypopnea_desat is not None else []
    events: List[Event] = []
    for ev in root.iter("ScoredEvent"):
        kind = map_concept(ev.findtext("EventConcept") or "")
        if kind is None:
            continue
        start = float(ev.findtext("Start") or 0)
        dur = float(ev.findtext("Duration") or 0)
        if kind == "hypopnea" and hypopnea_desat is not None:
            end_win = start + dur + desat_window
            if not any(start <= d <= end_win for d in desats):
                continue
        s, e = int(round(start)), int(round(start + dur))
        if n_seconds is not None and s >= n_seconds:
            continue
        events.append(Event(start=s, end=e if n_seconds is None else min(e, n_seconds), type=kind))
    events.sort(key=lambda x: x.start)
    return events


def nsrr_sleep_hours(xml_path_or_text: str) -> Optional[float]:
    """Total sleep time (hours) from the scored sleep stages in an NSRR XML file."""
    text = xml_path_or_text
    if not text.lstrip().startswith("<"):
        text = Path(xml_path_or_text).read_text(encoding="utf-8", errors="ignore")
    root = ET.fromstring(text)
    total, seen = 0.0, False
    for ev in root.iter("ScoredEvent"):
        if "stages" not in (ev.findtext("EventType") or "").lower():
            continue
        seen = True
        c = (ev.findtext("EventConcept") or "").lower()
        if c.startswith("wake") or "unscored" in c or "movement" in c:
            continue
        total += float(ev.findtext("Duration") or 0)
    return round(total / 3600, 3) if seen else None


def load_ahi_table(csv_path: str, id_col: str, ahi_col: str) -> Dict[str, float]:
    """Night-level AHI from the dataset CSV. Pick id_col / ahi_col from the NSRR
    data dictionary (Section 6.3 of the plan) and keep the same choice everywhere."""
    out: Dict[str, float] = {}
    with open(csv_path, newline="", encoding="utf-8", errors="ignore") as fh:
        for row in csv.DictReader(fh):
            try:
                out[str(row[id_col]).strip()] = float(row[ahi_col])
            except (KeyError, ValueError):
                continue
    return out


def anon_id(source: str, original: str) -> str:
    return f"{source}-{hashlib.sha1(original.encode()).hexdigest()[:10]}"


def convert_record(edf_path: str, xml_path: str, source: str, original_id: str,
                   ahi: Optional[float] = None, ahi_definition: Optional[str] = None,
                   hypopnea_desat: Optional[float] = 3.0) -> Recording:
    spo2, hr, start, info = read_spo2_hr(edf_path)
    spo2, hr = clean(spo2, hr)
    events = parse_nsrr_xml(xml_path, n_seconds=len(spo2), hypopnea_desat=hypopnea_desat)
    return from_arrays(
        recording_id=anon_id(source, original_id),
        source=Source(type=source, original_id=original_id),
        spo2=spo2, hr=hr, start_time=None,  # dataset dates are shifted; not needed
        labels=Labels(events=events, ahi=ahi, ahi_definition=ahi_definition,
                      sleep_hours=nsrr_sleep_hours(xml_path)),
    )
