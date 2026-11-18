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


def parse_nsrr_xml(xml_path_or_text: str, n_seconds: Optional[int] = None) -> List[Event]:
    text = xml_path_or_text
    if not text.lstrip().startswith("<"):
        text = Path(xml_path_or_text).read_text(encoding="utf-8", errors="ignore")
    root = ET.fromstring(text)
    events: List[Event] = []
    for ev in root.iter("ScoredEvent"):
        concept = (ev.findtext("EventConcept") or "")
        kind = map_concept(concept)
        if kind is None:
            continue
        start = float(ev.findtext("Start") or 0)
        dur = float(ev.findtext("Duration") or 0)
        s, e = int(round(start)), int(round(start + dur))
        if n_seconds is not None and s >= n_seconds:
            continue
        events.append(Event(start=s, end=e if n_seconds is None else min(e, n_seconds), type=kind))
    events.sort(key=lambda x: x.start)
    return events


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
                   ahi: Optional[float] = None, ahi_definition: Optional[str] = None) -> Recording:
    spo2, hr, start, info = read_spo2_hr(edf_path)
    spo2, hr = clean(spo2, hr)
    events = parse_nsrr_xml(xml_path, n_seconds=len(spo2))
    return from_arrays(
        recording_id=anon_id(source, original_id),
        source=Source(type=source, original_id=original_id),
        spo2=spo2, hr=hr, start_time=None,  # dataset dates are shifted; not needed
        labels=Labels(events=events, ahi=ahi, ahi_definition=ahi_definition),
    )
