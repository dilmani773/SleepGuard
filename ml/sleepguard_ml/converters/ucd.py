"""St. Vincent's University Hospital / UCD Sleep Apnea Database (PhysioNet) -> SGR.

Files per subject: ucddbXXX.rec (EDF format), ucddbXXX_respevt.txt (respiratory events).
Night-level AHI is in SubjectDetails (spreadsheet on the PhysioNet page).
The events file is a fixed-width text table. Run `debug_respevt(path)` on one file
first and check the parsed events against the raw text before converting all files.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from ..io import from_arrays
from ..preprocess import clean
from ..schemas import Event, Labels, Recording, Source
from .edf import read_spo2_hr

TIME_RE = re.compile(r"^\s*(\d{1,2}):(\d{2}):(\d{2})")


def map_ucd_type(t: str) -> Optional[str]:
    t = t.upper()
    if t.startswith("HYP"):
        return "hypopnea"
    if "APNEA" in t or t.startswith("APN"):
        if t.endswith("-O") or "OBS" in t:
            return "obstructive_apnea"
        if t.endswith("-C") or "CEN" in t:
            return "central_apnea"
        if t.endswith("-M") or "MIX" in t:
            return "mixed_apnea"
        return "obstructive_apnea"
    return None


def _columns(header: str):
    """Start positions of each header word in a fixed-width table."""
    return [(m.group(0), m.start()) for m in re.finditer(r"\S+", header)]


def parse_respevt(text: str, rec_start: Optional[datetime] = None) -> List[Event]:
    lines = text.splitlines()
    head_i = next((i for i, l in enumerate(lines) if "time" in l.lower() and "type" in l.lower()), None)
    if head_i is None:
        raise ValueError("header line with 'Time' and 'Type' not found")
    cols = _columns(lines[head_i])
    names = [c[0].lower() for c in cols]
    starts = [c[1] for c in cols] + [10_000]

    def field(line, name):
        if name not in names:
            return ""
        k = names.index(name)
        return line[starts[k]:starts[k + 1]].strip()

    events, prev = [], None
    day_offset = 0
    for line in lines[head_i + 1:]:
        m = TIME_RE.match(line)
        if not m:
            continue
        h, mi, s = map(int, m.groups())
        clock = h * 3600 + mi * 60 + s
        if prev is not None and clock < prev:  # passed midnight
            day_offset += 86400
        prev = clock
        tokens = line.split()
        kind = map_ucd_type(field(line, "type") or tokens[1])
        if kind is None:
            continue
        dur_txt = field(line, "duration")
        dur = int(float(dur_txt)) if dur_txt.replace(".", "", 1).isdigit() else 10
        abs_sec = clock + day_offset
        if rec_start is not None:
            start_sec = rec_start.hour * 3600 + rec_start.minute * 60 + rec_start.second
            offset = abs_sec - start_sec
            if offset < 0:
                offset += 86400
        else:
            offset = abs_sec
        events.append(Event(start=offset, end=offset + dur, type=kind))
    return events


def debug_respevt(path: str, rec_start: Optional[datetime] = None, n: int = 5) -> None:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    print("\n".join(text.splitlines()[:n + 4]))
    print("---- parsed ----")
    for ev in parse_respevt(text, rec_start)[:n]:
        print(ev)


def convert_subject(rec_path: str, respevt_path: str, ahi: Optional[float] = None) -> Recording:
    spo2, hr, start, info = read_spo2_hr(rec_path)
    spo2, hr = clean(spo2, hr)
    text = Path(respevt_path).read_text(encoding="utf-8", errors="ignore")
    events = [e for e in parse_respevt(text, start) if e.start < len(spo2)]
    name = Path(rec_path).stem
    return from_arrays(
        recording_id=f"ucd-{name}", source=Source(type="ucd", original_id=name),
        spo2=spo2, hr=hr, start_time=None,
        labels=Labels(events=events, ahi=ahi, ahi_definition="UCD SubjectDetails AHI"),
    )
