from datetime import datetime

from sleepguard_ml.converters.nsrr import parse_nsrr_xml
from sleepguard_ml.converters.ucd import parse_respevt

NSRR_XML = """<?xml version="1.0"?><PSGAnnotation><ScoredEvents>
<ScoredEvent><EventType>Stages|Stages</EventType><EventConcept>Wake|0</EventConcept><Start>0</Start><Duration>30</Duration></ScoredEvent>
<ScoredEvent><EventType>Respiratory|Respiratory</EventType><EventConcept>Obstructive apnea|Obstructive Apnea</EventConcept><Start>120.5</Start><Duration>15</Duration></ScoredEvent>
<ScoredEvent><EventType>Respiratory|Respiratory</EventType><EventConcept>Hypopnea|Hypopnea</EventConcept><Start>300</Start><Duration>22</Duration></ScoredEvent>
<ScoredEvent><EventType>Respiratory|Respiratory</EventType><EventConcept>SpO2 desaturation|SpO2 desaturation</EventConcept><Start>310</Start><Duration>20</Duration></ScoredEvent>
</ScoredEvents></PSGAnnotation>"""


def test_nsrr_xml():
    ev = parse_nsrr_xml(NSRR_XML)
    assert [e.type for e in ev] == ["obstructive_apnea", "hypopnea"]
    assert ev[0].start == 120 and ev[0].end == 136


RESPEVT = """Time      Type    PB/CS  Duration  Low   %Drop  Snore  Arousal
23:59:50  HYP-O   PB     12        91    4      -      -
00:00:30  APNEA-O        20        88    7      -      Yes
"""


def test_ucd_midnight_and_offset():
    ev = parse_respevt(RESPEVT, rec_start=datetime(2000, 1, 1, 23, 59, 0))
    assert [e.type for e in ev] == ["hypopnea", "obstructive_apnea"]
    assert ev[0].start == 50 and ev[1].start == 90 and ev[1].end == 110
