from datetime import datetime

from sleepguard_ml.converters.nsrr import nsrr_sleep_hours, parse_nsrr_xml
from sleepguard_ml.converters.ucd import parse_respevt

NSRR_XML = """<?xml version="1.0"?><PSGAnnotation><ScoredEvents>
<ScoredEvent><EventType>Stages|Stages</EventType><EventConcept>Wake|0</EventConcept><Start>0</Start><Duration>30</Duration></ScoredEvent>
<ScoredEvent><EventType>Respiratory|Respiratory</EventType><EventConcept>Obstructive apnea|Obstructive Apnea</EventConcept><Start>120.5</Start><Duration>15</Duration></ScoredEvent>
<ScoredEvent><EventType>Respiratory|Respiratory</EventType><EventConcept>Hypopnea|Hypopnea</EventConcept><Start>300</Start><Duration>22</Duration></ScoredEvent>
<ScoredEvent><EventType>Respiratory|Respiratory</EventType><EventConcept>SpO2 desaturation|SpO2 desaturation</EventConcept><Start>310</Start><Duration>20</Duration><SpO2Nadir>91</SpO2Nadir><SpO2Baseline>95</SpO2Baseline></ScoredEvent>
<ScoredEvent><EventType>Respiratory|Respiratory</EventType><EventConcept>Hypopnea|Hypopnea</EventConcept><Start>600</Start><Duration>20</Duration></ScoredEvent>
<ScoredEvent><EventType>Respiratory|Respiratory</EventType><EventConcept>SpO2 desaturation|SpO2 desaturation</EventConcept><Start>615</Start><Duration>15</Duration><SpO2Nadir>94</SpO2Nadir><SpO2Baseline>96</SpO2Baseline></ScoredEvent>
</ScoredEvents></PSGAnnotation>"""


def test_nsrr_xml():
    ev = parse_nsrr_xml(NSRR_XML)  # default 3%: hypopnea at 600 has only a 2% desat
    assert [e.type for e in ev] == ["obstructive_apnea", "hypopnea"]
    assert ev[0].start == 120 and ev[0].end == 136 and ev[1].start == 300


def test_nsrr_xml_desat_rules():
    assert len(parse_nsrr_xml(NSRR_XML, hypopnea_desat=None)) == 3   # all hypopneas
    assert len(parse_nsrr_xml(NSRR_XML, hypopnea_desat=4.0)) == 2    # 4% drop still ok for 300
    assert len(parse_nsrr_xml(NSRR_XML, hypopnea_desat=5.0)) == 1    # apnea only


RESPEVT = """Time      Type    PB/CS  Duration  Low   %Drop  Snore  Arousal
23:59:50  HYP-O   PB     12        91    4      -      -
00:00:30  APNEA-O        20        88    7      -      Yes
"""


def test_ucd_midnight_and_offset():
    ev = parse_respevt(RESPEVT, rec_start=datetime(2000, 1, 1, 23, 59, 0))
    assert [e.type for e in ev] == ["hypopnea", "obstructive_apnea"]
    assert ev[0].start == 50 and ev[1].start == 90 and ev[1].end == 110


def test_nsrr_sleep_hours():
    xml = """<PSGAnnotation><ScoredEvents>
<ScoredEvent><EventType>Stages|Stages</EventType><EventConcept>Wake|0</EventConcept><Start>0</Start><Duration>1800</Duration></ScoredEvent>
<ScoredEvent><EventType>Stages|Stages</EventType><EventConcept>Stage 2 sleep|2</EventConcept><Start>1800</Start><Duration>3600</Duration></ScoredEvent>
<ScoredEvent><EventType>Stages|Stages</EventType><EventConcept>REM sleep|5</EventConcept><Start>5400</Start><Duration>1800</Duration></ScoredEvent>
</ScoredEvents></PSGAnnotation>"""
    assert nsrr_sleep_hours(xml) == 1.5
    assert nsrr_sleep_hours(NSRR_XML) == 0.0  # only a Wake epoch
