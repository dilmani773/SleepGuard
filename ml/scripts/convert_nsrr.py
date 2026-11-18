"""Convert SHHS / MESA to SGR Parquet. NSRR data user's secure machine ONLY.

    python scripts/convert_nsrr.py --source shhs --edf-dir ../data/raw/shhs/edfs/shhs1 \
        --xml-dir ../data/raw/shhs/annotations-events-nsrr/shhs1 --out ../data/sgr/shhs \
        --ahi-csv ../data/raw/shhs/datasets/shhs1-dataset.csv --id-col nsrrid --ahi-col ahi_a0h3

Choose <AHI_VARIABLE> from the NSRR data dictionary (plan Section 6.3) and write it in the model card.
"""
import argparse
import re
from pathlib import Path

from sleepguard_ml.converters.nsrr import convert_record, load_ahi_table
from sleepguard_ml.io import save_parquet

p = argparse.ArgumentParser()
p.add_argument("--source", choices=["shhs", "mesa"], required=True)
p.add_argument("--edf-dir", required=True)
p.add_argument("--xml-dir", required=True)
p.add_argument("--out", required=True)
p.add_argument("--ahi-csv")
p.add_argument("--id-col")
p.add_argument("--ahi-col")
p.add_argument("--desat", type=float, default=3.0, help="hypopnea needs this %% SpO2 drop (3 for ahi_a0h3, 4 for ahi_a0h4)")
p.add_argument("--limit", type=int, default=0, help="convert only the first N files (testing)")
a = p.parse_args()

ahi = load_ahi_table(a.ahi_csv, a.id_col, a.ahi_col) if a.ahi_csv else {}
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
edfs = sorted(Path(a.edf_dir).glob("*.edf"))
if a.limit:
    edfs = edfs[:a.limit]
ok = fail = 0
for edf in edfs:
    xml = Path(a.xml_dir) / f"{edf.stem}-nsrr.xml"
    if not xml.exists():
        print(f"skip {edf.name}: no {xml.name}")
        continue
    m = re.search(r"(\d+)$", edf.stem)
    subject = m.group(1).lstrip("0") if m else edf.stem
    try:
        rec = convert_record(str(edf), str(xml), a.source, edf.stem,
                             ahi=ahi.get(subject) or ahi.get(edf.stem), ahi_definition=a.ahi_col,
                             hypopnea_desat=a.desat)
        save_parquet(rec, out / f"{rec.recording_id}.parquet")
        ok += 1
    except Exception as e:
        print(f"FAILED {edf.name}: {e}")
        fail += 1
print(f"done: {ok} converted, {fail} failed")
