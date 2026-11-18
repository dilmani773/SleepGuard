"""Convert the PhysioNet St. Vincent's / UCD database to SGR Parquet.

    python scripts/convert_ucd.py --src ../data/raw/ucd --out ../data/sgr/ucd --ahi-csv ../data/raw/ucd/SubjectDetails.csv

SubjectDetails comes as a spreadsheet: save it as CSV first. Pass the column names
for the subject ID and AHI if they differ from the defaults below.
"""
import argparse
import csv
from pathlib import Path

from sleepguard_ml.converters.ucd import convert_subject
from sleepguard_ml.io import save_parquet

p = argparse.ArgumentParser()
p.add_argument("--src", required=True)
p.add_argument("--out", required=True)
p.add_argument("--ahi-csv")
p.add_argument("--id-col", default="Study Number")
p.add_argument("--ahi-col", default="AHI")
a = p.parse_args()

ahi = {}
if a.ahi_csv:
    with open(a.ahi_csv, newline="", encoding="utf-8", errors="ignore") as fh:
        for row in csv.DictReader(fh):
            try:
                ahi[row[a.id_col].strip().lower()] = float(row[a.ahi_col])
            except (KeyError, ValueError):
                pass

src, out = Path(a.src), Path(a.out)
out.mkdir(parents=True, exist_ok=True)
for rec_path in sorted(src.glob("*.rec")) + sorted(src.glob("*.edf")):
    evt = rec_path.with_name(rec_path.stem + "_respevt.txt")
    if not evt.exists():
        print(f"skip {rec_path.name}: no events file")
        continue
    try:
        rec = convert_subject(str(rec_path), str(evt), ahi.get(rec_path.stem.lower()))
    except Exception as e:  # keep going, report at the end
        print(f"FAILED {rec_path.name}: {e}")
        continue
    save_parquet(rec, out / f"{rec.recording_id}.parquet")
    n_valid = sum(v is not None for v in rec.spo2) / 3600
    print(f"{rec.recording_id}: {n_valid:.1f} valid h, {len(rec.labels.events)} events, AHI={rec.labels.ahi}")
