"""Make synthetic nights for app / backend / dashboard testing.

    python scripts/make_synthetic.py --out ../data/synthetic --n 40
    python scripts/make_synthetic.py --examples ../docs/interfaces/examples
"""
import argparse
from pathlib import Path

from sleepguard_ml.io import save_json, save_parquet
from sleepguard_ml.synthetic import make_night

SEVERITIES = ["normal", "mild", "moderate", "severe"]

p = argparse.ArgumentParser()
p.add_argument("--out", default="../data/synthetic")
p.add_argument("--n", type=int, default=40, help="nights in total, spread over 4 severities")
p.add_argument("--hours", type=float, default=7.0)
p.add_argument("--examples", help="also write one JSON per severity to this folder")
a = p.parse_args()

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
for i in range(a.n):
    sev = SEVERITIES[i % 4]
    rec = make_night(sev, hours=a.hours, seed=i)
    save_parquet(rec, out / f"{rec.recording_id}.parquet")
print(f"wrote {a.n} nights to {out}")

if a.examples:
    ex = Path(a.examples)
    ex.mkdir(parents=True, exist_ok=True)
    for k, sev in enumerate(SEVERITIES):
        rec = make_night(sev, hours=a.hours, seed=1000 + k)
        save_json(rec, ex / f"sgr_example_{sev}.json")
    print(f"wrote 4 example SGR JSON files to {ex}")
