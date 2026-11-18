"""Download, convert and clean up NSRR nights in batches (NSRR data user's machine ONLY).

For each night: download the EDF + NSRR XML with the official `nsrr` tool,
convert to SGR Parquet, then DELETE the EDF to save disk (~40 MB each).
The XML is kept (small). Safe to stop and restart: finished nights are skipped.

    pip install nsrr
    python scripts/batch_nsrr.py --n 1000 --dry-run      # show the plan
    python scripts/batch_nsrr.py --n 1000                # run (hours; leave it overnight)

Token: copy yours from https://sleepdata.org/token into ../data/nsrr_token.txt (git-ignored).
"""
import argparse
import csv
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from sleepguard_ml.converters.nsrr import convert_record
from sleepguard_ml.io import save_parquet

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=1000, help="number of nights to have in total")
p.add_argument("--seed", type=int, default=42, help="random sample of subjects, fixed seed")
p.add_argument("--data", default="../data")
p.add_argument("--csv", default="raw/shhs/datasets/shhs1-dataset-0.21.0.csv")
p.add_argument("--ahi-col", default="ahi_a0h3")
p.add_argument("--desat", type=float, default=3.0)
p.add_argument("--keep-edf", action="store_true")
p.add_argument("--dry-run", action="store_true")
a = p.parse_args()

data = Path(a.data).resolve()
token = data / "nsrr_token.txt"
out = data / "sgr" / "shhs"
xml_dir = data / "raw" / "shhs" / "annotations-events-nsrr" / "shhs1"
edf_dir = data / "raw" / "shhs" / "edfs" / "shhs1"
tmp = data / "raw" / "_download_tmp"
log_path = data / "sgr" / "shhs_batch_log.csv"
for d in (out, xml_dir, edf_dir, tmp):
    d.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(data / a.csv, low_memory=False)
ahi = dict(zip(df["nsrrid"].astype(int), df[a.ahi_col]))
ids = sorted(int(i) for i in df["nsrrid"].dropna())
random.Random(a.seed).shuffle(ids)
ids = ids[: a.n]

done = {f.stem for f in out.glob("*.parquet")}
done_names = set()
if log_path.exists():
    with open(log_path, newline="") as fh:
        done_names = {r["name"] for r in csv.DictReader(fh) if r["status"] == "ok"}
todo = [i for i in ids if f"shhs1-{i}" not in done_names]
print(f"target {len(ids)} nights, already done {len(ids) - len(todo)}, to do {len(todo)}")
if a.dry_run:
    print("first 10 to do:", todo[:10])
    sys.exit(0)
if not token.exists():
    sys.exit(f"token file missing: {token}")
if shutil.which("nsrr") is None:
    sys.exit("`nsrr` command not found. Run: pip install nsrr")


def fetch(remote: str, dest: Path) -> Path:
    """Download one file with the nsrr tool into tmp, then move it to dest."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    for f in tmp.rglob("*"):
        if f.is_file():
            f.unlink()
    r = subprocess.run(["nsrr", "-d", remote, "--token-file", str(token)],
                       cwd=tmp, capture_output=True, text=True)
    hits = [f for f in tmp.rglob(Path(remote).name) if f.is_file() and f.stat().st_size > 0]
    if not hits:
        raise RuntimeError(f"download failed for {remote}: {(r.stdout + r.stderr)[-300:]}")
    shutil.move(str(hits[0]), dest)
    return dest


new_log = not log_path.exists()
with open(log_path, "a", newline="") as fh:
    w = csv.writer(fh)
    if new_log:
        w.writerow(["name", "status", "valid_h", "sleep_h", "events", "ahi", "seconds", "note"])
    for k, sid in enumerate(todo, 1):
        name, t0 = f"shhs1-{sid}", time.time()
        try:
            edf = fetch(f"shhs/polysomnography/edfs/shhs1/{name}.edf", edf_dir / f"{name}.edf")
            xml = fetch(f"shhs/polysomnography/annotations-events-nsrr/shhs1/{name}-nsrr.xml",
                        xml_dir / f"{name}-nsrr.xml")
            rec = convert_record(str(edf), str(xml), "shhs", name, ahi=ahi.get(sid),
                                 ahi_definition=a.ahi_col, hypopnea_desat=a.desat)
            save_parquet(rec, out / f"{rec.recording_id}.parquet")
            vh = sum(v is not None for v in rec.spo2) / 3600
            w.writerow([name, "ok", f"{vh:.2f}", rec.labels.sleep_hours, len(rec.labels.events),
                        rec.labels.ahi, f"{time.time() - t0:.0f}", ""])
            if not a.keep_edf:
                edf.unlink()
            print(f"[{k}/{len(todo)}] {name} ok  valid_h={vh:.1f} ahi={rec.labels.ahi}")
        except Exception as e:  # keep going; failures are logged and retried next run
            w.writerow([name, "failed", "", "", "", "", f"{time.time() - t0:.0f}", str(e)[:200]])
            print(f"[{k}/{len(todo)}] {name} FAILED: {e}")
        fh.flush()
print("finished. log:", log_path)
