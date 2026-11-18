"""Download, convert and clean up NSRR nights in batches (NSRR data user's machine ONLY).

For each night: download the EDF + NSRR XML from sleepdata.org with your token,
convert to SGR Parquet, then DELETE the EDF to save disk (~40 MB each).
The XML is kept (small). Safe to stop and restart: finished nights are skipped.

    python scripts/batch_nsrr.py --n 1000 --dry-run      # show the plan
    python scripts/batch_nsrr.py --n 1000                # run (hours; leave it overnight)

Token: copy yours from https://sleepdata.org/token into ../data/nsrr_token.txt (git-ignored).
"""
import argparse
import csv
import random
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

try:  # use the Windows / macOS certificate store (fixes CERTIFICATE_VERIFY_FAILED)
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

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
p.add_argument("--client", default="sleepguard-py", help="client tag in the download URL")
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
TOKEN = token.read_text(encoding="utf-8").strip()


def fetch(remote: str, dest: Path, retries: int = 3) -> Path:
    """Download sleepdata.org/datasets/shhs/files/<remote> into dest.
    URL pattern from the official NSRR R package (nsrr_download_url). Token never printed."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dataset, path = remote.split("/", 1)
    url = f"https://sleepdata.org/datasets/{dataset}/files/a/{TOKEN}/m/{a.client}/{path}"
    part = tmp / (dest.name + ".part")
    last = ""
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(part, "wb") as fh:
                shutil.copyfileobj(r, fh, length=1 << 20)
            if part.stat().st_size == 0:
                raise RuntimeError("empty file")
            with open(part, "rb") as fh:
                head = fh.read(200).lower()
            if b"<html" in head or b"<!doctype" in head:
                raise RuntimeError("got a web page instead of the file (token or access problem)")
            shutil.move(str(part), dest)
            return dest
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            if e.code in (401, 403, 404):
                break
        except Exception as e:
            last = str(e).replace(TOKEN, "<token>")
        time.sleep(3 * attempt)
    if part.exists():
        part.unlink()
    raise RuntimeError(f"download failed for {remote}: {last}")


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
