"""Download, convert and clean up NSRR nights in batches (NSRR data user's machine ONLY).

For each night: download the EDF + NSRR XML from sleepdata.org with your token,
convert to SGR Parquet, then DELETE the EDF to save disk (~40 MB each).
The XML is kept (small). Safe to stop and restart: finished nights are skipped.

    python scripts/batch_nsrr.py --n 1000 --dry-run                 # SHHS: show the plan
    python scripts/batch_nsrr.py --n 1000                           # SHHS: run (hours)
    python scripts/batch_nsrr.py --dataset mesa --n 200 --dry-run   # MESA (external test set)

Token: copy yours from https://sleepdata.org/token into ../data/nsrr_token.txt (git-ignored).
"""
import argparse
import csv
import random
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
p.add_argument("--dataset", choices=["shhs", "mesa"], default="shhs")
p.add_argument("--n", type=int, default=1000, help="number of nights to have in total")
p.add_argument("--seed", type=int, default=42, help="random sample of subjects, fixed seed")
p.add_argument("--data", default="../data")
p.add_argument("--csv", default=None, help="dataset CSV (default depends on --dataset)")
p.add_argument("--ahi-col", default="ahi_a0h3")
p.add_argument("--desat", type=float, default=3.0)
p.add_argument("--keep-edf", action="store_true")
p.add_argument("--dry-run", action="store_true")
p.add_argument("--workers", type=int, default=4, help="nights downloaded at the same time")
p.add_argument("--client", default="sleepguard-py", help="client tag in the download URL")
a = p.parse_args()

# Per-dataset layout on sleepdata.org. Both use ahi_a0h3 (all apneas + hypopneas with >= 3% desat).
DS = {
    "shhs": dict(csv="raw/shhs/datasets/shhs1-dataset-0.21.0.csv", id_col="nsrrid",
                 name=lambda i: f"shhs1-{i}",
                 edf=lambda n: f"shhs/polysomnography/edfs/shhs1/{n}.edf",
                 xml=lambda n: f"shhs/polysomnography/annotations-events-nsrr/shhs1/{n}-nsrr.xml",
                 sub="shhs1"),
    "mesa": dict(csv="raw/mesa/datasets/mesa-sleep-dataset-0.8.0.csv", id_col="mesaid",
                 name=lambda i: f"mesa-sleep-{i:04d}",
                 edf=lambda n: f"mesa/polysomnography/edfs/{n}.edf",
                 xml=lambda n: f"mesa/polysomnography/annotations-events-nsrr/{n}-nsrr.xml",
                 sub=""),
}[a.dataset]

data = Path(a.data).resolve()
token = data / "nsrr_token.txt"
out = data / "sgr" / a.dataset
xml_dir = data / "raw" / a.dataset / "annotations-events-nsrr" / DS["sub"]
edf_dir = data / "raw" / a.dataset / "edfs" / DS["sub"]
tmp = data / "raw" / "_download_tmp"
log_path = data / "sgr" / f"{a.dataset}_batch_log.csv"
for d in (out, xml_dir, edf_dir, tmp):
    d.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(data / (a.csv or DS["csv"]), low_memory=False)
for col in (DS["id_col"], a.ahi_col):
    if col not in df.columns:
        sys.exit(f"column '{col}' not in the CSV. Columns with 'ahi': {[c for c in df.columns if 'ahi' in c.lower()][:20]}")
df = df[df[DS["id_col"]].notna()]
if a.dataset != "shhs":  # SHHS keeps its original sample so finished nights are still recognised
    df = df[df[a.ahi_col].notna()]  # only people who had a PSG
ahi = dict(zip(df[DS["id_col"]].astype(int), df[a.ahi_col]))
ids = sorted(ahi)
random.Random(a.seed).shuffle(ids)
ids = ids[: a.n]

done = {f.stem for f in out.glob("*.parquet")}
done_names = set()
if log_path.exists():
    with open(log_path, newline="") as fh:
        done_names = {r["name"] for r in csv.DictReader(fh) if r["status"] == "ok"}
todo = [i for i in ids if DS["name"](i) not in done_names]
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
                expected = int(r.headers.get("Content-Length") or 0)
                shutil.copyfileobj(r, fh, length=1 << 20)
            got = part.stat().st_size
            if got == 0:
                raise RuntimeError("empty file")
            if expected and got != expected:  # connection dropped mid-file
                raise RuntimeError(f"incomplete download ({got} of {expected} bytes)")
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


def process(sid: int):
    name, t0 = DS["name"](sid), time.time()
    edf = fetch(DS["edf"](name), edf_dir / f"{name}.edf")
    xml = fetch(DS["xml"](name), xml_dir / f"{name}-nsrr.xml")
    try:
        rec = convert_record(str(edf), str(xml), a.dataset, name, ahi=ahi.get(sid),
                             ahi_definition=a.ahi_col, hypopnea_desat=a.desat)
    except Exception:
        edf.unlink(missing_ok=True)  # broken file: delete so the next run downloads it again
        raise
    save_parquet(rec, out / f"{rec.recording_id}.parquet")
    if not a.keep_edf:
        edf.unlink()
    vh = sum(v is not None for v in rec.spo2) / 3600
    return [name, "ok", f"{vh:.2f}", rec.labels.sleep_hours, len(rec.labels.events),
            rec.labels.ahi, f"{time.time() - t0:.0f}", ""]


lock = threading.Lock()
new_log = not log_path.exists()
t_start = time.time()
with open(log_path, "a", newline="") as fh, ThreadPoolExecutor(max_workers=a.workers) as pool:
    w = csv.writer(fh)
    if new_log:
        w.writerow(["name", "status", "valid_h", "sleep_h", "events", "ahi", "seconds", "note"])
    futures = {pool.submit(process, sid): sid for sid in todo}
    for k, fut in enumerate(as_completed(futures), 1):
        sid = futures[fut]
        try:
            row = fut.result()
            msg = f"ok  ahi={row[5]}"
        except Exception as e:  # keep going; failures are logged and retried next run
            row = [DS["name"](sid), "failed", "", "", "", "", "", str(e)[:200]]
            msg = f"FAILED: {e}"
        with lock:
            w.writerow(row)
            fh.flush()
        rate = (time.time() - t_start) / k
        left_h = rate * (len(todo) - k) / 3600
        print(f"[{k}/{len(todo)}] {row[0]} {msg}   (~{left_h:.1f} h left)")
print("finished. log:", log_path)
