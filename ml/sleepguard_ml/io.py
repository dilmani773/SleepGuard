"""Read and write SGR recordings (JSON for exchange, Parquet for training)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from .schemas import Labels, Recording, Source

PathLike = Union[str, Path]


def save_json(rec: Recording, path: PathLike) -> None:
    Path(path).write_text(rec.model_dump_json(), encoding="utf-8")


def load_json(path: PathLike) -> Recording:
    return Recording.model_validate_json(Path(path).read_text(encoding="utf-8"))


def save_parquet(rec: Recording, path: PathLike) -> None:
    """Signals as columns; metadata and labels in the Parquet schema metadata."""
    df = pd.DataFrame({
        "spo2": pd.array(rec.spo2, dtype="Int16"),
        "hr": pd.array(rec.hr, dtype="Int16"),
        "quality": pd.array(rec.quality if rec.quality is not None else [None] * rec.n_seconds, dtype="Int16"),
    })
    meta = rec.model_dump(mode="json", exclude={"spo2", "hr", "quality"})
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(df, preserve_index=False)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), b"sgr": json.dumps(meta).encode()})
    pq.write_table(table, path)


def load_parquet(path: PathLike) -> Recording:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    meta = json.loads(table.schema.metadata[b"sgr"])
    df = table.to_pandas()

    def col(name):
        return [None if pd.isna(v) else int(v) for v in df[name]]

    quality = col("quality")
    if all(q is None for q in quality):
        quality = None
    return Recording(**meta, spo2=col("spo2"), hr=col("hr"), quality=quality)


def to_arrays(rec: Recording):
    """SpO2 and HR as float arrays with NaN for invalid samples."""
    spo2 = np.array([np.nan if v is None else v for v in rec.spo2], dtype=float)
    hr = np.array([np.nan if v is None else v for v in rec.hr], dtype=float)
    return spo2, hr


def from_arrays(recording_id: str, source: Source, spo2, hr, quality=None,
                labels: Labels | None = None, start_time=None) -> Recording:
    def ints(a):
        return [None if (v is None or not np.isfinite(v)) else int(round(v)) for v in a]

    return Recording(
        recording_id=recording_id, source=source, start_time=start_time,
        spo2=ints(spo2), hr=ints(hr),
        quality=None if quality is None else ints(quality), labels=labels,
    )
