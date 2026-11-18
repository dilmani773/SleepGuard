"""Participant-level train / val / test split, stratified by AHI severity."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .metrics import to_severity


def make_split(ids: Sequence[str], ahi: Sequence[float], seed: int = 42,
               frac=(0.70, 0.15, 0.15)) -> Dict[str, List[str]]:
    rng = np.random.default_rng(seed)
    sev = np.array(to_severity(ahi))
    out = {"train": [], "val": [], "test": []}
    ids = np.array(ids)
    for s in np.unique(sev):
        g = ids[sev == s]
        g = g[rng.permutation(len(g))]
        n_tr = int(round(frac[0] * len(g)))
        n_va = int(round(frac[1] * len(g)))
        out["train"] += g[:n_tr].tolist()
        out["val"] += g[n_tr:n_tr + n_va].tolist()
        out["test"] += g[n_tr + n_va:].tolist()
    return out


def load_or_make(path, ids, ahi, seed: int = 42) -> Dict[str, List[str]]:
    """New recordings that are not in the saved split are added to TRAIN only,
    so the val / test sets never change once created."""
    p = Path(path)
    if p.exists():
        split = json.loads(p.read_text())
        known = set(sum(split.values(), []))
        new = [i for i in ids if i not in known]
        if new:
            split["train"] += new
            p.write_text(json.dumps(split, indent=1))
        return split
    split = make_split(ids, ahi, seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(split, indent=1))
    return split
