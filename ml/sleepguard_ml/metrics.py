"""Evaluation metrics used in the paper (Section 10.1 of the plan)."""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, roc_auc_score

from .summary import CUTOFFS

SEVERITY_ORDER = ["normal", "mild", "moderate", "severe"]


# ---------- event level ----------
def match_events(true: Sequence[Tuple[int, int]], pred: Sequence[Tuple[int, int]],
                 tolerance: int = 30) -> Dict[str, float]:
    """A predicted event matches a true event if they overlap, allowing `tolerance`
    seconds because SpO2 dips come 10 to 30 s after the breathing event."""
    used, tp = set(), 0
    for ps, pe in pred:
        for i, (ts, te) in enumerate(true):
            if i in used:
                continue
            if ps <= te + tolerance and pe >= ts:
                used.add(i)
                tp += 1
                break
    fp, fn = len(pred) - tp, len(true) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


# ---------- night level ----------
def icc_2_1(a: np.ndarray, b: np.ndarray) -> float:
    """Two-way random, absolute agreement, single measure ICC(2,1)."""
    x = np.column_stack([a, b]).astype(float)
    n, k = x.shape
    grand = x.mean()
    msr = k * ((x.mean(1) - grand) ** 2).sum() / (n - 1)
    msc = n * ((x.mean(0) - grand) ** 2).sum() / (k - 1)
    sse = ((x - x.mean(1, keepdims=True) - x.mean(0, keepdims=True) + grand) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def ahi_agreement(true_ahi: Sequence[float], pred_ahi: Sequence[float]) -> Dict[str, float]:
    t, p = np.asarray(true_ahi, float), np.asarray(pred_ahi, float)
    diff = p - t
    bias, sd = diff.mean(), diff.std(ddof=1)
    return {
        "mae": float(np.abs(diff).mean()),
        "pearson_r": float(np.corrcoef(t, p)[0, 1]),
        "icc": icc_2_1(t, p),
        "bias": float(bias),
        "loa_low": float(bias - 1.96 * sd),
        "loa_high": float(bias + 1.96 * sd),
    }


def to_severity(ahi: Sequence[float]) -> List[str]:
    out = []
    for a in ahi:
        out.append("normal" if a < CUTOFFS[0] else "mild" if a < CUTOFFS[1]
                   else "moderate" if a < CUTOFFS[2] else "severe")
    return out


def severity_metrics(true_ahi, pred_ahi) -> Dict:
    ts, ps = to_severity(true_ahi), to_severity(pred_ahi)
    return {
        "confusion": confusion_matrix(ts, ps, labels=SEVERITY_ORDER).tolist(),
        "labels": SEVERITY_ORDER,
        "kappa_linear": float(cohen_kappa_score(ts, ps, labels=SEVERITY_ORDER, weights="linear")),
        "accuracy": float(np.mean(np.array(ts) == np.array(ps))),
    }


def screening(true_ahi, pred_ahi, cutoff: float = 15.0) -> Dict[str, float]:
    t = np.asarray(true_ahi, float) >= cutoff
    score = np.asarray(pred_ahi, float)
    p = score >= cutoff
    tp, tn = int((t & p).sum()), int((~t & ~p).sum())
    fp, fn = int((~t & p).sum()), int((t & ~p).sum())
    auc = float(roc_auc_score(t, score)) if 0 < t.sum() < len(t) else float("nan")
    return {
        "cutoff": cutoff,
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "auc": auc, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def bootstrap_ci(metric: Callable[[np.ndarray, np.ndarray], float], true, pred,
                 n: int = 1000, seed: int = 0) -> Tuple[float, float]:
    """95% CI by resampling nights."""
    rng = np.random.default_rng(seed)
    t, p = np.asarray(true, float), np.asarray(pred, float)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(t), len(t))
        try:
            v = metric(t[idx], p[idx])
        except ValueError:
            continue
        if np.isfinite(v):
            vals.append(v)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
