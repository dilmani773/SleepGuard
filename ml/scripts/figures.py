"""Paper figures from the FROZEN final model (aggregate results; no participant signals).

    pip install matplotlib
    python scripts/figures.py

Saves PNGs (300 dpi) to ../data/results/figures/:
  fig1_scatter.png       predicted vs PSG AHI (SHHS test, MESA)
  fig2_bland_altman.png  agreement
  fig3_roc.png           ROC at AHI >= 15, SleepGuard vs ODI3 baseline
  fig4_robustness.png    MAE and referral sensitivity under cheap-sensor problems (needs robustness.json)
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from sleepguard_ml.baseline import BaselineModel, analyze_baseline, fit_calibration
from sleepguard_ml.io import load_parquet
from sleepguard_ml.m1 import M1Model, analyze_m1
from sleepguard_ml.summary import REFER_CUTOFF

p = argparse.ArgumentParser()
p.add_argument("--shhs", default="../data/sgr/shhs")
p.add_argument("--mesa", default="../data/sgr/mesa")
p.add_argument("--split", default="../data/splits/shhs_split.json")
p.add_argument("--model", default="../data/models/m1_final")
p.add_argument("--results", default="../data/results")
a = p.parse_args()

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans"})
C1, C2, GREY = "#2b6cb0", "#c05621", "#888888"
out = Path(a.results) / "figures"
out.mkdir(parents=True, exist_ok=True)


def load(folder):
    return [r for r in (load_parquet(f) for f in sorted(Path(folder).glob("*.parquet")))
            if r.labels is not None and r.labels.ahi is not None]


split = json.loads(Path(a.split).read_text())
shhs = {r.recording_id: r for r in load(a.shhs)}
train = [shhs[i] for i in split["train"] + split["val"] if i in shhs]
sets = {"SHHS test": [shhs[i] for i in split["test"] if i in shhs], "MESA (external)": load(a.mesa)}
m1 = M1Model.load(a.model)
m0 = fit_calibration(train, BaselineModel(drop=3.0))  # same baseline as final_eval

data = {}
for name, recs in sets.items():
    data[name] = dict(true=np.array([r.labels.ahi for r in recs]),
                      m1=np.array([analyze_m1(r, m1).ahi_est or 0.0 for r in recs]),
                      m0=np.array([analyze_baseline(r, m0).ahi_est or 0.0 for r in recs]))
    print(f"{name}: {len(recs)} nights")

# Fig 1: scatter
fig, axs = plt.subplots(1, 2, figsize=(7, 3.4), sharex=True, sharey=True)
for ax, (name, d), c in zip(axs, data.items(), (C1, C2)):
    lim = max(d["true"].max(), d["m1"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color=GREY, lw=0.8, ls="--")
    for v in (5, 15, 30):
        ax.axvline(v, color=GREY, lw=0.4, alpha=0.5)
        ax.axhline(v, color=GREY, lw=0.4, alpha=0.5)
    ax.scatter(d["true"], d["m1"], s=10, color=c, alpha=0.7, edgecolor="none")
    r = np.corrcoef(d["true"], d["m1"])[0, 1]
    ax.set_title(f"{name} (n = {len(d['true'])}, r = {r:.2f})")
    ax.set_xlabel("PSG AHI (events/h)")
axs[0].set_ylabel("SleepGuard AHI (events/h)")
fig.tight_layout()
fig.savefig(out / "fig1_scatter.png", dpi=300)

# Fig 2: Bland-Altman
fig, axs = plt.subplots(1, 2, figsize=(7, 3.2), sharey=True)
for ax, (name, d), c in zip(axs, data.items(), (C1, C2)):
    mean, diff = (d["true"] + d["m1"]) / 2, d["m1"] - d["true"]
    b, sd = diff.mean(), diff.std(ddof=1)
    ax.scatter(mean, diff, s=10, color=c, alpha=0.7, edgecolor="none")
    for y, ls in ((b, "-"), (b + 1.96 * sd, "--"), (b - 1.96 * sd, "--")):
        ax.axhline(y, color=GREY, lw=0.8, ls=ls)
    ax.text(0.98, 0.95, f"bias {b:+.1f}\nLoA {b - 1.96 * sd:.1f} to {b + 1.96 * sd:.1f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8)
    ax.set_title(name)
    ax.set_xlabel("Mean of PSG and SleepGuard AHI")
axs[0].set_ylabel("SleepGuard minus PSG AHI")
fig.tight_layout()
fig.savefig(out / "fig2_bland_altman.png", dpi=300)

# Fig 3: ROC at AHI >= 15
fig, axs = plt.subplots(1, 2, figsize=(7, 3.4), sharey=True)
for ax, (name, d) in zip(axs, data.items()):
    y = d["true"] >= 15
    for key, lab, c in (("m1", "SleepGuard", C1), ("m0", "ODI3 baseline", GREY)):
        fpr, tpr, _ = roc_curve(y, d[key])
        ax.plot(fpr, tpr, color=c, lw=1.5, label=f"{lab} (AUC {roc_auc_score(y, d[key]):.2f})")
    q = d["m1"] >= REFER_CUTOFF
    ax.plot(1 - (~q[~y]).mean(), q[y].mean(), "o", color=C2, ms=6,
            label=f"Referral rule (AHI ≥ {REFER_CUTOFF:g})")
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.6, ls=":")
    ax.set_title(f"{name}: AHI ≥ 15")
    ax.set_xlabel("1 − specificity")
    ax.legend(loc="lower right", fontsize=7, frameon=False)
axs[0].set_ylabel("Sensitivity")
fig.tight_layout()
fig.savefig(out / "fig3_roc.png", dpi=300)

# Fig 4: robustness
rob = Path(a.results) / "robustness.json"
if rob.exists():
    r = json.loads(rob.read_text())["sets"]
    conds = list(next(iter(r.values())).keys())
    yy = np.arange(len(conds))
    fig, axs = plt.subplots(1, 3, figsize=(9, 3.6), sharey=True)
    for k, (key, label) in enumerate((("mae", "MAE (events/h)"), ("refer_sens", "Referral sensitivity"),
                                      ("refer_spec", "Referral specificity"))):
        for j, (sname, c) in enumerate(zip(r, (C1, C2))):
            axs[k].plot([r[sname][cn][key] for cn in conds], yy + (j - 0.5) * 0.25, "o", color=c, ms=5,
                        label={"shhs_test": "SHHS test", "mesa_external": "MESA (external)"}.get(sname, sname))
        axs[k].set_xlabel(label)
        axs[k].grid(axis="x", lw=0.3, alpha=0.5)
    axs[1].set_xlim(0.5, 1.02)
    axs[2].set_xlim(-0.02, 1.02)
    axs[0].set_yticks(yy, conds)
    axs[0].invert_yaxis()
    axs[1].legend(loc="lower left", fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig4_robustness.png", dpi=300)
else:
    print("robustness.json not found: run scripts/robustness.py first for fig 4")

print(f"saved figures to {out}")
