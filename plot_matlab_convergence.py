"""
plot_matlab_convergence.py
==========================
Reads the fold_N_convergence.csv files from matlab_method/results/ and
produces a figure showing training and validation RMSE across all five folds.

Output: figures/matlab_convergence.png  (upload to Overleaf as matlab_convergence.png)

Usage:
    python plot_matlab_convergence.py
"""

import os
import csv
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join("matlab_method", "results")
OUT_PATH    = "figures/matlab_convergence.png"

FOLD_COLOURS = ["#4C8EDA", "#e64a19", "#2ca02c", "#9467bd", "#8c564b"]

os.makedirs("figures", exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 5.5))

for fold_idx in range(1, 6):
    fname  = os.path.join(RESULTS_DIR, f"fold_{fold_idx}_convergence.csv")
    col    = FOLD_COLOURS[fold_idx - 1]
    epochs, train_rmse, val_rmse = [], [], []

    with open(fname, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ep = int(row["Epoch"])
            except (KeyError, ValueError):
                continue
            epochs.append(ep)

            try:
                tr = float(row["TrainRMSE"])
            except (ValueError, KeyError):
                tr = float("nan")
            train_rmse.append(tr)

            try:
                vr = float(row["ValRMSE"])
            except (ValueError, KeyError):
                vr = float("nan")
            val_rmse.append(vr)

    ax.plot(epochs, train_rmse, color=col, linestyle="--", linewidth=1.2, alpha=0.6)
    ax.plot(epochs, val_rmse,   color=col, linestyle="-",  linewidth=1.8)

# Formatting
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("RMSE (fractional hours)", fontsize=11)
ax.set_title("Training & Validation RMSE per Epoch", fontsize=12, fontweight="bold")
ax.grid(linestyle="--", alpha=0.4)
ax.set_xlim(left=1)

from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], color="gray", linestyle="--", label="Train"),
    Line2D([0], [0], color="gray", linestyle="-",  label="Validation"),
]
ax.legend(handles=handles, fontsize=10)

fig.suptitle("MATLAB SqueezeNet CNN — Convergence Curves (5-Fold CV)",
             fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {OUT_PATH}")
