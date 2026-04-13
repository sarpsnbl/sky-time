"""
visualize_training.py
=====================
Reads the JSONL log written by main.py during training and produces:

  Page 1 — Training curves
    • Train / val MAE (minutes) per epoch, per fold
    • Train / val loss per epoch, per fold

  Page 2 — Guess vs Actual scatter (best fold)
    • Each point is one validation image coloured by error magnitude
    • Diagonal = perfect prediction; dashed lines = ±30 min bands
    • Hardest N images shown as thumbnail strip with predicted/actual overlay

Usage
-----
    python visualize_training.py
    python visualize_training.py --log checkpoints/train_log.jsonl
    python visualize_training.py --log checkpoints/train_log.jsonl \\
                                 --out training_report.png --hard 12

The log file is written by the modified main.py during evaluate_with_log().
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image as PILImage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


MINUTES_PER_DAY = 1440.0


def _fmt(m: float) -> str:
    h, mn = divmod(int(round(m)) % 1440, 60)
    return f"{h:02d}:{mn:02d}"


def _cyclic_diff(a: float, b: float) -> float:
    """Cyclic absolute difference in minutes."""
    d = abs(a - b)
    return min(d, MINUTES_PER_DAY - d)


# ---------------------------------------------------------------------------
# Load log
# ---------------------------------------------------------------------------

def load_log(log_path: str):
    """
    Returns:
        epoch_records : list of dicts  (one per epoch per fold)
        image_records : list of dicts  (one per validation image, last epoch)
    """
    epoch_records = []
    image_records = []

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["type"] == "epoch":
                epoch_records.append(rec)
            elif rec["type"] == "image":
                image_records.append(rec)

    return epoch_records, image_records


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------

_FOLD_COLOURS = ["#4C8EDA", "#e64a19", "#2ca02c", "#9467bd", "#8c564b"]


def _plot_curves(epoch_records, fig, gs_row):
    """Two sub-panels: MAE and Loss, one line per fold."""
    ax_mae  = fig.add_subplot(gs_row[0])
    ax_loss = fig.add_subplot(gs_row[1])

    folds = sorted({r["fold"] for r in epoch_records})

    for fold in folds:
        col  = _FOLD_COLOURS[fold % len(_FOLD_COLOURS)]
        recs = sorted([r for r in epoch_records if r["fold"] == fold],
                      key=lambda r: r["epoch"])
        epochs     = [r["epoch"]     for r in recs]
        train_mae  = [r["train_mae"] for r in recs]
        val_mae    = [r["val_mae"]   for r in recs]
        train_loss = [r["train_loss"] for r in recs]
        val_loss   = [r["val_loss"]   for r in recs]

        lbl = f"fold {fold}"
        ax_mae.plot(epochs, train_mae, color=col, linestyle="--",
                    linewidth=1.2, alpha=0.6, label=f"{lbl} train")
        ax_mae.plot(epochs, val_mae,   color=col, linestyle="-",
                    linewidth=1.8, label=f"{lbl} val")

        ax_loss.plot(epochs, train_loss, color=col, linestyle="--",
                     linewidth=1.2, alpha=0.6)
        ax_loss.plot(epochs, val_loss,   color=col, linestyle="-",
                     linewidth=1.8)

        # mark best epoch
        best_ep  = recs[int(np.argmin(val_mae))]
        ax_mae.scatter([best_ep["epoch"]], [best_ep["val_mae"]],
                       color=col, s=60, zorder=5, marker="*")

    ax_mae.set_xlabel("Epoch", fontsize=10)
    ax_mae.set_ylabel("MAE (minutes)", fontsize=10)
    ax_mae.set_title("MAE per Epoch", fontsize=11)
    ax_mae.grid(linestyle="--", alpha=0.4)
    ax_mae.legend(fontsize=7, ncol=2)

    ax_loss.set_xlabel("Epoch", fontsize=10)
    ax_loss.set_ylabel("Loss (MSE)", fontsize=10)
    ax_loss.set_title("Loss per Epoch", fontsize=11)
    ax_loss.grid(linestyle="--", alpha=0.4)

    # dashed/solid legend
    handles = [
        plt.Line2D([0], [0], color="gray", linestyle="--", label="train"),
        plt.Line2D([0], [0], color="gray", linestyle="-",  label="val"),
    ]
    ax_loss.legend(handles=handles, fontsize=8)


def _plot_scatter(image_records, fig, gs_cell):
    """Guess vs actual scatter coloured by error, with ±30-min bands."""
    ax = fig.add_subplot(gs_cell)

    pred_h   = np.array([r["pred_min"] / 60.0 for r in image_records])
    actual_h = np.array([r["actual_min"] / 60.0 for r in image_records])
    errors   = np.array([_cyclic_diff(r["pred_min"], r["actual_min"])
                         for r in image_records])

    sc = ax.scatter(actual_h, pred_h, c=errors, cmap="RdYlGn_r",
                    vmin=0, vmax=120, s=18, alpha=0.7, linewidths=0)
    cb = plt.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("Error (min)", fontsize=9)

    # perfect-prediction diagonal
    ax.plot([0, 24], [0, 24], color="black", linewidth=1.0,
            linestyle="-", label="perfect")
    # ±30 min bands
    for offset, ls in [(0.5, ":"), (-0.5, ":")]:
        ax.plot([0, 24], [offset, 24 + offset], color="gray",
                linewidth=0.8, linestyle=ls)
    ax.fill_between([0, 24], [-0.5, 23.5], [0.5, 24.5],
                    alpha=0.08, color="green", label="±30 min")

    mean_err = errors.mean()
    median_err = float(np.median(errors))
    ax.set_title(
        f"Predicted vs Actual  |  "
        f"mean err {mean_err:.1f} min  |  median {median_err:.1f} min",
        fontsize=10,
    )
    ax.set_xlabel("Actual Time (h)", fontsize=10)
    ax.set_ylabel("Predicted Time (h)", fontsize=10)
    ax.set_xlim(0, 24); ax.set_ylim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xticklabels([f"{h:02d}h" for h in range(0, 25, 3)], fontsize=8)
    ax.set_yticks(range(0, 25, 3))
    ax.set_yticklabels([f"{h:02d}h" for h in range(0, 25, 3)], fontsize=8)
    ax.grid(linestyle="--", alpha=0.3)
    ax.legend(fontsize=8)

    return errors


def _plot_hardest(image_records, errors, n_hard: int, fig, gs_row):
    """Thumbnail strip of the N hardest images with error annotations."""
    order   = np.argsort(errors)[::-1][:n_hard]
    hardest = [image_records[i] for i in order]
    errs    = errors[order]

    axes = [fig.add_subplot(gs_row[i]) for i in range(n_hard)]

    for ax, rec, err in zip(axes, hardest, errs):
        img_path = rec.get("path", "")
        loaded   = False
        if img_path and os.path.exists(img_path):
            try:
                img = PILImage.open(img_path).convert("RGB")
                img.thumbnail((160, 160), PILImage.Resampling.LANCZOS)
                ax.imshow(np.asarray(img))
                loaded = True
            except Exception:
                pass
        if not loaded:
            ax.set_facecolor("#dddddd")
            ax.text(0.5, 0.5, "no image", transform=ax.transAxes,
                    ha="center", va="center", fontsize=7, color="gray")

        ax.set_xticks([]); ax.set_yticks([])
        # colour frame by error severity
        frame_col = "#d32f2f" if err > 120 else "#f57f17" if err > 60 else "#388e3c"
        for spine in ax.spines.values():
            spine.set_edgecolor(frame_col)
            spine.set_linewidth(2.5)

        fname = Path(img_path).name if img_path else "?"
        title = (
            f"{fname[:18]}\n"
            f"pred {_fmt(rec['pred_min'])} / actual {_fmt(rec['actual_min'])}\n"
            f"err {err:.0f} min"
        )
        ax.set_title(title, fontsize=6.5, pad=2)


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------

def plot_training(log_path: str, out_path: str, n_hard: int = 10) -> None:
    epoch_records, image_records = load_log(log_path)

    if not epoch_records:
        print("WARNING: No epoch records found in log.")
    if not image_records:
        print("WARNING: No image records found in log — "
              "scatter and hardest panels will be empty.")

    has_images = len(image_records) > 0

    # ── figure layout ─────────────────────────────────────────────────────
    # Row 0: training curves (2 cols)
    # Row 1: scatter (left) + hardest strip (right, n_hard cols)
    n_hard_actual = min(n_hard, len(image_records)) if has_images else 0

    fig = plt.figure(figsize=(max(18, 2 * n_hard_actual), 14),
                     constrained_layout=True)
    fig.suptitle("Training Report", fontsize=14, fontweight="bold")

    outer = mgridspec.GridSpec(2, 1, figure=fig,
                               height_ratios=[1.0, 1.4], hspace=0.35)

    # row 0: curves
    gs_top = mgridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[0], wspace=0.25
    )
    if epoch_records:
        _plot_curves(epoch_records, fig, gs_top)

    # row 1: scatter + hardest strip
    gs_bot = mgridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[1], height_ratios=[1.6, 1], hspace=0.45
    )

    errors = np.array([])
    if has_images:
        errors = _plot_scatter(image_records, fig, gs_bot[0])

    if n_hard_actual > 0:
        gs_strip = mgridspec.GridSpecFromSubplotSpec(
            1, n_hard_actual, subplot_spec=gs_bot[1], wspace=0.12
        )
        _plot_hardest(image_records, errors, n_hard_actual, fig, gs_strip)

        # strip title
        ax_strip_title = fig.add_subplot(gs_bot[1])
        ax_strip_title.set_axis_off()
        ax_strip_title.set_title(
            f"Hardest {n_hard_actual} Validation Images (by cyclic MAE)",
            fontsize=10, pad=4,
        )

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize training curves and per-image diagnostics."
    )
    parser.add_argument(
        "--log", default=os.path.join("checkpoints", "train_log.jsonl"),
        help="Path to the JSONL log written by main.py "
             "(default: checkpoints/train_log.jsonl)"
    )
    parser.add_argument("--out", default="training_report.png")
    parser.add_argument("--hard", type=int, default=10,
                        help="Number of hardest images to show (default: 10)")
    args = parser.parse_args()

    if not os.path.exists(args.log):
        print(f"ERROR: Log file '{args.log}' not found.")
        print("  Run main.py first, or pass --log <path>.")
        sys.exit(1)

    plot_training(args.log, args.out, n_hard=args.hard)


if __name__ == "__main__":
    main()