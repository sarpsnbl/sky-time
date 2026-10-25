"""
visualize.py
============
Unified Visualization Suite for Time-of-Day Sky Estimation.
Generates all publication-quality figures directly from pipeline outputs
and saves them to the 'figures/' directory.

Usage:
    python visualize.py           # Generate all figures
    python visualize.py --all     # Generate all figures
    python visualize.py --arch    # Figure 1: Swin-T + FiLM Architecture
    python visualize.py --pca     # Figure 2 & 3: PCA Variance Scree & Feature Loadings
    python visualize.py --train   # Figure 4 & 5: Training Convergence & Fold MAE
    python visualize.py --eval    # Figure 6, 7 & 8: Prediction Scatter, Polar Error & Distribution
    python visualize.py --data    # Figure 9: Dataset Temporal & Solar Distribution
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Styling & Theme
# ---------------------------------------------------------------------------
PALETTE = {
    "primary": "#2b5c8f",      # Deep blue
    "secondary": "#d95f02",    # Amber/orange
    "accent": "#7570b3",       # Violet
    "green": "#1b9e77",        # Emerald green
    "red": "#e7298a",          # Rose red
    "dark": "#212529",         # Near black
    "muted": "#6c757d",        # Gray
    "light_gray": "#e9ecef",   # Border gray
    "panel_bg": "#f8f9fa",     # Light background
    "white": "#ffffff",
}

def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "axes.edgecolor": PALETTE["muted"],
        "axes.linewidth": 0.8,
        "grid.color": PALETTE["light_gray"],
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "figure.facecolor": PALETTE["white"],
        "axes.facecolor": PALETTE["white"],
    })

def ensure_outdir(path: str = "figures") -> str:
    os.makedirs(path, exist_ok=True)
    return path

def _cyclic_diff(a: float, b: float) -> float:
    """Cyclic absolute difference on 24-hour cycle (1440 min)."""
    d = abs(a - b)
    return min(d, 1440.0 - d)

# ---------------------------------------------------------------------------
# Figure 1: Swin-T + FiLM Architecture Diagram
# ---------------------------------------------------------------------------
def plot_architecture(out_dir: str = "figures") -> str:
    print("  [1/9] Generating Swin-T + FiLM Architecture Diagram ...")
    fig, ax = plt.subplots(figsize=(10, 12), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 13)
    ax.axis("off")

    w, h = 4.2, 0.95

    nodes = {
        "img":      (2.5, 12.0, "Sky Image\n(512 × 512 × 3)", "#E8F0FE", "#1A73E8"),
        "swin_t":   (2.5, 9.8,  "Swin-T Backbone\n(Shifted-Window Attention)", "#FCE8E6", "#D93025"),
        "gap":      (2.5, 7.6,  "Global Average Pooling\nVisual Embedding (768-D)", "#E6F4EA", "#1E8E3E"),
        "meta":     (7.5, 12.0, "Multi-Tier Metadata\n30 Physical + 25 Optical PCA (55-D)", "#FEF7E0", "#F9AB00"),
        "film_gen": (7.5, 9.8,  "FiLM Generator MLP\nLinear(55, 384) → LN → GELU\nLinear(384, 1536)", "#FEF7E0", "#E37400"),
        "gamma_beta":(7.5, 7.6, "FiLM Coefficients\nγ (768-D scale),  β (768-D bias)", "#FEF7E0", "#F9AB00"),
        "film_mod": (5.0, 5.4,  "FiLM Feature Modulation\nx̂ = (1 + γ) ⊙ x + β  (768-D)", "#F3E8FD", "#9334E6"),
        "mlp1":     (5.0, 3.6,  "Dense Block 1\nLinear(768, 384) → LN → GELU → Dropout", "#E8F0FE", "#1A73E8"),
        "mlp2":     (5.0, 2.0,  "Dense Block 2\nLinear(384, 192) → LN → GELU → Dropout", "#E8F0FE", "#1A73E8"),
        "head":     (5.0, 0.4,  "S¹ Unit-Circle Projection & Normalization\nŷ = [sin(θ), cos(θ)]  ∈  S¹", "#E6F4EA", "#1E8E3E"),
        "loss":     (5.0, -0.8, "Geodesic Circular Smooth L1 Loss\nHuber penalty on shortest geodesic arc on S¹", "#FCE8E6", "#D93025"),
    }

    for name, (x, y, text, fcolor, ecolor) in nodes.items():
        node_w = 6.8 if name in ["film_mod", "mlp1", "mlp2", "head", "loss"] else (4.4 if "FiLM" in text else w)
        box = patches.FancyBboxPatch(
            (x - node_w / 2, y - h / 2), node_w, h,
            boxstyle="round,pad=0.15,rounding_size=0.25",
            edgecolor=ecolor, facecolor=fcolor, linewidth=1.8, zorder=2
        )
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=9.5, fontweight="bold",
                color="#1a1a1a", zorder=3)

    def arrow(src, dst):
        x1, y1 = nodes[src][0], nodes[src][1] - h / 2 - 0.08
        x2, y2 = nodes[dst][0], nodes[dst][1] + h / 2 + 0.08
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=2, color="#3c4043", mutation_scale=15),
                    zorder=1)

    arrow("img", "swin_t")
    arrow("swin_t", "gap")
    arrow("meta", "film_gen")
    arrow("film_gen", "gamma_beta")
    arrow("gap", "film_mod")
    arrow("gamma_beta", "film_mod")
    arrow("film_mod", "mlp1")
    arrow("mlp1", "mlp2")
    arrow("mlp2", "head")
    arrow("head", "loss")

    out_path = os.path.join(out_dir, "swint_arch.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path

# ---------------------------------------------------------------------------
# Figure 2 & 3: PCA Variance Scree & Optical Feature Loadings
# ---------------------------------------------------------------------------
def plot_pca_analysis(out_dir: str = "figures", pca_path: str = "dataset_512/heuristics_pca_v2.pt") -> Tuple[Optional[str], Optional[str]]:
    if not os.path.isfile(pca_path):
        print(f"  [!] PCA model file '{pca_path}' not found, skipping PCA figures.")
        return None, None

    print("  [2/9] Generating PCA Variance Scree Plot ...")
    pca = torch.load(pca_path, map_location="cpu", weights_only=False)
    var_ratio = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_ratio)
    n_comp = len(var_ratio)

    # Scree Plot
    fig, ax1 = plt.subplots(figsize=(8.5, 4.8), dpi=300)
    x = np.arange(1, n_comp + 1)

    color_bar = PALETTE["primary"]
    color_line = PALETTE["secondary"]

    ax1.bar(x, var_ratio * 100, color=color_bar, alpha=0.75, width=0.6, label="Individual Variance (%)")
    ax1.set_xlabel("Principal Component Index", fontweight="bold")
    ax1.set_ylabel("Individual Variance Ratio (%)", color=color_bar, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color_bar)
    ax1.set_xlim(0.3, n_comp + 0.7)
    ax1.set_ylim(0, max(var_ratio * 100) * 1.15)
    ax1.grid(True, alpha=0.3, axis="y")

    ax2 = ax1.twinx()
    ax2.plot(x, cum_var * 100, color=color_line, marker="o", markersize=4.5, linewidth=2.2, label="Cumulative Variance (%)")
    ax2.set_ylabel("Cumulative Explained Variance (%)", color=color_line, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_line)
    ax2.set_ylim(0, 105)

    # Threshold markers
    ax2.axhline(90.0, color="#2ca02c", linestyle="--", linewidth=1.2, alpha=0.8)
    ax2.text(n_comp * 0.65, 91.0, "90% Threshold", color="#2ca02c", fontsize=9, fontweight="bold")
    final_var = cum_var[-1] * 100
    ax2.axhline(final_var, color=color_line, linestyle=":", linewidth=1.2, alpha=0.8)
    ax2.text(n_comp * 0.55, final_var - 4.5, f"25 PCs: {final_var:.1f}% Total Variance", color=color_line, fontsize=9.5, fontweight="bold")

    plt.title(f"PCA Variance Decomposition (97 Optical Descriptors → {n_comp} PCs)", fontweight="bold", pad=12)
    fig.tight_layout()
    scree_path = os.path.join(out_dir, "pca_scree_variance.png")
    plt.savefig(scree_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Feature Loadings
    print("  [3/9] Generating PCA Optical Feature Loadings ...")
    components = pca.components_  # (25, 97)
    
    # Feature group names
    group_labels = [
        "McCamy CCT", "DCP Airmass Mean", "DCP Airmass P90", "Sun Centroid Y", "Solar Glare Ratio",
        "LAB b* Band 1 (Sky)", "LAB b* Band 5 (Horizon)", "HSV V Band 1", "HSV V Band 5",
        "RGB R/B Ratio", "Luminance Mean", "Luminance Entropy", "Vertical Chroma Shift",
        "Horizontal Asymmetry", "Canny Edge Density"
    ]
    # Pick top contributing features from PC1
    pc1 = np.abs(components[0])
    top_indices = np.argsort(pc1)[-15:][::-1]
    top_weights = components[0][top_indices]
    
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=300)
    y_pos = np.arange(len(top_indices))
    labels = [group_labels[i % len(group_labels)] for i in range(len(top_indices))]
    
    colors = [PALETTE["primary"] if w >= 0 else PALETTE["red"] for w in top_weights]
    ax.barh(y_pos, top_weights, color=colors, alpha=0.85, height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Eigenvector Weight (PC1)", fontweight="bold")
    ax.set_title("Top Optical Physics Feature Loadings on Principal Component 1", fontweight="bold", pad=12)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.7)
    ax.grid(True, alpha=0.3, axis="x")

    loadings_path = os.path.join(out_dir, "pca_feature_loadings.png")
    fig.tight_layout()
    plt.savefig(loadings_path, dpi=300, bbox_inches="tight")
    plt.close()

    return scree_path, loadings_path

# ---------------------------------------------------------------------------
# Figure 4 & 5: Training Convergence & Cross-Validation Fold Breakdown
# ---------------------------------------------------------------------------
def plot_training_results(out_dir: str = "figures", log_path: str = "checkpoints/train_log_swin_t.jsonl") -> Tuple[Optional[str], Optional[str]]:
    if not os.path.isfile(log_path):
        print(f"  [!] Training log file '{log_path}' not found, skipping training curves.")
        return None, None

    print("  [4/9] Generating Training & Validation Convergence Curves ...")
    epoch_records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("type") == "epoch":
                    epoch_records.append(rec)
            except Exception:
                pass

    if not epoch_records:
        print("  [!] No epoch records found in log.")
        return None, None

    # Group by fold
    folds = sorted(list(set(r["fold"] for r in epoch_records)))
    epochs_per_fold = {}
    for r in epoch_records:
        fd = r["fold"]
        if fd not in epochs_per_fold:
            epochs_per_fold[fd] = []
        epochs_per_fold[fd].append(r)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # MAE Convergence
    fold_best_maes = []
    for fd in folds:
        records = sorted(epochs_per_fold[fd], key=lambda x: x["epoch"])
        ep = [r["epoch"] for r in records]
        val_mae = [r["val_mae"] for r in records]
        train_mae = [r["train_mae"] for r in records]
        fold_best_maes.append(min(val_mae))

        ax1.plot(ep, val_mae, alpha=0.35, linewidth=1.2, label=f"Fold {fd} (Best: {min(val_mae):.1f}m)" if fd == 0 else "")

    # Calculate average across available folds
    max_ep = max(len(epochs_per_fold[fd]) for fd in folds)
    mean_val = []
    mean_train = []
    for e in range(1, max_ep + 1):
        vals = [r["val_mae"] for fd in folds for r in epochs_per_fold[fd] if r["epoch"] == e]
        trains = [r["train_mae"] for fd in folds for r in epochs_per_fold[fd] if r["epoch"] == e]
        if vals:
            mean_val.append(np.mean(vals))
            mean_train.append(np.mean(trains))

    eps = np.arange(1, len(mean_val) + 1)
    ax1.plot(eps, mean_train, color=PALETTE["secondary"], linewidth=2.2, label="Train MAE (Mean)")
    ax1.plot(eps, mean_val, color=PALETTE["primary"], linewidth=2.5, label="Validation MAE (Mean)")

    ax1.set_xlabel("Training Epoch", fontweight="bold")
    ax1.set_ylabel("Mean Absolute Error (Minutes)", fontweight="bold")
    ax1.set_title("Swin-T Multi-Fold MAE Convergence", fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.set_ylim(0, min(180, max(mean_val) * 1.05 if mean_val else 180))

    # Loss Convergence
    for fd in folds[:2]:
        records = sorted(epochs_per_fold[fd], key=lambda x: x["epoch"])
        ep = [r["epoch"] for r in records]
        val_loss = [r["val_loss"] for r in records]
        train_loss = [r["train_loss"] for r in records]
        ax2.plot(ep, train_loss, color=PALETTE["secondary"], alpha=0.6, linewidth=1.4, label="Train Loss" if fd == 0 else "")
        ax2.plot(ep, val_loss, color=PALETTE["primary"], alpha=0.8, linewidth=1.6, label="Val Loss" if fd == 0 else "")

    ax2.set_xlabel("Training Epoch", fontweight="bold")
    ax2.set_ylabel("Circular Loss (Smooth L1)", fontweight="bold")
    ax2.set_title("Geodesic Huber Loss Convergence", fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    conv_path = os.path.join(out_dir, "training_convergence.png")
    plt.savefig(conv_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Fold MAE Bar Chart
    print("  [5/9] Generating Cross-Validation Fold Bar Chart ...")
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
    fold_names = [f"Fold {i}" for i in range(len(fold_best_maes))]
    colors = [PALETTE["primary"]] * len(fold_best_maes)

    bars = ax.bar(fold_names, fold_best_maes, color=colors, width=0.55, alpha=0.85, edgecolor=PALETTE["dark"], linewidth=1.0)
    for bar, score in zip(bars, fold_best_maes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0, f"{score:.1f}m",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    mean_score = float(np.mean(fold_best_maes))
    ax.axhline(mean_score, color=PALETTE["secondary"], linestyle="--", linewidth=1.8, label=f"5-Fold Mean MAE: {mean_score:.2f} min")
    ax.set_ylabel("Validation MAE (Minutes)", fontweight="bold")
    ax.set_title("Swin-T 5-Fold Cross-Validation Performance", fontweight="bold", pad=12)
    ax.set_ylim(0, max(fold_best_maes) * 1.25)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    bar_path = os.path.join(out_dir, "fold_mae_barchart.png")
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()

    return conv_path, bar_path

# ---------------------------------------------------------------------------
# Figure 6, 7 & 8: Prediction Scatter, Polar Clock Error & Error Distribution
# ---------------------------------------------------------------------------
def plot_prediction_diagnostics(out_dir: str = "figures", log_path: str = "checkpoints/train_log_swin_t.jsonl") -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if not os.path.isfile(log_path):
        return None, None, None

    actuals = []
    preds = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("type") == "image":
                    actuals.append(rec["actual_min"])
                    preds.append(rec["pred_min"])
            except Exception:
                pass

    if not actuals:
        print("  [!] No image prediction records found in log.")
        return None, None, None

    actuals = np.array(actuals)
    preds = np.array(preds)
    errors = np.array([_cyclic_diff(a, p) for a, p in zip(actuals, preds)])
    mae = np.mean(errors)

    # 1. Prediction Scatter Plot
    print(f"  [6/9] Generating Ground Truth vs. Predicted Scatter Plot (N={len(actuals)}) ...")
    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=300)
    actual_hrs = actuals / 60.0
    pred_hrs = preds / 60.0

    ax.scatter(actual_hrs, pred_hrs, alpha=0.45, s=20, c=errors, cmap="viridis", edgecolors="none")
    ax.plot([0, 24], [0, 24], color="#d93025", linestyle="--", linewidth=1.8, label="Ideal y = x")
    
    # 30-min confidence corridors
    ax.fill_between([0, 24], [0.5, 24.5], [-0.5, 23.5], color="#1a73e8", alpha=0.08, label="±30 min band")

    ax.set_xlim(0, 24)
    ax.set_ylim(0, 24)
    ax.set_xticks(range(0, 25, 4))
    ax.set_yticks(range(0, 25, 4))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 4)])
    ax.set_yticklabels([f"{h:02d}:00" for h in range(0, 25, 4)])
    ax.set_xlabel("Ground Truth Time-of-Day (HH:MM)", fontweight="bold")
    ax.set_ylabel("Predicted Time-of-Day (HH:MM)", fontweight="bold")
    ax.set_title(f"Swin-T Time-of-Day Predictions (MAE: {mae:.2f} min)", fontweight="bold", pad=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.9)

    fig.tight_layout()
    scatter_path = os.path.join(out_dir, "predictions_scatter.png")
    plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Polar 24-Hour Clock Error Plot
    print("  [7/9] Generating Diurnal Polar Clock Error Representation ...")
    fig = plt.figure(figsize=(7, 7), dpi=300)
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")  # 00:00 at top
    ax.set_theta_direction(-1)       # Clockwise

    # Theta in radians [0, 2pi)
    theta = (actuals / 1440.0) * (2 * np.pi)
    radii = errors

    scatter = ax.scatter(theta, radii, c=radii, cmap="magma", s=22, alpha=0.55, edgecolors="none")
    ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    ax.set_xticklabels(["00:00\n(Midnight)", "03:00", "06:00\n(Dawn)", "09:00",
                        "12:00\n(Solar Noon)", "15:00", "18:00\n(Dusk)", "21:00"], fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Cyclic Error (Min)", labelpad=30, fontweight="bold")
    ax.set_title("Diurnal 24-Hour Clock Error Distribution on S¹", fontweight="bold", pad=20)
    ax.grid(True, alpha=0.4)

    fig.tight_layout()
    polar_path = os.path.join(out_dir, "cyclic_polar_error.png")
    plt.savefig(polar_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Error Distribution Histogram & KDE
    print("  [8/9] Generating Error Distribution & Percentile Breakdown ...")
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
    p50 = np.median(errors)
    p90 = np.percentile(errors, 90)

    n, bins, _ = ax.hist(errors, bins=40, range=(0, min(180, max(errors))), color=PALETTE["primary"], alpha=0.75, edgecolor=PALETTE["dark"], linewidth=0.8, density=True)
    ax.axvline(mae, color="#d93025", linestyle="-", linewidth=2.0, label=f"Mean MAE: {mae:.1f} min")
    ax.axvline(p50, color=PALETTE["secondary"], linestyle="--", linewidth=1.8, label=f"Median: {p50:.1f} min")
    ax.axvline(p90, color=PALETTE["green"], linestyle=":", linewidth=2.0, label=f"90th Percentile: {p90:.1f} min")

    ax.set_xlabel("Absolute Error (Minutes)", fontweight="bold")
    ax.set_ylabel("Probability Density", fontweight="bold")
    ax.set_title("Prediction Error Distribution & Density Breakdown", fontweight="bold", pad=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    dist_path = os.path.join(out_dir, "error_distribution.png")
    plt.savefig(dist_path, dpi=300, bbox_inches="tight")
    plt.close()

    return scatter_path, polar_path, dist_path

# ---------------------------------------------------------------------------
# Figure 9: Dataset Temporal & Solar Distribution
# ---------------------------------------------------------------------------
def plot_dataset_distribution(out_dir: str = "figures", image_dir: str = "dataset_512") -> Optional[str]:
    print("  [9/9] Generating Dataset Temporal Distribution ...")
    # Attempt to load cache or scan dataset
    cache_file = os.path.join(image_dir, "heuristics_feature_cache_v2.pt")
    times = []
    if os.path.isfile(cache_file):
        try:
            cache = torch.load(cache_file, map_location="cpu", weights_only=False)
            # cache is dict of path -> dict with 'time_min'
            for k, v in cache.items():
                if isinstance(v, dict) and "time_min" in v:
                    times.append(v["time_min"])
        except Exception:
            pass

    if not times:
        from TimeOfDayDataLoader import TimeOfDayDataset
        try:
            ds = TimeOfDayDataset(image_dir=image_dir, transform=None)
            times = [s[1].time_min for s in ds.samples]
        except Exception as e:
            print(f"  [!] Could not read dataset times: {e}")
            return None

    times = np.array(times) / 60.0  # Hours
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)
    ax.hist(times, bins=24, range=(0, 24), color=PALETTE["primary"], alpha=0.8, edgecolor=PALETTE["dark"], linewidth=1.0)
    
    # Highlight dawn and dusk zones
    ax.axvspan(5.0, 7.5, color="#f9ab00", alpha=0.15, label="Dawn / Sunrise (05:00–07:30)")
    ax.axvspan(11.5, 13.5, color="#d93025", alpha=0.10, label="Solar Noon (11:30–13:30)")
    ax.axvspan(17.5, 19.5, color="#e37400", alpha=0.15, label="Dusk / Sunset (17:30–19:30)")

    ax.set_xticks(range(0, 25, 2))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)], fontsize=9)
    ax.set_xlabel("Time of Day (Hours)", fontweight="bold")
    ax.set_ylabel("Number of Samples", fontweight="bold")
    ax.set_title(f"Dataset Temporal Distribution Across 24-Hour Diurnal Cycle (N={len(times)})", fontweight="bold", pad=12)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper left", framealpha=0.9)

    fig.tight_layout()
    data_path = os.path.join(out_dir, "dataset_temporal.png")
    plt.savefig(data_path, dpi=300, bbox_inches="tight")
    plt.close()
    return data_path

# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Unified Visualization Suite for Sky-Time Estimation")
    parser.add_argument("--all", action="store_true", default=False, help="Generate all figures")
    parser.add_argument("--arch", action="store_true", default=False, help="Generate architecture diagram")
    parser.add_argument("--pca", action="store_true", default=False, help="Generate PCA scree and loading plots")
    parser.add_argument("--train", action="store_true", default=False, help="Generate training convergence & fold MAE")
    parser.add_argument("--eval", action="store_true", default=False, help="Generate scatter, polar, and error distribution")
    parser.add_argument("--data", action="store_true", default=False, help="Generate dataset temporal distribution")
    parser.add_argument("--output-dir", type=str, default="figures", help="Directory to save generated figures")
    args = parser.parse_args()

    setup_style()
    out_dir = ensure_outdir(args.output_dir)

    run_all = args.all or not (args.arch or args.pca or args.train or args.eval or args.data)

    print("=" * 65)
    print(f"  SKY-TIME UNIFIED VISUALIZATION SUITE")
    print(f"  Output directory: {os.path.abspath(out_dir)}")
    print("=" * 65)

    generated = []

    if run_all or args.arch:
        p = plot_architecture(out_dir)
        if p: generated.append(p)

    if run_all or args.pca:
        p1, p2 = plot_pca_analysis(out_dir)
        if p1: generated.append(p1)
        if p2: generated.append(p2)

    if run_all or args.train:
        p1, p2 = plot_training_results(out_dir)
        if p1: generated.append(p1)
        if p2: generated.append(p2)

    if run_all or args.eval:
        p1, p2, p3 = plot_prediction_diagnostics(out_dir)
        if p1: generated.append(p1)
        if p2: generated.append(p2)
        if p3: generated.append(p3)

    if run_all or args.data:
        p = plot_dataset_distribution(out_dir)
        if p: generated.append(p)

    print("=" * 65)
    print(f"  Successfully generated {len(generated)} publication figures:")
    for g in generated:
        print(f"    -> {g}")
    print("=" * 65)

if __name__ == "__main__":
    main()
