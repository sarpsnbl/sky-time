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
    fig, ax = plt.subplots(figsize=(10, 12.8), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.8, 12.8)
    ax.axis("off")

    w, h = 4.2, 0.95

    nodes = {
        "img":      (2.5, 12.0, "Sky Image\n(512 × 512 × 3)", "#E8F0FE", "#1A73E8"),
        "swin_t":   (2.5, 9.8,  "Swin-T Backbone\n(Shifted-Window Attention)", "#FCE8E6", "#D93025"),
        "gap":        (2.5, 7.6,  "Global Average Pooling\nVisual Embedding (768-D)", "#E6F4EA", "#1E8E3E"),
        "meta":       (7.5, 12.0, "Multi-Tier Metadata\n30 Physical + 25 Optical PCA (55-D)", "#FEF7E0", "#F9AB00"),
        "film_gen":   (7.5, 9.8,  "FiLM Generator MLP\nLinear(55, 384) → LN → GELU\nLinear(384, 1536)", "#FEF7E0", "#E37400"),
        "gamma_beta": (7.5, 7.6,  "FiLM Coefficients\nγ (768-D scale),  β (768-D bias)", "#FEF7E0", "#F9AB00"),
        "film_mod":   (5.0, 5.4,  "FiLM Feature Modulation\nx̂ = (1 + γ) ⊙ x + β  (768-D)", "#F3E8FD", "#9334E6"),
        "mlp1":       (5.0, 3.7,  "Dense Block 1\nLinear(768, 384) → LN → GELU → Dropout", "#E8F0FE", "#1A73E8"),
        "mlp2":       (5.0, 2.1,  "Dense Block 2\nLinear(384, 192) → LN → GELU → Dropout", "#E8F0FE", "#1A73E8"),
        "mixture_head": (5.0, 0.5, "Bimodal Circular Mixture Head\n[π₁, π₂] = Softmax(ℓ / τ)  |  v̂₁, v̂₂ ∈ S¹ × S¹", "#E6F4EA", "#1E8E3E"),
        "loss":       (5.0, -0.9, "Circular Mixture Loss (Softmin NLL)\nL_mix = -τ log ∑ πₖ exp(-L_Huber(v̂ₖ, y) / τ)", "#FCE8E6", "#D93025"),
    }

    for name, (x, y, text, fcolor, ecolor) in nodes.items():
        node_w = 7.4 if name in ["film_mod", "mlp1", "mlp2", "mixture_head", "loss"] else (4.4 if "FiLM" in text else w)
        box = patches.FancyBboxPatch(
            (x - node_w / 2, y - h / 2), node_w, h,
            boxstyle="round,pad=0.15,rounding_size=0.25",
            edgecolor=ecolor, facecolor=fcolor, linewidth=1.8, zorder=2
        )
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center", fontsize=9.2, fontweight="bold",
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
    arrow("mlp2", "mixture_head")
    arrow("mixture_head", "loss")

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

    # Threshold markers (staggered to completely eliminate text collisions)
    ax2.axhline(90.0, color="#2ca02c", linestyle="--", linewidth=1.2, alpha=0.8)
    ax2.text(2.0, 92.0, "90% Variance Threshold", color="#2ca02c", fontsize=9.5, fontweight="bold")
    final_var = cum_var[-1] * 100
    ax2.axhline(final_var, color=color_line, linestyle=":", linewidth=1.2, alpha=0.8)
    ax2.text(n_comp * 0.40, final_var + 2.5, f"25 PCs: {final_var:.1f}% Total Variance", color=color_line, fontsize=9.5, fontweight="bold")

    plt.title(f"PCA Variance Decomposition (97 Optical Descriptors → {n_comp} PCs)", fontweight="bold", pad=12)
    fig.tight_layout()
    scree_path = os.path.join(out_dir, "pca_scree_variance.png")
    plt.savefig(scree_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Feature Loadings
    print("  [3/9] Generating PCA Optical Feature Loadings ...")
    components = pca.components_  # (25, 97)
    
    # Accurate 97-feature names mapping
    feat_names = []
    for c in ['R', 'G', 'B']: feat_names += [f'{c} Mean', f'{c} Std']
    for c in ['H', 'S', 'V']: feat_names += [f'HSV {c} Mean', f'HSV {c} Std']
    for c in ['L*', 'a*', 'b*']: feat_names += [f'LAB {c} Mean', f'LAB {c} Std']
    for c in ['R', 'G', 'B']:
        for b in range(8): feat_names.append(f'RGB {c} Bin {b+1}')
    for c in ['H', 'S', 'V']:
        for b in range(8): feat_names.append(f'HSV {c} Bin {b+1}')
    feat_names += [
        'Zenith V Mean', 'Zenith V Std', 'Vert V Gradient', 'R/B Ratio',
        'Lum Mean', 'Lum Std', 'Lum Entropy', 'S Mean', 'S Std',
        'Canny Edge Density', 'Laplacian Texture Var', 'Vert Lum Gradient',
        'Horizontal Asymmetry', 'Zenith-Horizon Chroma Shift', 'McCamy CCT',
        'DCP Mean', 'DCP Std', 'DCP P90', 'Sun Centroid X', 'Sun Centroid Y', 'Solar Glare Ratio'
    ]
    for b in range(5): feat_names.append(f'LAB b* Band {b+1}')
    for b in range(5): feat_names.append(f'HSV V Band {b+1}')

    # Pick top contributing features from PC1
    pc1 = components[0]
    top_indices = np.argsort(np.abs(pc1))[-15:][::-1]
    top_weights = pc1[top_indices]
    
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=300)
    y_pos = np.arange(len(top_indices))
    labels = [feat_names[idx] for idx in top_indices]
    
    colors = [PALETTE["primary"] if w >= 0 else PALETTE["red"] for w in top_weights]
    ax.barh(y_pos, top_weights, color=colors, alpha=0.85, height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9.5)
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)

    # Distinct colors for the 5 folds
    fold_colors = ["#1a73e8", "#12b5cb", "#34a853", "#f9ab00", "#9334e6"]

    # MAE Convergence
    fold_best_maes = []
    for fd in folds:
        records = sorted(epochs_per_fold[fd], key=lambda x: x["epoch"])
        ep = [r["epoch"] for r in records]
        val_mae = [r["val_mae"] for r in records]
        best_v = min(val_mae)
        fold_best_maes.append(best_v)

        col = fold_colors[fd % len(fold_colors)]
        ax1.plot(ep, val_mae, alpha=0.35, linewidth=1.2, color=col, label=f"Fold {fd} (Min: {best_v:.1f}m)")

    # Calculate average across available folds
    max_ep = max(len(epochs_per_fold[fd]) for fd in folds)
    mean_val = []
    mean_train = []
    mean_val_loss = []
    mean_train_loss = []
    for e in range(1, max_ep + 1):
        vals = [r["val_mae"] for fd in folds for r in epochs_per_fold[fd] if r["epoch"] == e]
        trains = [r["train_mae"] for fd in folds for r in epochs_per_fold[fd] if r["epoch"] == e]
        v_losses = [r["val_loss"] for fd in folds for r in epochs_per_fold[fd] if r["epoch"] == e]
        t_losses = [r["train_loss"] for fd in folds for r in epochs_per_fold[fd] if r["epoch"] == e]
        if vals:
            mean_val.append(np.mean(vals))
            mean_train.append(np.mean(trains))
            mean_val_loss.append(np.mean(v_losses))
            mean_train_loss.append(np.mean(t_losses))

    eps = np.arange(1, len(mean_val) + 1)
    ax1.plot(eps, mean_train, color=PALETTE["secondary"], linewidth=2.2, label="Mean Train MAE")
    ax1.plot(eps, mean_val, color=PALETTE["dark"], linewidth=2.6, label=f"5-Fold Mean Val ({np.mean(fold_best_maes):.2f}m)")

    ax1.set_xlabel("Training Epoch", fontweight="bold")
    ax1.set_ylabel("Mean Absolute Error (Minutes)", fontweight="bold")
    ax1.set_title("Swin-T Multi-Fold MAE Convergence", fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", framealpha=0.92, fontsize=8.5)
    ax1.set_ylim(0, min(180, max(mean_val) * 1.05 if mean_val else 180))

    # Loss Convergence across all folds
    for fd in folds:
        records = sorted(epochs_per_fold[fd], key=lambda x: x["epoch"])
        ep = [r["epoch"] for r in records]
        val_loss = [r["val_loss"] for r in records]
        col = fold_colors[fd % len(fold_colors)]
        ax2.plot(ep, val_loss, color=col, alpha=0.25, linewidth=1.0)

    ax2.plot(eps, mean_train_loss, color=PALETTE["secondary"], linewidth=2.2, label="Mean Train Loss")
    ax2.plot(eps, mean_val_loss, color=PALETTE["primary"], linewidth=2.6, label="Mean Val Loss")

    ax2.set_xlabel("Training Epoch", fontweight="bold")
    ax2.set_ylabel("Circular Mixture Loss (Softmin NLL)", fontweight="bold")
    ax2.set_title("Circular Mixture Loss Convergence (5-Fold CV)", fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", framealpha=0.92, fontsize=9)
    ax2.set_ylim(bottom=0.0)

    fig.tight_layout()
    conv_path = os.path.join(out_dir, "training_convergence.png")
    plt.savefig(conv_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Fold MAE Bar Chart
    print("  [5/9] Generating Cross-Validation Fold Bar Chart ...")
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=300)
    fold_names = [f"Fold {i}" for i in range(len(fold_best_maes))]
    colors = [PALETTE["primary"]] * len(fold_best_maes)

    bars = ax.bar(fold_names, fold_best_maes, color=colors, width=0.52, alpha=0.85, edgecolor=PALETTE["dark"], linewidth=1.0)
    mean_score = float(np.mean(fold_best_maes))
    std_score = float(np.std(fold_best_maes))

    for bar, score in zip(bars, fold_best_maes):
        y_pos = bar.get_height() + 1.2
        if abs(y_pos - mean_score) < 1.8:
            y_pos = mean_score + 1.8
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos, f"{score:.1f}m",
                ha="center", va="bottom", fontsize=10.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#bdc1c6", alpha=0.95, linewidth=0.8))

    ax.axhline(mean_score, color=PALETTE["secondary"], linestyle="--", linewidth=2.0,
               label=f"5-Fold Mean MAE: {mean_score:.2f} ± {std_score:.2f} min")
    ax.set_ylabel("Validation MAE (Minutes)", fontweight="bold")
    ax.set_title("Swin-T 5-Fold Cross-Validation Performance", fontweight="bold", pad=12)
    ax.set_ylim(0, max(fold_best_maes) * 1.30)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper right", framealpha=0.95)

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
    mode1s = []
    mode2s = []
    prob1s = []
    prob2s = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("type") == "image":
                    actuals.append(rec["actual_min"])
                    preds.append(rec["pred_min"])
                    mode1s.append(rec.get("mode1_min", rec["pred_min"]))
                    mode2s.append(rec.get("mode2_min", rec["pred_min"]))
                    prob1s.append(rec.get("prob1", 1.0))
                    prob2s.append(rec.get("prob2", 0.0))
            except Exception:
                pass

    if not actuals:
        print("  [!] No image prediction records found in log.")
        return None, None, None

    actuals = np.array(actuals)
    preds = np.array(preds)
    mode1s = np.array(mode1s)
    mode2s = np.array(mode2s)
    prob1s = np.array(prob1s)
    prob2s = np.array(prob2s)

    errors_prim = np.array([_cyclic_diff(a, p) for a, p in zip(actuals, preds)])
    
    # Oracle Best-of-2 Disambiguation
    oracle_preds = []
    for a, m1, m2 in zip(actuals, mode1s, mode2s):
        e1 = _cyclic_diff(a, m1)
        e2 = _cyclic_diff(a, m2)
        oracle_preds.append(m1 if e1 <= e2 else m2)
    oracle_preds = np.array(oracle_preds)
    errors_oracle = np.array([_cyclic_diff(a, op) for a, op in zip(actuals, oracle_preds)])

    mae_prim = float(np.mean(errors_prim))
    med_prim = float(np.median(errors_prim))
    p75_prim = float(np.percentile(errors_prim, 75))
    flips_prim = int(np.sum(errors_prim > 300))
    pct_flips_prim = (flips_prim / len(errors_prim)) * 100.0

    mae_oracle = float(np.mean(errors_oracle))
    med_oracle = float(np.median(errors_oracle))
    p75_oracle = float(np.percentile(errors_oracle, 75))
    flips_oracle = int(np.sum(errors_oracle > 300))
    pct_flips_oracle = (flips_oracle / len(errors_oracle)) * 100.0

    # 1. Dual-Panel Prediction Scatter Plot (Primary vs Oracle)
    print(f"  [6/9] Generating Ground Truth vs. Predicted Scatter Plot (N={len(actuals)}) ...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 6.2), dpi=300)
    actual_hrs = actuals / 60.0
    pred_hrs = preds / 60.0
    oracle_hrs = oracle_preds / 60.0

    # Panel A: Primary Mode
    sc1 = ax1.scatter(actual_hrs, pred_hrs, alpha=0.45, s=20, c=errors_prim, cmap="viridis", vmin=0, vmax=180, edgecolors="none")
    ax1.plot([0, 24], [0, 24], color="#d93025", linestyle="--", linewidth=1.8, label="Ideal y = x")
    ax1.fill_between([0, 24], [0.5, 24.5], [-0.5, 23.5], color="#1a73e8", alpha=0.08, label="±30 min band")

    ax1.set_xlim(0, 24)
    ax1.set_ylim(0, 24)
    ax1.set_xticks(range(0, 25, 4))
    ax1.set_yticks(range(0, 25, 4))
    ax1.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 4)])
    ax1.set_yticklabels([f"{h:02d}:00" for h in range(0, 25, 4)])
    ax1.set_xlabel("Ground Truth Time-of-Day (HH:MM)", fontweight="bold")
    ax1.set_ylabel("Predicted Time-of-Day (HH:MM)", fontweight="bold")
    ax1.set_title("(a) Primary Mode Prediction (Highest Prior $\\pi_k$)", fontweight="bold", pad=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", framealpha=0.92, fontsize=8.5)

    stats_text1 = (f"Mean MAE: {mae_prim:.2f} min\n"
                   f"Median Error: {med_prim:.2f} min\n"
                   f"75th Percentile: {p75_prim:.2f} min\n"
                   f"AM/PM Flips (>300m): {flips_prim} ({pct_flips_prim:.2f}%)")
    ax1.text(0.97, 0.04, stats_text1, transform=ax1.transAxes, ha="right", va="bottom",
             fontsize=9, bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#bdc1c6", alpha=0.92))

    # Panel B: Oracle Mode
    sc2 = ax2.scatter(actual_hrs, oracle_hrs, alpha=0.45, s=20, c=errors_oracle, cmap="viridis", vmin=0, vmax=180, edgecolors="none")
    ax2.plot([0, 24], [0, 24], color="#d93025", linestyle="--", linewidth=1.8, label="Ideal y = x")
    ax2.fill_between([0, 24], [0.5, 24.5], [-0.5, 23.5], color="#1a73e8", alpha=0.08, label="±30 min band")

    ax2.set_xlim(0, 24)
    ax2.set_ylim(0, 24)
    ax2.set_xticks(range(0, 25, 4))
    ax2.set_yticks(range(0, 25, 4))
    ax2.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 4)])
    ax2.set_yticklabels([f"{h:02d}:00" for h in range(0, 25, 4)])
    ax2.set_xlabel("Ground Truth Time-of-Day (HH:MM)", fontweight="bold")
    ax2.set_ylabel("Predicted Time-of-Day (HH:MM)", fontweight="bold")
    ax2.set_title("(b) Dual-Mode Disambiguation (Oracle Best-of-2)", fontweight="bold", pad=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", framealpha=0.92, fontsize=8.5)

    stats_text2 = (f"Oracle MAE: {mae_oracle:.2f} min\n"
                   f"Oracle Median: {med_oracle:.2f} min\n"
                   f"75th Percentile: {p75_oracle:.2f} min\n"
                   f"AM/PM Flips (>300m): {flips_oracle} ({pct_flips_oracle:.2f}%)")
    ax2.text(0.97, 0.04, stats_text2, transform=ax2.transAxes, ha="right", va="bottom",
             fontsize=9, bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#bdc1c6", alpha=0.92))

    fig.subplots_adjust(right=0.90)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.70])
    cbar = fig.colorbar(sc2, cax=cbar_ax)
    cbar.set_label("Absolute Error (Minutes)", fontweight="bold")

    scatter_path = os.path.join(out_dir, "predictions_scatter.png")
    plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Polar 24-Hour Clock Error Plot
    print("  [7/9] Generating Diurnal Polar Clock Error Representation ...")
    fig = plt.figure(figsize=(7.5, 7.5), dpi=300)
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")  # 00:00 at top
    ax.set_theta_direction(-1)       # Clockwise

    # Theta in radians [0, 2pi)
    theta = (actuals / 1440.0) * (2 * np.pi)
    radii = errors_prim

    scatter = ax.scatter(theta, radii, c=radii, cmap="magma", s=22, alpha=0.55, edgecolors="none")
    ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    ax.set_xticklabels(["00:00\n(Midnight)", "03:00", "06:00\n(Dawn)", "09:00",
                        "12:00\n(Solar Noon)", "15:00", "18:00\n(Dusk)", "21:00"], fontsize=9.5, fontweight="bold")
    
    # Set radial label position to 22.5 degrees (01:30) so it never collides with 03:00 or Dusk/Dawn tick labels
    ax.set_rlabel_position(22.5)
    ax.set_yticks([60, 180, 360, 540, 720])
    ax.set_yticklabels(["60m", "180m", "360m", "540m", "720m"], fontsize=8.5, color="#5f6368")
    ax.set_ylim(0, 750)
    ax.set_title("Diurnal 24-Hour Clock Error Distribution on S¹", fontweight="bold", pad=22, fontsize=12)
    ax.grid(True, alpha=0.35)

    cbar = plt.colorbar(scatter, ax=ax, pad=0.08, fraction=0.046)
    cbar.set_label("Cyclic Error (Minutes)", fontweight="bold")

    fig.tight_layout()
    polar_path = os.path.join(out_dir, "cyclic_polar_error.png")
    plt.savefig(polar_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Dual-Panel Error Distribution & Cumulative Distribution (ECDF)
    print("  [8/9] Generating Error Distribution & Percentile Breakdown ...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)

    # Panel A: Core Error Density (0-60m zoom)
    core_errors = errors_prim[errors_prim <= 60.0]
    pct_core = (len(core_errors) / len(errors_prim)) * 100.0

    ax1.hist(core_errors, bins=30, range=(0, 60), color=PALETTE["primary"], alpha=0.75,
             edgecolor=PALETTE["dark"], linewidth=0.8, density=True)
    ax1.axvline(med_prim, color=PALETTE["secondary"], linestyle="-", linewidth=2.0, label=f"Median: {med_prim:.1f} min")
    ax1.axvline(p75_prim, color=PALETTE["green"], linestyle="--", linewidth=1.8, label=f"75th Percentile: {p75_prim:.1f} min")
    p80_prim = float(np.percentile(errors_prim, 80))
    ax1.axvline(p80_prim, color="#f9ab00", linestyle=":", linewidth=2.0, label=f"80th Percentile: {p80_prim:.1f} min")

    ax1.set_xlabel("Absolute Error (Minutes)", fontweight="bold")
    ax1.set_ylabel("Probability Density", fontweight="bold")
    ax1.set_title(f"(a) Core Error Density (0–60 min Zoom, {pct_core:.1f}% of Dataset)", fontweight="bold", pad=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", framealpha=0.92, fontsize=9)

    # Panel B: Full Diurnal Error ECDF on S¹ (0-720 min)
    sorted_prim = np.sort(errors_prim)
    ecdf_prim = np.arange(1, len(sorted_prim) + 1) / len(sorted_prim)

    sorted_oracle = np.sort(errors_oracle)
    ecdf_oracle = np.arange(1, len(sorted_oracle) + 1) / len(sorted_oracle)

    ax2.plot(sorted_prim, ecdf_prim * 100.0, color=PALETTE["primary"], linewidth=2.4,
             label=f"Primary Mode (MAE: {mae_prim:.1f}m, Med: {med_prim:.1f}m)")
    ax2.plot(sorted_oracle, ecdf_oracle * 100.0, color="#d93025", linewidth=2.2, linestyle="--",
             label=f"Oracle Disambiguation (MAE: {mae_oracle:.1f}m, Med: {med_oracle:.1f}m)")

    # Reference lines at 50%, 75%, 90%
    ax2.axhline(50, color="#80868b", linestyle=":", alpha=0.6, linewidth=1.0)
    ax2.axhline(75, color="#80868b", linestyle=":", alpha=0.6, linewidth=1.0)
    ax2.axhline(90, color="#80868b", linestyle=":", alpha=0.6, linewidth=1.0)
    ax2.text(680, 51, "50%", fontsize=8, color="#5f6368", ha="right")
    ax2.text(680, 76, "75%", fontsize=8, color="#5f6368", ha="right")
    ax2.text(680, 91, "90%", fontsize=8, color="#5f6368", ha="right")

    ax2.set_xlim(0, 720)
    ax2.set_ylim(0, 103)
    ax2.set_xlabel("Error Threshold (Minutes)", fontweight="bold")
    ax2.set_ylabel("Cumulative Percentage (%)", fontweight="bold")
    ax2.set_title("(b) Full Diurnal Error Cumulative Distribution (CDF on S¹)", fontweight="bold", pad=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right", framealpha=0.92, fontsize=9)

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
    ax.set_ylim(0, 360)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper left", framealpha=0.95, edgecolor="#bdc1c6")

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
