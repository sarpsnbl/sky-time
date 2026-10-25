"""
probe_boundaries.py
===================
Systematic discrete boundary probe for Swin-T hyperparameters.
Evaluates key boundary landmarks across learning rate, weight decay,
mixup, loss beta, dropout, augmentation, and freeze depth on dataset_224
to empirically determine optimal search spaces for continuous HPO.

Outputs:
  - checkpoints/boundary_probe_results.json
  - checkpoints/narrowed_search_space.json
  - figures/boundary_sensitivity_analysis.png
"""

import os
import sys
import copy
import time
import json
import argparse
from typing import Dict, List, Any

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config as cfg
from TimeOfDayDataLoader import (
    TimeOfDayDataset,
    create_dataloaders,
    get_transforms,
)
from Main import (
    TimeOfDayModel,
    get_loss_criterion,
    get_optimizer,
    get_scheduler,
    train_one_epoch,
    evaluate,
    build_and_compile_model,
)

def build_probe_candidates(baseline: dict) -> List[Dict[str, Any]]:
    """Builds 15 discrete landmark configurations anchored around the baseline."""
    candidates = []

    # 1. Anchor baseline
    c = copy.deepcopy(baseline)
    candidates.append({
        "id": "anchor_baseline",
        "category": "Baseline",
        "desc": "Current best parameters (Anchor)",
        "params": c,
    })

    # 2. Learning rate bounds
    c = copy.deepcopy(baseline)
    c["lr"] = 8.0e-5
    candidates.append({
        "id": "lr_low",
        "category": "Learning Rate",
        "desc": "LR = 8.0e-5 (Lower Bound)",
        "params": c,
    })

    c = copy.deepcopy(baseline)
    c["lr"] = 4.5e-4
    candidates.append({
        "id": "lr_high",
        "category": "Learning Rate",
        "desc": "LR = 4.5e-4 (Upper Bound)",
        "params": c,
    })

    # 3. Weight decay bounds
    c = copy.deepcopy(baseline)
    c["weight_decay"] = 0.005
    candidates.append({
        "id": "wd_low",
        "category": "Weight Decay",
        "desc": "WD = 0.005 (Light Regularization)",
        "params": c,
    })

    c = copy.deepcopy(baseline)
    c["weight_decay"] = 0.080
    candidates.append({
        "id": "wd_high",
        "category": "Weight Decay",
        "desc": "WD = 0.080 (Heavy Regularization)",
        "params": c,
    })

    # 4. Mixup alpha bounds
    c = copy.deepcopy(baseline)
    c["mixup_alpha"] = 0.0
    candidates.append({
        "id": "mixup_none",
        "category": "Mixup Alpha",
        "desc": "Mixup = 0.0 (No Regularization)",
        "params": c,
    })

    c = copy.deepcopy(baseline)
    c["mixup_alpha"] = 0.15
    candidates.append({
        "id": "mixup_high",
        "category": "Mixup Alpha",
        "desc": "Mixup = 0.15 (Aggressive Blending)",
        "params": c,
    })

    # 5. Huber loss transition beta bounds
    c = copy.deepcopy(baseline)
    c["beta_minutes"] = 5.0
    candidates.append({
        "id": "beta_tight",
        "category": "Huber Beta",
        "desc": "Beta = 5.0m (Strict L1)",
        "params": c,
    })

    c = copy.deepcopy(baseline)
    c["beta_minutes"] = 20.0
    candidates.append({
        "id": "beta_wide",
        "category": "Huber Beta",
        "desc": "Beta = 20.0m (Broad Quadratic Basin)",
        "params": c,
    })

    # 6. Dropout bounds
    c = copy.deepcopy(baseline)
    c["dropout"] = 0.02
    candidates.append({
        "id": "drop_low",
        "category": "Dropout",
        "desc": "Dropout = 0.02 (Minimal)",
        "params": c,
    })

    c = copy.deepcopy(baseline)
    c["dropout"] = 0.18
    candidates.append({
        "id": "drop_high",
        "category": "Dropout",
        "desc": "Dropout = 0.18 (Heavy)",
        "params": c,
    })

    # 7. Augmentation magnitude
    c = copy.deepcopy(baseline)
    c["aug_magnitude"] = "moderate"
    candidates.append({
        "id": "aug_moderate",
        "category": "Augmentation",
        "desc": "Aug = moderate (Standard Jitter)",
        "params": c,
    })

    # 8. Backbone freeze depth
    c = copy.deepcopy(baseline)
    c["freeze_until"] = "features.4"
    candidates.append({
        "id": "freeze_shallow",
        "category": "Freeze Depth",
        "desc": "Freeze = features.4 (More Fine-Tuned Layers)",
        "params": c,
    })

    # 9. Combinatorial Corner 1: Conservative (low LR, low WD, gentle mixup)
    c = copy.deepcopy(baseline)
    c["lr"] = 1.2e-4
    c["weight_decay"] = 0.015
    c["mixup_alpha"] = 0.03
    candidates.append({
        "id": "corner_conservative",
        "category": "Interaction",
        "desc": "Corner: Low LR (1.2e-4) + Low WD (0.015) + Mixup (0.03)",
        "params": c,
    })

    # 10. Combinatorial Corner 2: Aggressive (higher LR, high WD, stronger mixup)
    c = copy.deepcopy(baseline)
    c["lr"] = 3.5e-4
    c["weight_decay"] = 0.060
    c["mixup_alpha"] = 0.10
    candidates.append({
        "id": "corner_aggressive",
        "category": "Interaction",
        "desc": "Corner: High LR (3.5e-4) + High WD (0.060) + Mixup (0.10)",
        "params": c,
    })

    return candidates

def evaluate_candidate(
    cand: dict,
    epochs: int,
    image_dir: str,
    image_size: int,
    device: torch.device,
    seed: int = 42,
) -> dict:
    """Evaluates a single hyperparameter candidate on Fold 0."""
    params = cand["params"]

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    cfg.IMAGE_SIZE  = image_size
    cfg.IMAGE_DIR   = image_dir
    cfg.USE_COMPILE = False

    train_dataset = TimeOfDayDataset(
        image_dir=image_dir,
        target_size=image_size,
        transform=get_transforms(augment=True, target_size=image_size, magnitude=params.get("aug_magnitude", "heavy")),
    )
    val_dataset = TimeOfDayDataset(
        image_dir=image_dir,
        target_size=image_size,
        transform=get_transforms(augment=False, target_size=image_size),
    )

    batch_size = params.get("batch_size", 8)
    accum_steps = params.get("accum_steps", 1)

    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset, fold=0, n_splits=cfg.N_SPLITS,
        batch_size=batch_size, num_workers=cfg.NUM_WORKERS,
        val_ratio=cfg.VAL_RATIO, use_weighted_sampler=cfg.WEIGHTED_SAMPLER,
        persistent_workers=True,
    )

    model = build_and_compile_model(device, params)
    criterion = get_loss_criterion(cfg.LOSS_FUNCTION, params.get("beta_minutes", 10.0))
    optimizer = get_optimizer(model, params["lr"], params["weight_decay"])
    scheduler = get_scheduler(optimizer, epochs=epochs, eta_min=params.get("eta_min", 5.0e-6))
    scaler = torch.amp.GradScaler("cuda") if (cfg.USE_AMP and device.type == "cuda") else None

    history = {
        "epoch": [],
        "train_mae": [],
        "val_mae": [],
        "epoch_time": [],
    }

    best_val_mae = float("inf")
    best_epoch = -1

    for ep in range(epochs):
        t0 = time.time()
        _, train_mae = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            mixup_alpha=params.get("mixup_alpha", 0.05),
            label_noise=params.get("label_noise", 0.015),
            accum_steps=accum_steps,
        )
        _, val_mae = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        is_best = val_mae < best_val_mae
        if is_best:
            best_val_mae = val_mae
            best_epoch = ep + 1

        history["epoch"].append(ep + 1)
        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)
        history["epoch_time"].append(elapsed)

        marker = " ★ best" if is_best else ""
        print(f"    Ep {ep+1:>2}/{epochs} | Train: {train_mae:6.1f}m | Val: {val_mae:6.1f}m | {elapsed:4.1f}s{marker}", flush=True)

    del model, optimizer, train_loader, val_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = {
        "id": cand["id"],
        "category": cand["category"],
        "desc": cand["desc"],
        "params": params,
        "best_val_mae": float(best_val_mae),
        "best_epoch": int(best_epoch),
        "final_val_mae": float(history["val_mae"][-1]),
        "final_train_mae": float(history["train_mae"][-1]),
        "mean_epoch_time": float(np.mean(history["epoch_time"])),
        "history": history,
    }
    return result

def plot_sensitivity_analysis(results: List[dict], anchor_mae: float, out_path: str):
    """Generates publication-quality sensitivity plot showing impact of each boundary."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    
    # Filter out baseline
    non_baseline = [r for r in results if r["id"] != "anchor_baseline"]
    
    # Sort by Delta from Anchor
    deltas = [r["best_val_mae"] - anchor_mae for r in non_baseline]
    labels = [r["desc"] for r in non_baseline]
    categories = [r["category"] for r in non_baseline]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300, gridspec_kw={"width_ratios": [1.4, 1.0]})
    
    # Left: Horizontal bar chart of Delta MAE
    y_pos = np.arange(len(labels))
    colors = ["#2ecc71" if d <= 0 else "#e74c3c" for d in deltas]
    
    ax1.barh(y_pos, deltas, color=colors, height=0.65, edgecolor="black", linewidth=0.8, alpha=0.9)
    ax1.axvline(0, color="black", linestyle="--", linewidth=1.2, label=f"Anchor Baseline ({anchor_mae:.1f}m)")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=9.5, fontweight="medium")
    ax1.invert_yaxis()  # Top down
    ax1.set_xlabel(r"$\Delta$ Validation MAE relative to Anchor (Minutes, lower is better)", fontsize=11, fontweight="bold")
    ax1.set_title("Hyperparameter Boundary Sensitivity vs. Anchor", fontsize=13, fontweight="bold", pad=12)
    ax1.grid(axis="x", alpha=0.3, linestyle=":")
    ax1.legend(loc="lower right", framealpha=0.9)

    for i, d in enumerate(deltas):
        text = f"{d:+.1f}m"
        x_pos = d + (0.3 if d >= 0 else -0.3)
        ha = "left" if d >= 0 else "right"
        ax1.text(x_pos, i, text, va="center", ha=ha, fontsize=9, fontweight="bold",
                 color="#c0392b" if d > 0 else "#27ae60")

    # Right: Categorical breakdown by feature
    cat_map = {}
    for r in results:
        cat = r["category"]
        if cat not in cat_map:
            cat_map[cat] = []
        cat_map[cat].append((r["id"], r["best_val_mae"]))

    # Bar chart of best achieved per category
    cat_names = []
    cat_best_maes = []
    cat_worst_maes = []

    for cat, items in cat_map.items():
        if cat in ["Baseline", "Interaction"]:
            continue
        cat_names.append(cat)
        maes = [m for _, m in items]
        cat_best_maes.append(min(maes))
        cat_worst_maes.append(max(maes))

    x = np.arange(len(cat_names))
    width = 0.35
    ax2.bar(x - width/2, cat_best_maes, width, label="Best at Boundary", color="#3498db", edgecolor="black", linewidth=0.8)
    ax2.bar(x + width/2, cat_worst_maes, width, label="Worst at Boundary", color="#95a5a6", edgecolor="black", linewidth=0.8)
    ax2.axhline(anchor_mae, color="red", linestyle="--", linewidth=1.2, label=f"Anchor ({anchor_mae:.1f}m)")
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(cat_names, rotation=35, ha="right", fontsize=9.5, fontweight="medium")
    ax2.set_ylabel("Validation MAE (Minutes)", fontsize=11, fontweight="bold")
    ax2.set_title("Boundary Dispersion Across Parameters", fontsize=13, fontweight="bold", pad=12)
    ax2.grid(axis="y", alpha=0.3, linestyle=":")
    ax2.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Sensitivity plot saved -> {out_path}", flush=True)

def derive_narrowed_search_space(results: List[dict], anchor_mae: float) -> dict:
    """Derives focused continuous HPO search space based on empirical boundary response."""
    by_id = {r["id"]: r["best_val_mae"] for r in results}

    # 1. Learning Rate
    lr_low_mae = by_id.get("lr_low", 999)
    lr_high_mae = by_id.get("lr_high", 999)
    if lr_high_mae < lr_low_mae and lr_high_mae < anchor_mae:
        lr_range = [2.0e-4, 5.0e-4]
    elif lr_low_mae < lr_high_mae and lr_low_mae < anchor_mae:
        lr_range = [7.0e-5, 2.5e-4]
    else:
        lr_range = [1.2e-4, 3.5e-4]

    # 2. Weight Decay
    wd_low_mae = by_id.get("wd_low", 999)
    wd_high_mae = by_id.get("wd_high", 999)
    if wd_low_mae < wd_high_mae:
        wd_range = [0.005, 0.040]
    else:
        wd_range = [0.020, 0.075]

    # 3. Mixup Alpha
    mix_none_mae = by_id.get("mixup_none", 999)
    mix_high_mae = by_id.get("mixup_high", 999)
    if mix_none_mae < mix_high_mae:
        mixup_range = [0.0, 0.07]
    else:
        mixup_range = [0.03, 0.12]

    # 4. Beta Minutes
    beta_tight_mae = by_id.get("beta_tight", 999)
    beta_wide_mae = by_id.get("beta_wide", 999)
    if beta_tight_mae < beta_wide_mae:
        beta_range = [4.0, 12.0]
    else:
        beta_range = [8.0, 20.0]

    # 5. Dropout
    drop_low_mae = by_id.get("drop_low", 999)
    drop_high_mae = by_id.get("drop_high", 999)
    if drop_low_mae < drop_high_mae:
        drop_range = [0.01, 0.10]
    else:
        drop_range = [0.06, 0.18]

    # 6. Categoricals
    aug_mod_mae = by_id.get("aug_moderate", 999)
    aug_choices = ["heavy"] if aug_mod_mae > anchor_mae else ["moderate", "heavy"]

    freeze_shallow_mae = by_id.get("freeze_shallow", 999)
    freeze_choices = ["features.6"] if freeze_shallow_mae > anchor_mae else ["features.4", "features.6"]

    narrowed_space = {
        "model": "swin_t",
        "rationale": "Derived from discrete boundary sensitivity probe on dataset_224",
        "search_space": {
            "lr": {"type": "float", "low": lr_range[0], "high": lr_range[1], "log": True},
            "weight_decay": {"type": "float", "low": wd_range[0], "high": wd_range[1], "log": True},
            "mixup_alpha": {"type": "float", "low": mixup_range[0], "high": mixup_range[1]},
            "beta_minutes": {"type": "float", "low": beta_range[0], "high": beta_range[1]},
            "dropout": {"type": "float", "low": drop_range[0], "high": drop_range[1]},
            "aug_magnitude": {"type": "categorical", "choices": aug_choices},
            "freeze_until": {"type": "categorical", "choices": freeze_choices},
            "eta_min": {"type": "float", "low": 2.0e-6, "high": 8.0e-6, "log": True},
            "label_noise": {"type": "float", "low": 0.005, "high": 0.025},
        }
    }
    return narrowed_space

def main():
    parser = argparse.ArgumentParser(description="Systematic Discrete Boundary Probe")
    parser.add_argument("--epochs", type=int, default=12, help="Epochs per candidate configuration")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--image-dir", type=str, default="dataset_224", help="Dataset directory")
    parser.add_argument("--image-size", type=int, default=224, help="Target image size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70, flush=True)
    print("  SYSTEMATIC HYPERPARAMETER BOUNDARY PROBE", flush=True)
    print(f"  Device: {device} | Dataset: {args.image_dir} ({args.image_size}px) | Epochs: {args.epochs}", flush=True)
    print("=" * 70, flush=True)

    baseline = cfg.load_params_from_json()
    baseline["batch_size"] = args.batch_size
    baseline["accum_steps"] = 1

    candidates = build_probe_candidates(baseline)
    n_cands = len(candidates)
    print(f"Total candidates to evaluate: {n_cands}", flush=True)

    results = []
    total_start = time.time()

    for idx, cand in enumerate(candidates):
        print("\n" + "=" * 65, flush=True)
        print(f"  [{idx+1:>2}/{n_cands}] Candidate: {cand['id']} ({cand['category']})", flush=True)
        print(f"  Description: {cand['desc']}", flush=True)
        print("=" * 65, flush=True)

        res = evaluate_candidate(
            cand=cand,
            epochs=args.epochs,
            image_dir=args.image_dir,
            image_size=args.image_size,
            device=device,
            seed=42,
        )
        results.append(res)

        print(f"  -> Best Val MAE: {res['best_val_mae']:.2f}m (Epoch {res['best_epoch']}) | Avg epoch: {res['mean_epoch_time']:.1f}s", flush=True)

    total_time = time.time() - total_start
    anchor_mae = next(r["best_val_mae"] for r in results if r["id"] == "anchor_baseline")

    # Summary table
    print("\n" + "=" * 80, flush=True)
    print("  BOUNDARY PROBE SUMMARY & SENSITIVITY BREAKDOWN", flush=True)
    print("=" * 80, flush=True)
    print(f" {'ID':<20} | {'Category':<14} | {'Best Val':^10} | {'Delta':^10} | {'Rank':^6}", flush=True)
    print("-" * 80, flush=True)

    sorted_results = sorted(results, key=lambda x: x["best_val_mae"])
    rank_map = {r["id"]: i + 1 for i, r in enumerate(sorted_results)}

    for r in results:
        delta = r["best_val_mae"] - anchor_mae
        delta_str = f"{delta:+.2f}m" if r["id"] != "anchor_baseline" else "ANCHOR"
        print(f" {r['id']:<20} | {r['category']:<14} | {r['best_val_mae']:>7.2f}m  | {delta_str:>9} | #{rank_map[r['id']]:<4}", flush=True)

    print("-" * 80, flush=True)
    print(f"Total Probe Runtime: {total_time/60:.1f} minutes", flush=True)

    # Save detailed JSON
    json_path = os.path.join(cfg.OUTPUT_DIR, "boundary_probe_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "epochs": args.epochs,
            "anchor_mae": anchor_mae,
            "candidates": results,
        }, f, indent=4)
    print(f"  Results saved -> {json_path}", flush=True)

    # Plot figure
    plot_path = os.path.join("figures", "boundary_sensitivity_analysis.png")
    plot_sensitivity_analysis(results, anchor_mae, plot_path)

    # Derive narrowed search space
    narrowed = derive_narrowed_search_space(results, anchor_mae)
    space_path = os.path.join(cfg.OUTPUT_DIR, "narrowed_search_space.json")
    with open(space_path, "w", encoding="utf-8") as f:
        json.dump(narrowed, f, indent=4)
    print(f"  Narrowed search space saved -> {space_path}", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
