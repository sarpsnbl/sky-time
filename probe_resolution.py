"""
probe_resolution.py
===================
Controlled empirical probe to measure the resolution mismatch
between 224px and 512px training on Swin-T under identical hyperparameters,
data split (Fold 0), and random seed.

Usage:
    python probe_resolution.py [--epochs 15] [--batch-size 8]
"""

import os
import sys
import time
import argparse
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
    get_metadata_dim,
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

def run_training_probe(
    image_size: int,
    image_dir: str,
    epochs: int,
    params: dict,
    device: torch.device,
    seed: int = 42,
) -> dict:
    print("\n" + "-" * 60)
    print(f"  LAUNCHING PROBE RUN: {image_size}px  ({image_dir})")
    print(f"  Epochs: {epochs} | Batch Size: {params['batch_size']} | Seed: {seed}")
    print("-" * 60, flush=True)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    cfg.IMAGE_SIZE  = image_size
    cfg.IMAGE_DIR   = image_dir
    cfg.USE_COMPILE = False  # Avoid SymPy modular indexing compilation delay on non-standard Swin window sizes

    train_dataset = TimeOfDayDataset(
        image_dir=image_dir,
        target_size=image_size,
        transform=get_transforms(augment=True, target_size=image_size, magnitude=params["aug_magnitude"]),
    )
    val_dataset = TimeOfDayDataset(
        image_dir=image_dir,
        target_size=image_size,
        transform=get_transforms(augment=False, target_size=image_size),
    )

    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset, fold=0, n_splits=cfg.N_SPLITS,
        batch_size=params["batch_size"], num_workers=cfg.NUM_WORKERS,
        val_ratio=cfg.VAL_RATIO, use_weighted_sampler=cfg.WEIGHTED_SAMPLER,
        persistent_workers=True,
    )

    model = build_and_compile_model(device, params)
    criterion = get_loss_criterion(cfg.LOSS_FUNCTION, params.get("beta_minutes", 10.0))
    optimizer = get_optimizer(model, params["lr"], params["weight_decay"])
    scheduler = get_scheduler(optimizer, epochs=epochs, eta_min=params["eta_min"])
    scaler = torch.amp.GradScaler("cuda") if (cfg.USE_AMP and device.type == "cuda") else None

    history = {
        "epoch": [],
        "train_mae": [],
        "val_mae": [],
        "epoch_time": [],
    }

    best_val_mae = float("inf")

    for ep in range(epochs):
        t0 = time.time()
        _, train_mae = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            mixup_alpha=params["mixup_alpha"], label_noise=params["label_noise"],
            accum_steps=params["accum_steps"]
        )
        _, val_mae = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        is_best = val_mae < best_val_mae
        if is_best:
            best_val_mae = val_mae

        history["epoch"].append(ep + 1)
        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)
        history["epoch_time"].append(elapsed)

        marker = " ★ best" if is_best else ""
        print(f"  [{image_size:>3}px] Ep {ep+1:>2}/{epochs} | Train: {train_mae:6.1f}m | Val: {val_mae:6.1f}m | {elapsed:4.1f}s{marker}", flush=True)

    del model, optimizer, train_loader, val_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    history["best_val_mae"] = best_val_mae
    history["mean_epoch_time"] = float(np.mean(history["epoch_time"]))
    return history

def main():
    parser = argparse.ArgumentParser(description="Probe Resolution Mismatch (224px vs 512px)")
    parser.add_argument("--epochs", type=int, default=8, help="Number of probe epochs per resolution")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for controlled test")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65, flush=True)
    print("  RESOLUTION MISMATCH PROBE: 224px vs. 512px", flush=True)
    print(f"  Device: {device}")
    print("=" * 65)

    # Use standard representative parameters
    params = cfg.load_params_from_json()
    params["batch_size"] = args.batch_size
    params["accum_steps"] = 1

    # 1. Run 224px
    hist_224 = run_training_probe(
        image_size=224,
        image_dir="dataset_224",
        epochs=args.epochs,
        params=params,
        device=device,
        seed=42,
    )

    # 2. Run 512px
    hist_512 = run_training_probe(
        image_size=512,
        image_dir="dataset_512",
        epochs=args.epochs,
        params=params,
        device=device,
        seed=42,
    )

    # 3. Comparative Analysis
    print("\n" + "=" * 65)
    print("  PROBE RESULTS COMPARISON TABLE")
    print("=" * 65)
    print(f" {'Epoch':^6} | {'224px Val':^11} | {'512px Val':^11} | {'Delta (Diff)':^14} | {'Speed Ratio':^12}")
    print("-" * 65)

    deltas = []
    for e in range(args.epochs):
        v224 = hist_224["val_mae"][e]
        v512 = hist_512["val_mae"][e]
        diff = v512 - v224
        deltas.append(diff)
        sign = "+" if diff >= 0 else ""
        t224 = hist_224["epoch_time"][e]
        t512 = hist_512["epoch_time"][e]
        ratio = t512 / max(0.1, t224)
        print(f" {e+1:^6} | {v224:>9.1f}m | {v512:>9.1f}m | {sign:>5}{diff:>6.1f}m     | {ratio:>9.1f}x")

    print("-" * 65)
    print(f" Best Val MAE (224px): {hist_224['best_val_mae']:.2f} min (avg epoch: {hist_224['mean_epoch_time']:.1f}s)")
    print(f" Best Val MAE (512px): {hist_512['best_val_mae']:.2f} min (avg epoch: {hist_512['mean_epoch_time']:.1f}s)")
    
    mean_abs_delta = float(np.mean(np.abs(deltas)))
    final_delta = abs(hist_512['best_val_mae'] - hist_224['best_val_mae'])
    
    # Trajectory correlation
    corr = np.corrcoef(hist_224["val_mae"], hist_512["val_mae"])[0, 1]
    
    print(f"\n  Trajectory Correlation (r): {corr:.4f}")
    print(f"  Mean Absolute Epoch Delta:  {mean_abs_delta:.2f} min")
    print(f"  Best-Score Delta:           {final_delta:.2f} min")
    print(f"  Throughput Speedup (224px): {hist_512['mean_epoch_time']/hist_224['mean_epoch_time']:.1f}x faster")
    print("=" * 65)

    # 4. Generate Visualization
    os.makedirs("figures", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    eps = np.arange(1, args.epochs + 1)
    ax1.plot(eps, hist_224["val_mae"], color="#1A73E8", marker="o", linewidth=2.0, label="224px Val MAE")
    ax1.plot(eps, hist_512["val_mae"], color="#D93025", marker="s", linewidth=2.0, label="512px Val MAE")
    ax1.plot(eps, hist_224["train_mae"], color="#1A73E8", linestyle="--", alpha=0.5, label="224px Train")
    ax1.plot(eps, hist_512["train_mae"], color="#D93025", linestyle="--", alpha=0.5, label="512px Train")
    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylabel("MAE (Minutes)", fontweight="bold")
    ax1.set_title(f"Convergence Trajectory: 224px vs. 512px (r={corr:.3f})", fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    ax2.bar(eps, deltas, color=["#D93025" if d > 0 else "#1E8E3E" for d in deltas], width=0.6, alpha=0.8)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Epoch", fontweight="bold")
    ax2.set_ylabel("Delta: Val MAE (512px - 224px) [min]", fontweight="bold")
    ax2.set_title(f"Per-Epoch Resolution Discrepancy (Mean |Δ|: {mean_abs_delta:.1f}m)", fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    plot_path = "figures/resolution_mismatch_probe.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Comparison plot saved -> {plot_path}")

if __name__ == "__main__":
    main()
