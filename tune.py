"""
tune.py
=======
Optional Optuna hyperparameter optimisation for Time-of-Day Estimation.

Usage
-----
Run a fresh study (SQLite backend so it survives crashes / can be resumed):

    python tune.py

Resume an existing study with the same name:

    python tune.py            # Optuna automatically continues where it left off

After tuning, the best trial's parameters are printed and the best checkpoint
is saved to  checkpoints/optuna_best.pt  so you can drop them straight into
config.py and run main.py as normal.

Searchable parameters
---------------------
    lr              log-uniform  [1e-5, 3e-3]
    weight_decay    log-uniform  [1e-5, 1e-2]
    hidden_dim      categorical  [128, 256, 512]
    dropout         uniform      [0.1, 0.5]
    batch_size      categorical  [16, 32, 64]
    freeze_until    categorical  ["layer1", "layer2", "layer3", "layer4"]
"""

import gc
import os
import time
import argparse
import multiprocessing as mp
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
except ImportError:
    raise ImportError(
        "Optuna is required for hyperparameter tuning.\n"
        "Install it with:  pip install optuna"
    )

from config import Config as cfg
from TimeOfDayDataLoader import (
    TimeOfDayDataset,
    create_dataloaders,
    decode_time_tensor,
    get_transforms,
    MINUTES_PER_DAY,
)
from main import (
    TimeOfDayModel,
    CyclicMSELoss,
    cyclic_mae_minutes,
    train_one_epoch,
    evaluate,
    save_checkpoint,
)


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def objective(trial: optuna.Trial, device: torch.device, dataset: TimeOfDayDataset) -> float:
    """
    One Optuna trial = one full short training run.
    Uses the pre-loaded dataset passed from run_study.
    """

    # ── Suggest hyperparameters ────────────────────────────────────────────
    lr           = trial.suggest_float("lr",           1e-3, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 5e-4, log=True)
    hidden_dim   = trial.suggest_categorical("hidden_dim",   [180, 256, 320])
    dropout      = trial.suggest_float("dropout",      0.2,  0.4)
    batch_size   = trial.suggest_categorical("batch_size",   [16, 24, 32])
    freeze_until = trial.suggest_categorical(
        "freeze_until", ["layer1", "layer2", "layer3", "layer4"]
    )

    # ── Dataloaders ───────────────────────────────────────────────────
    train_loader, val_loader = create_dataloaders(
        dataset,
        fold=cfg.FOLD,
        n_splits=cfg.N_SPLITS,
        batch_size=batch_size,
        num_workers=cfg.NUM_WORKERS,
        val_ratio=cfg.VAL_RATIO,
        use_weighted_sampler=cfg.WEIGHTED_SAMPLER,
    )

    # ── Model ──────────────────────────────────────────────────────────────
    model = TimeOfDayModel(
        pretrained=cfg.PRETRAINED,
        freeze_until=freeze_until,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    criterion = CyclicMSELoss()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=cfg.OPTUNA_EPOCHS, eta_min=1e-6
    )
    scaler = (
        torch.cuda.amp.GradScaler()
        if (cfg.USE_AMP and device.type == "cuda")
        else None
    )

    # ── Short training run with pruning ───────────────────────────────────
    best_val_mae = float("inf")

    for epoch in range(cfg.OPTUNA_EPOCHS):
        train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_mae = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        trial.report(val_mae, epoch)
        if trial.should_prune():
            # Clean up before pruning to free memory
            del model, optimizer, train_loader, val_loader
            gc.collect()
            torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()

        if val_mae < best_val_mae:
            best_val_mae = val_mae

    # 3. Final cleanup for this trial
    del model, optimizer, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()

    return best_val_mae


# ---------------------------------------------------------------------------
# Study runner
# ---------------------------------------------------------------------------

def run_study(device: torch.device) -> None:
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print(f"Loading dataset from {cfg.IMAGE_DIR}...")
    train_tf = get_transforms(augment=True)
    shared_dataset = TimeOfDayDataset(image_dir=cfg.IMAGE_DIR, transform=train_tf)

    storage_path = os.path.join(cfg.OUTPUT_DIR, "optuna_study.db")
    storage_url  = f"sqlite:///{storage_path}"

    study = optuna.create_study(
        study_name=cfg.OPTUNA_STUDY_NAME,
        storage=storage_url,
        load_if_exists=True,
        direction="minimize",
        sampler=TPESampler(seed=cfg.SEED),
        pruner=MedianPruner(
            n_startup_trials=cfg.OPTUNA_N_STARTUP_TRIALS,
            n_warmup_steps=max(1, cfg.OPTUNA_EPOCHS // 4),
        ),
    )

    completed = len([t for t in study.trials if t.state.is_finished()])
    remaining = cfg.OPTUNA_N_TRIALS - completed

    remaining = cfg.OPTUNA_N_TRIALS - completed
    print(f"\n{'='*60}")
    print(f"  Optuna HPO  —  study: '{cfg.OPTUNA_STUDY_NAME}'")
    print(f"  Storage    : {storage_path}")
    print(f"  Completed  : {completed} / {cfg.OPTUNA_N_TRIALS} trials")
    print(f"  Remaining  : {remaining}")
    print(f"  Epochs/trial: {cfg.OPTUNA_EPOCHS}")
    print(f"  Device     : {device}")
    print(f"{'='*60}\n")

    if remaining <= 0:
        print("All trials already completed. Showing best result.\n")
    else:
        study.optimize(
            lambda trial: objective(trial, device, shared_dataset),
            n_trials=remaining,
            timeout=cfg.OPTUNA_TIMEOUT_SECONDS,
            show_progress_bar=True,
            gc_after_trial=True,
        )

    # ── Results ────────────────────────────────────────────────────────────
    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"  Best trial : #{best.number}")
    print(f"  Val MAE    : {best.value:.2f} min  ({best.value/60:.2f} h)")
    print(f"\n  Suggested config.py overrides:")
    print(f"  {'─'*40}")
    for k, v in best.params.items():
        print(f"    {k:20s} = {v!r}")
    print(f"{'='*60}\n")

    # Retrain best config to save a proper checkpoint
    _save_best_checkpoint(best.params, device)


def _save_best_checkpoint(params: dict, device: torch.device) -> None:
    """Retrain the best config for cfg.EPOCHS and save a checkpoint."""
    print("Retraining best configuration for full epochs …")

    train_tf = get_transforms(augment=True)
    dataset  = TimeOfDayDataset(image_dir=cfg.IMAGE_DIR, transform=train_tf)

    train_loader, val_loader = create_dataloaders(
        dataset,
        fold=cfg.FOLD,
        n_splits=cfg.N_SPLITS,
        batch_size=params["batch_size"],
        num_workers=cfg.NUM_WORKERS,
        val_ratio=cfg.VAL_RATIO,
        use_weighted_sampler=cfg.WEIGHTED_SAMPLER,
    )

    model = TimeOfDayModel(
        pretrained=cfg.PRETRAINED,
        freeze_until=params["freeze_until"],
        hidden_dim=params["hidden_dim"],
        dropout=params["dropout"],
    ).to(device)

    criterion = CyclicMSELoss()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=1e-6)
    scaler    = (
        torch.cuda.amp.GradScaler()
        if (cfg.USE_AMP and device.type == "cuda")
        else None
    )

    best_val_mae  = float("inf")
    best_ckpt     = os.path.join(cfg.OUTPUT_DIR, "optuna_best.pt")

    print(f"\n{'Epoch':>6}  {'Train MAE':>10}  {'Val MAE':>10}  {'Time':>7}")
    print("-" * 40)

    for epoch in range(cfg.EPOCHS):
        t0 = time.time()
        train_loss, train_mae = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_loss, val_mae = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"{epoch+1:>6}  {train_mae:>9.2f}m  {val_mae:>9.2f}m  {elapsed:>6.1f}s")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            save_checkpoint(model, optimizer, scheduler, epoch + 1,
                            val_mae, best_ckpt)
            print(f"  ★ New best: {best_val_mae:.2f} min")

    print(f"\nBest checkpoint saved → {best_ckpt}")
    print(f"Best val MAE          : {best_val_mae:.2f} min")


if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)

    torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )

    run_study(device)