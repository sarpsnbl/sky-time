"""
tune.py
=======
Optuna hyperparameter optimisation for Time-of-Day Estimation.
Searches over convnext_tiny and convnext_small only.

Usage
-----
    python tune.py
    python tune.py --model convnext_tiny   # restrict to one variant
"""

import argparse
import gc
import os
import time
import multiprocessing as mp
from typing import Optional

import numpy as np
import torch
from torch.optim import AdamW

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
except ImportError:
    raise ImportError("Install optuna:  pip install optuna")

from config import Config as cfg
from TimeOfDayDataLoader import (
    TimeOfDayDataset,
    create_dataloaders,
    get_transforms,
    MINUTES_PER_DAY,
)
from main import (
    TimeOfDayModel,
    CyclicMSELoss,
    train_one_epoch,
    evaluate,
    save_checkpoint,
    get_scheduler,
)

SUPPORTED_MODELS = ["convnext_tiny", "convnext_small"]


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

def get_search_space(trial: optuna.Trial, model_name: str) -> dict:
    """
    Optuna search space tailored for N=2500 with ConvNeXt backbones.
    Relaxed regularization to allow deeper feature learning.
    """
    # --- 1. Learning Rate & Optimizer ---
    # Shifted down slightly. With more data, smoother/slower convergence is safer.
    lr           = trial.suggest_float("lr",           5e-5, 1e-3, log=True)
    # Relaxed weight decay significantly. We don't need to choke the weights anymore.
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 5e-2, log=True) 
    eta_min      = trial.suggest_float("eta_min",      1e-7, 1e-5, log=True)

    # --- 2. Architecture & Dropout ---
    # Shifted up. The MLP can handle a wider representation now.
    hidden_dim   = trial.suggest_categorical("hidden_dim", [384, 512, 768])
    # Lowered the dropout floor. The model can trust its pathways more.
    dropout      = trial.suggest_float("dropout",      0.10, 0.40)
    
    # Unfreezing deeper. You have the data to train earlier stages of ConvNeXt safely.
    freeze_until = trial.suggest_categorical(
        "freeze_until", ["features.2", "features.4", "features.5", "freatures.6"]
    )

    # --- 3. Augmentations ---
    # Re-introducing 'none'. With 2500 native images, heavy augmentation might just be noise.
    aug_magnitude = trial.suggest_categorical("aug_magnitude", ["none", "light", "moderate"])

    # --- 4. Label Noise & Mixup ---
    # STILL HARDCODED TO 0.0: Linear mixup math still breaks cyclic (sin/cos) targets.
    mixup_alpha  = 0.0  
    
    label_noise  = trial.suggest_float("label_noise",  0.0, 0.03)

    return {
        "lr":            lr,
        "weight_decay":  weight_decay,
        "hidden_dim":    hidden_dim,
        "dropout":       dropout,
        "freeze_until":  freeze_until,
        "eta_min":       eta_min,
        "aug_magnitude": aug_magnitude,
        "mixup_alpha":   mixup_alpha,
        "label_noise":   label_noise,
        "batch_size":    16,
    }


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def objective(
    trial:      optuna.Trial,
    device:     torch.device,
    train_dataset:    TimeOfDayDataset,
    val_dataset:    TimeOfDayDataset,
    model_list: list,
) -> float:
    model_name = trial.suggest_categorical("model", model_list)
    cfg.MODEL  = model_name

    params = get_search_space(trial, model_name)

    fold_maes = []

    for fold_idx in range(cfg.OPTUNA_CV_FOLDS):
        train_loader, val_loader = create_dataloaders(
            train_dataset,
            val_dataset,
            fold=fold_idx,
            n_splits=cfg.N_SPLITS,
            batch_size=params["batch_size"],
            num_workers=cfg.NUM_WORKERS,
            val_ratio=cfg.VAL_RATIO,
            use_weighted_sampler=cfg.WEIGHTED_SAMPLER,
            persistent_workers=True,
        )

        try:
            model = TimeOfDayModel(
                pretrained=cfg.PRETRAINED,
                freeze_until=params["freeze_until"],
                hidden_dim=params["hidden_dim"],
                dropout=params["dropout"],
            ).to(device)
        except Exception as exc:
            print(f"  Trial {trial.number}: model build failed ({exc}), pruning.")
            raise optuna.exceptions.TrialPruned()

        criterion = CyclicMSELoss()
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=params["lr"],
            weight_decay=params["weight_decay"],
        )
        scheduler = get_scheduler(
            optimizer, epochs=cfg.OPTUNA_EPOCHS, eta_min=params["eta_min"],
        )
        scaler = (
            torch.amp.GradScaler('cuda')
            if (cfg.USE_AMP and device.type == "cuda") else None
        )

        best_fold_val_mae   = float("inf")
        best_fold_train_mae = float("inf")
        best_fold_epoch     = -1

        for epoch in range(cfg.OPTUNA_EPOCHS):
            _, train_mae = train_one_epoch(
                model, train_loader, optimizer, criterion, device, scaler,
                mixup_alpha=params["mixup_alpha"],
                label_noise=params["label_noise"],
            )
            _, val_mae = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            if val_mae < best_fold_val_mae:
                best_fold_val_mae   = val_mae
                best_fold_train_mae = train_mae
                best_fold_epoch     = epoch + 1

            step = fold_idx * cfg.OPTUNA_EPOCHS + epoch
            trial.report(val_mae, step)
            if trial.should_prune():
                del model, optimizer, train_loader, val_loader
                gc.collect()
                torch.cuda.empty_cache()
                raise optuna.exceptions.TrialPruned()

        fold_maes.append({
            "val_mae":   best_fold_val_mae,
            "train_mae": best_fold_train_mae,
            "epoch":     best_fold_epoch,
        })

        del model, optimizer, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()

    mean_mae = float(np.mean([r["val_mae"] for r in fold_maes]))

    print(f"\n  Trial {trial.number} [{model_name}] completed:")
    for i, r in enumerate(fold_maes):
        print(f"    Fold {i} | Best epoch: {r['epoch']:>2} | "
              f"Train MAE: {r['train_mae']:.2f} | Val MAE: {r['val_mae']:.2f}")
    print(f"  → Mean val MAE: {mean_mae:.2f}\n")

    return mean_mae


# ---------------------------------------------------------------------------
# Study runner
# ---------------------------------------------------------------------------

def run_study(device: torch.device, model_list: list) -> None:
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    study_name   = "tod_hpo_convnext"
    storage_path = os.path.join(cfg.OUTPUT_DIR, "optuna_convnext.db")
    storage_url  = f"sqlite:///{storage_path}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="minimize",
        sampler=TPESampler(seed=cfg.SEED, n_startup_trials=cfg.OPTUNA_N_STARTUP_TRIALS),
        pruner=MedianPruner(
            n_startup_trials=cfg.OPTUNA_N_STARTUP_TRIALS,
            n_warmup_steps=max(1, cfg.OPTUNA_EPOCHS // 4),
        ),
    )

    completed = len([t for t in study.trials if t.state.is_finished()])
    remaining = cfg.OPTUNA_N_TRIALS - completed

    print(f"\n{'='*60}")
    print(f"  Models       : {', '.join(model_list)}")
    print(f"  Study        : '{study_name}'")
    print(f"  Storage      : {storage_path}")
    print(f"  Completed    : {completed} / {cfg.OPTUNA_N_TRIALS} trials")
    print(f"  Remaining    : {remaining}")
    print(f"  Epochs/trial : {cfg.OPTUNA_EPOCHS}  ×  {cfg.OPTUNA_CV_FOLDS} fold(s)")
    print(f"  Device       : {device}")
    print(f"{'='*60}\n")

    print(f"Loading dataset from {cfg.IMAGE_DIR} …")
    train_dataset = TimeOfDayDataset(
            image_dir=cfg.IMAGE_DIR,
            transform=get_transforms(augment=True, magnitude=cfg.AUG_MAGNITUDE),
        )
    val_dataset = TimeOfDayDataset(
            image_dir=cfg.IMAGE_DIR,
            transform=get_transforms(augment=False),
        )

    if remaining > 0:
        study.optimize(
            lambda trial: objective(trial, device, train_dataset, val_dataset, model_list),
            n_trials=remaining,
            timeout=cfg.OPTUNA_TIMEOUT_SECONDS,
            show_progress_bar=True,
            gc_after_trial=True,
        )

    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"  Best trial   : #{best.number}")
    print(f"  Val MAE      : {best.value:.2f} min  ({best.value/60:.2f} h)")
    print(f"\n  Suggested config.py overrides:")
    print(f"  {'─'*40}")
    for k, v in best.params.items():
        print(f"    {k:20s} = {v!r}")
    print(f"{'='*60}\n")

    _retrain_best(best.params, device)


# ---------------------------------------------------------------------------
# Retrain best config
# ---------------------------------------------------------------------------

def _retrain_best(params: dict, device: torch.device) -> None:
    model_name = params.get("model", cfg.MODEL)
    cfg.MODEL  = model_name

    print(f"Retraining best config ({model_name}) for {cfg.EPOCHS} epochs …")

    dataset = TimeOfDayDataset(
        image_dir=cfg.IMAGE_DIR,
        transform=get_transforms(augment=True, magnitude=params.get("aug_magnitude", "none")),
    )
    train_loader, val_loader = create_dataloaders(
        dataset,
        fold=cfg.FOLD,
        n_splits=cfg.N_SPLITS,
        batch_size=params.get("batch_size", cfg.BATCH_SIZE),
        num_workers=cfg.NUM_WORKERS,
        val_ratio=cfg.VAL_RATIO,
        use_weighted_sampler=cfg.WEIGHTED_SAMPLER,
        persistent_workers=True,
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
    scheduler  = get_scheduler(optimizer, epochs=cfg.EPOCHS, eta_min=params["eta_min"])
    scaler = (
        torch.amp.GradScaler('cuda')
        if (cfg.USE_AMP and device.type == "cuda") else None
    )

    best_val_mae = float("inf")
    best_ckpt    = os.path.join(cfg.OUTPUT_DIR, "optuna_best_convnext.pt")

    print(f"\n{'Epoch':>6}  {'Train MAE':>10}  {'Val MAE':>10}  {'LR':>10}  {'Time':>7}")
    print("-" * 50)

    for epoch in range(cfg.EPOCHS):
        t0 = time.time()
        _, train_mae = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            mixup_alpha=params.get("mixup_alpha", cfg.MIXUP_ALPHA),
            label_noise=params.get("label_noise", cfg.LABEL_NOISE_STD),
        )
        _, val_mae = evaluate(
            model, val_loader, criterion, device,
            use_tta=cfg.TTA_ENABLED, tta_passes=cfg.TTA_FLIPS,
        )
        scheduler.step()

        print(f"{epoch+1:>6}  {train_mae:>9.2f}m  {val_mae:>9.2f}m  "
              f"{scheduler.get_last_lr()[0]:>10.2e}  {time.time()-t0:>6.1f}s")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            save_checkpoint(model, optimizer, scheduler, epoch + 1, val_mae, best_ckpt)
            print(f"  ★ New best: {best_val_mae:.2f} min")

    print(f"\nBest checkpoint → {best_ckpt}")
    print(f"Best val MAE    : {best_val_mae:.2f} min")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna HPO — ConvNeXt")
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        default=None,
        help="Restrict search to one ConvNeXt variant (default: both).",
    )
    args = parser.parse_args()

    model_list = [args.model] if args.model else SUPPORTED_MODELS

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

    run_study(device, model_list)