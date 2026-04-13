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
import logging
import os
import time
import multiprocessing as mp
from typing import Optional

import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm

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
    setup_logging,
    log,
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
    lr           = trial.suggest_float("lr",           5e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 5e-2, log=True)
    eta_min      = trial.suggest_float("eta_min",      1e-7, 1e-5, log=True)

    hidden_dim   = trial.suggest_categorical("hidden_dim", [384, 512, 768])
    dropout      = trial.suggest_float("dropout",      0.10, 0.40)

    freeze_until = trial.suggest_categorical(
        "freeze_until", ["features.2", "features.4", "features.5", "features.6"]
    )

    aug_magnitude = trial.suggest_categorical("aug_magnitude", ["none", "light", "moderate"])

    # Linear mixup math breaks cyclic (sin/cos) targets — keep at 0.
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
    trial:         optuna.Trial,
    device:        torch.device,
    train_dataset: TimeOfDayDataset,
    val_dataset:   TimeOfDayDataset,
    model_list:    list,
    trial_pbar:    tqdm,
) -> float:
    model_name = trial.suggest_categorical("model", model_list)
    cfg.MODEL  = model_name

    params = get_search_space(trial, model_name)

    log.debug(
        f"Trial {trial.number} [{model_name}] params: "
        + ", ".join(f"{k}={v}" for k, v in params.items() if k != "batch_size")
    )

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
            log.warning(f"Trial {trial.number}: model build failed ({exc}), pruning.")
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

        epoch_pbar = tqdm(
            range(cfg.OPTUNA_EPOCHS),
            desc=f"  T{trial.number} F{fold_idx}",
            unit="ep",
            leave=False,
            dynamic_ncols=True,
            colour="yellow",
        )

        for epoch in epoch_pbar:
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

            epoch_pbar.set_postfix(
                tr=f"{train_mae:.1f}m",
                val=f"{val_mae:.1f}m",
                best=f"{best_fold_val_mae:.1f}m",
                refresh=False,
            )

            step = fold_idx * cfg.OPTUNA_EPOCHS + epoch
            trial.report(val_mae, step)
            if trial.should_prune():
                epoch_pbar.close()
                log.debug(f"Trial {trial.number} pruned at epoch {epoch + 1}.")
                del model, optimizer, train_loader, val_loader
                gc.collect()
                torch.cuda.empty_cache()
                raise optuna.exceptions.TrialPruned()

        epoch_pbar.close()

        fold_maes.append({
            "val_mae":   best_fold_val_mae,
            "train_mae": best_fold_train_mae,
            "epoch":     best_fold_epoch,
        })

        del model, optimizer, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()

    mean_mae = float(np.mean([r["val_mae"] for r in fold_maes]))

    # One compact log line per completed trial
    fold_summary = "  ".join(
        f"F{i} ep{r['epoch']} tr{r['train_mae']:.1f}/val{r['val_mae']:.1f}m"
        for i, r in enumerate(fold_maes)
    )
    log.info(
        f"Trial {trial.number:>3} [{model_name:<15}]  mean val MAE {mean_mae:6.2f}m  |  {fold_summary}"
    )

    # Update the outer trial progress bar
    trial_pbar.set_postfix(best=f"{trial.study.best_value:.2f}m" if trial.study.best_trial else "n/a", refresh=False)
    trial_pbar.update(1)

    return mean_mae


# ---------------------------------------------------------------------------
# Study runner
# ---------------------------------------------------------------------------

def run_study(device: torch.device, model_list: list) -> None:
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    study_name   = "tod_hpo_convnext"
    storage_path = os.path.join(cfg.OUTPUT_DIR, "optuna_convnext.db")
    storage_url  = f"sqlite:///{storage_path}"

    # Silence Optuna's own verbose logging — our logger handles it
    optuna.logging.set_verbosity(optuna.logging.WARNING)

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

    log.info("=" * 60)
    log.info(f"Models        : {', '.join(model_list)}")
    log.info(f"Study         : '{study_name}'")
    log.info(f"Storage       : {storage_path}")
    log.info(f"Completed     : {completed} / {cfg.OPTUNA_N_TRIALS} trials")
    log.info(f"Remaining     : {remaining}")
    log.info(f"Epochs/trial  : {cfg.OPTUNA_EPOCHS}  x  {cfg.OPTUNA_CV_FOLDS} fold(s)")
    log.info(f"Device        : {device}")
    log.info("=" * 60)

    log.info(f"Loading dataset from {cfg.IMAGE_DIR} ...")
    train_dataset = TimeOfDayDataset(
        image_dir=cfg.IMAGE_DIR,
        transform=get_transforms(augment=True, magnitude=cfg.AUG_MAGNITUDE),
    )
    val_dataset = TimeOfDayDataset(
        image_dir=cfg.IMAGE_DIR,
        transform=get_transforms(augment=False),
    )

    if remaining > 0:
        with tqdm(
            total=remaining,
            desc="Optuna trials",
            unit="trial",
            dynamic_ncols=True,
            colour="green",
        ) as trial_pbar:
            study.optimize(
                lambda trial: objective(
                    trial, device, train_dataset, val_dataset, model_list, trial_pbar
                ),
                n_trials=remaining,
                timeout=cfg.OPTUNA_TIMEOUT_SECONDS,
                show_progress_bar=False,   # we manage our own bar
                gc_after_trial=True,
            )

    best = study.best_trial
    log.info("=" * 60)
    log.info(f"Best trial    : #{best.number}")
    log.info(f"Val MAE       : {best.value:.2f} min  ({best.value/60:.2f} h)")
    log.info("Suggested config.py overrides:")
    for k, v in best.params.items():
        log.info(f"  {k:<22} = {v!r}")
    log.info("=" * 60)

    _retrain_best(best.params, device)


# ---------------------------------------------------------------------------
# Retrain best config
# ---------------------------------------------------------------------------

def _retrain_best(params: dict, device: torch.device) -> None:
    model_name = params.get("model", cfg.MODEL)
    cfg.MODEL  = model_name

    log.info(f"Retraining best config ({model_name}) for {cfg.EPOCHS} epochs ...")

    train_dataset = TimeOfDayDataset(
        image_dir=cfg.IMAGE_DIR,
        transform=get_transforms(augment=True, magnitude=params.get("aug_magnitude", "none")),
    )
    val_dataset = TimeOfDayDataset(
        image_dir=cfg.IMAGE_DIR,
        transform=get_transforms(augment=False),
    )
    train_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
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

    n_train_batches = len(train_loader)
    n_val_batches   = len(val_loader)
    total_batches   = n_train_batches + n_val_batches

    for epoch in range(cfg.EPOCHS):
        t0 = time.time()

        desc = f"Retrain  Ep {epoch+1:>3}/{cfg.EPOCHS}"
        with tqdm(
            total=total_batches,
            desc=desc,
            unit="batch",
            leave=False,
            dynamic_ncols=True,
            colour="cyan",
        ) as pbar:
            pbar.set_description(f"{desc} [train]")
            _, train_mae = train_one_epoch(
                model, train_loader, optimizer, criterion, device, scaler,
                mixup_alpha=params.get("mixup_alpha", cfg.MIXUP_ALPHA),
                label_noise=params.get("label_noise", cfg.LABEL_NOISE_STD),
                pbar=pbar,
            )
            pbar.set_description(f"{desc} [val]  ")
            _, val_mae = evaluate(
                model, val_loader, criterion, device,
                use_tta=cfg.TTA_ENABLED, tta_passes=cfg.TTA_FLIPS,
                pbar=pbar,
            )

        scheduler.step()
        elapsed  = time.time() - t0
        is_best  = val_mae < best_val_mae

        log.info(
            f"Retrain  Ep {epoch+1:>3}/{cfg.EPOCHS}  "
            f"MAE {train_mae:6.1f}/{val_mae:6.1f}m  "
            f"lr {scheduler.get_last_lr()[0]:.2e}  {elapsed:.1f}s"
            + ("  * best" if is_best else "")
        )

        if is_best:
            best_val_mae = val_mae
            save_checkpoint(model, optimizer, scheduler, epoch + 1, val_mae, best_ckpt)

    log.info(f"Best checkpoint -> {best_ckpt}")
    log.info(f"Best val MAE    : {best_val_mae:.2f} min")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna HPO -- ConvNeXt")
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

    setup_logging(cfg.OUTPUT_DIR)
    run_study(device, model_list)