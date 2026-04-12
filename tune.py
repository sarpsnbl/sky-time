"""
tune.py
=======
Optuna hyperparameter optimisation for Time-of-Day Estimation.

Changes vs previous version
----------------------------
* Per-architecture search spaces via get_search_space().
  Each backbone family gets LR ranges, freeze strategies, aug levels, and
  warmup_epochs tuned to its training dynamics.
* Separate Optuna study per backbone family (study_name = "tod_hpo_<family>").
  Keeps TPE's internal model clean — mixing architectures confuses the sampler.
* model name is sampled inside the objective; cfg.MODEL is patched per-trial.
* OPTUNA_MODELS in config controls which backbones are included in the sweep.
* get_scheduler() is called with warmup_epochs so ViT / Swin get warmup.
* _save_best_checkpoint() now passes warmup_epochs and model name through.
* OPTUNA_N_STARTUP_TRIALS bumped to 8 (more random exploration before TPE).
* batch_size remains fixed at 24 but is also searchable for large backbones
  (convnext_base, vit_b_16) that may OOM — set to [16, 24].

Usage
-----
    python tune.py                         # runs all families in OPTUNA_MODELS
    python tune.py --family convnext       # runs only the convnext family
    python tune.py --family vit            # runs only ViT
"""

import argparse
import gc
import os
import time
import multiprocessing as mp
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
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
    cyclic_mae_minutes,
    train_one_epoch,
    evaluate,
    save_checkpoint,
    get_scheduler,
)


# ---------------------------------------------------------------------------
# Architecture families
# ---------------------------------------------------------------------------

# Maps a short family name to the list of model strings to include in that study.
# Edit OPTUNA_MODELS in config.py to control which families are run.
FAMILY_MODELS: Dict[str, list] = {
    "convnext":     ["convnext_tiny", "convnext_small"],
    "efficientnet": ["efficientnet_b3"],
    "swin":         ["swin_t"],
    "vit":          ["vit_b_16"],
    "resnet":       ["resnet50"],
}

# Default warmup epochs per model (used when warmup_epochs is not searchable)
_DEFAULT_WARMUP: Dict[str, int] = {
    "resnet50":        0,
    "efficientnet_b3": 0,
    "efficientnet_b4": 0,
    "convnext_tiny":   0,
    "convnext_small":  0,
    "convnext_base":   0,
    "swin_t":          3,
    "swin_s":          3,
    "vit_b_16":        5,
}


# ---------------------------------------------------------------------------
# Per-architecture search spaces
# ---------------------------------------------------------------------------

def get_search_space(trial: optuna.Trial, model_name: str) -> dict:
    """
    Suggest architecture-appropriate hyperparameters.

    Returns a flat dict with all keys needed to build and train a trial model.
    Keys always present:
        lr, weight_decay, hidden_dim, dropout, freeze_until,
        eta_min, mixup_alpha, label_noise, aug_magnitude,
        warmup_epochs, batch_size
    """
    name = model_name.lower()

    # ── ConvNeXt (tiny / small / base) ───────────────────────────────────
    if name in ("convnext_tiny", "convnext_small", "convnext_base"):
        batch_size = 24 if name != "convnext_base" else trial.suggest_categorical(
            "batch_size", [16, 24]
        )
        return {
            "lr": trial.suggest_float("lr", 2e-3, 3.5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 2e-4, 5e-4, log=True),
            "hidden_dim": 320,
            "dropout": trial.suggest_float("dropout", 0.14, 0.18),
            "freeze_until": "features.0",
            "eta_min": trial.suggest_float("eta_min", 1e-7, 8e-7, log=True),
            "aug_magnitude": "none",
            "mixup_alpha": trial.suggest_float("mixup_alpha", 0.12, 0.20),
            "label_noise": trial.suggest_float("label_noise", 0.03, 0.05),
            "warmup_epochs": 0,
            "batch_size": 24,
        }

    # ── EfficientNet (B3 / B4) ───────────────────────────────────────────
    elif name in ("efficientnet_b3", "efficientnet_b4"):
        return {
            "lr":           trial.suggest_float("lr",           1e-3, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 5e-4, log=True),
            "hidden_dim":   hidden_dim,
            "dropout":      trial.suggest_float("dropout",      0.2,  0.4),
            "freeze_until": trial.suggest_categorical("freeze_until",
                                ["features.4", "features.5", "features.6"]),
            "eta_min":      eta_min,
            "mixup_alpha":  trial.suggest_float("mixup_alpha",  0.0,  0.2),
            "label_noise":  label_noise,
            "aug_magnitude":trial.suggest_categorical("aug_magnitude", ["light", "medium"]),
            "warmup_epochs":0,
            "batch_size":   24,
        }

    # ── Swin Transformer (swin_t / swin_s) ───────────────────────────────
    elif name in ("swin_t", "swin_s"):
        return {
            "lr":           trial.suggest_float("lr",           5e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-4, 5e-3, log=True),
            "hidden_dim":   trial.suggest_categorical("hidden_dim_swin", [320, 384, 512]),
            "dropout":      trial.suggest_float("dropout",      0.1,  0.3),
            # Swin stage names inside backbone.features: 0=patch_embed, 2=stage1,
            # 4=stage2, 6=stage3  (odd indices are PatchMerging layers)
            "freeze_until": trial.suggest_categorical("freeze_until",
                                ["0.", "2.", "4."]),
            "eta_min":      trial.suggest_float("eta_min",      1e-6, 1e-5, log=True),
            "mixup_alpha":  trial.suggest_float("mixup_alpha",  0.0,  0.3),
            "label_noise":  trial.suggest_float("label_noise",  0.01, 0.05),
            "aug_magnitude":trial.suggest_categorical("aug_magnitude", ["light", "medium"]),
            "warmup_epochs":trial.suggest_categorical("warmup_epochs", [0, 3, 5]),
            "batch_size":   24,
        }

    # ── Vision Transformer (vit_b_16) ────────────────────────────────────
    elif name == "vit_b_16":
        # Freeze first N encoder layers; the freeze_until string selects
        # by matching a substring of parameter names.
        # encoder_layer_9 → train layers 9, 10, 11 (last 3 of 12)
        # encoder_layer_6 → train layers 6–11 (last 6 of 12)
        return {
            "lr":           trial.suggest_float("lr",           1e-4, 1e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True),
            "hidden_dim":   trial.suggest_categorical("hidden_dim_vit", [384, 512]),
            "dropout":      trial.suggest_float("dropout",      0.1,  0.4),
            "freeze_until": trial.suggest_categorical("freeze_until",
                                ["encoder_layer_6", "encoder_layer_9"]),
            "eta_min":      trial.suggest_float("eta_min",      1e-6, 1e-5, log=True),
            "mixup_alpha":  trial.suggest_float("mixup_alpha",  0.2,  0.5),
            "label_noise":  trial.suggest_float("label_noise",  0.01, 0.03),
            "aug_magnitude":trial.suggest_categorical("aug_magnitude", ["medium", "heavy"]),
            "warmup_epochs":trial.suggest_categorical("warmup_epochs", [3, 5]),
            "batch_size":   trial.suggest_categorical("batch_size", [16, 24]),
        }

    # ── ResNet-50 (baseline) ─────────────────────────────────────────────
    else:  # resnet50
        return {
            "lr":           trial.suggest_float("lr",           1e-3, 8e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-3, log=True),
            "hidden_dim":   hidden_dim,
            "dropout":      trial.suggest_float("dropout",      0.2,  0.4),
            "freeze_until": trial.suggest_categorical("freeze_until",
                                ["layer2", "layer3", "layer4"]),
            "eta_min":      eta_min,
            "mixup_alpha":  trial.suggest_float("mixup_alpha",  0.0,  0.3),
            "label_noise":  label_noise,
            "aug_magnitude":trial.suggest_categorical("aug_magnitude",
                                ["light", "medium", "heavy"]),
            "warmup_epochs":0,
            "batch_size":   24,
        }


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def objective(
    trial:      optuna.Trial,
    device:     torch.device,
    dataset:    TimeOfDayDataset,
    model_list: list,
) -> float:
    """
    One trial = one short training run averaged over OPTUNA_CV_FOLDS folds.
    The model architecture is sampled from model_list.
    Pruning fires after each epoch if the running MAE is too high.
    """
    # Sample architecture first so the rest of the space is conditioned on it
    model_name = trial.suggest_categorical("model", model_list)

    # Patch the global config so TimeOfDayModel.__init__ reads the right name
    cfg.MODEL = model_name

    params = get_search_space(trial, model_name)

    n_cv_folds = cfg.OPTUNA_CV_FOLDS
    fold_results = []

    for fold_idx in range(n_cv_folds):
        train_tf = get_transforms(augment=True, magnitude=params["aug_magnitude"])
        dataset.transform = train_tf

        train_loader, val_loader = create_dataloaders(
            dataset,
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
            # If the model fails to build (e.g. bad freeze_until for this arch)
            # prune the trial gracefully rather than crashing the study.
            print(f"  Trial {trial.number}: model build failed ({exc}), pruning.")
            raise optuna.exceptions.TrialPruned()

        criterion = CyclicMSELoss()
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=params["lr"],
            weight_decay=params["weight_decay"],
        )
        scheduler = get_scheduler(
            optimizer,
            epochs=cfg.OPTUNA_EPOCHS,
            eta_min=params["eta_min"],
            warmup_epochs=params["warmup_epochs"],
        )
        scaler = (
            torch.amp.GradScaler('cuda')
            if (cfg.USE_AMP and device.type == "cuda") else None
        )

        # Track the best metrics for this specific fold
        best_fold_val_mae = float("inf")
        best_fold_train_mae = float("inf")
        best_fold_epoch = -1

        for epoch in range(cfg.OPTUNA_EPOCHS):
            # Capture train_mae from the training loop
            _, train_mae = train_one_epoch(
                model, train_loader, optimizer, criterion, device, scaler,
                mixup_alpha=params["mixup_alpha"],
                label_noise=params["label_noise"],
            )
            _, val_mae = evaluate(
                model, val_loader, criterion, device,
                use_tta=False, tta_passes=1,
            )
            scheduler.step()

            # Update best metrics if validation improves
            if val_mae < best_fold_val_mae:
                best_fold_val_mae = val_mae
                best_fold_train_mae = train_mae
                best_fold_epoch = epoch + 1

            step = fold_idx * cfg.OPTUNA_EPOCHS + epoch
            trial.report(val_mae, step)
            if trial.should_prune():
                del model, optimizer, train_loader, val_loader
                gc.collect()
                torch.cuda.empty_cache()
                raise optuna.exceptions.TrialPruned()

        # Save the detailed results for this fold
        fold_results.append({
            "val_mae": best_fold_val_mae,
            "train_mae": best_fold_train_mae,
            "epoch": best_fold_epoch
        })
        
        del model, optimizer, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()

    mean_mae = float(np.mean([res["val_mae"] for res in fold_results]))
    
    # Print out the detailed summary for the trial
    print(f"\n  Trial {trial.number} [{model_name}] Completed:")
    for i, res in enumerate(fold_results):
        print(f"    Fold {i} | Best Epoch: {res['epoch']:>2} | Train MAE: {res['train_mae']:.2f} | Val MAE: {res['val_mae']:.2f}")
    print(f"  → Trial Mean Val MAE: {mean_mae:.2f}\n")
    
    return mean_mae


# ---------------------------------------------------------------------------
# Per-family study runner
# ---------------------------------------------------------------------------

def run_family_study(
    family:     str,
    model_list: list,
    device:     torch.device,
    dataset:    TimeOfDayDataset,
) -> None:
    study_name   = f"tod_hpo_{family}"
    storage_path = os.path.join(cfg.OUTPUT_DIR, f"optuna_{family}.db")
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
    print(f"  Family       : {family}  ({', '.join(model_list)})")
    print(f"  Study        : '{study_name}'")
    print(f"  Storage      : {storage_path}")
    print(f"  Completed    : {completed} / {cfg.OPTUNA_N_TRIALS} trials")
    print(f"  Remaining    : {remaining}")
    print(f"  Epochs/trial : {cfg.OPTUNA_EPOCHS}  ×  {cfg.OPTUNA_CV_FOLDS} fold(s)")
    print(f"  Device       : {device}")
    print(f"{'='*60}\n")

    if remaining <= 0:
        print("All trials already completed for this family.\n")
    else:
        study.optimize(
            lambda trial: objective(trial, device, dataset, model_list),
            n_trials=remaining,
            timeout=cfg.OPTUNA_TIMEOUT_SECONDS,
            show_progress_bar=True,
            gc_after_trial=True,
        )

    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"  [{family}] Best trial : #{best.number}")
    print(f"  [{family}] Val MAE    : {best.value:.2f} min  ({best.value/60:.2f} h)")
    print(f"\n  Suggested config.py overrides:")
    print(f"  {'─'*40}")
    for k, v in best.params.items():
        print(f"    {k:20s} = {v!r}")
    print(f"{'='*60}\n")

    _save_best_checkpoint(best.params, device, family=family)


# ---------------------------------------------------------------------------
# Retrain best config
# ---------------------------------------------------------------------------

def _save_best_checkpoint(params: dict, device: torch.device, family: str = "") -> None:
    model_name = params.get("model", cfg.MODEL)
    cfg.MODEL  = model_name   # ensure correct arch is built

    print(f"Retraining best [{family}] config ({model_name}) for full epochs …")

    train_tf = get_transforms(
        augment=True,
        magnitude=params.get("aug_magnitude", cfg.AUG_MAGNITUDE),
    )
    dataset = TimeOfDayDataset(image_dir=cfg.IMAGE_DIR, transform=train_tf)

    batch_size = params.get("batch_size", cfg.BATCH_SIZE)

    train_loader, val_loader = create_dataloaders(
        dataset,
        fold=cfg.FOLD,
        n_splits=cfg.N_SPLITS,
        batch_size=batch_size,
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
    warmup_epochs = params.get("warmup_epochs", _DEFAULT_WARMUP.get(model_name, 0))
    scheduler = get_scheduler(
        optimizer,
        epochs=cfg.EPOCHS,
        eta_min=params.get("eta_min", cfg.ETA_MIN),
        warmup_epochs=warmup_epochs,
    )
    scaler = (
        torch.amp.GradScaler('cuda')
        if (cfg.USE_AMP and device.type == "cuda") else None
    )

    best_val_mae = float("inf")
    suffix       = f"_{family}" if family else ""
    best_ckpt    = os.path.join(cfg.OUTPUT_DIR, f"optuna_best{suffix}.pt")
    mixup_alpha  = params.get("mixup_alpha", cfg.MIXUP_ALPHA)
    label_noise  = params.get("label_noise", cfg.LABEL_NOISE_STD)

    print(f"\n{'Epoch':>6}  {'Train MAE':>10}  {'Val MAE':>10}  {'LR':>10}  {'Time':>7}")
    print("-" * 50)

    for epoch in range(cfg.EPOCHS):
        t0 = time.time()
        _, train_mae = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            mixup_alpha=mixup_alpha, label_noise=label_noise,
        )
        _, val_mae = evaluate(
            model, val_loader, criterion, device,
            use_tta=cfg.TTA_ENABLED, tta_passes=cfg.TTA_FLIPS,
        )
        scheduler.step()
        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        print(f"{epoch+1:>6}  {train_mae:>9.2f}m  {val_mae:>9.2f}m  "
              f"{lr_now:>10.2e}  {elapsed:>6.1f}s")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            save_checkpoint(model, optimizer, scheduler, epoch + 1,
                            val_mae, best_ckpt)
            print(f"  ★ New best: {best_val_mae:.2f} min")

    print(f"\nBest checkpoint → {best_ckpt}")
    print(f"Best val MAE    : {best_val_mae:.2f} min")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_study(device: torch.device, families: Optional[list] = None) -> None:
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Determine which families to run
    target_families = families or cfg.OPTUNA_MODELS   # list of family names
    if not target_families:
        raise ValueError(
            "No families specified. Set cfg.OPTUNA_MODELS or pass --family."
        )

    print(f"Loading dataset from {cfg.IMAGE_DIR} …")
    shared_dataset = TimeOfDayDataset(
        image_dir=cfg.IMAGE_DIR,
        transform=get_transforms(augment=True, magnitude=cfg.AUG_MAGNITUDE),
    )

    for family in target_families:
        if family not in FAMILY_MODELS:
            print(f"  WARNING: Unknown family '{family}', skipping.")
            continue
        model_list = FAMILY_MODELS[family]
        run_family_study(family, model_list, device, shared_dataset)

    # Print cross-family summary
    print(f"\n{'='*60}")
    print("  Cross-family summary")
    print(f"  {'─'*40}")
    results = []
    for family in target_families:
        db_path = os.path.join(cfg.OUTPUT_DIR, f"optuna_{family}.db")
        if not os.path.exists(db_path):
            continue
        try:
            study = optuna.load_study(
                study_name=f"tod_hpo_{family}",
                storage=f"sqlite:///{db_path}",
            )
            best = study.best_trial
            results.append((best.value, family, best.params))
        except Exception:
            pass

    results.sort()
    for val_mae, family, params in results:
        model = params.get("model", "?")
        print(f"  {family:15s} ({model:20s}) → {val_mae:.2f} min")

    if results:
        best_mae, best_family, best_params = results[0]
        print(f"\n  Overall winner : {best_family}  ({best_params.get('model', '?')})")
        print(f"  Best Val MAE   : {best_mae:.2f} min  ({best_mae/60:.2f} h)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna HPO for Time-of-Day Estimation")
    parser.add_argument(
        "--family",
        nargs="+",
        default=None,
        help=(
            "One or more backbone families to tune. "
            "Choices: convnext, efficientnet, swin, vit, resnet. "
            "Defaults to cfg.OPTUNA_MODELS."
        ),
    )
    args = parser.parse_args()

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

    run_study(device, families=args.family)