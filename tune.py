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


# ---------------------------------------------------------------------------
# Tqdm-safe logging
# ---------------------------------------------------------------------------

class _TqdmHandler(logging.StreamHandler):
    """Routes log records through tqdm.write() so they never clobber bars."""
    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


def setup_logging(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-5s  %(message)s",
                            datefmt="%H:%M:%S")
    root = logging.getLogger("tune")
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    console = _TqdmHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    root.addHandler(console)

    fh = logging.FileHandler(os.path.join(output_dir, "tune.log"), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)


log = logging.getLogger("tune")

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

SUPPORTED_MODELS = ["convnext_tiny"]


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

def get_search_space(trial: optuna.Trial, model_name: str) -> dict:
    """
    Optuna search space tailored for N=2500 with ConvNeXt backbones.
    Relaxed regularization to allow deeper feature learning.
    """
    freeze_until = "features.6"
    mixup_alpha  = 0.0
    hidden_dim   = trial.suggest_categorical("hidden_dim", [256, 384])
    lr           = trial.suggest_float("lr",           2e-4, 6e-4,  log=True)
    weight_decay = trial.suggest_float("weight_decay", 5e-3, 5e-2,  log=True)
    eta_min      = trial.suggest_float("eta_min",      2e-6, 8e-6,  log=True)
    dropout      = trial.suggest_float("dropout",      0.10, 0.25)
    label_noise  = trial.suggest_float("label_noise",  0.0,  0.008)
    aug_magnitude = trial.suggest_categorical("aug_magnitude", ["moderate", "heavy"])

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
    trial_pbar:    "Optional[tqdm]" = None,
) -> float:
    model_name = trial.suggest_categorical("model", model_list)
    cfg.MODEL  = model_name

    params = get_search_space(trial, model_name)

    _w = 54
    lines = [
        f"┌{'─' * _w}┐",
        f"│  Trial {trial.number:>3}  [{model_name}]{' ' * (_w - 18 - len(model_name))}│",
        f"├{'─' * _w}┤",
    ]
    display = {k: v for k, v in params.items() if k != "batch_size"}
    for k, v in display.items():
        val_str = f"{v:.2e}" if isinstance(v, float) and abs(v) < 0.01 else str(v)
        row = f"│  {k:<20}  {val_str}"
        lines.append(row + " " * (_w - len(row) + 1) + "│")
    lines.append(f"└{'─' * _w}┘")
    tqdm.write("\n".join(lines))

    fold_maes = []

    for fold_idx in range(cfg.OPTUNA_CV_FOLDS):
        # Capture create_dataloaders' print() output via tqdm.write so it
        # doesn't corrupt the outer trial bar.
        n_train = sum(1 for _ in range(len(train_dataset)))  # just a size ref
        tqdm.write(f"  Fold {fold_idx}  |  building dataloaders …")
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
            tqdm.write(f"  ✗ Trial {trial.number}: model build failed ({exc}), pruning.")
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

        # No inner tqdm bar — print one compact line per epoch via tqdm.write
        # so the outer trial_pbar is the only live bar on screen.
        log_every = max(1, cfg.OPTUNA_EPOCHS // 10)   # ~10 lines per trial

        for epoch in range(cfg.OPTUNA_EPOCHS):
            _, train_mae = train_one_epoch(
                model, train_loader, optimizer, criterion, device, scaler,
                mixup_alpha=params["mixup_alpha"],
                label_noise=params["label_noise"],
            )
            _, val_mae = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            is_best_epoch = val_mae < best_fold_val_mae
            if is_best_epoch:
                best_fold_val_mae   = val_mae
                best_fold_train_mae = train_mae
                best_fold_epoch     = epoch + 1

            # Print a status line at regular intervals and on best epochs
            if is_best_epoch or (epoch + 1) % log_every == 0:
                marker = " ★" if is_best_epoch else ""
                tqdm.write(
                    f"  T{trial.number:>3} F{fold_idx} "
                    f"ep {epoch+1:>3}/{cfg.OPTUNA_EPOCHS}  "
                    f"tr {train_mae:>6.1f}m  val {val_mae:>6.1f}m  "
                    f"best {best_fold_val_mae:>6.1f}m{marker}"
                )

            # Update the outer trial bar postfix so the operator can glance
            # at overall progress without any nested bar noise.
            if trial_pbar is not None:
                trial_pbar.set_postfix(
                    T=trial.number,
                    ep=f"{epoch+1}/{cfg.OPTUNA_EPOCHS}",
                    best=f"{best_fold_val_mae:.1f}m",
                    refresh=False,
                )

            step = fold_idx * cfg.OPTUNA_EPOCHS + epoch
            trial.report(val_mae, step)
            if trial.should_prune():
                tqdm.write(
                    f"  ✗ Trial {trial.number} pruned at epoch {epoch + 1}  "
                    f"(best val so far: {best_fold_val_mae:.1f}m)"
                )
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

    try:
        study_best = min(
            t.value for t in trial.study.trials
            if t.value is not None
        )
    except (ValueError, AttributeError):
        study_best = mean_mae

    is_study_best = mean_mae <= study_best

    _w = 54
    header = f"  Trial {trial.number:>3} [{model_name}]  —  result"
    lines = [
        f"┌{'─' * _w}┐",
        f"│  {header:<{_w - 2}}│",
        f"├{'─' * 6}┬{'─' * 10}┬{'─' * 10}┬{'─' * 8}┬{'─' * (_w - 38)}┤",
        f"│ {'fold':^4} │ {'train MAE':^8} │ {'val MAE':^8} │ {'epoch':^6} │ {'gap':^{_w - 40}} │",
        f"├{'─' * 6}┼{'─' * 10}┼{'─' * 10}┼{'─' * 8}┼{'─' * (_w - 38)}┤",
    ]
    for i, r in enumerate(fold_maes):
        gap = r["val_mae"] - r["train_mae"]
        gap_str = f"+{gap:.1f}m" if gap >= 0 else f"{gap:.1f}m"
        lines.append(
            f"│  {i:>2}  │ {r['train_mae']:>7.1f}m │ {r['val_mae']:>7.1f}m "
            f"│  ep {r['epoch']:>2}  │ {gap_str:^{_w - 40}} │"
        )
    lines.append(f"├{'─' * 6}┴{'─' * 10}┴{'─' * 10}┴{'─' * 8}┴{'─' * (_w - 38)}┤")
    best_marker = "  ★ NEW BEST" if is_study_best else ""
    mean_line = f"  mean val MAE : {mean_mae:.2f}m{best_marker}"
    lines.append(f"│  {mean_line:<{_w - 2}}│")
    lines.append(f"└{'─' * _w}┘")
    tqdm.write("\n".join(lines))

    log.info(
        f"Trial {trial.number:>3} [{model_name:<15}]  "
        f"mean val {mean_mae:.2f}m"
        + ("  ★ best" if is_study_best else "")
    )

    return mean_mae


# ---------------------------------------------------------------------------
# Study runner
# ---------------------------------------------------------------------------

def run_study(device: torch.device, model_list: list) -> None:
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    study_name   = "tod_hpo_convnext"
    storage_path = os.path.join(cfg.OUTPUT_DIR, "optuna_convnext.db")
    storage_url  = f"sqlite:///{storage_path}"

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

            def update_pbar_callback(study: optuna.Study, trial: optuna.Trial):
                trial_pbar.update(1)
                try:
                    best_val = study.best_value
                    trial_pbar.set_postfix(
                        study_best=f"{best_val:.2f}m", refresh=False
                    )
                except ValueError:
                    pass

            study.optimize(
                lambda trial: objective(
                    trial, device, train_dataset, val_dataset, model_list,
                    trial_pbar=trial_pbar,
                ),
                n_trials=remaining,
                timeout=cfg.OPTUNA_TIMEOUT_SECONDS,
                show_progress_bar=False,
                gc_after_trial=True,
                callbacks=[update_pbar_callback],
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