"""
tune.py
=======
Optuna hyperparameter optimisation for Time-of-Day Estimation.
Searches over convnext_tiny and convnext_small only.
"""
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
import argparse
import gc
import logging
import time
import multiprocessing as mp
from typing import Optional

import numpy as np
import torch
import torch._dynamo
torch._dynamo.config.cache_size_limit = 32
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
)
from main import (
    CyclicMSELoss,
    train_one_epoch,
    evaluate,
    save_checkpoint,
    get_scheduler,
    get_optimizer,
    build_and_compile_model
)

SUPPORTED_MODELS = ["convnext_tiny"]

# ---------------------------------------------------------------------------
# Tqdm-safe logging
# ---------------------------------------------------------------------------
class _TqdmHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)

def setup_logging(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-5s  %(message)s", datefmt="%H:%M:%S")
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

# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------
def get_search_space(trial: optuna.Trial, model_name: str) -> dict:
    # LR anchored around 3e-4, but bounded to prevent instability
    lr           = trial.suggest_float("lr",           1.5e-4, 4.5e-4, log=True)
    
    # Shifted up: Previous best was 0.038, let it explore heavier regularization
    weight_decay = trial.suggest_float("weight_decay", 1.5e-2, 6.5e-2, log=True)
    
    # Standard cosine annealing minimums
    eta_min      = trial.suggest_float("eta_min",      5.0e-6, 2.0e-5, log=True)
    
    # Widened slightly: 512px might need a bit more dropout than 448px
    dropout      = trial.suggest_float("dropout", 0.1, 0.25)
    
    # Shifted up: It loved noise in the 448px run
    label_noise = trial.suggest_float("label_noise", 0.02, 0.05)

    mixup_alpha = trial.suggest_float("mixup_alpha", 0.1, 0.3)

    aug_magnitude = trial.suggest_categorical("aug_magnitude", ["moderate", "heavy"])

    # Let Optuna decide how deep to train the backbone
    freeze_until = trial.suggest_categorical("freeze_until", ["features.4", "features.6"])

    # Fixed parameters
    hidden_dim    = 384

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
        "batch_size":    cfg.BATCH_SIZE,
        "accum_steps":   cfg.ACCUM_STEPS,
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
    display = {k: v for k, v in params.items() if k not in ["batch_size", "accum_steps"]}
    for k, v in display.items():
        val_str = f"{v:.2e}" if isinstance(v, float) and abs(v) < 0.01 else str(v)
        row = f"│  {k:<20}  {val_str}"
        lines.append(row + " " * (_w - len(row) + 1) + "│")
    lines.append(f"└{'─' * _w}┘")
    log.info("\n" + "\n".join(lines))

    fold_maes = []

    for fold_idx in range(cfg.OPTUNA_CV_FOLDS):
        log.info(f"  Fold {fold_idx}  |  building dataloaders …")
        
        train_loader, val_loader = create_dataloaders(
            train_dataset, val_dataset, fold=fold_idx, n_splits=cfg.N_SPLITS,
            batch_size=params["batch_size"], num_workers=cfg.NUM_WORKERS,
            val_ratio=cfg.VAL_RATIO, use_weighted_sampler=cfg.WEIGHTED_SAMPLER,
            persistent_workers=True,
        )

        try:
            model = build_and_compile_model(device, params)
        except Exception as exc:
            log.info(f"  ✗ Trial {trial.number}: model build failed ({exc}), pruning.")
            raise optuna.exceptions.TrialPruned()

        criterion = CyclicMSELoss()
        optimizer = get_optimizer(model, params["lr"], params["weight_decay"])
        scheduler = get_scheduler(optimizer, epochs=cfg.OPTUNA_EPOCHS, eta_min=params["eta_min"])
        scaler = torch.amp.GradScaler('cuda') if (cfg.USE_AMP and device.type == "cuda") else None

        best_fold_val_mae   = float("inf")
        best_fold_train_mae = float("inf")
        best_fold_epoch     = -1

        log_every = max(1, cfg.OPTUNA_EPOCHS // 10)

        for epoch in range(cfg.OPTUNA_EPOCHS):
            _, train_mae = train_one_epoch(
                model, train_loader, optimizer, criterion, device, scaler,
                mixup_alpha=params["mixup_alpha"], label_noise=params["label_noise"], 
                accum_steps=params["accum_steps"]
            )
            _, val_mae = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            is_best_epoch = val_mae < best_fold_val_mae
            if is_best_epoch:
                best_fold_val_mae   = val_mae
                best_fold_train_mae = train_mae
                best_fold_epoch     = epoch + 1

            if is_best_epoch or (epoch + 1) % log_every == 0:
                marker = " ★" if is_best_epoch else ""
                log.info(
                    f"  T{trial.number:>3} F{fold_idx} "
                    f"ep {epoch+1:>3}/{cfg.OPTUNA_EPOCHS}  "
                    f"tr {train_mae:>6.1f}m  val {val_mae:>6.1f}m  "
                    f"best {best_fold_val_mae:>6.1f}m{marker}"
                )

            if trial_pbar is not None:
                trial_pbar.set_postfix(T=trial.number, ep=f"{epoch+1}/{cfg.OPTUNA_EPOCHS}", best=f"{best_fold_val_mae:.1f}m", refresh=False)

            step = fold_idx * cfg.OPTUNA_EPOCHS + epoch
            trial.report(val_mae, step)
            if trial.should_prune():
                log.info(f"  ✗ Trial {trial.number} pruned at epoch {epoch + 1} (best val: {best_fold_val_mae:.1f}m)")
                del model, optimizer, train_loader, val_loader
                gc.collect()
                torch.cuda.empty_cache()
                raise optuna.exceptions.TrialPruned()

        fold_maes.append({
            "val_mae": best_fold_val_mae, "train_mae": best_fold_train_mae, "epoch": best_fold_epoch,
        })

        del model, optimizer, train_loader, val_loader
        gc.collect()
        torch.cuda.empty_cache()

    mean_mae = float(np.mean([r["val_mae"] for r in fold_maes]))

    try:
        study_best = min(t.value for t in trial.study.trials if t.value is not None)
    except (ValueError, AttributeError):
        study_best = mean_mae

    is_study_best = mean_mae <= study_best

    _w = 54
    header = f"  Trial {trial.number:>3} [{model_name}]  —  result"
    lines = [
        f"┌{'─' * _w}┐", f"│  {header:<{_w - 2}}│",
        f"├{'─' * 6}┬{'─' * 10}┬{'─' * 10}┬{'─' * 8}┬{'─' * (_w - 38)}┤",
        f"│ {'fold':^4} │ {'train MAE':^8} │ {'val MAE':^8} │ {'epoch':^6} │ {'gap':^{_w - 40}} │",
        f"├{'─' * 6}┼{'─' * 10}┼{'─' * 10}┼{'─' * 8}┼{'─' * (_w - 38)}┤",
    ]
    for i, r in enumerate(fold_maes):
        gap = r["val_mae"] - r["train_mae"]
        gap_str = f"+{gap:.1f}m" if gap >= 0 else f"{gap:.1f}m"
        lines.append(f"│  {i:>2}  │ {r['train_mae']:>7.1f}m │ {r['val_mae']:>7.1f}m │  ep {r['epoch']:>2}  │ {gap_str:^{_w - 40}} │")
    lines.append(f"├{'─' * 6}┴{'─' * 10}┴{'─' * 10}┴{'─' * 8}┴{'─' * (_w - 38)}┤")
    best_marker = "  ★ NEW BEST" if is_study_best else ""
    lines.append(f"│  mean val MAE : {mean_mae:.2f}m{best_marker:<{_w - 24}}│")
    lines.append(f"└{'─' * _w}┘")
    log.info("\n" + "\n".join(lines))

    param_str = ", ".join(f"{k}={v}" for k, v in params.items() if k not in ["batch_size", "accum_steps"])
    log.info(f"Trial {trial.number:>3} [{model_name:<15}] mean val {mean_mae:.2f}m" + ("  ★ best" if is_study_best else "") + f"  | Params: {param_str}")

    return mean_mae

# ---------------------------------------------------------------------------
# Study runner & Retrain best
# ---------------------------------------------------------------------------
def run_study(device: torch.device, model_list: list) -> None:
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    study_name   = "tod_hpo_convnext"
    storage_path = os.path.join(cfg.OUTPUT_DIR, "optuna_convnext.db")
    storage_url  = f"sqlite:///{storage_path}"
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name=study_name, storage=storage_url, load_if_exists=True, direction="minimize",
        sampler=TPESampler(seed=cfg.SEED, n_startup_trials=cfg.OPTUNA_N_STARTUP_TRIALS),
        pruner=MedianPruner(n_startup_trials=cfg.OPTUNA_N_STARTUP_TRIALS, n_warmup_steps=max(18, cfg.OPTUNA_EPOCHS // 4)),
    )

    completed = len([t for t in study.trials if t.state.is_finished()])
    remaining = cfg.OPTUNA_N_TRIALS - completed

    log.info("=" * 60)
    log.info(f"Models        : {', '.join(model_list)}")
    log.info(f"Study         : '{study_name}'")
    log.info(f"Completed     : {completed} / {cfg.OPTUNA_N_TRIALS} trials")
    log.info(f"Device        : {device}")
    log.info(f"Epochs        : {cfg.OPTUNA_EPOCHS}")
    log.info(f"CV Folds      : {cfg.OPTUNA_CV_FOLDS}")
    log.info(f"Batch Size    : {cfg.BATCH_SIZE} (accum steps: {cfg.ACCUM_STEPS}, effective: {cfg.BATCH_SIZE * cfg.ACCUM_STEPS})")
    log.info(f"Image Size    : {cfg.IMAGE_SIZE}px")
    log.info(f"Optimizers    : Compile={cfg.USE_COMPILE}, 8-bit={cfg.USE_8BIT_OPTIM}, NHWC={cfg.USE_CHANNELS_LAST}")
    log.info("=" * 60)

    train_dataset = TimeOfDayDataset(image_dir=cfg.IMAGE_DIR, transform=get_transforms(augment=True, magnitude=cfg.AUG_MAGNITUDE))
    val_dataset   = TimeOfDayDataset(image_dir=cfg.IMAGE_DIR, transform=get_transforms(augment=False))

    if remaining > 0:
        with tqdm(total=remaining, desc="Optuna trials", unit="trial", dynamic_ncols=True, colour="green") as trial_pbar:
            def update_pbar_callback(study: optuna.Study, trial: optuna.Trial):
                trial_pbar.update(1)
                try: trial_pbar.set_postfix(study_best=f"{study.best_value:.2f}m", refresh=False)
                except ValueError: pass

            study.optimize(
                lambda trial: objective(trial, device, train_dataset, val_dataset, model_list, trial_pbar=trial_pbar),
                n_trials=remaining, timeout=cfg.OPTUNA_TIMEOUT_SECONDS, show_progress_bar=False, gc_after_trial=True, callbacks=[update_pbar_callback],
            )

    best = study.best_trial
    log.info("=" * 60)
    log.info(f"Best trial    : #{best.number}")
    log.info(f"Val MAE       : {best.value:.2f} min")
    log.info("=" * 60)
    _retrain_best(best.params, device)

def _retrain_best(params: dict, device: torch.device) -> None:
    model_name = params.get("model", cfg.MODEL)
    cfg.MODEL  = model_name
    log.info(f"Retraining best config ({model_name}) for {cfg.EPOCHS} epochs ...")

    train_dataset = TimeOfDayDataset(image_dir=cfg.IMAGE_DIR, transform=get_transforms(augment=True, magnitude=params.get("aug_magnitude", "none")))
    val_dataset   = TimeOfDayDataset(image_dir=cfg.IMAGE_DIR, transform=get_transforms(augment=False))
    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset, fold=cfg.FOLD, n_splits=cfg.N_SPLITS,
        batch_size=params.get("batch_size", cfg.BATCH_SIZE), num_workers=cfg.NUM_WORKERS,
        val_ratio=cfg.VAL_RATIO, use_weighted_sampler=cfg.WEIGHTED_SAMPLER, persistent_workers=True,
    )

    model = build_and_compile_model(device, params)
    criterion = CyclicMSELoss()
    optimizer = get_optimizer(model, params["lr"], params["weight_decay"])
    scheduler = get_scheduler(optimizer, epochs=cfg.EPOCHS, eta_min=params["eta_min"])
    scaler = torch.amp.GradScaler('cuda') if (cfg.USE_AMP and device.type == "cuda") else None

    best_val_mae, best_ckpt = float("inf"), os.path.join(cfg.OUTPUT_DIR, "optuna_best_convnext.pt")
    total_batches = len(train_loader) + len(val_loader)

    for epoch in range(cfg.EPOCHS):
        t0, desc = time.time(), f"Retrain  Ep {epoch+1:>3}/{cfg.EPOCHS}"
        with tqdm(total=total_batches, desc=desc, unit="batch", leave=False, dynamic_ncols=True, colour="cyan") as pbar:
            pbar.set_description(f"{desc} [train]")
            _, train_mae = train_one_epoch(
                model, train_loader, optimizer, criterion, device, scaler,
                mixup_alpha=params.get("mixup_alpha", cfg.MIXUP_ALPHA), label_noise=params.get("label_noise", cfg.LABEL_NOISE_STD), 
                pbar=pbar, accum_steps=params.get("accum_steps", cfg.ACCUM_STEPS)
            )
            pbar.set_description(f"{desc} [val]  ")
            _, val_mae = evaluate(model, val_loader, criterion, device, use_tta=cfg.TTA_ENABLED, tta_passes=cfg.TTA_FLIPS, pbar=pbar)

        scheduler.step()
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            save_checkpoint(model, optimizer, scheduler, epoch + 1, val_mae, best_ckpt)

        log.info(f"Retrain  Ep {epoch+1:>3}/{cfg.EPOCHS}  MAE {train_mae:6.1f}/{val_mae:6.1f}m  lr {scheduler.get_last_lr()[0]:.2e}  {time.time()-t0:.1f}s" + ("  * best" if val_mae == best_val_mae else ""))

    log.info(f"Best val MAE    : {best_val_mae:.2f} min")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna HPO -- ConvNeXt")
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default=None)
    args = parser.parse_args()

    model_list = [args.model] if args.model else SUPPORTED_MODELS
    if mp.get_start_method(allow_none=True) != "spawn": mp.set_start_method("spawn", force=True)

    torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(cfg.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    setup_logging(cfg.OUTPUT_DIR)
    run_study(device, model_list)