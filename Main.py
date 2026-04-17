"""
main.py
=======
Training and evaluation pipeline for:
    Deep Learning-Based Time-of-Day Estimation
    from Sky Images and EXIF Calendar Dates
"""
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
import json as _json
import logging
import time
from typing import List, Optional, Tuple
import multiprocessing as mp

import torch
import torch._dynamo
torch._dynamo.config.cache_size_limit = 32
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config as cfg
from TimeOfDayDataLoader import (
    TimeOfDayDataset,
    create_dataloaders,
    decode_time_tensor,
    get_metadata_dim,
    get_transforms,
    minutes_to_hhmm,
    tta_predict,
    MINUTES_PER_DAY,
)

try:
    import torchvision.models as models
    from torchvision.models import (
        ConvNeXt_Tiny_Weights,
        ConvNeXt_Small_Weights,
    )
    _TORCHVISION_AVAILABLE = True
except ImportError:
    _TORCHVISION_AVAILABLE = False

try:
    import bitsandbytes as bnb
    _BNB_AVAILABLE = True
except ImportError:
    _BNB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("tod")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-5s  %(message)s",
        datefmt="%H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(log_dir, "train.log"), mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s  %(levelname)-5s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)

    return logger

log = logging.getLogger("tod")


# ---------------------------------------------------------------------------
# Scheduler factory & Optimizers
# ---------------------------------------------------------------------------
def get_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs:    int,
    eta_min:   float,
) -> CosineAnnealingLR:
    return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)

def get_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    params = filter(lambda p: p.requires_grad, model.parameters())
    if cfg.USE_8BIT_OPTIM and _BNB_AVAILABLE:
        log.info("Using bitsandbytes 8-bit AdamW")
        return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=weight_decay)
    else:
        if cfg.USE_8BIT_OPTIM and not _BNB_AVAILABLE:
            log.warning("8-bit AdamW requested but bitsandbytes not installed. Falling back to standard AdamW.")
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class MetadataFusionMLP(nn.Module):
    def __init__(
        self,
        image_feat_dim: int   = 768,
        metadata_dim:   int   = 6,
        hidden_dim:     int   = 256,
        dropout:        float = 0.3,
    ):
        super().__init__()
        in_dim = image_feat_dim + metadata_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),   
        )

    def forward(self, image_feats: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([image_feats, metadata], dim=1))

class TimeOfDayModel(nn.Module):
    _FEAT_DIM = 768   

    def __init__(
        self,
        pretrained:   bool  = True,
        freeze_until: str   = "features.4",
        hidden_dim:   int   = 256,
        dropout:      float = 0.3,
        metadata_dim: Optional[int] = None,
    ):
        super().__init__()
        if not _TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is required.")

        if metadata_dim is None:
            metadata_dim = get_metadata_dim()

        self._build_encoder(pretrained)
        self._freeze_layers(freeze_until)

        self.fusion = MetadataFusionMLP(
            image_feat_dim=self._FEAT_DIM,
            metadata_dim=metadata_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def _build_encoder(self, pretrained: bool) -> None:
        name = cfg.MODEL.lower()
        if name == "convnext_tiny":
            weights  = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.convnext_tiny(weights=weights)
        elif name == "convnext_small":
            weights  = ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.convnext_small(weights=weights)
        else:
            raise ValueError(f"Unsupported model '{cfg.MODEL}'")
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])

    def _freeze_layers(self, freeze_until: str) -> None:
        freeze = True
        for name, param in self.encoder.named_parameters():
            if freeze_until in name:
                freeze = False
            param.requires_grad = not freeze

    def unfreeze_all(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self, images: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        x = self.encoder(images).flatten(start_dim=1)
        return self.fusion(x, metadata)

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ---------------------------------------------------------------------------
# Loss & metrics
# ---------------------------------------------------------------------------
class CyclicMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.mse(pred, target)

def cyclic_mae_minutes(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_min   = decode_time_tensor(pred)
    target_min = decode_time_tensor(target)
    diff = torch.abs(pred_min - target_min)
    return torch.min(diff, MINUTES_PER_DAY - diff).mean()

# ---------------------------------------------------------------------------
# Mixup & label noise
# ---------------------------------------------------------------------------
def mixup_batch(
    images:   torch.Tensor,
    metadata: torch.Tensor,
    targets:  torch.Tensor,
    alpha:    float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if alpha <= 0.0:
        return images, metadata, targets
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    idx = torch.randperm(images.size(0), device=images.device)
    return (
        lam * images   + (1 - lam) * images[idx],
        lam * metadata + (1 - lam) * metadata[idx],
        lam * targets  + (1 - lam) * targets[idx],
    )

def add_label_noise(targets: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    if std <= 0.0:
        return targets
    return targets + torch.randn_like(targets) * std

# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------
def train_one_epoch(
    model:       TimeOfDayModel,
    loader:      DataLoader,
    optimizer:   torch.optim.Optimizer,
    criterion:   nn.Module,
    device:      torch.device,
    scaler:      Optional[torch.cuda.amp.GradScaler] = None,
    mixup_alpha: float = 0.0,
    label_noise: float = 0.0,
    pbar:        Optional[tqdm] = None,
    accum_steps: int = 1,
) -> Tuple[float, float]:
    model.train()
    total_loss = total_mae = 0.0

    optimizer.zero_grad()

    for idx, (images, metadata, targets) in enumerate(loader):
        # Channels last optimization
        mem_fmt = torch.channels_last if cfg.USE_CHANNELS_LAST else torch.contiguous_format
        images = images.to(device, non_blocking=True, memory_format=mem_fmt)
        metadata = metadata.to(device, non_blocking=True)
        targets  = targets.to(device, non_blocking=True)

        images, metadata, targets = mixup_batch(images, metadata, targets, alpha=mixup_alpha)
        targets = add_label_noise(targets, std=label_noise)

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                preds = model(images, metadata)
                loss  = criterion(preds, targets) / accum_steps
            scaler.scale(loss).backward()
            
            if (idx + 1) % accum_steps == 0 or (idx + 1) == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            preds = model(images, metadata)
            loss  = criterion(preds, targets) / accum_steps
            loss.backward()
            
            if (idx + 1) % accum_steps == 0 or (idx + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        # Re-multiply by accum_steps so logging reflects actual batch loss
        batch_loss = loss.item() * accum_steps
        batch_mae  = cyclic_mae_minutes(preds.detach().cpu(), targets.detach().cpu()).item()
        
        total_loss += batch_loss
        total_mae  += batch_mae

        if pbar is not None:
            pbar.set_postfix(loss=f"{batch_loss:.4f}", mae=f"{batch_mae:.1f}m", refresh=False)
            pbar.update(1)

    n = len(loader)
    return total_loss / n, total_mae / n

@torch.no_grad()
def evaluate(
    model:      TimeOfDayModel,
    loader:     DataLoader,
    criterion:  nn.Module,
    device:     torch.device,
    use_tta:    bool = False,
    tta_passes: int  = 4,
    pbar:       Optional[tqdm] = None,
) -> Tuple[float, float]:
    model.eval()
    total_loss = total_mae = 0.0
    mem_fmt = torch.channels_last if cfg.USE_CHANNELS_LAST else torch.contiguous_format

    for images, metadata, targets in loader:
        images   = images.to(device, non_blocking=True, memory_format=mem_fmt)
        metadata = metadata.to(device, non_blocking=True)
        targets  = targets.to(device, non_blocking=True)

        if cfg.USE_AMP and device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                preds = (
                    tta_predict(model, images, metadata, n_passes=tta_passes)
                    if (use_tta and tta_passes > 1) else
                    model(images, metadata)
                )
                loss = criterion(preds, targets)
        else:
            preds = (
                tta_predict(model, images, metadata, n_passes=tta_passes)
                if (use_tta and tta_passes > 1) else
                model(images, metadata)
            )
            loss = criterion(preds, targets)

        total_loss += loss.item()
        total_mae  += cyclic_mae_minutes(preds.cpu(), targets.cpu()).item()

        if pbar is not None:
            pbar.update(1)

    n = len(loader)
    return total_loss / n, total_mae / n

@torch.no_grad()
def evaluate_with_log(
    model:      TimeOfDayModel,
    loader:     DataLoader,
    criterion:  nn.Module,
    device:     torch.device,
    use_tta:    bool = False,
    tta_passes: int  = 4,
) -> Tuple[float, float, List[str], List[float], List[float]]:
    model.eval()
    total_loss = total_mae = 0.0
    mem_fmt = torch.channels_last if cfg.USE_CHANNELS_LAST else torch.contiguous_format

    all_paths:   List[str]   = []
    all_preds:   List[float] = []
    all_actuals: List[float] = []

    subset  = loader.dataset
    dataset = subset.dataset
    path_iter = iter(dataset.samples[i][0] for i in subset.indices)

    for images, metadata, targets in loader:
        images   = images.to(device, non_blocking=True, memory_format=mem_fmt)
        metadata = metadata.to(device, non_blocking=True)
        targets  = targets.to(device, non_blocking=True)

        if cfg.USE_AMP and device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                preds = (
                    tta_predict(model, images, metadata, n_passes=tta_passes)
                    if (use_tta and tta_passes > 1) else
                    model(images, metadata)
                )
                loss = criterion(preds, targets)
        else:
            preds = (
                tta_predict(model, images, metadata, n_passes=tta_passes)
                if (use_tta and tta_passes > 1) else
                model(images, metadata)
            )
            loss = criterion(preds, targets)

        total_loss += loss.item()
        total_mae  += cyclic_mae_minutes(preds.cpu(), targets.cpu()).item()

        all_preds.extend(decode_time_tensor(preds.cpu()).tolist())
        all_actuals.extend(decode_time_tensor(targets.cpu()).tolist())
        all_paths.extend(next(path_iter) for _ in range(images.size(0)))

    n = len(loader)
    return total_loss / n, total_mae / n, all_paths, all_preds, all_actuals

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def save_checkpoint(model, optimizer, scheduler, epoch, val_mae, path):
    torch.save({
        "epoch":     epoch,
        "val_mae":   val_mae,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, path)
    log.info(f"Checkpoint saved -> {path}")

def load_checkpoint(path, model, optimizer=None, scheduler=None,
                    device=torch.device("cpu")) -> int:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    log.info(f"Loaded checkpoint '{path}'  (epoch {ckpt['epoch']}, "
             f"val MAE {ckpt['val_mae']:.2f} min)")
    return ckpt["epoch"]

# ---------------------------------------------------------------------------
# Training log helpers
# ---------------------------------------------------------------------------
def _log_epoch(log_path, fold, epoch, train_loss, train_mae, val_loss, val_mae):
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    record = {
        "type": "epoch", "fold": fold, "epoch": epoch,
        "train_loss": train_loss, "train_mae": float(train_mae),
        "val_loss": val_loss, "val_mae": float(val_mae),
    }
    with open(log_path, "a") as f:
        f.write(_json.dumps(record) + "\n")

def _log_images(log_path, image_paths, pred_mins, actual_mins):
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a") as f:
        for path, pred, actual in zip(image_paths, pred_mins, actual_mins):
            f.write(_json.dumps({
                "type": "image", "path": path,
                "pred_min": float(pred), "actual_min": float(actual),
            }) + "\n")

# ---------------------------------------------------------------------------
# Model Setup Helper
# ---------------------------------------------------------------------------
def build_and_compile_model(device: torch.device, params: dict = None) -> TimeOfDayModel:
    model = TimeOfDayModel(
        pretrained=cfg.PRETRAINED,
        freeze_until=params["freeze_until"] if params else cfg.FREEZE_UNTIL,
        hidden_dim=params["hidden_dim"] if params else cfg.HIDDEN_DIM,
        dropout=params["dropout"] if params else cfg.DROPOUT,
    ).to(device)
    
    if cfg.USE_CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)
        
    if cfg.USE_COMPILE and int(torch.__version__.split('.')[0]) >= 2:
        log.info("Compiling model with torch.compile() ...")
        model = torch.compile(model)
        
    return model

# ---------------------------------------------------------------------------
# Single-fold training
# ---------------------------------------------------------------------------
def train_fold(fold: int, device: torch.device) -> float:
    jsonl_path = os.path.join(cfg.OUTPUT_DIR, "train_log.jsonl")

    train_dataset = TimeOfDayDataset(
        image_dir=cfg.IMAGE_DIR,
        transform=get_transforms(augment=True, magnitude=cfg.AUG_MAGNITUDE),
    )
    val_dataset = TimeOfDayDataset(
        image_dir=cfg.IMAGE_DIR,
        transform=get_transforms(augment=False),
    )

    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset, fold=fold, n_splits=cfg.N_SPLITS,
        batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS,
        val_ratio=cfg.VAL_RATIO, use_weighted_sampler=cfg.WEIGHTED_SAMPLER,
    )

    model = build_and_compile_model(device)
    log.info(f"Fold {fold} | trainable params: {model.count_trainable_params():,}")

    criterion = CyclicMSELoss()
    optimizer = get_optimizer(model, cfg.LR, cfg.WEIGHT_DECAY)
    scheduler = get_scheduler(optimizer, epochs=cfg.EPOCHS, eta_min=cfg.ETA_MIN)
    scaler = torch.amp.GradScaler('cuda') if (cfg.USE_AMP and device.type == "cuda") else None

    start_epoch = 0
    if cfg.CHECKPOINT:
        start_epoch = load_checkpoint(cfg.CHECKPOINT, model, optimizer, scheduler, device)

    best_val_mae = float("inf")
    best_ckpt    = os.path.join(cfg.OUTPUT_DIR, f"best_fold{fold}.pt")

    n_train_batches = len(train_loader)
    n_val_batches   = len(val_loader)
    total_batches   = n_train_batches + n_val_batches

    for epoch in range(start_epoch, cfg.EPOCHS):
        t0 = time.time()

        if cfg.UNFREEZE_EPOCH is not None and epoch == cfg.UNFREEZE_EPOCH:
            # We must access the original uncompiled model's method if it was compiled
            raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            raw_model.unfreeze_all()
            optimizer = get_optimizer(model, cfg.LR * 0.1, cfg.WEIGHT_DECAY)
            scheduler = get_scheduler(optimizer, epochs=cfg.EPOCHS - epoch, eta_min=cfg.ETA_MIN)
            log.info(f"Backbone unfrozen at epoch {epoch + 1} (LR x 0.1)")

        desc = f"Fold {fold}  Ep {epoch+1:>3}/{cfg.EPOCHS}"
        with tqdm(total=total_batches, desc=desc, unit="batch", leave=False, dynamic_ncols=True, colour="cyan") as pbar:
            pbar.set_description(f"{desc} [train]")
            train_loss, train_mae = train_one_epoch(
                model, train_loader, optimizer, criterion, device, scaler,
                mixup_alpha=cfg.MIXUP_ALPHA, label_noise=cfg.LABEL_NOISE_STD, pbar=pbar, accum_steps=cfg.ACCUM_STEPS
            )
            pbar.set_description(f"{desc} [val]  ")
            val_loss, val_mae = evaluate(
                model, val_loader, criterion, device,
                use_tta=cfg.TTA_ENABLED, tta_passes=cfg.TTA_FLIPS, pbar=pbar,
            )

        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]
        is_best = val_mae < best_val_mae

        log.info(
            f"Fold {fold}  Ep {epoch+1:>3}/{cfg.EPOCHS}  "
            f"loss {train_loss:.5f}/{val_loss:.5f}  "
            f"MAE {train_mae:6.1f}/{val_mae:6.1f}m  "
            f"lr {lr_now:.2e}  {elapsed:.1f}s"
            + ("  * best" if is_best else "")
        )

        _log_epoch(jsonl_path, fold, epoch + 1, train_loss, train_mae, val_loss, val_mae)

        if is_best:
            best_val_mae = val_mae
            save_checkpoint(model, optimizer, scheduler, epoch + 1, val_mae, best_ckpt)

    load_checkpoint(best_ckpt, model, device=device)
    _, _, img_paths, pred_mins, actual_mins = evaluate_with_log(
        model, val_loader, criterion, device,
        use_tta=cfg.TTA_ENABLED, tta_passes=cfg.TTA_FLIPS,
    )
    _log_images(jsonl_path, img_paths, pred_mins, actual_mins)
    log.info(f"Per-image diagnostics logged -> {jsonl_path}")

    save_checkpoint(model, optimizer, scheduler, cfg.EPOCHS, val_mae,
                    os.path.join(cfg.OUTPUT_DIR, f"last_fold{fold}.pt"))

    log.info(f"Fold {fold} complete -- best val MAE: {best_val_mae:.2f} min ")
    return best_val_mae

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    setup_logging(cfg.OUTPUT_DIR)

    torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )

    log.info("=" * 60)
    log.info("Time-of-Day Estimation  --  ConvNeXt")
    log.info(f"Device      : {device}")
    log.info(f"Model       : {cfg.MODEL}")
    log.info(f"Image size  : {cfg.IMAGE_SIZE}")
    log.info(f"Batch Size  : Physical {cfg.BATCH_SIZE} | Effective {cfg.BATCH_SIZE * cfg.ACCUM_STEPS}")
    log.info(f"Optimizers  : Compile={cfg.USE_COMPILE}, AMP={cfg.USE_AMP}, 8-bit={cfg.USE_8BIT_OPTIM}, NHWC={cfg.USE_CHANNELS_LAST}")
    log.info("=" * 60)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    if cfg.EVAL_ONLY:
        assert cfg.CHECKPOINT, "EVAL_ONLY requires a CHECKPOINT path."
        dataset = TimeOfDayDataset(image_dir=cfg.IMAGE_DIR, transform=get_transforms(augment=False))
        _, val_loader = create_dataloaders(
            dataset, dataset, fold=cfg.FOLD, n_splits=cfg.N_SPLITS,
            batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS,
        )
        model = build_and_compile_model(device)
        load_checkpoint(cfg.CHECKPOINT, model, device=device)
        criterion = CyclicMSELoss()
        val_loss, val_mae = evaluate(
            model, val_loader, criterion, device,
            use_tta=cfg.TTA_ENABLED, tta_passes=cfg.TTA_FLIPS,
        )
        log.info(f"Eval -- loss: {val_loss:.6f}  |  MAE: {val_mae:.2f} min")
        return

    if cfg.TRAIN_ALL_FOLDS:
        fold_maes = []
        for fold in range(cfg.N_SPLITS):
            log.info(f"{'='*20}  FOLD {fold + 1}/{cfg.N_SPLITS}  {'='*20}")
            fold_maes.append(train_fold(fold, device))

        log.info("=" * 60)
        log.info("Cross-validation results")
        for i, m in enumerate(fold_maes):
            log.info(f"  Fold {i}: {m:.2f} min")
        log.info(f"  Mean MAE: {sum(fold_maes)/len(fold_maes):.2f} min")
        log.info("=" * 60)
    else:
        log.info(f"Fold: {cfg.FOLD} / {cfg.N_SPLITS}")
        train_fold(cfg.FOLD, device)

if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    main()