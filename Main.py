"""
main.py
=======
Training and evaluation pipeline for:

    Deep Learning-Based Time-of-Day Estimation
    from Sky Images and EXIF Calendar Dates

Architecture
------------
A ConvNeXt backbone extracts visual features. A small MLP fuses the image
embedding with calendar metadata (from EXIF). The combined representation is
projected to 2 outputs: (sin_t, cos_t), encoding time-of-day cyclically so
23:59 and 00:01 are nearby.

Supported backbones
-------------------
  convnext_tiny, convnext_small

Loss
----
Mean squared error on the (sin_t, cos_t) pair.
Primary metric: Mean Absolute Error in *minutes* (cyclic-aware).

Usage
-----
Adjust settings in config.py, then:
    python main.py
"""

import json as _json
import os
import time
from typing import List, Optional, Tuple
import multiprocessing as mp

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

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


# ---------------------------------------------------------------------------
# Scheduler factory
# ---------------------------------------------------------------------------

def get_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs:    int,
    eta_min:   float,
) -> CosineAnnealingLR:
    return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)


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
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),   # → (sin_t, cos_t)
        )

    def forward(self, image_feats: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([image_feats, metadata], dim=1))


class TimeOfDayModel(nn.Module):
    """
    ConvNeXt backbone + fusion MLP for cyclic time-of-day regression.

    The backbone is frozen up to `freeze_until` during early training;
    call unfreeze_all() at UNFREEZE_EPOCH to open the whole network.

    Encoder: Sequential(*backbone.children()[:-1])
    Forward:  encoder(x).flatten(1) → (B, 768)
    """

    _FEAT_DIM = 768   # both convnext_tiny and convnext_small output 768-d

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
            raise ValueError(
                f"Unsupported model '{cfg.MODEL}'. Choose 'convnext_tiny' or 'convnext_small'."
            )
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])

    def _freeze_layers(self, freeze_until: str) -> None:
        """
        Freeze all encoder parameters up to (but not including) the first
        parameter whose name contains `freeze_until`.
        Use ConvNeXt stage names, e.g. "features.4" to train from stage 4 onward.
        """
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
) -> Tuple[float, float]:
    model.train()
    total_loss = total_mae = 0.0

    for images, metadata, targets in loader:
        images   = images.to(device,   non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)
        targets  = targets.to(device,  non_blocking=True)

        images, metadata, targets = mixup_batch(images, metadata, targets, alpha=mixup_alpha)
        targets = add_label_noise(targets, std=label_noise)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                preds = model(images, metadata)
                loss  = criterion(preds, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(images, metadata)
            loss  = criterion(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        total_mae  += cyclic_mae_minutes(preds.detach().cpu(), targets.detach().cpu())

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
) -> Tuple[float, float]:
    model.eval()
    total_loss = total_mae = 0.0

    for images, metadata, targets in loader:
        images   = images.to(device,   non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)
        targets  = targets.to(device,  non_blocking=True)

        preds = (
            tta_predict(model, images, metadata, n_passes=tta_passes)
            if (use_tta and tta_passes > 1) else
            model(images, metadata)
        )
        total_loss += criterion(preds, targets).item()
        total_mae  += cyclic_mae_minutes(preds.cpu(), targets.cpu())

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
    """
    Like evaluate(), but also returns per-image paths and minute predictions
    for visualize_training.py.
    """
    model.eval()
    total_loss = total_mae = 0.0

    all_paths:   List[str]   = []
    all_preds:   List[float] = []
    all_actuals: List[float] = []

    subset  = loader.dataset
    dataset = subset.dataset
    path_iter = iter(dataset.samples[i][0] for i in subset.indices)

    for images, metadata, targets in loader:
        images   = images.to(device,   non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)
        targets  = targets.to(device,  non_blocking=True)

        preds = (
            tta_predict(model, images, metadata, n_passes=tta_passes)
            if (use_tta and tta_passes > 1) else
            model(images, metadata)
        )
        total_loss += criterion(preds, targets).item()
        total_mae  += cyclic_mae_minutes(preds.cpu(), targets.cpu())

        all_preds.extend(decode_time_tensor(preds.cpu()).tolist())
        all_actuals.extend(decode_time_tensor(targets.cpu()).tolist())
        all_paths.extend(next(path_iter) for _ in range(images.size(0)))

    n = len(loader)
    return total_loss / n, total_mae / n, all_paths, all_preds, all_actuals


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_single(
    model:      TimeOfDayModel,
    image_path: str,
    device:     torch.device,
    month:      Optional[int]   = None,
    day:        Optional[int]   = None,
    year:       int             = 2024,
    latitude:   Optional[float] = None,
    longitude:  Optional[float] = None,
    use_tta:    bool            = True,
    tta_passes: int             = 4,
) -> str:
    from PIL import Image as PILImage
    from TimeOfDayDataLoader import TimeOfDayLabel, ImageFeatureExtractor, _day_of_year

    transform = get_transforms(augment=False)
    img = PILImage.open(image_path).convert("RGB")
    img = img.resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), PILImage.Resampling.LANCZOS)

    image_features = ImageFeatureExtractor.extract(img) if cfg.USE_IMAGE_FEATURES else None
    image_tensor   = transform(img).unsqueeze(0).to(device)

    label = TimeOfDayLabel(
        time_min=0, month=month,
        day_of_year=_day_of_year(month, day, year),
        latitude=latitude, longitude=longitude,
        image_features=image_features,
    )
    meta_tensor = label.to_metadata_tensor().unsqueeze(0).to(device)

    model.eval()
    pred = (
        tta_predict(model, image_tensor, meta_tensor, n_passes=tta_passes)
        if (use_tta and tta_passes > 1) else
        model(image_tensor, meta_tensor)
    )
    return minutes_to_hhmm(decode_time_tensor(pred.cpu())[0].item())


@torch.no_grad()
def predict_ensemble(
    models:     List[TimeOfDayModel],
    image_path: str,
    device:     torch.device,
    month:      Optional[int]   = None,
    day:        Optional[int]   = None,
    year:       int             = 2024,
    latitude:   Optional[float] = None,
    longitude:  Optional[float] = None,
    use_tta:    bool            = True,
    tta_passes: int             = 4,
) -> str:
    from PIL import Image as PILImage
    from TimeOfDayDataLoader import TimeOfDayLabel, ImageFeatureExtractor, _day_of_year

    transform = get_transforms(augment=False)
    img = PILImage.open(image_path).convert("RGB")
    img = img.resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), PILImage.Resampling.LANCZOS)

    image_features = ImageFeatureExtractor.extract(img) if cfg.USE_IMAGE_FEATURES else None
    image_tensor   = transform(img).unsqueeze(0).to(device)

    label = TimeOfDayLabel(
        time_min=0, month=month,
        day_of_year=_day_of_year(month, day, year),
        latitude=latitude, longitude=longitude,
        image_features=image_features,
    )
    meta_tensor = label.to_metadata_tensor().unsqueeze(0).to(device)

    all_preds = []
    for m in models:
        m.eval()
        p = (
            tta_predict(m, image_tensor, meta_tensor, n_passes=tta_passes)
            if (use_tta and tta_passes > 1) else
            m(image_tensor, meta_tensor)
        )
        all_preds.append(p.cpu())

    avg_pred = torch.stack(all_preds, dim=0).mean(dim=0)
    return minutes_to_hhmm(decode_time_tensor(avg_pred)[0].item())


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
    print(f"  ✓ checkpoint saved → {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None,
                    device=torch.device("cpu")) -> int:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    print(f"Loaded checkpoint '{path}'  (epoch {ckpt['epoch']}, "
          f"val MAE {ckpt['val_mae']:.2f} min)")
    return ckpt["epoch"]


# ---------------------------------------------------------------------------
# Training log helpers
# ---------------------------------------------------------------------------

def _log_epoch(
    log_path: str, fold: int, epoch: int,
    train_loss: float, train_mae: float,
    val_loss: float, val_mae: float,
) -> None:
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    record = {
        "type": "epoch",
        "fold": fold, "epoch": epoch,
        "train_loss": train_loss, "train_mae": float(train_mae),
        "val_loss":   val_loss,   "val_mae":   float(val_mae),
    }
    with open(log_path, "a") as f:
        f.write(_json.dumps(record) + "\n")


def _log_images(
    log_path: str,
    image_paths: List[str],
    pred_mins:   List[float],
    actual_mins: List[float],
) -> None:
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a") as f:
        for path, pred, actual in zip(image_paths, pred_mins, actual_mins):
            f.write(_json.dumps({
                "type": "image",
                "path": path,
                "pred_min": float(pred),
                "actual_min": float(actual),
            }) + "\n")


# ---------------------------------------------------------------------------
# Single-fold training
# ---------------------------------------------------------------------------

def train_fold(fold: int, device: torch.device) -> float:
    log_path = os.path.join(cfg.OUTPUT_DIR, "train_log.jsonl")

    train_dataset = TimeOfDayDataset(
        image_dir=cfg.IMAGE_DIR,
        transform=get_transforms(augment=True, magnitude=cfg.AUG_MAGNITUDE),
    )
    val_dataset   = TimeOfDayDataset(
        image_dir=cfg.IMAGE_DIR,
        transform=get_transforms(augment=False),
    )

    train_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        fold=fold,
        n_splits=cfg.N_SPLITS,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        val_ratio=cfg.VAL_RATIO,
        use_weighted_sampler=cfg.WEIGHTED_SAMPLER,
    )

    model = TimeOfDayModel(
        pretrained=cfg.PRETRAINED,
        freeze_until=cfg.FREEZE_UNTIL,
        hidden_dim=cfg.HIDDEN_DIM,
        dropout=cfg.DROPOUT,
    ).to(device)
    print(f"\n  Trainable parameters (fold {fold}): {model.count_trainable_params():,}")

    criterion = CyclicMSELoss()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = get_scheduler(optimizer, epochs=cfg.EPOCHS, eta_min=cfg.ETA_MIN)
    scaler = (
        torch.amp.GradScaler('cuda')
        if (cfg.USE_AMP and device.type == "cuda") else None
    )

    start_epoch = 0
    if cfg.CHECKPOINT:
        start_epoch = load_checkpoint(cfg.CHECKPOINT, model, optimizer, scheduler, device)

    best_val_mae = float("inf")
    best_ckpt    = os.path.join(cfg.OUTPUT_DIR, f"best_fold{fold}.pt")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train MAE':>10}  "
          f"{'Val Loss':>10}  {'Val MAE':>10}  {'LR':>10}  {'Time':>7}")
    print("-" * 72)

    for epoch in range(start_epoch, cfg.EPOCHS):
        t0 = time.time()

        if cfg.UNFREEZE_EPOCH is not None and epoch == cfg.UNFREEZE_EPOCH:
            model.unfreeze_all()
            optimizer = AdamW(
                model.parameters(), lr=cfg.LR * 0.1, weight_decay=cfg.WEIGHT_DECAY,
            )
            scheduler = get_scheduler(
                optimizer, epochs=cfg.EPOCHS - epoch, eta_min=cfg.ETA_MIN,
            )
            print(f"  ↑ Backbone unfrozen at epoch {epoch + 1} (LR×0.1)")

        train_loss, train_mae = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            mixup_alpha=cfg.MIXUP_ALPHA,
            label_noise=cfg.LABEL_NOISE_STD,
        )
        val_loss, val_mae = evaluate(
            model, val_loader, criterion, device,
            use_tta=cfg.TTA_ENABLED, tta_passes=cfg.TTA_FLIPS,
        )
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        print(f"{epoch+1:>6}  {train_loss:>10.6f}  {train_mae:>9.2f}m  "
              f"{val_loss:>10.6f}  {val_mae:>9.2f}m  {lr_now:>10.2e}  "
              f"{elapsed:>6.1f}s")

        _log_epoch(log_path, fold, epoch + 1, train_loss, train_mae, val_loss, val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            save_checkpoint(model, optimizer, scheduler, epoch + 1, val_mae, best_ckpt)
            print(f"  ★ New best val MAE: {best_val_mae:.2f} min "
                  f"({minutes_to_hhmm(best_val_mae)} error)")

    load_checkpoint(best_ckpt, model, device=device)
    _, _, img_paths, pred_mins, actual_mins = evaluate_with_log(
        model, val_loader, criterion, device,
        use_tta=cfg.TTA_ENABLED, tta_passes=cfg.TTA_FLIPS,
    )
    _log_images(log_path, img_paths, pred_mins, actual_mins)
    print(f"  Per-image diagnostics logged → {log_path}")

    save_checkpoint(model, optimizer, scheduler, cfg.EPOCHS, val_mae,
                    os.path.join(cfg.OUTPUT_DIR, f"last_fold{fold}.pt"))

    print(f"\n  [Fold {fold}] Best val MAE : {best_val_mae:.2f} min")
    return best_val_mae


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )

    print(f"\n{'='*60}")
    print(f"  Time-of-Day Estimation  —  ConvNeXt")
    print(f"  Device     : {device}")
    print(f"  Model      : {cfg.MODEL}")
    print(f"  Image size : {cfg.IMAGE_SIZE}")
    print(f"  Metadata   : {get_metadata_dim()}d  "
          f"(image features: {'on' if cfg.USE_IMAGE_FEATURES else 'off'})")
    print(f"  Augment    : {cfg.AUG_MAGNITUDE}")
    print(f"  Mixup α    : {cfg.MIXUP_ALPHA}")
    print(f"  Label noise: {cfg.LABEL_NOISE_STD}")
    print(f"  TTA        : {'on' if cfg.TTA_ENABLED else 'off'}  "
          f"(passes={cfg.TTA_FLIPS})")
    print(f"  All folds  : {cfg.TRAIN_ALL_FOLDS}")
    print(f"{'='*60}\n")

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    if cfg.EVAL_ONLY:
        assert cfg.CHECKPOINT, "EVAL_ONLY requires a CHECKPOINT path."
        dataset = TimeOfDayDataset(
            image_dir=cfg.IMAGE_DIR, transform=get_transforms(augment=False)
        )
        _, val_loader = create_dataloaders(
            dataset, fold=cfg.FOLD, n_splits=cfg.N_SPLITS,
            batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS,
        )
        model = TimeOfDayModel(
            pretrained=False, freeze_until=cfg.FREEZE_UNTIL,
            hidden_dim=cfg.HIDDEN_DIM, dropout=cfg.DROPOUT,
        ).to(device)
        load_checkpoint(cfg.CHECKPOINT, model, device=device)
        criterion = CyclicMSELoss()
        val_loss, val_mae = evaluate(
            model, val_loader, criterion, device,
            use_tta=cfg.TTA_ENABLED, tta_passes=cfg.TTA_FLIPS,
        )
        print(f"\nEval — loss: {val_loss:.6f}  |  MAE: {val_mae:.2f} min ({val_mae/60:.2f} h)")
        return

    if cfg.TRAIN_ALL_FOLDS:
        fold_maes = []
        for fold in range(cfg.N_SPLITS):
            print(f"\n{'='*60}  FOLD {fold + 1} / {cfg.N_SPLITS}  {'='*60}")
            fold_maes.append(train_fold(fold, device))
        print(f"\n{'='*60}")
        print(f"  Cross-validation results")
        for i, m in enumerate(fold_maes):
            print(f"    Fold {i}: {m:.2f} min")
        print(f"  Mean MAE : {sum(fold_maes)/len(fold_maes):.2f} min")
        print(f"{'='*60}\n")
    else:
        print(f"  Fold : {cfg.FOLD} / {cfg.N_SPLITS}")
        train_fold(cfg.FOLD, device)


if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    main()