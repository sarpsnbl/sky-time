"""
main.py
=======
Training and evaluation pipeline for:

    Deep Learning-Based Time-of-Day Estimation
    from Sky Images and EXIF Calendar Dates

Architecture
------------
A ResNet-50 backbone extracts visual features from the image.
A small MLP fuses the image embedding with calendar metadata (from EXIF).
The combined representation is projected to 2 outputs: (sin_t, cos_t),
encoding time-of-day cyclically so that 23:59 and 00:01 are nearby.

Loss
----
Mean squared error on the (sin_t, cos_t) pair.
The primary evaluation metric is Mean Absolute Error in *minutes*,
decoded from the cyclic representation.

Usage
-----
Adjust settings in config.py, then run:
python main.py
"""

import os
import time
from typing import Optional, Tuple
import multiprocessing as mp

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# Import the new config
from config import Config as cfg

from TimeOfDayDataLoader import (
    TimeOfDayDataset,
    create_dataloaders,
    decode_time_tensor,
    get_transforms,
    minutes_to_hhmm,
    extract_exif_datetime,
    MINUTES_PER_DAY,
)

try:
    from torchvision.models import resnet50, ResNet50_Weights
    _TORCHVISION_AVAILABLE = True
except ImportError:
    _TORCHVISION_AVAILABLE = False


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MetadataFusionMLP(nn.Module):
    def __init__(
        self,
        image_feat_dim: int = 2048,
        metadata_dim:   int = 6,
        hidden_dim:     int = 256,
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
        x = torch.cat([image_feats, metadata], dim=1)
        return self.net(x)


class TimeOfDayModel(nn.Module):
    def __init__(
        self,
        pretrained:   bool  = True,
        freeze_until: str   = "layer2",   
        hidden_dim:   int   = 256,
        dropout:      float = 0.3,
        metadata_dim: int   = 6,
    ):
        super().__init__()

        if not _TORCHVISION_AVAILABLE:
            raise ImportError(
                "torchvision is required. Install with: pip install torchvision"
            )

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)

        self.encoder = nn.Sequential(*list(backbone.children())[:-1])  # → (B, 2048, 1, 1)
        self._freeze_layers(freeze_until)

        self.fusion = MetadataFusionMLP(
            image_feat_dim=2048,
            metadata_dim=metadata_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def _freeze_layers(self, freeze_until: str) -> None:
        freeze = True
        for name, param in self.encoder.named_parameters():
            if freeze_until in name:
                freeze = False
            param.requires_grad = not freeze

    def forward(self, images: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(images)          # (B, 2048, 1, 1)
        feats = feats.flatten(start_dim=1)    # (B, 2048)
        return self.fusion(feats, metadata)   # (B, 2)

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


def cyclic_mae_minutes(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_min   = decode_time_tensor(pred)
    target_min = decode_time_tensor(target)

    diff = torch.abs(pred_min - target_min)
    diff = torch.min(diff, MINUTES_PER_DAY - diff)   
    return diff.mean().item()


# ---------------------------------------------------------------------------
# Training loop helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:       TimeOfDayModel,
    loader:      DataLoader,
    optimizer:   torch.optim.Optimizer,
    criterion:   nn.Module,
    device:      torch.device,
    scaler:      Optional[torch.cuda.amp.GradScaler] = None,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_mae  = 0.0
    n_batches  = 0

    for images, metadata, targets in loader:
        images   = images.to(device,   non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)
        targets  = targets.to(device,  non_blocking=True)

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

        mae = cyclic_mae_minutes(preds.detach().cpu(), targets.detach().cpu())
        total_loss += loss.item()
        total_mae  += mae
        n_batches  += 1

    return total_loss / n_batches, total_mae / n_batches


@torch.no_grad()
def evaluate(
    model:     TimeOfDayModel,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_mae  = 0.0
    n_batches  = 0

    for images, metadata, targets in loader:
        images   = images.to(device,   non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)
        targets  = targets.to(device,  non_blocking=True)

        preds = model(images, metadata)
        loss  = criterion(preds, targets)

        mae = cyclic_mae_minutes(preds.cpu(), targets.cpu())
        total_loss += loss.item()
        total_mae  += mae
        n_batches  += 1

    return total_loss / n_batches, total_mae / n_batches


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_single(
    model:     TimeOfDayModel,
    image_path: str,
    device:    torch.device,
    month:     Optional[int] = None,
    day:       Optional[int] = None,
    year:      int = 2024,
    latitude:  Optional[float] = None,
    longitude: Optional[float] = None,
) -> str:
    """
    Run inference on a single image. Extracts the date from EXIF automatically,
    or falls back to manually provided `month` and `day`.
    """
    from PIL import Image as PILImage
    from TimeOfDayDataLoader import TimeOfDayLabel, _day_of_year

    if month is None or day is None:
        exif_data = extract_exif_datetime(image_path)
        if exif_data is None:
            raise ValueError(f"No EXIF data in '{image_path}'. Must provide month/day manually.")
        _, month, day, year = exif_data

    transform = get_transforms(augment=False)
    img = PILImage.open(image_path).convert("RGB")
    img = img.resize((224, 224), PILImage.Resampling.LANCZOS)
    image_tensor = transform(img).unsqueeze(0).to(device)

    doy   = _day_of_year(month, day, year)
    label = TimeOfDayLabel(
        time_min=0,   
        month=month,
        day_of_year=doy,
        latitude=latitude,
        longitude=longitude,
    )
    meta_tensor = label.to_metadata_tensor().unsqueeze(0).to(device)

    model.eval()
    pred = model(image_tensor, meta_tensor)           
    pred_min = decode_time_tensor(pred.cpu())[0].item()
    return minutes_to_hhmm(pred_min)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model:      TimeOfDayModel,
    optimizer:  torch.optim.Optimizer,
    scheduler:  torch.optim.lr_scheduler._LRScheduler,
    epoch:      int,
    val_mae:    float,
    path:       str,
) -> None:
    torch.save({
        "epoch":      epoch,
        "val_mae":    val_mae,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "scheduler":  scheduler.state_dict(),
    }, path)
    print(f"  ✓ checkpoint saved → {path}")


def load_checkpoint(
    path:      str,
    model:     TimeOfDayModel,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device:    torch.device = torch.device("cpu"),
) -> int:
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
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # --- reproducibility ------------------------------------------------------
    torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"\n{'='*60}")
    print(f"  Time-of-Day Estimation")
    print(f"  Device : {device}")
    print(f"  Fold   : {cfg.FOLD} / {cfg.N_SPLITS}")
    print(f"{'='*60}\n")

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # --- dataset --------------------------------------------------------------
    train_tf = get_transforms(augment=True)

    dataset = TimeOfDayDataset(
        image_dir=cfg.IMAGE_DIR,
        transform=train_tf,
    )

    train_loader, val_loader = create_dataloaders(
        dataset,
        fold=cfg.FOLD,
        n_splits=cfg.N_SPLITS,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        val_ratio=cfg.VAL_RATIO,
        use_weighted_sampler=cfg.WEIGHTED_SAMPLER,
    )

    # --- model ----------------------------------------------------------------
    model = TimeOfDayModel(
        pretrained=cfg.PRETRAINED,
        freeze_until=cfg.FREEZE_UNTIL,
        hidden_dim=cfg.HIDDEN_DIM,
        dropout=cfg.DROPOUT,
    ).to(device)

    print(f"\nTrainable parameters: {model.count_trainable_params():,}")

    criterion = CyclicMSELoss()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=1e-6)
    scaler    = (
        torch.cuda.amp.GradScaler()
        if (cfg.USE_AMP and device.type == "cuda")
        else None
    )

    start_epoch = 0
    if cfg.CHECKPOINT:
        start_epoch = load_checkpoint(
            cfg.CHECKPOINT, model, optimizer, scheduler, device
        )

    # --- evaluation-only mode -------------------------------------------------
    if cfg.EVAL_ONLY:
        val_loss, val_mae = evaluate(model, val_loader, criterion, device)
        print(f"\nEvaluation results:")
        print(f"  Val loss : {val_loss:.6f}")
        print(f"  Val MAE  : {val_mae:.2f} minutes  "
              f"({val_mae/60:.2f} hours)")
        return

    # --- training loop --------------------------------------------------------
    best_val_mae = float("inf")
    best_ckpt    = os.path.join(cfg.OUTPUT_DIR, f"best_fold{cfg.FOLD}.pt")
    last_ckpt    = os.path.join(cfg.OUTPUT_DIR, f"last_fold{cfg.FOLD}.pt")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train MAE':>10}  "
          f"{'Val Loss':>10}  {'Val MAE':>10}  {'LR':>10}  {'Time':>7}")
    print("-" * 72)

    for epoch in range(start_epoch, cfg.EPOCHS):
        t0 = time.time()

        train_loss, train_mae = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_loss, val_mae = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        print(f"{epoch+1:>6}  {train_loss:>10.6f}  {train_mae:>9.2f}m  "
              f"{val_loss:>10.6f}  {val_mae:>9.2f}m  {lr_now:>10.2e}  "
              f"{elapsed:>6.1f}s")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            save_checkpoint(model, optimizer, scheduler, epoch + 1,
                            val_mae, best_ckpt)
            print(f"  ★ New best val MAE: {best_val_mae:.2f} min "
                  f"({minutes_to_hhmm(best_val_mae)} error)")

    save_checkpoint(model, optimizer, scheduler, cfg.EPOCHS,
                    val_mae, last_ckpt)

    print(f"\nTraining complete.")
    print(f"  Best val MAE : {best_val_mae:.2f} minutes")
    print(f"  Best checkpoint : {best_ckpt}")

if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)

    main()