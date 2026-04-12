"""
main.py
=======
Training and evaluation pipeline for:

    Deep Learning-Based Time-of-Day Estimation
    from Sky Images and EXIF Calendar Dates

Architecture
------------
A CNN / Transformer backbone extracts visual features.  A small MLP fuses the
image embedding with calendar metadata (from EXIF).  The combined representation
is projected to 2 outputs: (sin_t, cos_t), encoding time-of-day cyclically so
23:59 and 00:01 are nearby.

Supported backbones
-------------------
  resnet50, efficientnet_b3, efficientnet_b4,
  convnext_tiny, convnext_small, convnext_base,
  swin_t, swin_s,
  vit_b_16

Changes vs previous version
----------------------------
* get_scheduler() helper: SequentialLR warmup + CosineAnnealing for ViT / Swin.
* TimeOfDayModel now handles ViT and Swin feature extraction properly:
    - ViT uses backbone.encoder + CLS token extraction instead of children()[:-1]
    - Swin uses backbone.features + adaptive pool instead of children()[:-1]
  All other backbones continue to use the sequential children slice.
* _freeze_layers() is architecture-aware: ViT freezes named encoder layers,
  Swin freezes named feature stages, CNNs/ResNets use substring matching.
* unfreeze_all() works correctly across all backbone families.
* EfficientNet-B4 added.
* Swin-S added.
* WARMUP_EPOCHS added to Config (used automatically for vit_b_16 / swin_*).
* train_fold() uses get_scheduler() and respects WARMUP_EPOCHS.
* _save_best_checkpoint() in tune.py passes warmup_epochs through.

Loss
----
Mean squared error on the (sin_t, cos_t) pair.
Primary metric: Mean Absolute Error in *minutes* (cyclic-aware).

Usage
-----
Adjust settings in config.py, then:
    python main.py
"""

import os
import time
from typing import List, Optional, Tuple
import multiprocessing as mp

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)
from torch.utils.data import DataLoader

from config import Config as cfg
from TimeOfDayDataLoader import (
    TimeOfDayDataset,
    create_dataloaders,
    decode_time_tensor,
    get_metadata_dim,
    get_transforms,
    minutes_to_hhmm,
    extract_exif_datetime,
    tta_predict,
    MINUTES_PER_DAY,
)

try:
    import torchvision.models as models
    from torchvision.models import (
        ResNet50_Weights,
        EfficientNet_B3_Weights,
        EfficientNet_B4_Weights,
        ConvNeXt_Tiny_Weights,
        ConvNeXt_Small_Weights,
        ConvNeXt_Base_Weights,
        Swin_T_Weights,
        Swin_S_Weights,
        ViT_B_16_Weights,
    )
    _TORCHVISION_AVAILABLE = True
except ImportError:
    _TORCHVISION_AVAILABLE = False


# ---------------------------------------------------------------------------
# Scheduler factory
# ---------------------------------------------------------------------------

def get_scheduler(
    optimizer:     torch.optim.Optimizer,
    epochs:        int,
    eta_min:       float,
    warmup_epochs: int = 0,
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Returns a CosineAnnealingLR scheduler, optionally preceded by a linear
    warmup phase implemented via SequentialLR.

    Parameters
    ----------
    optimizer     : the optimizer to schedule
    epochs        : total training epochs (warmup + cosine)
    eta_min       : minimum LR at the end of the cosine phase
    warmup_epochs : number of linear warmup epochs (0 = no warmup)
                    ViT and Swin should use 3–5; CNNs can use 0.
    """
    if warmup_epochs > 0:
        warmup = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs - warmup_epochs),
            eta_min=eta_min,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
    return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

# Backbone families that need special feature-extraction handling
_SWIN_MODELS  = {"swin_t", "swin_s"}
_VIT_MODELS   = {"vit_b_16"}
_TRANSFORMER_MODELS = _SWIN_MODELS | _VIT_MODELS


class MetadataFusionMLP(nn.Module):
    def __init__(
        self,
        image_feat_dim: int   = 2048,
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
        x = torch.cat([image_feats, metadata], dim=1)
        return self.net(x)


class TimeOfDayModel(nn.Module):
    """
    Backbone + fusion MLP for cyclic time-of-day regression.

    The backbone is frozen up to `freeze_until` during early training;
    call unfreeze_all() at UNFREEZE_EPOCH to open the whole network.

    Encoder contracts
    -----------------
    * CNN backbones (ResNet, EfficientNet, ConvNeXt):
        self.encoder = Sequential(*backbone.children()[:-1])
        forward: encoder(x).flatten(1)  → (B, feat_dim)

    * Swin Transformer:
        self.encoder  = backbone.features  (patch embed + 4 stages)
        self._pool    = AdaptiveAvgPool2d(1)
        forward: pool(encoder(x)).flatten(1)  → (B, 768 or 1024)

    * ViT-B/16:
        self.encoder      = backbone.encoder  (transformer blocks)
        self._vit_proj    = backbone.conv_proj  (patch embedding)
        self._vit_cls     = backbone.class_token
        self._vit_pos_emb = backbone.encoder.pos_embedding
        forward: extract CLS token after full transformer  → (B, 768)
    """

    def __init__(
        self,
        pretrained:    bool  = True,
        freeze_until:  str   = "layer2",
        hidden_dim:    int   = 256,
        dropout:       float = 0.3,
        metadata_dim:  Optional[int] = None,
    ):
        super().__init__()
        if not _TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is required.")

        if metadata_dim is None:
            metadata_dim = get_metadata_dim()

        self._model_name = cfg.MODEL.lower()
        self._pool       = None   # used by Swin only

        feat_dim = self._build_encoder(pretrained)
        self._freeze_layers(freeze_until)

        self.fusion = MetadataFusionMLP(
            image_feat_dim=feat_dim,
            metadata_dim=metadata_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    # ── Encoder construction ─────────────────────────────────────────────

    def _build_encoder(self, pretrained: bool) -> int:
        """Instantiate self.encoder (and helpers for ViT/Swin). Returns feat_dim."""
        name = self._model_name

        # ── ResNet ───────────────────────────────────────────────────────
        if name == "resnet50":
            weights  = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = models.resnet50(weights=weights)
            self.encoder = nn.Sequential(*list(backbone.children())[:-1])
            return 2048

        # ── EfficientNet ─────────────────────────────────────────────────
        elif name == "efficientnet_b3":
            weights  = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.efficientnet_b3(weights=weights)
            self.encoder = nn.Sequential(*list(backbone.children())[:-1])
            return 1536

        elif name == "efficientnet_b4":
            weights  = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.efficientnet_b4(weights=weights)
            self.encoder = nn.Sequential(*list(backbone.children())[:-1])
            return 1792

        # ── ConvNeXt ─────────────────────────────────────────────────────
        elif name == "convnext_tiny":
            weights  = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.convnext_tiny(weights=weights)
            self.encoder = nn.Sequential(*list(backbone.children())[:-1])
            return 768

        elif name == "convnext_small":
            weights  = ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.convnext_small(weights=weights)
            self.encoder = nn.Sequential(*list(backbone.children())[:-1])
            return 768

        elif name == "convnext_base":
            weights  = ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.convnext_base(weights=weights)
            self.encoder = nn.Sequential(*list(backbone.children())[:-1])
            return 1024

        # ── Swin Transformer ─────────────────────────────────────────────
        elif name == "swin_t":
            weights  = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.swin_t(weights=weights)
            # backbone.features: patch embed + 4 stages (each a Sequential)
            self.encoder = backbone.features
            self._pool   = nn.AdaptiveAvgPool2d(1)
            return 768

        elif name == "swin_s":
            weights  = Swin_S_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.swin_s(weights=weights)
            self.encoder = backbone.features
            self._pool   = nn.AdaptiveAvgPool2d(1)
            return 768

        # ── Vision Transformer ───────────────────────────────────────────
        elif name == "vit_b_16":
            weights  = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.vit_b_16(weights=weights)
            # Store the components we need for a proper forward pass
            self._vit_proj    = backbone.conv_proj          # patch embedding
            self._vit_cls     = backbone.class_token        # (1, 1, 768)
            self._vit_pos_emb = backbone.encoder.pos_embedding
            self.encoder      = backbone.encoder            # transformer + ln
            return 768

        else:
            raise ValueError(
                f"Model '{cfg.MODEL}' not supported. Choose from: "
                "resnet50, efficientnet_b3, efficientnet_b4, "
                "convnext_tiny, convnext_small, convnext_base, "
                "swin_t, swin_s, vit_b_16"
            )

    # ── Freezing ─────────────────────────────────────────────────────────

    def _freeze_layers(self, freeze_until: str) -> None:
        """
        Freeze backbone parameters up to (but not including) the first
        parameter whose name contains `freeze_until`.

        For ViT, `freeze_until` should be a layer index string such as
        "encoder_layer_9" (freeze layers 0–8, train 9–11).
        For Swin, use stage names like "features.4" (train from stage 4 on).
        For CNNs, use "layer2", "layer3", "blocks.5" etc.
        """
        freeze = True
        named = list(self.encoder.named_parameters())

        # ViT also has patch-projection and CLS token params
        if self._model_name in _VIT_MODELS:
            vit_extras = (
                list(self._vit_proj.named_parameters()) +
                [("class_token", self._vit_cls)] +
                [("pos_embedding", self._vit_pos_emb)]
            )
            named = vit_extras + named   # extras come first → always frozen until freeze_until hits

        for name, param in named:
            if freeze_until in name:
                freeze = False
            param.requires_grad = not freeze

    def unfreeze_all(self) -> None:
        """Unfreeze every backbone parameter."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        if self._model_name in _VIT_MODELS:
            for param in self._vit_proj.parameters():
                param.requires_grad = True
            self._vit_cls.requires_grad     = True
            self._vit_pos_emb.requires_grad = True

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(self, images: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        name = self._model_name

        if name in _SWIN_MODELS:
            # encoder output: (B, H, W, C) → permute → pool → flatten
            x = self.encoder(images)          # (B, H', W', C)
            x = x.permute(0, 3, 1, 2)        # (B, C, H', W')
            x = self._pool(x).flatten(1)      # (B, C)

        elif name in _VIT_MODELS:
            # Replicate ViT._process_input + encoder forward
            x  = self._vit_proj(images)       # (B, C, gh, gw)
            B, C, gh, gw = x.shape
            x  = x.flatten(2).transpose(1, 2) # (B, gh*gw, C)
            cls = self._vit_cls.expand(B, -1, -1)  # (B, 1, C)
            x  = torch.cat([cls, x], dim=1)   # (B, 1+gh*gw, C)
            x  = x + self._vit_pos_emb        # positional embedding
            x  = self.encoder(x)              # (B, 1+gh*gw, C)
            x  = x[:, 0]                      # CLS token → (B, C)

        else:
            x = self.encoder(images)
            x = x.flatten(start_dim=1)

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
    diff = torch.min(diff, MINUTES_PER_DAY - diff)
    return diff.mean()


# ---------------------------------------------------------------------------
# Mixup helper
# ---------------------------------------------------------------------------

def mixup_batch(
    images:   torch.Tensor,
    metadata: torch.Tensor,
    targets:  torch.Tensor,
    alpha:    float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Applies Mixup to a batch.  Targets are (sin_t, cos_t) pairs — linearly
    interpolating them is valid since we re-normalise nothing (the loss is MSE).
    """
    if alpha <= 0.0:
        return images, metadata, targets

    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    B   = images.size(0)
    idx = torch.randperm(B, device=images.device)

    images   = lam * images   + (1 - lam) * images[idx]
    metadata = lam * metadata + (1 - lam) * metadata[idx]
    targets  = lam * targets  + (1 - lam) * targets[idx]
    return images, metadata, targets


# ---------------------------------------------------------------------------
# Label noise helper
# ---------------------------------------------------------------------------

def add_label_noise(targets: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    """Add small Gaussian noise to (sin_t, cos_t) targets as regularisation."""
    if std <= 0.0:
        return targets
    return targets + torch.randn_like(targets) * std


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
    mixup_alpha: float = 0.0,
    label_noise: float = 0.0,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_mae  = 0.0
    n_batches  = len(loader)

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

        mae = cyclic_mae_minutes(preds.detach().cpu(), targets.detach().cpu())
        total_loss += loss.item()
        total_mae  += mae

    return total_loss / n_batches, total_mae / n_batches


@torch.no_grad()
def evaluate(
    model:       TimeOfDayModel,
    loader:      DataLoader,
    criterion:   nn.Module,
    device:      torch.device,
    use_tta:     bool = False,
    tta_passes:  int  = 4,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_mae  = 0.0
    n_batches  = 0

    for images, metadata, targets in loader:
        images   = images.to(device,   non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)
        targets  = targets.to(device,  non_blocking=True)

        if use_tta and tta_passes > 1:
            preds = tta_predict(model, images, metadata, n_passes=tta_passes)
        else:
            preds = model(images, metadata)

        loss = criterion(preds, targets)
        mae  = cyclic_mae_minutes(preds.cpu(), targets.cpu())
        total_loss += loss.item()
        total_mae  += mae
        n_batches  += 1

    return total_loss / n_batches, total_mae / n_batches


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

    doy   = _day_of_year(month, day, year)
    label = TimeOfDayLabel(time_min=0, month=month, day_of_year=doy,
                           latitude=latitude, longitude=longitude,
                           image_features=image_features)
    meta_tensor = label.to_metadata_tensor().unsqueeze(0).to(device)

    model.eval()
    if use_tta and tta_passes > 1:
        pred = tta_predict(model, image_tensor, meta_tensor, n_passes=tta_passes)
    else:
        pred = model(image_tensor, meta_tensor)

    pred_min = decode_time_tensor(pred.cpu())[0].item()
    return minutes_to_hhmm(pred_min)


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
    """Average (sin_t, cos_t) predictions across multiple fold models."""
    from PIL import Image as PILImage
    from TimeOfDayDataLoader import TimeOfDayLabel, ImageFeatureExtractor, _day_of_year

    transform = get_transforms(augment=False)
    img = PILImage.open(image_path).convert("RGB")
    img = img.resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), PILImage.Resampling.LANCZOS)

    image_features = ImageFeatureExtractor.extract(img) if cfg.USE_IMAGE_FEATURES else None
    image_tensor   = transform(img).unsqueeze(0).to(device)

    doy   = _day_of_year(month, day, year)
    label = TimeOfDayLabel(time_min=0, month=month, day_of_year=doy,
                           latitude=latitude, longitude=longitude,
                           image_features=image_features)
    meta_tensor = label.to_metadata_tensor().unsqueeze(0).to(device)

    all_preds = []
    for m in models:
        m.eval()
        if use_tta and tta_passes > 1:
            p = tta_predict(m, image_tensor, meta_tensor, n_passes=tta_passes)
        else:
            p = m(image_tensor, meta_tensor)
        all_preds.append(p.cpu())

    avg_pred = torch.stack(all_preds, dim=0).mean(dim=0)
    pred_min = decode_time_tensor(avg_pred)[0].item()
    return minutes_to_hhmm(pred_min)


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
# Single-fold training
# ---------------------------------------------------------------------------

def train_fold(fold: int, device: torch.device) -> float:
    """Train one fold. Returns best val MAE for that fold."""
    train_tf = get_transforms(augment=True, magnitude=cfg.AUG_MAGNITUDE)
    dataset  = TimeOfDayDataset(image_dir=cfg.IMAGE_DIR, transform=train_tf)

    train_loader, val_loader = create_dataloaders(
        dataset,
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
    scheduler = get_scheduler(
        optimizer,
        epochs=cfg.EPOCHS,
        eta_min=cfg.ETA_MIN,
        warmup_epochs=cfg.WARMUP_EPOCHS,
    )
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

        # Scheduled backbone unfreeze
        if cfg.UNFREEZE_EPOCH is not None and epoch == cfg.UNFREEZE_EPOCH:
            model.unfreeze_all()
            optimizer = AdamW(
                model.parameters(),
                lr=cfg.LR * 0.1,
                weight_decay=cfg.WEIGHT_DECAY,
            )
            # Reset scheduler for the remaining epochs after unfreeze
            remaining = cfg.EPOCHS - epoch
            scheduler = get_scheduler(
                optimizer,
                epochs=remaining,
                eta_min=cfg.ETA_MIN,
                warmup_epochs=0,
            )
            print(f"  ↑ Backbone unfrozen at epoch {epoch+1} (LR×0.1)")

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

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            save_checkpoint(model, optimizer, scheduler, epoch + 1,
                            val_mae, best_ckpt)
            print(f"  ★ New best val MAE: {best_val_mae:.2f} min "
                  f"({minutes_to_hhmm(best_val_mae)} error)")

    last_ckpt = os.path.join(cfg.OUTPUT_DIR, f"last_fold{fold}.pt")
    save_checkpoint(model, optimizer, scheduler, cfg.EPOCHS, val_mae, last_ckpt)

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
    print(f"  Time-of-Day Estimation")
    print(f"  Device     : {device}")
    print(f"  Model      : {cfg.MODEL}")
    print(f"  Image size  : {cfg.IMAGE_SIZE}")
    print(f"  Metadata   : {get_metadata_dim()}d  "
          f"(image features: {'on' if cfg.USE_IMAGE_FEATURES else 'off'})")
    print(f"  Augment    : {cfg.AUG_MAGNITUDE}")
    print(f"  Mixup α    : {cfg.MIXUP_ALPHA}")
    print(f"  Label noise: {cfg.LABEL_NOISE_STD}")
    print(f"  Warmup     : {cfg.WARMUP_EPOCHS} epoch(s)")
    print(f"  TTA        : {'on' if cfg.TTA_ENABLED else 'off'}  "
          f"(passes={cfg.TTA_FLIPS})")
    print(f"  All folds  : {cfg.TRAIN_ALL_FOLDS}")
    print(f"{'='*60}\n")

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # ── Evaluation-only mode ──────────────────────────────────────────────
    if cfg.EVAL_ONLY:
        assert cfg.CHECKPOINT, "EVAL_ONLY requires a CHECKPOINT path."
        val_tf   = get_transforms(augment=False)
        dataset  = TimeOfDayDataset(image_dir=cfg.IMAGE_DIR, transform=val_tf)
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
        print(f"\nEvaluation — Val loss: {val_loss:.6f}  |  "
              f"Val MAE: {val_mae:.2f} min ({val_mae/60:.2f} h)")
        return

    # ── Training ──────────────────────────────────────────────────────────
    if cfg.TRAIN_ALL_FOLDS:
        fold_maes = []
        for fold in range(cfg.N_SPLITS):
            print(f"\n{'='*60}")
            print(f"  FOLD {fold + 1} / {cfg.N_SPLITS}")
            print(f"{'='*60}")
            mae = train_fold(fold, device)
            fold_maes.append(mae)

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