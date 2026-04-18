"""
ensemble.py
===========
Loads multiple fold checkpoints and produces a final ensemble for:

  1. Batch evaluation  — runs the ensemble over the validation dataset
                         and reports mean cyclic MAE.

  2. Single-image inference — predicts time-of-day for one image using
                              all loaded models and averages their (sin,cos)
                              outputs before decoding.

  3. Model soup — merges all model weights into one averaged checkpoint
                  (weight averaging). Single model at inference time.

Usage
-----
  python ensemble.py --mode eval
  python ensemble.py --mode predict --image path/to/photo.jpg
  python ensemble.py --mode soup --out checkpoints/soup.pt
  python ensemble.py --mode eval --checkpoints checkpoints/best_fold0.pt checkpoints/best_fold2.pt
"""

import argparse
import os
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config as cfg
from main import (
    TimeOfDayModel,
    CyclicMSELoss,
    cyclic_mae_minutes,
    load_checkpoint,
)
from TimeOfDayDataLoader import (
    TimeOfDayDataset,
    create_dataloaders,
    decode_time_tensor,
    extract_exif_data,
    get_transforms,
    minutes_to_hhmm,
    MINUTES_PER_DAY,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discover_checkpoints(ckpt_dir: str) -> List[str]:
    """Auto-discovers best_fold*.pt files, falling back to last_fold*.pt."""
    ckpt_dir = Path(ckpt_dir)
    paths = sorted(ckpt_dir.glob("best_fold*.pt"))
    if not paths:
        paths = sorted(ckpt_dir.glob("last_fold*.pt"))
    if not paths:
        raise FileNotFoundError(
            f"No best_fold*.pt or last_fold*.pt checkpoints found in '{ckpt_dir}'."
        )
    return [str(p) for p in paths]


def load_ensemble(checkpoint_paths: List[str], device: torch.device) -> List[TimeOfDayModel]:
    """Instantiates and loads one model per checkpoint. All set to eval mode."""
    models = []
    for path in checkpoint_paths:
        model = TimeOfDayModel(
            pretrained=False,
            freeze_until=cfg.FREEZE_UNTIL,
            hidden_dim=cfg.HIDDEN_DIM,
            dropout=cfg.DROPOUT,
        ).to(device)
        load_checkpoint(path, model, device=device)
        model.eval()
        models.append(model)
        print(f"  Loaded: {path}")
    return models


@torch.no_grad()
def ensemble_predict(
    models:   List[TimeOfDayModel],
    images:   torch.Tensor,
    metadata: torch.Tensor,
) -> torch.Tensor:
    """
    Averages (sin_t, cos_t) outputs across all models, then re-normalises
    onto the unit circle for stable atan2 decoding.
    """
    all_preds = [model(images, metadata) for model in models]
    avg = torch.stack(all_preds, dim=0).mean(dim=0)
    norm = avg.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return avg / norm


# ---------------------------------------------------------------------------
# Mode 1: Batch evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_eval(args, device):
    checkpoint_paths = args.checkpoints or discover_checkpoints(cfg.OUTPUT_DIR)
    print(f"\nEnsemble of {len(checkpoint_paths)} checkpoint(s):")
    models = load_ensemble(checkpoint_paths, device)

    dataset = TimeOfDayDataset(
        image_dir=cfg.IMAGE_DIR,
        transform=get_transforms(augment=False),
    )
    _, val_loader = create_dataloaders(
        dataset, dataset,
        fold=cfg.FOLD,
        n_splits=cfg.N_SPLITS,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        val_ratio=cfg.VAL_RATIO,
    )

    criterion = CyclicMSELoss()
    total_loss = total_mae = 0.0

    for images, metadata, targets in val_loader:
        images   = images.to(device, non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)
        targets  = targets.to(device, non_blocking=True)

        preds = ensemble_predict(models, images, metadata)
        total_loss += criterion(preds, targets).item()
        total_mae  += cyclic_mae_minutes(preds.cpu(), targets.cpu())

    n   = len(val_loader)
    mae = float(total_mae / n)
    print(f"\nEnsemble Evaluation")
    print(f"  Models   : {len(models)}")
    print(f"  Val loss : {total_loss/n:.6f}")
    print(f"  Val MAE  : {mae:.2f} min  ({mae/60:.2f} h)")


# ---------------------------------------------------------------------------
# Mode 2: Single-image prediction
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_predict(args, device):
    from PIL import Image as PILImage
    from TimeOfDayDataLoader import TimeOfDayLabel, ImageFeatureExtractor, _day_of_year

    checkpoint_paths = args.checkpoints or discover_checkpoints(cfg.OUTPUT_DIR)
    print(f"\nEnsemble of {len(checkpoint_paths)} checkpoint(s):")
    models = load_ensemble(checkpoint_paths, device)

    transform = get_transforms(augment=False)
    img = PILImage.open(args.image).convert("RGB")
    img = img.resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), PILImage.Resampling.LANCZOS)
    image_tensor = transform(img).unsqueeze(0).to(device)

    exif_data = extract_exif_data(args.image)
    if exif_data is not None:
        _, month, day, year, lat, lon = exif_data
        doy = _day_of_year(month, day, year)
        print(f"  EXIF date : {year}-{month:02d}-{day:02d}  (day {doy})")
        if lat is not None:
            print(f"  EXIF GPS  : {lat:.4f}, {lon:.4f}")
    else:
        month, doy, lat, lon = 1, 1, None, None
        print("  EXIF date : not found, calendar metadata zeroed out")

    image_features = ImageFeatureExtractor.extract(img) if cfg.USE_IMAGE_FEATURES else None
    label = TimeOfDayLabel(
        time_min=0, month=month, day_of_year=doy,
        latitude=lat, longitude=lon,
        image_features=image_features,
    )
    meta_tensor = label.to_metadata_tensor().unsqueeze(0).to(device)

    pred = ensemble_predict(models, image_tensor, meta_tensor)
    result = minutes_to_hhmm(decode_time_tensor(pred.cpu())[0].item())

    print(f"\nImage     : {args.image}")
    print(f"Predicted : {result}")


# ---------------------------------------------------------------------------
# Mode 3: Model soup (weight averaging)
# ---------------------------------------------------------------------------

def run_soup(args, device):
    """
    Averages all checkpoint weights into a single 'soup' model.
    Nearly as accurate as the full ensemble but only one model at inference time.
    """
    checkpoint_paths = args.checkpoints or discover_checkpoints(cfg.OUTPUT_DIR)
    print(f"\nModel soup from {len(checkpoint_paths)} checkpoint(s):")
    for p in checkpoint_paths:
        print(f"  {p}")

    state_dicts = [
        torch.load(p, map_location=device)["model"]
        for p in checkpoint_paths
    ]

    soup_state = {
        key: torch.stack([sd[key].float() for sd in state_dicts], dim=0).mean(dim=0)
        for key in state_dicts[0]
    }

    soup_model = TimeOfDayModel(
        pretrained=False,
        freeze_until=cfg.FREEZE_UNTIL,
        hidden_dim=cfg.HIDDEN_DIM,
        dropout=cfg.DROPOUT,
    ).to(device)
    soup_model.load_state_dict(soup_state)
    soup_model.eval()
    print("  Weight averaging complete.")

    # Quick eval
    dataset = TimeOfDayDataset(
        image_dir=cfg.IMAGE_DIR,
        transform=get_transforms(augment=False),
    )
    _, val_loader = create_dataloaders(
        dataset, dataset,
        fold=cfg.FOLD,
        n_splits=cfg.N_SPLITS,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        val_ratio=cfg.VAL_RATIO,
    )
    criterion = CyclicMSELoss()
    total_loss = total_mae = 0.0

    with torch.no_grad():
        for images, metadata, targets in val_loader:
            images   = images.to(device)
            metadata = metadata.to(device)
            targets  = targets.to(device)
            preds    = soup_model(images, metadata)
            total_loss += criterion(preds, targets).item()
            total_mae  += cyclic_mae_minutes(preds.cpu(), targets.cpu())

    n   = len(val_loader)
    mae = float(total_mae / n)
    print(f"\nSoup Evaluation")
    print(f"  Val loss : {total_loss/n:.6f}")
    print(f"  Val MAE  : {mae:.2f} min  ({mae/60:.2f} h)")

    out_path = args.out or os.path.join(cfg.OUTPUT_DIR, "soup.pt")
    torch.save({"model": soup_state, "epoch": 0, "val_mae": mae}, out_path)
    print(f"\n  Saved → {out_path}")

# ---------------------------------------------------------------------------
# Mode 4: Per-image audit (full dataset scan for cleaning)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_audit(args, device):
    """
    Runs the ensemble over the FULL dataset (train + val) and reports
    per-image cyclic error, sorted worst-first. Use this to find images
    worth removing before retraining — not for benchmarking.
    """
    checkpoint_paths = args.checkpoints or discover_checkpoints(cfg.OUTPUT_DIR)
    print(f"\nAudit mode — scanning full dataset with {len(checkpoint_paths)} model(s).")
    print("NOTE: this includes training images. Use for data cleaning only.\n")
    models = load_ensemble(checkpoint_paths, device)

    dataset = TimeOfDayDataset(
        image_dir=cfg.IMAGE_DIR,
        transform=get_transforms(augment=False),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
    )

    all_paths, all_errors, all_preds, all_actuals = [], [], [], []
    sample_iter = iter(path for path, _ in dataset.samples)

    for images, metadata, targets in loader:
        images   = images.to(device, non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)

        preds   = ensemble_predict(models, images, metadata)
        pred_m  = decode_time_tensor(preds.cpu())
        actual_m = decode_time_tensor(targets)

        diff = torch.abs(pred_m - actual_m)
        errors = torch.min(diff, MINUTES_PER_DAY - diff)

        all_preds.extend(pred_m.tolist())
        all_actuals.extend(actual_m.tolist())
        all_errors.extend(errors.tolist())
        all_paths.extend(next(sample_iter) for _ in range(images.size(0)))

    rows = sorted(zip(all_errors, all_paths, all_preds, all_actuals), reverse=True)

    print(f"{'Error':>8}  {'Predicted':>9}  {'Actual':>9}  Path")
    print("-" * 80)
    for err, path, pred, actual in rows:
        print(f"{err:>7.1f}m  {minutes_to_hhmm(pred):>9}  {minutes_to_hhmm(actual):>9}  {path}")

    if args.out:
        import csv
        with open(args.out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["error_min", "predicted", "actual", "path"])
            for err, path, pred, actual in rows:
                writer.writerow([f"{err:.1f}", minutes_to_hhmm(pred), minutes_to_hhmm(actual), path])
        print(f"\nSaved → {args.out}")

    mae = sum(all_errors) / len(all_errors)
    print(f"\nFull-dataset MAE : {mae:.2f} min (not a valid benchmark)")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ensemble inference and model soup.")
    parser.add_argument("--mode", choices=["eval", "predict", "soup", "audit"], required=True)
    parser.add_argument("--checkpoints", nargs="+", default=None,
                        help="Explicit checkpoint paths (default: auto-discover from config OUTPUT_DIR).")
    parser.add_argument("--image", default=None, help="Image path for predict mode.")
    parser.add_argument("--out",   default=None, help="Output path for soup checkpoint.")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")

    if args.mode == "eval":
        run_eval(args, device)
    elif args.mode == "predict":
        if not args.image:
            parser.error("--image is required for predict mode.")
        run_predict(args, device)
    elif args.mode == "soup":
        run_soup(args, device)
    elif args.mode == "audit":
        run_audit(args, device)


if __name__ == "__main__":
    main()