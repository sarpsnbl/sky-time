"""
sample_run.py
=============
Executes a sample training and validation run on the real dataset_512
using the complete 108-dimensional FiLM architecture, unit-circle normalization,
and AngularCosineLoss.
"""
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np

from config import Config as cfg
from TimeOfDayDataLoader import (
    TimeOfDayDataset,
    get_transforms,
    decode_time_tensor,
    minutes_to_hhmm,
    MINUTES_PER_DAY,
)
from Main import (
    TimeOfDayModel,
    AngularCosineLoss,
    cyclic_mae_minutes,
    mixup_batch,
    add_label_noise,
)

def run_sample():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==================================================")
    print(f"Starting Sample Run on dataset_512")
    print(f"Device: {device}")
    print(f"==================================================")

    # 1. Load Dataset (instantaneous via cached features)
    t0 = time.time()
    full_dataset = TimeOfDayDataset(
        image_dir=cfg.IMAGE_DIR,
        transform=get_transforms(augment=True, magnitude="light"),
    )
    val_dataset = TimeOfDayDataset(
        image_dir=cfg.IMAGE_DIR,
        transform=get_transforms(augment=False),
    )
    print(f"Dataset loaded: {len(full_dataset)} total images in {time.time()-t0:.2f}s")

    # 2. Create Train / Val subsets for sample execution
    total_samples = len(full_dataset)
    indices = np.random.RandomState(42).permutation(total_samples)
    
    # Use 128 samples for train and 32 samples for validation to run quickly on CPU
    train_size = min(128, int(total_samples * 0.8))
    val_size   = min(32, total_samples - train_size)
    
    train_idx = indices[:train_size]
    val_idx   = indices[train_size:train_size + val_size]

    train_loader = DataLoader(
        Subset(full_dataset, train_idx),
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_idx),
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    print(f"Train samples: {len(train_idx)} | Val samples: {len(val_idx)}")

    # 3. Build Model with FiLM and 108-dim metadata
    print("\nInitializing TimeOfDayModel (ConvNeXt-Tiny + FiLMFusionHead)...")
    # For CPU sample run, convnext_tiny runs fast and stably
    cfg.MODEL = "convnext_tiny"
    model = TimeOfDayModel(
        pretrained=True,
        freeze_until="features.5",
        hidden_dim=256,
        dropout=0.1,
        metadata_dim=108,
        use_film=True,
    ).to(device)

    print(f"Model successfully built. Trainable parameters: {model.count_trainable_params():,}")

    criterion = AngularCosineLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=3e-4,
        weight_decay=1e-2,
    )

    # 4. Run 2 Training Epochs
    for epoch in range(1, 3):
        epoch_t0 = time.time()
        model.train()
        running_loss = 0.0
        running_mae = 0.0

        for b_idx, (images, metadata, targets) in enumerate(train_loader):
            images   = images.to(device)
            metadata = metadata.to(device)
            targets  = targets.to(device)

            # Apply circular mixup and label noise
            images, metadata, targets = mixup_batch(images, metadata, targets, alpha=0.1)
            targets = add_label_noise(targets, std=0.01)

            optimizer.zero_grad()
            preds = model(images, metadata)

            loss = criterion(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            mae = cyclic_mae_minutes(preds.detach(), targets.detach())
            running_loss += loss.item()
            running_mae  += mae.item()

        n_train = len(train_loader)
        train_loss = running_loss / n_train
        train_mae  = running_mae / n_train

        # Validation
        model.eval()
        val_loss = 0.0
        val_mae  = 0.0
        sample_preds = []

        with torch.no_grad():
            for images, metadata, targets in val_loader:
                images   = images.to(device)
                metadata = metadata.to(device)
                targets  = targets.to(device)

                preds = model(images, metadata)
                loss = criterion(preds, targets)
                mae = cyclic_mae_minutes(preds, targets)

                val_loss += loss.item()
                val_mae  += mae.item()

                if len(sample_preds) < 5:
                    p_mins = decode_time_tensor(preds).cpu().numpy()
                    t_mins = decode_time_tensor(targets).cpu().numpy()
                    for p, t in zip(p_mins, t_mins):
                        sample_preds.append((p, t))

        n_val = len(val_loader)
        val_loss /= n_val
        val_mae  /= n_val

        elapsed = time.time() - epoch_t0
        print(f"\n--- Epoch {epoch}/2 [{elapsed:.1f}s] ---")
        print(f"  Train Loss: {train_loss:.4f} | Train MAE: {train_mae:.1f} min ({train_mae/60:.2f} hrs)")
        print(f"  Val   Loss: {val_loss:.4f} | Val   MAE: {val_mae:.1f} min ({val_mae/60:.2f} hrs)")

    # 5. Display sample predictions
    print(f"\n==================================================")
    print(f"Sample Validation Predictions (Predicted vs Actual):")
    print(f"{'Pred Time':<12} | {'Actual Time':<12} | {'Circ Error':<12}")
    print(f"--------------------------------------------------")
    for p, a in sample_preds[:5]:
        diff = abs(p - a)
        err = min(diff, MINUTES_PER_DAY - diff)
        print(f"{minutes_to_hhmm(p):<12} | {minutes_to_hhmm(a):<12} | {err:.1f} min ({err/60:.2f}h)")
    print(f"==================================================")
    print(f"Sample run completed successfully!")

if __name__ == "__main__":
    run_sample()
