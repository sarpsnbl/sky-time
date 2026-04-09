"""
TimeOfDayDataLoader.py
======================
Dataset loading, augmentation, and DataLoader creation for
time-of-day regression from sky/outdoor images + EXIF date metadata.

Problem
-------
Predict the exact time of day (in minutes since midnight, 0–1439) from:
  - An outdoor / sky photograph
  - The calendar date on which the photo was taken (extracted via EXIF)

Expected folder layout
----------------------
data/
    image_001.jpg          <- any supported format with EXIF data
    image_002.png
    ...
"""

import math
import os
import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ExifTags
from sklearn.model_selection import KFold, ShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

# HEIC/HEIF support (optional – install pillow-heif if needed)
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    _HEIC_SUPPORTED = True
except ImportError:
    _HEIC_SUPPORTED = False

MINUTES_PER_DAY = 1440.0
DAYS_PER_YEAR   = 365.25


# ---------------------------------------------------------------------------
# Cyclic encoding helpers
# ---------------------------------------------------------------------------

def cyclic_encode(value: float, period: float) -> Tuple[float, float]:
    angle = 2.0 * math.pi * value / period
    return math.sin(angle), math.cos(angle)

def cyclic_decode(sin_val: float, cos_val: float, period: float) -> float:
    angle = math.atan2(sin_val, cos_val)
    if angle < 0:
        angle += 2.0 * math.pi
    return angle * period / (2.0 * math.pi)

def decode_time_tensor(pred: torch.Tensor) -> torch.Tensor:
    angles = torch.atan2(pred[:, 0], pred[:, 1])
    angles = torch.where(angles < 0, angles + 2 * math.pi, angles)
    return angles * MINUTES_PER_DAY / (2 * math.pi)


# ---------------------------------------------------------------------------
# EXIF Parsing
# ---------------------------------------------------------------------------

def _day_of_year(month: int, day: int, year: int = 2024) -> int:
    """Approximate day-of-year (1–365). Uses a fixed leap year by default."""
    days_before = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    leap_offset = 1 if (month > 2 and year % 4 == 0) else 0
    return days_before[month - 1] + day + leap_offset

def extract_exif_datetime(image_path: str) -> Optional[Tuple[float, int, int, int]]:
    """
    Extracts time in minutes, month, day, and year from image EXIF data.
    Returns None if EXIF data is missing or unparseable.
    """
    try:
        with Image.open(image_path) as img:
            exif = img._getexif()
            if not exif:
                return None
            
            # 36867 is DateTimeOriginal, 306 is DateTime
            dt_str = exif.get(36867) or exif.get(306)
            if not dt_str:
                return None

            # Standard EXIF format: "YYYY:MM:DD HH:MM:SS"
            date_part, time_part = dt_str.split(" ")
            year, month, day = map(int, date_part.split(":"))
            hour, minute, second = map(int, time_part.split(":"))

            time_min = float(hour * 60 + minute + second / 60.0)
            return time_min, month, day, year
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Label entry
# ---------------------------------------------------------------------------

class TimeOfDayLabel:
    """Holds all metadata for a single sample."""

    __slots__ = ("time_min", "month", "day_of_year", "latitude", "longitude")

    def __init__(
        self,
        time_min: float,
        month: int,
        day_of_year: int,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
    ):
        self.time_min    = float(time_min)
        self.month       = int(month)
        self.day_of_year = int(day_of_year)
        self.latitude    = latitude
        self.longitude   = longitude

    def to_metadata_tensor(self) -> torch.Tensor:
        sin_m, cos_m   = cyclic_encode(self.month,       12.0)
        sin_d, cos_d   = cyclic_encode(self.day_of_year, DAYS_PER_YEAR)
        lat_norm = (self.latitude  / 90.0)  if self.latitude  is not None else 0.0
        lon_norm = (self.longitude / 180.0) if self.longitude is not None else 0.0
        return torch.tensor(
            [sin_m, cos_m, sin_d, cos_d, lat_norm, lon_norm],
            dtype=torch.float32,
        )

    def to_target_tensor(self) -> torch.Tensor:
        sin_t, cos_t = cyclic_encode(self.time_min, MINUTES_PER_DAY)
        return torch.tensor([sin_t, cos_t], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TimeOfDayDataset(Dataset):
    VALID_EXTENSIONS: frozenset = frozenset(
        {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp",
         ".heic", ".heif"}
    )

    def __init__(
        self,
        image_dir: str,
        transform: Optional[Callable] = None,
        target_size: int = 224,
        random_seed: int = 42,
    ):
        self.image_dir   = image_dir
        self.transform   = transform
        self.target_size = target_size
        self.random_seed = random_seed

        self.samples: List[Tuple[str, TimeOfDayLabel]] = []
        self._build_dataset()

        print(f"TimeOfDayDataset: {len(self.samples)} valid samples loaded "
              f"from '{image_dir}'")
        if self.samples:
            self._print_stats()

    def _build_dataset(self) -> None:
        random.seed(self.random_seed)
        
        filenames = [f for f in os.listdir(self.image_dir) if self._is_valid_file(f)]
        skipped = 0

        for fname in sorted(filenames):
            path = os.path.join(self.image_dir, fname)
            exif_data = extract_exif_datetime(path)
            
            if exif_data is None:
                skipped += 1
                continue
                
            time_min, month, day, year = exif_data
            doy = _day_of_year(month, day, year)
            
            label = TimeOfDayLabel(
                time_min=time_min, 
                month=month, 
                day_of_year=doy
            )
            self.samples.append((path, label))

        if skipped > 0:
            print(f"  WARNING: Skipped {skipped} image(s) missing valid EXIF date/time data.")

        if not self.samples:
            raise RuntimeError(
                "No valid images with EXIF data found. Check your image directory."
            )

    def _is_valid_file(self, filename: str) -> bool:
        ext = os.path.splitext(filename.lower())[1]
        if ext in {".heic", ".heif"} and not _HEIC_SUPPORTED:
            return False
        return ext in self.VALID_EXTENSIONS

    def _letterbox_resize(self, image: Image.Image) -> Image.Image:
        ow, oh = image.size
        scale  = min(self.target_size / ow, self.target_size / oh)
        nw, nh = int(ow * scale), int(oh * scale)
        resized = image.resize((nw, nh), Image.Resampling.LANCZOS)
        canvas  = Image.new("RGB", (self.target_size, self.target_size), (0, 0, 0))
        canvas.paste(resized, ((self.target_size - nw) // 2,
                               (self.target_size - nh) // 2))
        return canvas

    def _print_stats(self) -> None:
        times = [lbl.time_min for _, lbl in self.samples]
        times_arr = np.array(times)
        print(f"  Time-of-day range : {times_arr.min():.0f}–{times_arr.max():.0f} min "
              f"({int(times_arr.min())//60:02d}:{int(times_arr.min())%60:02d}"
              f"–{int(times_arr.max())//60:02d}:{int(times_arr.max())%60:02d})")
        print(f"  Mean / Std        : {times_arr.mean():.1f} / {times_arr.std():.1f} min")

    @property
    def raw_times(self) -> np.ndarray:
        return np.array([lbl.time_min for _, lbl in self.samples])

    def get_sample_weight(self) -> torch.Tensor:
        times  = self.raw_times
        hours  = (times / 60).astype(int) % 24
        counts = np.bincount(hours, minlength=24).astype(float)
        counts = np.where(counts == 0, 1.0, counts)   
        weights = 1.0 / counts[hours]
        weights /= weights.sum()
        return torch.from_numpy(weights.astype(np.float32))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_path, label = self.samples[idx]

        try:
            # Re-open the image (avoiding EXIF orientation issues using convert)
            image = Image.open(img_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = self._letterbox_resize(image)
        except Exception as exc:
            print(f"  ERROR loading '{img_path}': {exc} – using black fallback.")
            image = Image.new("RGB", (self.target_size, self.target_size), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        metadata = label.to_metadata_tensor()
        target   = label.to_target_tensor()

        return image, metadata, target


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_transforms(augment: bool = True) -> transforms.Compose:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225],
    )

    if augment:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(
                brightness=0.25, contrast=0.2, saturation=0.25, hue=0.04
            ),
            transforms.RandomResizedCrop(size=224, scale=(0.85, 1.0)),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

def minutes_to_hhmm(minutes: float) -> str:
    total = int(round(minutes)) % MINUTES_PER_DAY
    h, m  = divmod(int(total), 60)
    return f"{h:02d}:{m:02d}"


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_dataloaders(
    dataset: TimeOfDayDataset,
    fold:        int   = 0,
    n_splits:    int   = 5,
    batch_size:  int   = 32,
    num_workers: int   = 4,
    val_ratio:   float = 0.2,
    use_weighted_sampler: bool = False,
) -> Tuple[DataLoader, DataLoader]:

    indices = np.arange(len(dataset))

    if n_splits == 1:
        splitter = ShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)
        train_idx, val_idx = next(splitter.split(indices))
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        for fold_num, (tr, va) in enumerate(kf.split(indices)):
            if fold_num == fold:
                train_idx, val_idx = tr, va
                break
        else:
            raise ValueError(f"Fold {fold} not found; must be between 0 and {n_splits - 1}.")

    sampler = None
    shuffle_train = True
    if use_weighted_sampler:
        from torch.utils.data import WeightedRandomSampler
        all_weights = dataset.get_sample_weight()
        train_weights = all_weights[train_idx]
        sampler = WeightedRandomSampler(
            weights=train_weights,
            num_samples=len(train_idx),
            replacement=True,
        )
        shuffle_train = False 

    _pw = num_workers > 0
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=_pw,
        prefetch_factor=2 if _pw else None,
    )

    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=_pw,
        prefetch_factor=2 if _pw else None,
    )

    print(f"\nFold {fold}  |  train: {len(train_idx)} samples  |  "
          f"val: {len(val_idx)} samples")
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TimeOfDayDataLoader smoke-test")
    parser.add_argument("image_dir",  help="Flat folder containing images with EXIF")
    parser.add_argument("--batch-size",  type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--fold",        type=int, default=0)
    parser.add_argument("--n-splits",    type=int, default=5)
    args = parser.parse_args()

    train_tf = get_transforms(augment=True)
    dataset  = TimeOfDayDataset(
        image_dir=args.image_dir,
        transform=train_tf,
    )

    train_loader, val_loader = create_dataloaders(
        dataset,
        fold=args.fold,
        n_splits=args.n_splits,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    images, metadata, targets = next(iter(train_loader))
    print(f"\nSample batch shapes:")
    print(f"  images   : {images.shape}")
    print(f"  metadata : {metadata.shape}")
    print(f"  targets  : {targets.shape}")

    decoded = decode_time_tensor(targets)
    print(f"\nFirst 5 decoded times (minutes): {decoded[:5].tolist()}")
    print(f"First 5 decoded times (HH:MM)  : "
          f"{[minutes_to_hhmm(t.item()) for t in decoded[:5]]}")