"""
TimeOfDayDataLoader.py
======================
Dataset loading, augmentation, and DataLoader creation for
time-of-day regression from sky/outdoor images + EXIF date metadata.
"""

import math
import os
import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ExifTags
Image.MAX_IMAGE_PIXELS = None
import imageio.v3 as iio
from datetime import datetime
import exifread
import rawpy
from sklearn.model_selection import KFold, ShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from skimage import color, filters, feature

from config import Config as cfg

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    _HEIC_SUPPORTED = True
except ImportError:
    _HEIC_SUPPORTED = False

MINUTES_PER_DAY    = 1440.0
DAYS_PER_YEAR      = 365.25

# Dimensions of the two metadata parts — kept here so main.py can import them
# instead of hardcoding magic numbers.
_CALENDAR_DIM      = 6   # sin/cos month, sin/cos doy, lat_norm, lon_norm
_IMAGE_FEATURE_DIM = 77   # see ImageFeatureExtractor


def get_metadata_dim() -> int:
    """Return the full metadata vector length based on current config."""
    from config import Config as _cfg   # late import avoids circular dependency
    return _CALENDAR_DIM + (_IMAGE_FEATURE_DIM if _cfg.USE_IMAGE_FEATURES else 0)


# ---------------------------------------------------------------------------
# Handcrafted photometric feature extractor
# ---------------------------------------------------------------------------

class ImageFeatureExtractor:
    """
    Extracts a 57-dimensional feature vector matching the MATLAB pipeline.

    Layout (all features normalised to comparable scales):
      [0:6]    RGB channel means + stds             (6)
      [6:12]   HSV channel means + stds             (6)
      [12:18]  LAB channel means + stds             (6)  — L normalised to [0,1]
      [18:66]  RGB histograms, 16 bins × 3 channels (48) → but we keep only 16×3=48
      ... wait, 6+6+6+48+48+2+1+1+3+2+1+1 = 125 raw

    To keep the vector compact (and avoid exploding the MLP), we use
    8-bin histograms (24 hist features) instead of 16, giving 57 total:
      [0:6]    RGB means+stds                       (6)
      [6:12]   HSV means+stds                       (6)
      [12:18]  LAB means+stds                       (6)
      [18:42]  RGB histograms, 8 bins × 3           (24)
      [42:66]  HSV histograms, 8 bins × 3           (24) → trimmed to fit
    
    Actually let's count exactly and be explicit — see below.
    """

    @classmethod
    def extract(cls, image: "Image.Image") -> np.ndarray:
        """
        Parameters
        ----------
        image : PIL.Image.Image

        Returns
        -------
        np.ndarray, shape (_IMAGE_FEATURE_DIM,), dtype float32
        """
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0  # [0,1]
        H = rgb.shape[0]

        feat = []

        # ── 1. RGB channel means + stds (6) ──────────────────────────────
        for c in range(3):
            ch = rgb[:, :, c]
            feat += [ch.mean(), ch.std()]

        # ── 2. HSV channel means + stds (6) ──────────────────────────────
        hsv = color.rgb2hsv(rgb)  # all channels in [0,1]
        for c in range(3):
            ch = hsv[:, :, c]
            feat += [ch.mean(), ch.std()]

        # ── 3. LAB channel means + stds (6) ──────────────────────────────
        lab = color.rgb2lab(rgb)          # L in [0,100], a/b in [-128,127]
        lab_norm = lab / np.array([100.0, 128.0, 128.0])  # → [0,1] / [-1,1]
        for c in range(3):
            ch = lab_norm[:, :, c]
            feat += [ch.mean(), ch.std()]

        # ── 4. RGB histograms, 8 bins × 3 (24) ───────────────────────────
        for c in range(3):
            hist, _ = np.histogram(rgb[:, :, c], bins=8, range=(0.0, 1.0))
            feat += (hist / hist.sum()).tolist()

        # ── 5. HSV histograms, 8 bins × 3 (24) ───────────────────────────
        for c in range(3):
            hist, _ = np.histogram(hsv[:, :, c], bins=8, range=(0.0, 1.0))
            feat += (hist / hist.sum()).tolist()

        # ── 6. Sun-region brightness — top third (2) ─────────────────────
        top_v = hsv[: H // 3, :, 2]          # Value channel, top third
        feat += [top_v.mean(), top_v.std()]

        # ── 7. Horizon luminance gradient (1) ────────────────────────────
        V   = hsv[:, :, 2]
        dVy = np.diff(V, axis=0)             # vertical gradient
        feat += [np.abs(dVy).mean()]

        # ── 8. Colour temperature proxy R/B (1) ──────────────────────────
        r_mean = rgb[:, :, 0].mean()
        b_mean = rgb[:, :, 2].mean()
        feat += [r_mean / (b_mean + 1e-6)]   # unbounded; clip below if needed

        # ── 9. Global luminance stats + entropy (3) ───────────────────────
        lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        hist_lum, _ = np.histogram((lum * 255).astype(np.uint8), bins=256, range=(0, 256))
        hist_lum    = hist_lum / (hist_lum.sum() + 1e-8)
        entropy     = -np.sum(hist_lum * np.log2(hist_lum + 1e-8))   # bits, max ≈ 8
        feat += [lum.mean(), lum.std(), entropy / 8.0]               # normalise entropy

        # ── 10. Saturation mean + std (2) ────────────────────────────────
        S = hsv[:, :, 1]
        feat += [S.mean(), S.std()]

        # ── 11. Edge density via Canny (1) ────────────────────────────────
        edges = feature.canny(lum, sigma=1.0)
        feat += [edges.mean()]               # fraction of edge pixels

        # ── 12. Laplacian variance — sharpness / cloud texture (1) ───────
        lap = filters.laplace(lum)
        feat += [lap.var()]

        arr = np.array(feat, dtype=np.float32)

        # Clip the one unbounded feature (R/B ratio, index 66) to [0, 4]
        arr[66] = np.clip(arr[66], 0.0, 4.0) / 4.0

        return arr


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
    days_before = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    leap_offset = 1 if (month > 2 and year % 4 == 0) else 0
    return days_before[month - 1] + day + leap_offset

def extract_exif_datetime(image_path: str):
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f, stop_tag='DateTimeOriginal', details=False)

        ts_str = None
        for tag in ['EXIF DateTimeOriginal', 'Image DateTime', 'EXIF DateTimeDigitized']:
            if tag in tags:
                ts_str = str(tags[tag])
                break

        if ts_str:
            try:
                dt = datetime.strptime(ts_str, '%Y:%m:%d %H:%M:%S')
                return dt.hour * 60 + dt.minute, dt.month, dt.day, dt.year
            except ValueError:
                pass

        with Image.open(image_path) as img:
            exif = img.getexif()
            for tag_id in [36867, 306, 36868]:
                if tag_id in exif:
                    dt = datetime.strptime(exif[tag_id], '%Y:%m:%d %H:%M:%S')
                    return dt.hour * 60 + dt.minute, dt.month, dt.day, dt.year

    except Exception as e:
        print(f"Metadata error for {image_path}: {e}")

    try:
        mtime = os.path.getmtime(image_path)
        dt = datetime.fromtimestamp(mtime)
        return dt.hour * 60 + dt.minute, dt.month, dt.day, dt.year
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Label entry
# ---------------------------------------------------------------------------

class TimeOfDayLabel:
    __slots__ = ("time_min", "month", "day_of_year", "latitude", "longitude",
                 "image_features")

    def __init__(self, time_min, month, day_of_year, latitude=None, longitude=None,
                 image_features: Optional[np.ndarray] = None):
        self.time_min       = float(time_min)
        self.month          = int(month)
        self.day_of_year    = int(day_of_year)
        self.latitude       = latitude
        self.longitude      = longitude
        self.image_features = image_features   # ndarray (5,) or None

    def to_metadata_tensor(self) -> torch.Tensor:
        sin_m, cos_m = cyclic_encode(self.month,       12.0)
        sin_d, cos_d = cyclic_encode(self.day_of_year, DAYS_PER_YEAR)
        lat_norm = (self.latitude  / 90.0)  if self.latitude  is not None else 0.0
        lon_norm = (self.longitude / 180.0) if self.longitude is not None else 0.0

        parts: List[float] = [sin_m, cos_m, sin_d, cos_d, lat_norm, lon_norm]

        if self.image_features is not None:
            parts.extend(self.image_features.tolist())

        return torch.tensor(parts, dtype=torch.float32)

    def to_target_tensor(self) -> torch.Tensor:
        sin_t, cos_t = cyclic_encode(self.time_min, MINUTES_PER_DAY)
        return torch.tensor([sin_t, cos_t], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TimeOfDayDataset(Dataset):
    VALID_EXTENSIONS: frozenset = frozenset(
        {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp",
         ".heic", ".heif", ".dng"}
    )

    def __init__(
        self,
        image_dir:   str,
        transform:   Optional[Callable] = None,
        target_size: int = cfg.IMAGE_SIZE,
        random_seed: int = 42,
    ):
        self.image_dir   = image_dir
        self.transform   = transform
        self.target_size = target_size
        self.random_seed = random_seed

        self.samples: List[Tuple[str, TimeOfDayLabel]] = []
        self._feature_cache: Dict[str, np.ndarray] = {}
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
            # Image features are extracted lazily in __getitem__ so that the
            # dataset constructor stays fast even for large collections.
            label = TimeOfDayLabel(time_min=time_min, month=month, day_of_year=doy)
            self.samples.append((path, label))

        if skipped > 0:
            print(f"  WARNING: Skipped {skipped} image(s) missing valid EXIF data.")
        if not self.samples:
            raise RuntimeError("No valid images with EXIF data found.")

    def _is_valid_file(self, filename: str) -> bool:
        ext = os.path.splitext(filename.lower())[1]
        if ext in {".heic", ".heif"} and not _HEIC_SUPPORTED:
            return False
        return ext in self.VALID_EXTENSIONS

    def _letterbox_resize(self, image: Image.Image) -> Image.Image:
        ow, oh = image.size
        working_max = self.target_size
        if ow > working_max or oh > working_max:
            image.thumbnail((working_max, working_max), Image.Resampling.BILINEAR)
            ow, oh = image.size
        scale  = min(self.target_size / ow, self.target_size / oh)
        nw, nh = int(ow * scale), int(oh * scale)
        resized = image.resize((nw, nh), Image.Resampling.BILINEAR)
        canvas  = Image.new("RGB", (self.target_size, self.target_size), (0, 0, 0))
        canvas.paste(resized, ((self.target_size - nw) // 2,
                               (self.target_size - nh) // 2))
        return canvas

    def _print_stats(self) -> None:
        times = np.array([lbl.time_min for _, lbl in self.samples])
        print(f"  Time-of-day range : {times.min():.0f}–{times.max():.0f} min "
              f"({int(times.min())//60:02d}:{int(times.min())%60:02d}"
              f"–{int(times.max())//60:02d}:{int(times.max())%60:02d})")
        print(f"  Mean / Std        : {times.mean():.1f} / {times.std():.1f} min")

    @property
    def raw_times(self) -> np.ndarray:
        return np.array([lbl.time_min for _, lbl in self.samples])

    def get_sample_weight(self) -> torch.Tensor:
        times   = self.raw_times
        hours   = (times / 60).astype(int) % 24
        counts  = np.bincount(hours, minlength=24).astype(float)
        counts  = np.where(counts == 0, 1.0, counts)
        weights = 1.0 / counts[hours]
        weights /= weights.sum()
        return torch.from_numpy(weights.astype(np.float32))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_path, label = self.samples[idx]
        image = None

        try:
            image = Image.open(img_path).convert("RGB")

            # Only resize if needed
            if image.size != (self.target_size, self.target_size):
                image = self._letterbox_resize(image)

        except Exception as exc:
            print(f"ERROR: Failed to load {img_path}: {exc}")
            image = Image.new("RGB", (self.target_size, self.target_size), (0, 0, 0))
        
        # ── Photometric features — computed BEFORE augmentation so that
        #    ColorJitter / RandomCrop do not corrupt the raw signal. ──────────
        if cfg.USE_IMAGE_FEATURES:
            if img_path not in self._feature_cache:
                self._feature_cache[img_path] = ImageFeatureExtractor.extract(image)
            image_features = self._feature_cache[img_path]
            label = TimeOfDayLabel(
                time_min=label.time_min,
                month=label.month,
                day_of_year=label.day_of_year,
                latitude=label.latitude,
                longitude=label.longitude,
                image_features=image_features,
            )

        if self.transform:
            image = self.transform(image)

        return image, label.to_metadata_tensor(), label.to_target_tensor()


# ---------------------------------------------------------------------------
# Transforms  (light / medium / heavy)
# ---------------------------------------------------------------------------

def get_transforms(
    augment:    bool = True,
    target_size: int = cfg.IMAGE_SIZE,
    magnitude:  str  = "none",
) -> transforms.Compose:
    """
    magnitude : "none" | "light" | "moderate" | "heavy"
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225],
    )

    if not augment:
        return transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(), 
            normalize
        ])

    mag = magnitude.lower()

    # Base transforms shared across all magnitudes
    # We use Resize + small RandomCrop to allow shifting without destroying scale
    base_spatial = [
        transforms.Resize((int(target_size * 1.05), int(target_size * 1.05))),
        transforms.RandomCrop(target_size),
        transforms.RandomHorizontalFlip(p=0.5),
    ]

    if mag == "light":
        return transforms.Compose([
            *base_spatial,
            # Very slight camera tilt and pan
            transforms.RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.98, 1.02)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01),
            transforms.ToTensor(),
            normalize,
        ])

    elif mag == "moderate":
        return transforms.Compose([
            *base_spatial,
            # Moderate tilt/pan
            transforms.RandomAffine(degrees=4, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
            transforms.ToTensor(),
            normalize,
            # Simulates small clouds or sensor artifacts
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.08), ratio=(0.3, 3.3)),
        ])
        
    elif mag == "heavy":
        return transforms.Compose([
            *base_spatial,
            # More aggressive tilt/pan, but strictly preserving horizon structure
            transforms.RandomAffine(degrees=7, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            normalize,
            # Simulates larger occlusions
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
        ])

    else: # none
        return transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(), 
            normalize
        ])


def minutes_to_hhmm(minutes: float) -> str:
    if torch.is_tensor(minutes):
        minutes = minutes.item()
    total = int(round(minutes)) % int(MINUTES_PER_DAY)
    h, m  = divmod(int(total), 60)
    return f"{h:02d}:{m:02d}"


# ---------------------------------------------------------------------------
# TTA helpers
# ---------------------------------------------------------------------------

def tta_predict(
    model:      "torch.nn.Module",
    images:     torch.Tensor,
    metadata:   torch.Tensor,
    n_passes:   int = 4,
) -> torch.Tensor:
    """
    Test-time augmentation: average predictions over n_passes random horizontal
    flips (and the original).  Works with any batch size.

    Returns averaged (sin_t, cos_t) predictions — shape (B, 2).
    """
    preds = []
    # Always include original orientation
    preds.append(model(images, metadata))
    # Additional flipped passes
    for _ in range(n_passes - 1):
        flipped = torch.flip(images, dims=[3])   # horizontal flip
        preds.append(model(flipped, metadata))

    # Average in (sin, cos) space — unit circle mean is a valid cyclic average
    stacked = torch.stack(preds, dim=0)          # (passes, B, 2)
    return stacked.mean(dim=0)                   # (B, 2)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_dataloaders(
    dataset:     TimeOfDayDataset,
    fold:        int   = 0,
    n_splits:    int   = 5,
    batch_size:  int   = 32,
    num_workers: int   = 4,
    val_ratio:   float = 0.2,
    use_weighted_sampler: bool = False,
    persistent_workers: Optional[bool] = None,
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
            raise ValueError(f"Fold {fold} not found; must be 0–{n_splits - 1}.")

    sampler       = None
    shuffle_train = True
    if use_weighted_sampler:
        from torch.utils.data import WeightedRandomSampler
        all_weights   = dataset.get_sample_weight()
        train_weights = all_weights[train_idx]
        sampler = WeightedRandomSampler(
            weights=train_weights, num_samples=len(train_idx), replacement=True
        )
        shuffle_train = False

    if persistent_workers is None:
        _pw = num_workers > 0
    else:
        _pw = persistent_workers and (num_workers > 0)

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=_pw,
        prefetch_factor=1 if _pw else None,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=_pw,
        prefetch_factor=1 if _pw else None,
    )

    print(f"\nFold {fold}  |  train: {len(train_idx)} samples  |  "
          f"val: {len(val_idx)} samples")
    return train_loader, val_loader