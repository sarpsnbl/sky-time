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
from torchvision.transforms import v2

from config import Config as cfg

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    _HEIC_SUPPORTED = True
except ImportError:
    _HEIC_SUPPORTED = False

MINUTES_PER_DAY    = 1440.0
DAYS_PER_YEAR      = 365.25

_CALENDAR_DIM      = 6   
_IMAGE_FEATURE_DIM = 77   

def get_metadata_dim() -> int:
    from config import Config as _cfg   
    return _CALENDAR_DIM + (_IMAGE_FEATURE_DIM if _cfg.USE_IMAGE_FEATURES else 0)


# ---------------------------------------------------------------------------
# Handcrafted photometric feature extractor
# ---------------------------------------------------------------------------

class ImageFeatureExtractor:
    @classmethod
    def extract(cls, image: "Image.Image") -> np.ndarray:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0  
        H = rgb.shape[0]

        feat = []

        for c in range(3):
            ch = rgb[:, :, c]
            feat += [ch.mean(), ch.std()]

        hsv = color.rgb2hsv(rgb)  
        for c in range(3):
            ch = hsv[:, :, c]
            feat += [ch.mean(), ch.std()]

        lab = color.rgb2lab(rgb)          
        lab_norm = lab / np.array([100.0, 128.0, 128.0])  
        for c in range(3):
            ch = lab_norm[:, :, c]
            feat += [ch.mean(), ch.std()]

        for c in range(3):
            hist, _ = np.histogram(rgb[:, :, c], bins=8, range=(0.0, 1.0))
            feat += (hist / hist.sum()).tolist()

        for c in range(3):
            hist, _ = np.histogram(hsv[:, :, c], bins=8, range=(0.0, 1.0))
            feat += (hist / hist.sum()).tolist()

        top_v = hsv[: H // 3, :, 2]          
        feat += [top_v.mean(), top_v.std()]

        V   = hsv[:, :, 2]
        dVy = np.diff(V, axis=0)             
        feat += [np.abs(dVy).mean()]

        r_mean = rgb[:, :, 0].mean()
        b_mean = rgb[:, :, 2].mean()
        feat += [r_mean / (b_mean + 1e-6)]   

        lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        hist_lum, _ = np.histogram((lum * 255).astype(np.uint8), bins=256, range=(0, 256))
        hist_lum    = hist_lum / (hist_lum.sum() + 1e-8)
        entropy     = -np.sum(hist_lum * np.log2(hist_lum + 1e-8))   
        feat += [lum.mean(), lum.std(), entropy / 8.0]               

        S = hsv[:, :, 1]
        feat += [S.mean(), S.std()]

        edges = feature.canny(lum, sigma=1.0)
        feat += [edges.mean()]               

        lap = filters.laplace(lum)
        feat += [lap.var()]

        arr = np.array(feat, dtype=np.float32)
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

def _convert_gps_to_degrees(value):
    """Helper to convert EXIF GPS rationals to decimal degrees."""
    try:
        d = float(value.values[0].num) / float(value.values[0].den)
        m = float(value.values[1].num) / float(value.values[1].den)
        s = float(value.values[2].num) / float(value.values[2].den)
        return d + (m / 60.0) + (s / 3600.0)
    except Exception:
        return None

def extract_exif_data(image_path: str):
    """
    Extracts DateTime and GPS coordinates.
    Returns: (time_min, month, day, year, latitude, longitude) or None if no valid time found.
    """
    time_min, month, day, year = None, None, None, None
    lat, lon = None, None

    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)

        ts_str = None
        for tag in ['EXIF DateTimeOriginal', 'Image DateTime', 'EXIF DateTimeDigitized']:
            if tag in tags:
                ts_str = str(tags[tag])
                break

        if ts_str:
            try:
                dt = datetime.strptime(ts_str, '%Y:%m:%d %H:%M:%S')
                time_min = dt.hour * 60 + dt.minute
                month, day, year = dt.month, dt.day, dt.year
            except ValueError:
                pass

        if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
            lat_val = _convert_gps_to_degrees(tags['GPS GPSLatitude'])
            lon_val = _convert_gps_to_degrees(tags['GPS GPSLongitude'])
            
            lat_ref = str(tags.get('GPS GPSLatitudeRef', 'N'))
            lon_ref = str(tags.get('GPS GPSLongitudeRef', 'E'))

            if lat_val is not None and lon_val is not None:
                lat = lat_val if lat_ref == 'N' else -lat_val
                lon = lon_val if lon_ref == 'E' else -lon_val

    except Exception as e:
        print(f"Metadata error for {image_path}: {e}")

    if time_min is None:
        try:
            with Image.open(image_path) as img:
                exif = img.getexif()
                for tag_id in [36867, 306, 36868]:
                    if tag_id in exif:
                        dt = datetime.strptime(exif[tag_id], '%Y:%m:%d %H:%M:%S')
                        time_min = dt.hour * 60 + dt.minute
                        month, day, year = dt.month, dt.day, dt.year
                        break
        except Exception:
            pass

    if time_min is None:
        return None

    return time_min, month, day, year, lat, lon


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
        self.image_features = image_features   

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
            exif_data = extract_exif_data(path)
            
            if exif_data is None:
                skipped += 1
                continue
                
            time_min, month, day, year, lat, lon = exif_data
            doy = _day_of_year(month, day, year)
            
            label = TimeOfDayLabel(
                time_min=time_min, 
                month=month, 
                day_of_year=doy,
                latitude=lat,
                longitude=lon
            )
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

    def get_sample_weight(self, max_ratio: float = 10.0) -> torch.Tensor:
        """
        Calculates sample weights for the dataset, capping extreme outliers 
        to prevent sparse hour bins from dominating the sampler.
        """
        times   = self.raw_times
        hours   = (times / 60).astype(int) % 24
        counts  = np.bincount(hours, minlength=24).astype(float)
        
        counts  = np.where(counts == 0, 1.0, counts)
        
        weights = 1.0 / counts[hours]
        
        median_weight = np.median(weights)
        weights = np.clip(weights, a_min=None, a_max=median_weight * max_ratio)
        
        weights /= weights.sum()
        
        return torch.from_numpy(weights.astype(np.float32))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_path, label = self.samples[idx]
        image = None

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as exc:
            print(f"ERROR: Failed to load {img_path}: {exc}")
            image = Image.new("RGB", (self.target_size, self.target_size), (0, 0, 0))
        
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

        if image.size != (self.target_size, self.target_size):
            image = self._letterbox_resize(image)

        if self.transform:
            image = self.transform(image)

        return image, label.to_metadata_tensor(), label.to_target_tensor()

# ---------------------------------------------------------------------------
# Transforms  (light / moderate / heavy)
# ---------------------------------------------------------------------------

def get_transforms(
    augment:    bool = True,
    target_size: int = cfg.IMAGE_SIZE,
    magnitude:  str  = "none",
) -> v2.Compose:
    
    normalize = v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225],
    )

    base_resize = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((target_size, target_size), antialias=True),
    ]

    if not augment:
        return v2.Compose([*base_resize, normalize])

    mag = magnitude.lower()

    spatial = [
        v2.RandomResizedCrop(target_size, scale=(0.8, 1.0), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
    ]

    if mag == "light":
        return v2.Compose([
            *base_resize,
            *spatial,
            v2.RandAugment(num_ops=2, magnitude=5),
            normalize,
        ])

    elif mag == "moderate":
        return v2.Compose([
            *base_resize,
            *spatial,
            v2.RandAugment(num_ops=2, magnitude=9),
            normalize,
            v2.RandomErasing(p=0.15, scale=(0.02, 0.08)),
        ])
        
    elif mag == "heavy":
        return v2.Compose([
            *base_resize,
            *spatial,
            v2.RandAugment(num_ops=3, magnitude=12),
            normalize,
            v2.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ])

    return v2.Compose([*base_resize, normalize])


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
    preds = []
    preds.append(model(images, metadata))
    for _ in range(n_passes - 1):
        flipped = torch.flip(images, dims=[3])   
        preds.append(model(flipped, metadata))

    stacked = torch.stack(preds, dim=0)          
    return stacked.mean(dim=0)                   


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_dataloaders(
    train_dataset: TimeOfDayDataset,
    val_dataset:   TimeOfDayDataset,
    fold:          int   = 0,
    n_splits:      int   = 5,
    batch_size:    int   = 32,
    num_workers:   int   = 4,
    val_ratio:     float = 0.2,
    use_weighted_sampler: bool = False,
    persistent_workers: Optional[bool] = None,
) -> Tuple[DataLoader, DataLoader]:

    if len(train_dataset) != len(val_dataset):
        raise ValueError("train_dataset and val_dataset must have the same length.")

    indices = np.arange(len(train_dataset))

    splitter = ShuffleSplit(n_splits=n_splits, test_size=val_ratio, random_state=42)
    splits = list(splitter.split(indices))
    if fold >= len(splits):
        raise ValueError(f"Fold {fold} not found; must be 0–{n_splits - 1}.")
    train_idx, val_idx = splits[fold]

    sampler       = None
    shuffle_train = True
    if use_weighted_sampler:
        from torch.utils.data import WeightedRandomSampler
        all_weights   = train_dataset.get_sample_weight()
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
        Subset(train_dataset, train_idx),
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=_pw,
        prefetch_factor=1 if _pw else None,
    )
    
    val_loader = DataLoader(
        Subset(val_dataset, val_idx),
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