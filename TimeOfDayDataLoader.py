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
_IMAGE_FEATURE_DIM = 5   # see ImageFeatureExtractor


def get_metadata_dim() -> int:
    """Return the full metadata vector length based on current config."""
    from config import Config as _cfg   # late import avoids circular dependency
    return _CALENDAR_DIM + (_IMAGE_FEATURE_DIM if _cfg.USE_IMAGE_FEATURES else 0)


# ---------------------------------------------------------------------------
# Handcrafted photometric feature extractor
# ---------------------------------------------------------------------------

class ImageFeatureExtractor:
    """
    Extracts a small, interpretable feature vector from a PIL image.

    Features (all in [0, 1] after normalisation):
      0. brightness_p90   – 90th-percentile of the HSV Value channel.
                            Tracks sun elevation better than the mean.
      1. rb_ratio_norm    – R / (R + B), a colour-temperature proxy.
                            Dawn/dusk → warm (→ 1); midday clear sky → cool (→ 0).
      2. sky_saturation   – Mean HSV Saturation in the *top third* of the image
                            (likely sky). High at golden hour, low at noon/overcast.
      3. horizon_gradient – Signed luminance difference (top half − bottom half).
                            Positive early/late when sky is brighter than ground.
      4. dark_pixel_ratio – Fraction of pixels with V < 0.15.
                            Useful for distinguishing night from day.

    Design notes
    ------------
    * Features are computed on the raw PIL image **before** any torchvision
      augmentation, so ColorJitter / RandomCrop do not corrupt the signal.
    * All outputs are clipped to [0, 1] so they are on the same scale as the
      cyclic calendar features the model already receives.
    * We use plain NumPy / PIL — no OpenCV dependency required.
    """

    # RGB → luminance weights (ITU-R BT.601)
    _LUM_R = 0.299
    _LUM_G = 0.587
    _LUM_B = 0.114

    @staticmethod
    def _to_hsv_array(image: "Image.Image") -> np.ndarray:
        """Return H, S, V arrays each in [0, 1], shape (H, W)."""
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin

        # Value
        v = cmax

        # Saturation
        s = np.where(cmax > 0, delta / (cmax + 1e-8), 0.0)

        # Hue (we don't use hue as a feature, but compute for completeness)
        # Omitted to save work — only S and V are needed below.

        return s, v   # (H, W), (H, W)

    @classmethod
    def extract(cls, image: "Image.Image") -> np.ndarray:
        """
        Parameters
        ----------
        image : PIL.Image.Image  (any mode — converted internally)

        Returns
        -------
        np.ndarray, shape (_IMAGE_FEATURE_DIM,), dtype float32, values in [0, 1]
        """
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

        s, v = cls._to_hsv_array(image)
        h_px = v.shape[0]

        # 0. Brightness: 90th percentile of V (robust to small dark regions)
        brightness_p90 = float(np.percentile(v, 90))

        # 1. Colour-temperature proxy: R/(R+B), normalised to [0,1]
        r_mean = r.mean()
        b_mean = b.mean()
        rb_ratio_norm = float(r_mean / (r_mean + b_mean + 1e-8))   # already [0,1]

        # 2. Sky-region saturation: top third of the frame
        sky_s = s[: h_px // 3, :]
        sky_saturation = float(sky_s.mean())

        # 3. Horizon gradient: top-half mean V minus bottom-half mean V
        #    Divided by 2 to map the signed [-1, 1] range into [0, 1]
        top_v    = v[: h_px // 2, :].mean()
        bottom_v = v[h_px // 2 :, :].mean()
        horizon_gradient = float((top_v - bottom_v + 1.0) / 2.0)   # [0,1]

        # 4. Dark-pixel ratio: fraction with V < 0.15 (night / deep shadow)
        dark_pixel_ratio = float((v < 0.15).mean())

        features = np.array(
            [brightness_p90, rb_ratio_norm, sky_saturation,
             horizon_gradient, dark_pixel_ratio],
            dtype=np.float32,
        )
        return np.clip(features, 0.0, 1.0)


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
            if img_path.lower().endswith('.dng'):
                try:
                    with rawpy.imread(img_path) as raw:
                        rgb = raw.postprocess(use_camera_wb=True, half_size=True)
                        image = Image.fromarray(rgb)
                except Exception:
                    pass

            if image is None:
                try:
                    img_np = iio.imread(img_path)
                    image = Image.fromarray(img_np).convert("RGB")
                except Exception:
                    pass

            if image is None:
                image = Image.open(img_path).convert("RGB")

            if hasattr(image, "format") and image.format == "JPEG":
                image.draft("RGB", (self.target_size * 2, self.target_size * 2))

            image = self._letterbox_resize(image)

        except Exception as exc:
            print(f"  ERROR: All decoders failed for {img_path}: {exc}")
            image = Image.new("RGB", (self.target_size, self.target_size), (0, 0, 0))

        # ── Photometric features — computed BEFORE augmentation so that
        #    ColorJitter / RandomCrop do not corrupt the raw signal. ──────────
        if cfg.USE_IMAGE_FEATURES:
            image_features = ImageFeatureExtractor.extract(image)
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
    magnitude : "none" | "light" | "medium" | "heavy"
        Controls how aggressive the training augmentation is.
        Ignored when augment=False (validation / inference path).
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225],
    )

    if not augment:
        return transforms.Compose([transforms.ToTensor(), normalize])

    mag = magnitude.lower()

    if mag == "light":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.15, contrast=0.1, saturation=0.15, hue=0.02),
            transforms.RandomResizedCrop(size=target_size, scale=(0.9, 1.0)),
            transforms.ToTensor(),
            normalize,
        ])

    elif mag == "heavy":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.4, contrast=0.35, saturation=0.4, hue=0.08),
            transforms.RandomResizedCrop(size=target_size, scale=(0.7, 1.0)),
            transforms.RandomGrayscale(p=0.05),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
            transforms.ToTensor(),
            normalize,
        ])

    elif mag == "medium":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.25, contrast=0.2, saturation=0.25, hue=0.04),
            transforms.RandomResizedCrop(size=target_size, scale=(0.8, 1.0)),
            transforms.RandomGrayscale(p=0.02),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            transforms.ToTensor(),
            normalize,
        ])
    else: # none
        return transforms.Compose([transforms.ToTensor(), normalize])


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

    _pw = num_workers > 0
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