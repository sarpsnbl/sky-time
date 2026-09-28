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
try:
    import exifread
    _EXIFREAD_AVAILABLE = True
except ImportError:
    _EXIFREAD_AVAILABLE = False

try:
    import rawpy
    _RAWPY_AVAILABLE = True
except ImportError:
    _RAWPY_AVAILABLE = False
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

# 28 core astronomical, sensor, orientation, and calendar dimensions
_PHYSICAL_METADATA_DIM = 28
_CALENDAR_DIM          = _PHYSICAL_METADATA_DIM   # Backward compatibility alias
_IMAGE_FEATURE_DIM     = 80   # Handcrafted photometric descriptors (77 base + 3 atmospheric gradients)

def get_metadata_dim() -> int:
    from config import Config as _cfg   
    return _PHYSICAL_METADATA_DIM + (_IMAGE_FEATURE_DIM if _cfg.USE_IMAGE_FEATURES else 0)


# ---------------------------------------------------------------------------
# Handcrafted photometric feature extractor (79 Dimensions)
# ---------------------------------------------------------------------------

class ImageFeatureExtractor:
    @classmethod
    def extract(cls, image: "Image.Image") -> np.ndarray:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0  
        H, W = rgb.shape[:2]

        feat = []

        # 1. RGB statistics (mean, std x 3 = 6)
        for c in range(3):
            ch = rgb[:, :, c]
            feat += [ch.mean(), ch.std()]

        # 2. HSV statistics (mean, std x 3 = 6)
        hsv = color.rgb2hsv(rgb)  
        for c in range(3):
            ch = hsv[:, :, c]
            feat += [ch.mean(), ch.std()]

        # 3. LAB statistics (mean, std x 3 = 6)
        lab = color.rgb2lab(rgb)          
        lab_norm = lab / np.array([100.0, 128.0, 128.0])  
        for c in range(3):
            ch = lab_norm[:, :, c]
            feat += [ch.mean(), ch.std()]

        # 4. RGB Histograms (8 bins x 3 = 24)
        for c in range(3):
            hist, _ = np.histogram(rgb[:, :, c], bins=8, range=(0.0, 1.0))
            feat += (hist / (hist.sum() + 1e-8)).tolist()

        # 5. HSV Histograms (8 bins x 3 = 24)
        for c in range(3):
            hist, _ = np.histogram(hsv[:, :, c], bins=8, range=(0.0, 1.0))
            feat += (hist / (hist.sum() + 1e-8)).tolist()

        # 6. Sun-region brightness (top-third mean, std = 2)
        top_v = hsv[: max(1, H // 3), :, 2]          
        feat += [top_v.mean(), top_v.std()]

        # 7. Horizon luminance gradient (abs vertical gradient = 1)
        V   = hsv[:, :, 2]
        dVy = np.diff(V, axis=0)             
        feat += [np.abs(dVy).mean()]

        # 8. Colour temperature proxy (R/B ratio, clipped & normalized = 1)
        r_mean = rgb[:, :, 0].mean()
        b_mean = rgb[:, :, 2].mean()
        rb_ratio = float(r_mean / (b_mean + 1e-6))
        feat += [float(np.clip(rb_ratio, 0.0, 4.0) / 4.0)]

        # 9. Global luminance statistics (mean, std, entropy = 3)
        lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        hist_lum, _ = np.histogram((lum * 255).astype(np.uint8), bins=256, range=(0, 256))
        hist_lum    = hist_lum / (hist_lum.sum() + 1e-8)
        entropy     = -np.sum(hist_lum * np.log2(hist_lum + 1e-8))   
        feat += [lum.mean(), lum.std(), entropy / 8.0]               

        # 10. Saturation statistics (mean, std = 2)
        S = hsv[:, :, 1]
        feat += [S.mean(), S.std()]

        # 11. Edge density (Canny on luminance = 1)
        edges = feature.canny(lum, sigma=1.0)
        feat += [edges.mean()]               

        # 12. Laplacian variance (sharpness / cloud texture = 1)
        lap = filters.laplace(lum)
        feat += [lap.var()]

        # --- Enhanced Atmospheric Gradients (3 new features) ---
        # 13. Signed vertical luminance gradient (zenith vs horizon direction = 1)
        bot_v = hsv[max(0, 2 * H // 3) :, :, 2]
        feat += [float(top_v.mean() - bot_v.mean())]

        # 14. Horizontal solar asymmetry gradient (East vs West diffuse lighting = 1)
        left_v  = hsv[:, : max(1, W // 3), 2]
        right_v = hsv[:, max(0, 2 * W // 3) :, 2]
        feat += [float(left_v.mean() - right_v.mean())]

        # 15. Zenith-to-horizon chrominance shift (Rayleigh scattering b* proxy = 1)
        top_b = lab_norm[: max(1, H // 5), :, 2].mean()
        bot_b = lab_norm[max(0, 4 * H // 5) :, :, 2].mean()
        feat += [float(bot_b - top_b)]

        return np.array(feat, dtype=np.float32)

# ---------------------------------------------------------------------------
# Cyclic encoding helpers & Astrometric Engine
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

def compute_astronomy(
    month: int,
    doy: int,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    tz_offset: Optional[float] = None,
) -> Tuple[float, float, float, float, float]:
    """
    Computes zero-leakage astronomical parameters derived solely from calendar date
    and geographic location:
      - Solar Declination (degrees)
      - Equation of Time (minutes)
      - Theoretical Day Length (hours)
      - Theoretical Solar Noon (minutes from midnight in local clock time)
      - Maximum Solar Elevation at Noon (degrees)
    """
    ref_lat = lat if lat is not None else 38.5
    ref_lon = lon if lon is not None else 27.0
    ref_tz  = tz_offset if tz_offset is not None else 3.0

    # 1. Solar declination (-23.44° to +23.44°)
    delta_deg = -23.44 * math.cos(2.0 * math.pi * (doy + 10.0) / DAYS_PER_YEAR)
    delta_rad = math.radians(delta_deg)
    phi_rad   = math.radians(ref_lat)

    # 2. Equation of Time (minutes)
    B = 2.0 * math.pi * (doy - 81.0) / 364.0
    eot_min = 9.87 * math.sin(2.0 * B) - 7.53 * math.cos(B) - 1.5 * math.sin(B)

    # 3. Day length (hours)
    tan_prod = math.tan(phi_rad) * math.tan(delta_rad)
    tan_prod = max(-1.0, min(1.0, tan_prod))
    omega_0_rad = math.acos(-tan_prod)
    day_len_hours = (2.0 * math.degrees(omega_0_rad)) / 15.0

    # 4. Solar noon in minutes from midnight local clock time
    std_meridian = 15.0 * ref_tz
    lon_corr_min = 4.0 * (ref_lon - std_meridian)
    solar_noon_min = ((12.0 * 60.0) - lon_corr_min - eot_min) % MINUTES_PER_DAY

    # 5. Maximum solar elevation at noon (degrees)
    alpha_max_deg = max(0.0, 90.0 - abs(ref_lat - delta_deg))

    return delta_deg, eot_min, day_len_hours, solar_noon_min, alpha_max_deg

# ---------------------------------------------------------------------------
# EXIF Parsing & Metadata Container
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

class ExifMetadata:
    """
    Rich container for all optical, spatial, sensor, and temporal metadata.
    Unpacks as a 6-tuple (time_min, month, day, year, lat, lon) for seamless backward compatibility.
    """
    def __init__(
        self,
        time_min: float,
        month: int,
        day: int,
        year: int,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        day_of_week: int = 0,
        tz_offset: Optional[float] = None,
        heading: Optional[float] = None,
        altitude: Optional[float] = None,
        ev100: Optional[float] = None,
        brightness_val: Optional[float] = None,
        shutter_speed: Optional[float] = None,
        iso: Optional[float] = None,
        f_number: Optional[float] = None,
        exposure_bias: Optional[float] = None,
        fov: Optional[float] = None,
    ):
        self.time_min       = time_min
        self.month          = month
        self.day            = day
        self.year           = year
        self.lat            = lat
        self.lon            = lon
        self.day_of_week    = day_of_week
        self.tz_offset      = tz_offset
        self.heading        = heading
        self.altitude       = altitude
        self.ev100          = ev100
        self.brightness_val = brightness_val
        self.shutter_speed  = shutter_speed
        self.iso            = iso
        self.f_number       = f_number
        self.exposure_bias  = exposure_bias
        self.fov            = fov

    def __iter__(self):
        # Backward compatibility unpacking: time_min, month, day, year, lat, lon = exif_data
        return iter((self.time_min, self.month, self.day, self.year, self.lat, self.lon))

def extract_exif_data(image_path: str) -> Optional[ExifMetadata]:
    """
    Extracts DateTime, GPS, camera orientation, and photometric exposure physics from EXIF.
    Returns: ExifMetadata object (or None if no valid timestamp is found).
    """
    time_min, month, day, year, dow = None, None, None, None, 0
    lat, lon, heading, altitude = None, None, None, None
    tz_offset = None
    ev100, bv, shutter, iso, f_num, bias_val, fov_val = None, None, None, None, None, 0.0, None

    # Step 1: Attempt extraction via Pillow's _getexif() (fast and comprehensive)
    try:
        with Image.open(image_path) as img:
            exif = img._getexif() or {}

            # DateTime parsing (tags 36867 = DateTimeOriginal, 306 = DateTime, 36868 = DateTimeDigitized)
            for tag_id in [36867, 306, 36868]:
                if tag_id in exif:
                    try:
                        dt = datetime.strptime(str(exif[tag_id]), "%Y:%m:%d %H:%M:%S")
                        time_min = dt.hour * 60.0 + dt.minute + dt.second / 60.0
                        month, day, year = dt.month, dt.day, dt.year
                        dow = dt.weekday()
                        break
                    except ValueError:
                        pass

            # BrightnessValue (tag 37379)
            if 37379 in exif:
                try:
                    bv = float(exif[37379])
                except (ValueError, TypeError):
                    bv = None

            # Exposure settings (33434 = ExposureTime, 33437 = FNumber, 34855 = ISOSpeedRatings)
            t_val  = exif.get(33434)
            fn_val = exif.get(33437)
            s_val  = exif.get(34855)
            if t_val and fn_val and s_val:
                try:
                    shutter = float(t_val)
                    f_num   = float(fn_val)
                    iso     = float(s_val)
                    if shutter > 0 and f_num > 0 and iso > 0:
                        ev100 = math.log2((f_num ** 2) / shutter) - math.log2(iso / 100.0)
                except (ValueError, TypeError):
                    pass

            # Exposure Bias (tag 37380)
            if 37380 in exif:
                try:
                    bias_val = float(exif[37380])
                except (ValueError, TypeError):
                    pass

            # Field of View from FocalLengthIn35mmFilm (tag 41989)
            if 41989 in exif:
                try:
                    f35 = float(exif[41989])
                    if f35 > 0:
                        fov_val = 2.0 * math.atan(36.0 / (2.0 * f35))
                except (ValueError, TypeError):
                    pass

            # Timezone offset (tags 36881 = OffsetTimeOriginal, 36880 = OffsetTime, 36882 = OffsetTimeDigitized)
            for tid in [36881, 36880, 36882]:
                if tid in exif:
                    try:
                        raw_tz = str(exif[tid]).strip()
                        parts = raw_tz.split(":")
                        h = float(parts[0])
                        m = float(parts[1]) if len(parts) > 1 else 0.0
                        tz_offset = h + (m / 60.0 if h >= 0 else -m / 60.0)
                        break
                    except (ValueError, IndexError):
                        pass

            # GPS Info (tag 34853)
            gps = exif.get(34853)
            if isinstance(gps, dict):
                lat_ref = gps.get(1, "N")
                lat_raw = gps.get(2)
                lon_ref = gps.get(3, "E")
                lon_raw = gps.get(4)
                if lat_raw and lon_raw:
                    try:
                        lat = float(lat_raw[0]) + float(lat_raw[1]) / 60.0 + float(lat_raw[2]) / 3600.0
                        if lat_ref == "S": lat = -lat
                        lon = float(lon_raw[0]) + float(lon_raw[1]) / 60.0 + float(lon_raw[2]) / 3600.0
                        if lon_ref == "W": lon = -lon
                    except (ValueError, TypeError, IndexError):
                        pass
                h_raw = gps.get(17)
                if h_raw is not None:
                    try: heading = float(h_raw)
                    except (ValueError, TypeError): pass
                a_raw = gps.get(6)
                if a_raw is not None:
                    try: altitude = float(a_raw)
                    except (ValueError, TypeError): pass
    except Exception:
        pass

    # Step 2: Fallback to exifread if DateTime wasn't found or for DNG/RAW files
    if time_min is None and _EXIFREAD_AVAILABLE:
        try:
            with open(image_path, "rb") as f:
                tags = exifread.process_file(f, details=False)

            ts_str = None
            for tag in ["EXIF DateTimeOriginal", "Image DateTime", "EXIF DateTimeDigitized"]:
                if tag in tags:
                    ts_str = str(tags[tag])
                    break

            if ts_str:
                try:
                    dt = datetime.strptime(ts_str, "%Y:%m:%d %H:%M:%S")
                    time_min = dt.hour * 60.0 + dt.minute + dt.second / 60.0
                    month, day, year = dt.month, dt.day, dt.year
                    dow = dt.weekday()
                except ValueError:
                    pass

            if lat is None and "GPS GPSLatitude" in tags and "GPS GPSLongitude" in tags:
                lat_val = _convert_gps_to_degrees(tags["GPS GPSLatitude"])
                lon_val = _convert_gps_to_degrees(tags["GPS GPSLongitude"])
                lat_ref = str(tags.get("GPS GPSLatitudeRef", "N"))
                lon_ref = str(tags.get("GPS GPSLongitudeRef", "E"))
                if lat_val is not None and lon_val is not None:
                    lat = lat_val if lat_ref == "N" else -lat_val
                    lon = lon_val if lon_ref == "E" else -lon_val

            if tz_offset is None:
                for tz_tag in ["EXIF OffsetTimeOriginal", "EXIF OffsetTime"]:
                    if tz_tag in tags:
                        try:
                            parts = str(tags[tz_tag]).strip().split(":")
                            h = float(parts[0])
                            m = float(parts[1]) if len(parts) > 1 else 0.0
                            tz_offset = h + (m / 60.0 if h >= 0 else -m / 60.0)
                            break
                        except Exception:
                            pass
        except Exception as e:
            pass

    if time_min is None:
        return None

    return ExifMetadata(
        time_min=time_min,
        month=month,
        day=day,
        year=year,
        lat=lat,
        lon=lon,
        day_of_week=dow,
        tz_offset=tz_offset,
        heading=heading,
        altitude=altitude,
        ev100=ev100,
        brightness_val=bv,
        shutter_speed=shutter,
        iso=iso,
        f_number=f_num,
        exposure_bias=bias_val,
        fov=fov_val,
    )

# ---------------------------------------------------------------------------
# Label entry
# ---------------------------------------------------------------------------

class TimeOfDayLabel:
    __slots__ = (
        "time_min", "month", "day_of_year", "latitude", "longitude",
        "day_of_week", "tz_offset", "heading", "altitude",
        "ev100", "brightness_val", "shutter_speed", "iso", "f_number",
        "exposure_bias", "fov", "image_features"
    )

    def __init__(
        self,
        time_min: float,
        month: int,
        day_of_year: int,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        day_of_week: int = 0,
        tz_offset: Optional[float] = None,
        heading: Optional[float] = None,
        altitude: Optional[float] = None,
        ev100: Optional[float] = None,
        brightness_val: Optional[float] = None,
        shutter_speed: Optional[float] = None,
        iso: Optional[float] = None,
        f_number: Optional[float] = None,
        exposure_bias: Optional[float] = None,
        fov: Optional[float] = None,
        image_features: Optional[np.ndarray] = None,
    ):
        self.time_min       = float(time_min)
        self.month          = int(month)
        self.day_of_year    = int(day_of_year)
        self.latitude       = latitude
        self.longitude      = longitude
        self.day_of_week    = int(day_of_week)
        self.tz_offset      = tz_offset
        self.heading        = heading
        self.altitude       = altitude
        self.ev100          = ev100
        self.brightness_val = brightness_val
        self.shutter_speed  = shutter_speed
        self.iso            = iso
        self.f_number       = f_number
        self.exposure_bias  = exposure_bias
        self.fov            = fov
        self.image_features = image_features   

    def to_metadata_tensor(self) -> torch.Tensor:
        # Tier D: Calendar & Temporal Cycles (6 dims)
        sin_m, cos_m = cyclic_encode(self.month, 12.0)
        sin_d, cos_d = cyclic_encode(self.day_of_year, DAYS_PER_YEAR)
        sin_dow, cos_dow = cyclic_encode(float(self.day_of_week), 7.0)

        # Tier C: Astrometric Ephemeris (6 dims)
        delta_deg, eot_min, day_len_h, noon_min, alpha_max_deg = compute_astronomy(
            self.month, self.day_of_year, self.latitude, self.longitude, self.tz_offset
        )
        sin_noon, cos_noon = cyclic_encode(noon_min, MINUTES_PER_DAY)
        delta_norm    = float(delta_deg / 23.44)
        eot_norm      = float(np.clip(eot_min / 20.0, -1.0, 1.0))
        daylen_norm   = float(np.clip((day_len_h - 12.0) / 12.0, -1.0, 1.0))
        alphamax_norm = float(alpha_max_deg / 90.0)

        # Tier B: Geodetic, Spatial & Orientation (8 dims)
        has_gps     = 1.0 if (self.latitude is not None and self.longitude is not None) else 0.0
        lat_norm    = (self.latitude / 90.0) if self.latitude is not None else (38.5 / 90.0)
        lon_norm    = (self.longitude / 180.0) if self.longitude is not None else (27.0 / 180.0)
        tz_norm     = (self.tz_offset / 12.0) if self.tz_offset is not None else (3.0 / 12.0)
        has_heading = 1.0 if self.heading is not None else 0.0
        sin_head, cos_head = cyclic_encode(self.heading if self.heading is not None else 0.0, 360.0)
        if not has_heading:
            sin_head, cos_head = 0.0, 0.0
        alt_norm    = float(np.clip(math.log2(1.0 + max(0.0, self.altitude if self.altitude is not None else 0.0)) / 14.0, 0.0, 1.0))

        # Tier A: Photometric Hardware & Sensor Physics (8 dims)
        has_exp       = 1.0 if self.ev100 is not None else 0.0
        ev_norm       = float(np.clip((self.ev100 + 2.0) / 20.0, 0.0, 1.0)) if self.ev100 is not None else 0.5
        bv_norm       = float(np.clip((self.brightness_val + 5.0) / 20.0, 0.0, 1.0)) if self.brightness_val is not None else ev_norm
        shutter_norm  = float(np.clip((math.log2(self.shutter_speed) + 15.0) / 17.0, 0.0, 1.0)) if self.shutter_speed and self.shutter_speed > 0 else 0.5
        iso_norm      = float(np.clip(math.log2(self.iso / 100.0) / 6.0, 0.0, 1.0)) if self.iso and self.iso > 0 else 0.0
        aperture_norm = float(np.clip(self.f_number / 22.0, 0.0, 1.0)) if self.f_number and self.f_number > 0 else (1.8 / 22.0)
        bias_norm     = float(np.clip(self.exposure_bias / 3.0, -1.0, 1.0)) if self.exposure_bias is not None else 0.0
        fov_norm      = float(np.clip(self.fov / math.pi, 0.0, 1.0)) if self.fov is not None else (65.0 / 180.0)

        # 28 Core Metadata Dimensions
        core_parts: List[float] = [
            # Tier D (6)
            sin_m, cos_m, sin_d, cos_d, sin_dow, cos_dow,
            # Tier C (6)
            delta_norm, eot_norm, daylen_norm, sin_noon, cos_noon, alphamax_norm,
            # Tier B (8)
            has_gps, lat_norm, lon_norm, tz_norm, has_heading, sin_head, cos_head, alt_norm,
            # Tier A (8)
            has_exp, ev_norm, bv_norm, shutter_norm, iso_norm, aperture_norm, bias_norm, fov_norm,
        ]

        if self.image_features is not None:
            core_parts.extend(self.image_features.tolist())

        return torch.tensor(core_parts, dtype=torch.float32)

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
                
            doy = _day_of_year(exif_data.month, exif_data.day, exif_data.year)
            
            label = TimeOfDayLabel(
                time_min=exif_data.time_min, 
                month=exif_data.month, 
                day_of_year=doy,
                latitude=exif_data.lat,
                longitude=exif_data.lon,
                day_of_week=exif_data.day_of_week,
                tz_offset=exif_data.tz_offset,
                heading=exif_data.heading,
                altitude=exif_data.altitude,
                ev100=exif_data.ev100,
                brightness_val=exif_data.brightness_val,
                shutter_speed=exif_data.shutter_speed,
                iso=exif_data.iso,
                f_number=exif_data.f_number,
                exposure_bias=exif_data.exposure_bias,
                fov=exif_data.fov,
            )
            self.samples.append((path, label))

        if skipped > 0:
            print(f"  WARNING: Skipped {skipped} image(s) missing valid EXIF data.")
        if not self.samples:
            raise RuntimeError("No valid images with EXIF data found.")

        if cfg.USE_IMAGE_FEATURES:
            cache_file = os.path.join(self.image_dir, "heuristics_feature_cache.pt")
            if os.path.exists(cache_file):
                print(f"Loading cached image features from {cache_file}...")
                try:
                    self._feature_cache = torch.load(cache_file, weights_only=False)
                    print(f"Loaded {len(self._feature_cache)} cached feature vectors.")
                except Exception as e:
                    print(f"Failed to load cache ({e}), recomputing...")
                    self._feature_cache = {}

            missing_samples = [path for path, _ in self.samples if path not in self._feature_cache]
            if missing_samples:
                print(f"Pre-computing {len(missing_samples)} handcrafted image features (multi-threaded)...")
                from concurrent.futures import ThreadPoolExecutor
                from tqdm import tqdm

                def _process_one(p):
                    try:
                        with Image.open(p).convert("RGB") as img:
                            return p, ImageFeatureExtractor.extract(img)
                    except Exception as e:
                        return p, None

                max_workers = min(16, os.cpu_count() or 4)
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    for p, feat in tqdm(executor.map(_process_one, missing_samples), total=len(missing_samples), desc="Extracting features"):
                        if feat is not None:
                            self._feature_cache[p] = feat

                try:
                    torch.save(self._feature_cache, cache_file)
                    print(f"Saved feature cache to {cache_file}")
                except Exception as e:
                    print(f"Warning: could not save feature cache ({e})")

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
            label.image_features = self._feature_cache[img_path]

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
    mean_pred = stacked.mean(dim=0)
    return torch.nn.functional.normalize(mean_pred, p=2, dim=-1)


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