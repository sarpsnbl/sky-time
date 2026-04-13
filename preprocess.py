import os
from PIL import Image
from tqdm import tqdm
import imageio.v3 as iio
import piexif
import exifread
from datetime import datetime
Image.MAX_IMAGE_PIXELS = None

# HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    print("pillow-heif not found. HEIC files may fail.")

SOURCE_DIR = "dataset"
TARGET_DIR = "dataset_512"
TARGET_SIZE = 512


def _exif_bytes_from_source(src_path: str) -> bytes | None:
    """
    Extract EXIF from any source format and return piexif-compatible bytes
    suitable for embedding in a JPEG, or None if extraction fails.

    piexif.load() only understands JPEG/TIFF — it silently produces garbage
    or raises for HEIC/PNG/etc.  We therefore always read with exifread first
    (which handles HEIC correctly via pillow-heif), pull the datetime fields
    we need, and reconstruct a minimal EXIF dict that piexif can serialise.
    """
    # ── Step 1: read raw tags with exifread (works for HEIC, JPEG, TIFF…) ──
    try:
        with open(src_path, "rb") as f:
            tags = exifread.process_file(f, details=False)
    except Exception:
        tags = {}

    dt_str = None
    for tag in ("EXIF DateTimeOriginal", "Image DateTime", "EXIF DateTimeDigitized"):
        if tag in tags:
            dt_str = str(tags[tag])
            break

    if dt_str is None:
        # No datetime tag found at all — nothing useful to embed.
        return None

    # Validate the datetime string is parseable before we embed it.
    try:
        datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None

    dt_bytes = dt_str.encode("ascii") + b"\x00"   # null-terminated ASCII

    # Pull camera make/model from the exifread tags we already have.
    # These are preserved by piexif.load() for JPEGs (Step 2), but would
    # otherwise be silently dropped in the fallback path (Step 3).
    def _tag_bytes(key: str) -> bytes | None:
        val = tags.get(key)
        if val is None:
            return None
        return str(val).strip().encode("utf-8") + b"\x00"

    make_bytes  = _tag_bytes("Image Make")
    model_bytes = _tag_bytes("Image Model")

    # ── Step 2: try to carry over the full original EXIF for JPEG/TIFF ──
    # piexif.load() works fine for JPEG/TIFF sources; use it when we can
    # so we preserve GPS, camera model, etc.  For HEIC and other formats
    # it will raise — we catch that and fall back to the minimal dict below.
    ext = os.path.splitext(src_path.lower())[1]
    if ext in (".jpg", ".jpeg", ".tiff", ".tif"):
        try:
            exif_dict = piexif.load(src_path)
            # Ensure the datetime field is consistent with what exifread found.
            exif_dict.setdefault("Exif", {})[piexif.ExifIFD.DateTimeOriginal] = dt_bytes
            exif_dict.setdefault("0th",  {})[piexif.ImageIFD.DateTime]        = dt_bytes
            return piexif.dump(exif_dict)
        except Exception:
            pass   # fall through to minimal dict

    # ── Step 3: build a minimal EXIF dict that piexif can always serialise ──
    # This is the path taken for HEIC, PNG, DNG, and any JPEG whose EXIF
    # piexif couldn't parse cleanly.  We carry over make/model from the
    # exifread tags grabbed in Step 1 so camera info isn't lost.
    ifd0: dict = {piexif.ImageIFD.DateTime: dt_bytes}
    if make_bytes:
        ifd0[piexif.ImageIFD.Make]  = make_bytes
    if model_bytes:
        ifd0[piexif.ImageIFD.Model] = model_bytes

    exif_dict = {
        "0th":  ifd0,
        "Exif": {piexif.ExifIFD.DateTimeOriginal:  dt_bytes,
                 piexif.ExifIFD.DateTimeDigitized: dt_bytes},
        "GPS":  {},
        "1st":  {},
        "Interop": {},
    }
    try:
        return piexif.dump(exif_dict)
    except Exception:
        return None


def preprocess_dataset():
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)

    files = [f for f in os.listdir(SOURCE_DIR) if os.path.isfile(os.path.join(SOURCE_DIR, f))]
    print(f"Found {len(files)} files. Downscaling + copying EXIF …")

    n_ok = n_no_exif = n_skipped = 0

    for fname in tqdm(files):
        src_path  = os.path.join(SOURCE_DIR, fname)
        base_name = os.path.splitext(fname)[0]
        dst_path  = os.path.join(TARGET_DIR, f"{base_name}.jpg")

        if os.path.exists(dst_path):
            continue

        try:
            # 1. Load image
            if src_path.lower().endswith(".dng"):
                import rawpy
                with rawpy.imread(src_path) as raw:
                    rgb = raw.postprocess(use_camera_wb=True, half_size=True)
                    img = Image.fromarray(rgb)
            else:
                img_np = iio.imread(src_path)
                img    = Image.fromarray(img_np).convert("RGB")

            # 2. Resize
            img.thumbnail((TARGET_SIZE, TARGET_SIZE), Image.Resampling.LANCZOS)

            # 3. Extract EXIF — using the format-aware helper
            exif_bytes = _exif_bytes_from_source(src_path)

            if exif_bytes is None:
                n_no_exif += 1

            # 4. Save
            if exif_bytes:
                img.save(dst_path, "JPEG", quality=90, optimize=True, exif=exif_bytes)
            else:
                img.save(dst_path, "JPEG", quality=90, optimize=True)

            n_ok += 1

        except Exception as e:
            print(f"\nSkipping {fname}: {e}")
            n_skipped += 1

    print(f"\nDone.  Converted: {n_ok}  |  No EXIF found: {n_no_exif}  |  Skipped (error): {n_skipped}")


if __name__ == "__main__":
    preprocess_dataset()