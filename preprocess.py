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
TARGET_DIR = "dataset_224"
TARGET_SIZE = 224


def _exif_bytes_from_source(src_path: str) -> bytes | None:
    """
    Extract EXIF from any source format and return piexif-compatible bytes
    suitable for embedding in a JPEG, or None if extraction fails.
    """
    raw_exif = None

    # ── Step 1: Extract raw EXIF bytes via Pillow ──
    # pillow-heif patches Pillow, NOT exifread. We must use Pillow to parse
    # the HEIC container and extract the raw EXIF payload.
    try:
        with Image.open(src_path) as tmp_img:
            raw_exif = tmp_img.info.get("exif")
    except Exception:
        pass

    # ── Step 2: Try to carry over the full original EXIF ──
    # piexif.load() works with file paths (JPEG/TIFF) OR raw EXIF bytes (HEIC/PNG).
    try:
        if raw_exif:
            exif_dict = piexif.load(raw_exif)
        else:
            exif_dict = piexif.load(src_path)
            
        return piexif.dump(exif_dict)
    except Exception:
        pass  # Fall through to manual/minimal extraction if piexif chokes

    # ── Step 3: Fallback minimal EXIF dict using exifread ──
    # For DNGs or malformed EXIFs, we fall back to exifread.
    tags = {}
    try:
        if raw_exif:
            # exifread expects a TIFF header. Pillow's raw EXIF often starts with "Exif\x00\x00"
            exif_data = raw_exif[6:] if raw_exif.startswith(b"Exif\x00\x00") else raw_exif
            tags = exifread.process_file(io.BytesIO(exif_data), details=False)
        else:
            # Read directly from the file (works natively for DNG/TIFF/JPEG)
            with open(src_path, "rb") as f:
                tags = exifread.process_file(f, details=False)
    except Exception:
        pass

    dt_str = None
    for tag in ("EXIF DateTimeOriginal", "Image DateTime", "EXIF DateTimeDigitized"):
        if tag in tags:
            dt_str = str(tags[tag])
            break

    if dt_str is None:
        return None

    try:
        datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None

    dt_bytes = dt_str.encode("ascii") + b"\x00"

    def _tag_bytes(key: str) -> bytes | None:
        val = tags.get(key)
        return str(val).strip().encode("utf-8") + b"\x00" if val else None

    ifd0: dict = {piexif.ImageIFD.DateTime: dt_bytes}
    make_bytes = _tag_bytes("Image Make")
    model_bytes = _tag_bytes("Image Model")

    if make_bytes:
        ifd0[piexif.ImageIFD.Make] = make_bytes
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