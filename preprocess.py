import os
from PIL import Image
from tqdm import tqdm
import imageio.v3 as iio
import piexif
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

def preprocess_dataset():
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)

    files = [f for f in os.listdir(SOURCE_DIR) if os.path.isfile(os.path.join(SOURCE_DIR, f))]
    print(f"Found {len(files)} files. Downscaling + Copying EXIF...")

    for fname in tqdm(files):
        src_path = os.path.join(SOURCE_DIR, fname)
        base_name = os.path.splitext(fname)[0]
        dst_path = os.path.join(TARGET_DIR, f"{base_name}.jpg")

        if os.path.exists(dst_path):
            continue

        try:
            # 1. Load Image
            if src_path.lower().endswith('.dng'):
                import rawpy
                with rawpy.imread(src_path) as raw:
                    rgb = raw.postprocess(use_camera_wb=True, half_size=True)
                    img = Image.fromarray(rgb)
            else:
                img_np = iio.imread(src_path)
                img = Image.fromarray(img_np).convert("RGB")

            # 2. Resize
            img.thumbnail((TARGET_SIZE, TARGET_SIZE), Image.Resampling.LANCZOS)
            
            # 3. Handle EXIF Transfer
            exif_bytes = None
            try:
                # Extract EXIF from original
                exif_dict = piexif.load(src_path)
                # Convert to bytes for saving
                exif_bytes = piexif.dump(exif_dict)
            except Exception:
                # If file has no EXIF or format is unsupported (like some RAWs)
                pass

            # 4. Save with original EXIF
            if exif_bytes:
                img.save(dst_path, "JPEG", quality=90, optimize=True, exif=exif_bytes)
            else:
                img.save(dst_path, "JPEG", quality=90, optimize=True)

        except Exception as e:
            print(f"\nSkipping {fname}: {e}")

if __name__ == "__main__":
    preprocess_dataset()