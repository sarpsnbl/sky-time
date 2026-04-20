import os
import zipfile
from datetime import datetime
import io
from PIL import Image
import pillow_heif

# Enable HEIC support for Pillow
pillow_heif.register_heif_opener()

# ── Configuration ────────────────────────────────────────────────────────
DOWNLOADS_DIR = r"C:\Users\ssarp\Downloads"
OUT_DIR = r"C:\Users\ssarp\.vscode\projects\sky-time\dataset"

TARGET_SIZE = (224, 224) # From your MATLAB cfg.inputSize
TODAY_DATE = datetime(2026, 4, 14).date()
CUTOFF_DATE = datetime(2026, 4, 7).date()

# Ensure the dataset folder exists
os.makedirs(OUT_DIR, exist_ok=True)

def get_exif_date(img):
    """Extract DateTimeOriginal from PIL Image EXIF data."""
    exif_data = img.getexif()
    if not exif_data:
        return None
    
    # 36867 is DateTimeOriginal, 306 is DateTime
    date_str = exif_data.get(36867) or exif_data.get(306)
    if date_str:
        try:
            # Format is strictly 'YYYY:MM:DD HH:MM:SS'
            clean_str = str(date_str).strip()
            dt = datetime.strptime(clean_str, '%Y:%m:%d %H:%M:%S')
            return dt.date()
        except ValueError:
            return None
    return None

def process_zips():
    for filename in os.listdir(DOWNLOADS_DIR):
        if not filename.lower().endswith('.zip'):
            continue
            
        zip_path = os.path.join(DOWNLOADS_DIR, filename)
        
        # 1. Verify the zip was created/downloaded today
        mtime = datetime.fromtimestamp(os.path.getmtime(zip_path)).date()
        if mtime != TODAY_DATE:
            continue
            
        print(f"Scanning Archive: {filename} ...")
        
        with zipfile.ZipFile(zip_path, 'r') as z:
            for file_info in z.infolist():
                if file_info.is_dir():
                    continue
                    
                ext = os.path.splitext(file_info.filename)[1].lower()
                
                # 2. Skip DNGs and non-image files
                valid_exts = {'.jpg', '.jpeg', '.png', '.heic'}
                if ext not in valid_exts:
                    continue
                    
                try:
                    # Read file directly from zip into memory
                    with z.open(file_info) as f:
                        img_bytes = f.read()
                        
                    with Image.open(io.BytesIO(img_bytes)) as img:
                        # 3. Check EXIF date
                        taken_date = get_exif_date(img)
                        
                        # Skip if missing date or taken on/after April 7
                        if not taken_date or taken_date >= CUTOFF_DATE:
                            continue 
                            
                        # Extract raw EXIF binary data before resizing
                        exif_bytes = img.info.get('exif', b'')
                            
                        # 4. Scale down strictly to 224x224
                        resized_img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                        
                        # Flatten filename to prevent creating zip subdirectories in dataset folder
                        out_name = os.path.basename(file_info.filename)
                        out_path = os.path.join(OUT_DIR, out_name)
                        
                        # Convert RGBA/Palette to RGB so JPEG doesn't throw errors
                        if resized_img.mode in ('RGBA', 'P'):
                            resized_img = resized_img.convert('RGB')
                            
                        # 5. Save with exact original metadata block
                        if ext == '.heic':
                            # Save HEIC files as JPEGs so the standard EXIF block embeds cleanly
                            # (Your MATLAB loadDataset.m will read this flawlessly)
                            out_path = os.path.splitext(out_path)[0] + '.jpg'
                            resized_img.save(out_path, 'JPEG', quality=95, exif=exif_bytes)
                        elif ext in ['.jpg', '.jpeg']:
                            resized_img.save(out_path, 'JPEG', quality=95, exif=exif_bytes)
                        elif ext == '.png':
                            # PNGs handle metadata differently, pass the info dict natively
                            resized_img.save(out_path, 'PNG', pnginfo=img.info)
                            
                        print(f"  [+] Saved: {os.path.basename(out_path)} (Taken: {taken_date})")
                        
                except Exception as e:
                    print(f"  [-] Error processing {file_info.filename}: {e}")

if __name__ == "__main__":
    process_zips()
    print("\nExtraction complete.")