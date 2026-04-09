from PIL import Image
from PIL.ExifTags import TAGS
from pillow_heif import register_heif_opener

# This tells Pillow how to handle .heic files
register_heif_opener()

def get_image_metadata(image_path):
    try:
        img = Image.open(image_path)
        
        # HEIC files store EXIF similarly to JPEGs
        exif_data = img.getexif()
        
        if not exif_data:
            print(f"No EXIF metadata found in {image_path}")
            return

        print(f"{'Tag':<25} | {'Value'}")
        print("-" * 50)

        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, tag_id)
            print(f"{tag_name:<25} | {value}")

    except Exception as e:
        print(f"Error: {e}")

get_image_metadata("IMG_3278.heic")