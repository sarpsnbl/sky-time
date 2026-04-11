import os
import glob
from PIL import Image, ExifTags
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from datetime import datetime

# --- CONFIGURATION ---
DATASET_DIR = "dataset" # Change this to your folder path
SUPPORTED_FORMATS = ('*.jpg', '*.jpeg', '*.png', '*.heic', '*.heif')

def get_exif_data(image_path):
    """Extracts datetime and camera model from image EXIF."""
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if not exif_data:
            return None, None
            
        dt_str = None
        camera_model = "Unknown"
        
        for tag_id, value in exif_data.items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            if tag == 'DateTimeOriginal':
                dt_str = value
            elif tag == 'Model':
                camera_model = str(value).strip()
                
        if dt_str:
            # EXIF datetime format is usually "YYYY:MM:DD HH:MM:SS"
            dt_obj = datetime.strptime(dt_str, '%Y:%m:%d %H:%M:%S')
            return dt_obj, camera_model
    except Exception:
        pass
    return None, None

def get_dominant_color(image_path, n_colors=1):
    """Resizes image for speed and extracts dominant RGB color."""
    try:
        img = Image.open(image_path).convert('RGB')
        img.thumbnail((50, 50)) # Downscale heavily for performance
        pixels = np.array(img).reshape(-1, 3)
        kmeans = KMeans(n_clusters=n_colors, n_init=3, random_state=42)
        kmeans.fit(pixels)
        return kmeans.cluster_centers_[0].astype(int)
    except Exception:
        return np.array([0, 0, 0])

def get_season(month):
    if month in [3, 4, 5]: return 'Spring'
    elif month in [6, 7, 8]: return 'Summer'
    elif month in [9, 10, 11]: return 'Autumn'
    else: return 'Winter'

# --- 1. DATA EXTRACTION ---
print("Scanning dataset and extracting metadata... This might take a minute depending on folder size.")
data = []
image_files = []
for fmt in SUPPORTED_FORMATS:
    image_files.extend(glob.glob(os.path.join(DATASET_DIR, fmt)))
    image_files.extend(glob.glob(os.path.join(DATASET_DIR, fmt.upper())))

for img_path in image_files:
    dt_obj, camera_model = get_exif_data(img_path)
    if dt_obj:
        dom_color = get_dominant_color(img_path)
        data.append({
            'Filename': os.path.basename(img_path),
            'Datetime': dt_obj,
            'Month': dt_obj.month,
            'Hour': dt_obj.hour,
            'Season': get_season(dt_obj.month),
            'Camera': camera_model,
            'R': dom_color[0], 'G': dom_color[1], 'B': dom_color[2]
        })

df = pd.DataFrame(data)

if df.empty:
    print("No valid EXIF data found. Make sure your images retain their original metadata.")
    exit()

# --- 2. VISUALIZATIONS ---
sns.set_theme(style="whitegrid")

# Plot 1: Distribution by Month and Season
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.countplot(data=df, x='Month', palette='viridis', ax=axes[0])
axes[0].set_title('Photos per Month')
sns.countplot(data=df, x='Season', order=['Spring', 'Summer', 'Autumn', 'Winter'], palette='coolwarm', ax=axes[1])
axes[1].set_title('Photos per Season')
plt.tight_layout()
plt.show()

# Plot 2: Distribution by Hour
plt.figure(figsize=(12, 5))
sns.countplot(data=df, x='Hour', palette='magma')
plt.title('Photos per Hour of the Day')
plt.xticks(range(0, 24))
plt.show()

# Plot 3: Heatmap (Hour vs Month)
plt.figure(figsize=(12, 6))
heatmap_data = pd.crosstab(df['Month'], df['Hour'])
# Reindex to ensure all 24 hours and 12 months show even if empty
heatmap_data = heatmap_data.reindex(index=range(1, 13), columns=range(0, 24), fill_value=0)
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='d', linewidths=.5)
plt.title('Photo Density: Hour of the Day across Months')
plt.show()

# Plot 4: Color Palettes per Month
def plot_color_palette(grouped_data, group_col, title):
    unique_groups = sorted(grouped_data[group_col].unique())
    fig, axes = plt.subplots(len(unique_groups), 1, figsize=(10, len(unique_groups) * 0.8))
    if len(unique_groups) == 1: axes = [axes]
    
    for i, grp in enumerate(unique_groups):
        grp_pixels = grouped_data[grouped_data[group_col] == grp][['R', 'G', 'B']].values
        if len(grp_pixels) > 0:
            # Find top 5 dominant colors for this specific group
            kmeans = KMeans(n_clusters=min(5, len(grp_pixels)), n_init=3)
            kmeans.fit(grp_pixels)
            colors = kmeans.cluster_centers_ / 255.0
            
            axes[i].imshow([colors])
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            axes[i].set_ylabel(f"{grp}", rotation=0, labelpad=30, va='center')
            
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

print("Generating Color Palettes...")
plot_color_palette(df, 'Month', 'Dominant Sky Colors per Month')
plot_color_palette(df, 'Hour', 'Dominant Sky Colors per Hour')

# --- 3. GENERAL INFORMATION REPORT ---
print("\n" + "="*40)
print("📸 DATASET GENERAL REPORT")
print("="*40)
print(f"Total Images Processed (with EXIF): {len(df)}")
print(f"Total Images Missing EXIF/Dropped: {len(image_files) - len(df)}")

earliest_hour = df['Hour'].min()
latest_hour = df['Hour'].max() # Wait, max() is needed!
print(f"\nTime Coverage:")
print(f"- Earliest Photo Time: {earliest_hour}:00")
print(f"- Latest Photo Time: {latest_hour}:59")
print(f"- Most Active Hour: {df['Hour'].mode()[0]}:00")
print(f"- Most Active Month: {df['Month'].mode()[0]}")

print(f"\nDevice Variety (Top 5 Cameras):")
print(df['Camera'].value_counts().head(5).to_string())

# Data Gap Warning (Anything to help explain the dataset)
missing_hours = [h for h in range(24) if h not in df['Hour'].values]
if missing_hours:
    print(f"\n⚠️ Blind Spots Detected! You have ZERO photos for these hours: {missing_hours}")
else:
    print("\n✅ Good temporal coverage: You have at least one photo for every hour of the day.")