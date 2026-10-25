# Sky Time Estimation — MATLAB Pipeline
**Deep Learning-Based Time of Day Estimation from Sky Images and Calendar Dates**

Authors: 220401050 Alkım Gönenç Efe · 220401067 Damla Parlakyıldız · 230401114 Sarp Sünbül

---

## Requirements

| Toolbox | Used for |
|---|---|
| Image Processing Toolbox | `imread`, `imresize`, `imfinfo`, `rgb2hsv`, `rgb2lab`, `edge`, `imfilter` |
| Deep Learning Toolbox | `squeezenet`, `trainNetwork`, `layerGraph`, `imageDatastore`, `regressionLayer` |
| Statistics & ML Toolbox | `TreeBagger` (Random Forest), `fitrsvm` (SVR), `cvpartition` |

MATLAB **R2022a or newer** is recommended.  
HEIC support requires R2022a+. DNG/RAW requires the Image Processing Toolbox.

---

## Quick Start

```
1. Place all your sky images inside a folder named  dataset/
   (supported formats: .jpg  .jpeg  .png  .dng  .heic)

2. Open MATLAB, cd to this folder.

3. Run:  main
```

The pipeline will:
- Parse DateTime from each image's EXIF metadata
- Extract both deep (CNN) and classical features
- Run 5-fold cross-validation with three models
- Print per-image predictions and RMSE
- Save results to  sky_time_results.csv

---

## File Structure

```
sky_time/
├── main.m                     ← Entry point (run this)
├── loadDataset.m              ← EXIF DateTime reader; supports all formats
├── extractClassicalFeatures.m ← 145-dim hand-crafted feature extractor
├── buildMultiInputCNN.m       ← SqueezeNet + date-branch layerGraph
├── buildDatastore.m           ← Combined datastore for trainNetwork
├── predictCNN.m               ← Batch inference on test images
├── computeRMSE.m              ← Circular-clock RMSE
├── circularTimeDiff.m         ← |g − a| on a 24-hr clock
├── hoursToHHMM.m              ← Fractional hours → 'HH:MM' string
├── printModelResults.m        ← Summary printer
└── README.md
```

---

## Architecture

### Primary Model — Multi-Input CNN

```
IMAGE (224×224×3)               DATE (4 floats)
       │                              │
  SqueezeNet backbone           FC(32)→BN→ReLU
  (fire1–6 frozen,              FC(64)→ReLU
   fire7–9 fine-tuned)               │
  pool10 → Flatten [512]        [64]
       │                              │
       └──────── concat [576] ────────┘
                      │
              FC(256)→BN→ReLU
                 Dropout(0.4)
                FC(64)→ReLU
                   FC(1)
              RegressionLayer
```

### Date Encoding
The calendar date from `DateTime` is encoded as 4 circular features:

| Feature | Formula |
|---|---|
| sin(day-of-year) | sin(2π × doy/366) |
| cos(day-of-year) | cos(2π × doy/366) |
| sin(month) | sin(2π × (month−1)/12) |
| cos(month) | cos(2π × (month−1)/12) |

This lets the model learn that January and December are adjacent, and
correctly associate sun angles with seasons.

### Baseline Models
- **Random Forest** — 100 trees, `TreeBagger`, trained on 145-dim classical features + 4-dim date
- **SVR** — RBF kernel, `fitrsvm`, auto-scaled, trained on the same feature set

### Classical Features (145 dims)
- RGB / HSV / LAB channel statistics (mean + std) → 18
- RGB histogram (16 bins × 3 channels) → 48
- HSV histogram (16 bins × 3 channels) → 48
- Sun-region brightness (top ⅓ of frame) → 2
- Horizon luminance gradient → 1
- Colour temperature proxy (R/B ratio) → 1
- Global luminance (mean, std, entropy) → 3
- Saturation stats → 2
- Edge density (Canny) → 1
- Laplacian variance (cloud texture) → 1

---

## Output Format

```
Index   Guess       Actual      Abs Err(min)  Image
──────────────────────────────────────────────────────────────────────────
1       guess: 15:07   actual: 16:35   88.0 min     IMG_0042.jpg
2       guess: 08:22   actual: 08:15    7.0 min     sky_morning.heic
...
──────────────────────────────────────────────────────────────────────────
error margin: 34.17 min (RMSE over all folds)
```

Results are also saved to `sky_time_results.csv`.

---

## Cross-Validation

5-fold CV is used (`cvpartition`). All three models are trained and evaluated
on identical splits. The final RMSE reported is computed over **all test
predictions** concatenated across folds (not the mean of per-fold RMSEs),
which is the standard approach for small datasets.

---

## Notes on Image Formats

| Format | imread support | Notes |
|---|---|---|
| `.jpg` / `.jpeg` | All versions | Standard |
| `.png` | All versions | Standard |
| `.dng` | R2019b+ | Demosaiced by MATLAB automatically |
| `.heic` | R2022a+ | Requires Apple HEIC codec on Windows |

If HEIC images fail to load, install the **HEIF Image Extensions** from the
Microsoft Store (Windows) or ensure libheif is installed (Linux/macOS).

---

## Tips for Better Results

- **Collect diverse images**: dawn, morning, noon, afternoon, dusk — distribute
  timestamps evenly to avoid class imbalance in regression.
- **Sky must be visible**: images without sky content will confuse the model.
- **Consistent capture angle**: pointing the camera at roughly the same
  elevation angle across sessions helps the CNN learn consistent lighting cues.
- **More data → better RMSE**: with 50+ images, the CNN generalises well;
  with fewer, the Random Forest baseline may outperform it.
