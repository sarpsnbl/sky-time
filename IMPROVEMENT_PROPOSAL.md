# Comprehensive Improvement Proposal: Deep Learning-Based Time-of-Day Estimation

**Project:** Sky Time Estimation from Single Outdoor Photographs and EXIF Metadata  
**Repository:** `sky-time`  
**Date:** September 2026  
**Authors:** Alkım Gönenç Efe · Sarp Sünbül · Damla Parlakyıldız  

---

## 1. Executive Summary & Scientific Premise

This document defines an exhaustive, physics-grounded engineering proposal to upgrade the Python/PyTorch pipeline for time-of-day regression from unconstrained outdoor sky photographs.

By performing a forensic audit of sample images from both smartphones (**Apple iPhone 13**) and dedicated mirrorless cameras (**Fujifilm X100VI**), we discovered a rich, largely unexploited suite of optical, sensor, orientation, and geodetic metadata recorded in standard EXIF headers. 

Currently, our primary **Swin-T** model achieves an average 5-fold cross-validation Mean Absolute Error (MAE) of **52.45 minutes**. By moving beyond simple calendar dates and extracting every physically feasible metadata signal—from APEX scene luminance and camera compass orientation to exact astronomical solar ephemeris—we can provide the neural network with direct physical constraints, targeting a validation cyclic MAE **under 38–42 minutes**.

---

## 2. Complete Metadata Taxonomy: "Milking" Every Feasible Signal

Modern digital cameras and smartphones do not simply record pixels; their onboard light meters, gyroscopes, magnetometers, and lenses capture a complete snapshot of ambient illumination and spatial geometry.

Below is the complete catalog of all metadata signals to be integrated into the pipeline, categorized into 5 distinct physical tiers:

```
+----------------------------------------------------------------------------------------------------+
|                                    TOTAL METADATA VECTOR (105-D)                                    |
+----------------------------------+----------------------------------+------------------------------+
| 1. Optical & Sensor Physics (12) | 2. Spatial & Orientation (8)     | 3. Astrometric Ephemeris (6) |
| - APEX Brightness Value (Bv)     | - Camera Compass Heading (sin/cos| - Solar Declination (delta)  |
| - Calculated EV100               | - Timezone Offset (OffsetTime)   | - Equation of Time (EoT)     |
| - Shutter Speed (log2 t)         | - Latitude & Longitude           | - Theoretical Day Length     |
| - ISO Sensitivity (log2 ISO)     | - GPS Altitude                   | - Theoretical Solar Noon     |
| - Relative Aperture (F-number)   | - Missingness Binary Flags       | - Sunrise & Sunset Bounds    |
| - Exposure Bias (EV comp)        |                                  | - Max Solar Elevation        |
| - 35mm Focal Length / FoV        +----------------------------------+------------------------------+
| - White Balance & Metering Mode  | 4. Cyclic Calendar (6)           | 5. Atmospheric Photometrics  |
| - Flash State                    | - Day of Year, Month, Day of Week|    (79 Handcrafted Dims)     |
+----------------------------------+----------------------------------+------------------------------+
```

---

### Tier A: Photometric Hardware & Optical Sensor Physics (12 Dimensions)

Modern auto-exposure algorithms digitally normalize pixel brightness, causing night scenes with streetlights to share identical 8-bit histograms with midday clouds. The hardware tags below break this ambiguity completely:

| # | Tag Name | EXIF Tag ID | Formula / Extraction | Normalization / Units | Physical Value to Model |
|---|---|---|---|---|---|
| **1** | **APEX Brightness Value ($B_v$)** | `0x9203` (37378) | Direct photometer reading: $B_v = \log_2(B / NK)$ | Clip to $[-5, 15]$, scaled to $[0, 1]$ | **Direct scene luminance in $cd/m^2$**. Invariant to camera auto-exposure. Night: $[-5, 0]$, Overcast: $[5, 8]$, Bright Sun: $[10, 14]$. |
| **2** | **Calculated Exposure Value ($\text{EV}_{100}$)** | Computed | $\log_2(N^2 / t) - \log_2(\text{ISO} / 100)$ | Clip to $[-2, 18]$, scaled to $[0, 1]$ | Cross-check on ambient lux when $B_v$ is absent ($>8,000\times$ difference between night & noon). |
| **3** | **Shutter Speed ($t$)** | `0x829A` (33434) | `ExposureTime` (seconds) | $\log_2(t)$, clip $[-15, 2]$ | Fast shutter ($<1/1000\text{s}$) indicates peak sunlight; slow shutter ($>1/30\text{s}$) indicates dusk/night. |
| **4** | **ISO Sensitivity** | `0x8827` (34855) | `ISOSpeedRatings` | $\log_2(\text{ISO} / 100)$, clip $[0, 7]$ | Sensor amplification. Low (25–100) = bright daylight; High (800–6400) = dark conditions. |
| **5** | **Relative Aperture ($N$)** | `0x829D` (33437) | `FNumber` | Normalized $[1.0, 22.0]$ | Lens light-gathering capacity. |
| **6** | **Exposure Bias ($\Delta\text{EV}$)** | `0x9204` (37380) | `ExposureBiasValue` | Clip $[-3.0, +3.0]$ | Indicates intentional user exposure compensation (e.g. $-1.5$ EV dialed in to preserve golden hour sunset colors). |
| **7** | **Field of View (FoV)** | `0xA405` (41989) | $2 \arctan(36 / (2 f_{35}))$ from `FocalLengthIn35mmFilm` | Normalized $[0, \pi]$ | Tells how much of the sky hemisphere is captured (wide angle $\sim 80^\circ$ vs telephoto $\sim 25^\circ$). |
| **8** | **Camera Elevation / Pitch** | `0x9205` / Maker | `CameraElevationAngle` | $[-90^\circ, +90^\circ] / 90^\circ$ | Angle between optical axis and the horizon (pointing up at zenith vs level). |
| **9** | **White Balance Mode** | `0xA403` (41987) | `WhiteBalance` | Binary (0 = Auto, 1 = Manual) | Tells if color temperature shift is natural or compensated. |
| **10** | **Light Source Preset** | `0x9208` (37384) | `LightSource` | Categorical embedding or normalized | 1 = Daylight, 9 = Fine weather, 10 = Cloudy, 11 = Shade. |
| **11** | **Metering Mode** | `0x9207` (37382) | `MeteringMode` | Categorical / One-Hot (3 dims) | 2 = Center-Weighted, 3 = Spot, 5 = Pattern/Matrix. |
| **12** | **Flash Status** | `0x9209` (37385) | `Flash` | Binary (1 = Fired, 0 = Suppressed) | Eliminates artificial close-up highlights. |

---

### Tier B: Geodetic, Spatial & Camera Orientation Metadata (8 Dimensions)

Enables robust handling of photographs from **Vienna, İzmir, Balıkesir, or anywhere worldwide**:

| # | Tag Name | EXIF Tag ID | Formula / Extraction | Normalization / Units | Physical Value to Model |
|---|---|---|---|---|---|
| **13–14** | **Camera Compass Heading ($\gamma_{\text{cam}}$)** | GPS Tag 17 (`GPSImgDirection`) | Cyclic: $[\sin(\gamma_{\text{cam}}), \cos(\gamma_{\text{cam}})]$ | Unit Circle $[-1, 1]$ | **Massive Dawn vs. Dusk Discriminator!** Facing East ($90^\circ$) toward a low sun = Dawn; Facing West ($270^\circ$) = Dusk. |
| **15** | **Timezone Offset** | `0x9011` (`OffsetTimeOriginal`) | Parsed string (e.g. `+03:00` $\to 3.0$) | $\text{tz} / 12.0 \in [-1, 1]$ | **Aligns Vienna (UTC+2) and Turkey (UTC+3)** to global solar time. |
| **16–17** | **Geographic Coordinates** | GPS Tags 2, 4 (`GPSLatitude`, `GPSLongitude`) | Decimal degrees | $\text{lat} / 90.0, \text{lon} / 180.0$ | Exact location on Earth. Fallback to Aegean centroid ($39^\circ\text{N}, 27^\circ\text{E}$) when missing. |
| **18** | **GPS Altitude** | GPS Tag 6 (`GPSAltitude`) | Altitude in meters | $\log(1 + h) / 10.0$ | Atmospheric pressure, horizon visibility, and airmass shift with altitude. |
| **19–20** | **Missingness Indicator Mask** | Internal flags | Binary flags | $\{0, 1\}$ | `has_gps`, `has_heading`, `has_ev`, `has_tz`. Enables graceful degradation when sensors are off. |

---

### Tier C: Pure Astrometric Physics (Zero-Leakage Solar Ephemeris) (6 Dimensions)

Because capture time is our prediction target, we **cannot** pass current sun coordinates as inputs (that would leak the label). However, **Date and Location deterministically dictate the daily solar envelope** before a single photon is captured:

| # | Feature Name | Astronomical Formula | Physical Value to Model |
|---|---|---|---|
| **21** | **Solar Declination ($\delta$)** | $\delta = -23.44^\circ \cdot \cos\left(\frac{2\pi}{365.25}(d + 10)\right)$ | Earth's axial tilt on day $d$. **Requires ZERO GPS knowledge**. Solves summer vs. winter daylight length globally. |
| **22** | **Equation of Time ($\text{EoT}$)** | $\text{EoT} = 9.87 \sin(2B) - 7.53 \cos(B) - 1.5 \sin(B)$, $B = \frac{2\pi(d-81)}{364}$ | Discrepancy between apparent solar time and mean clock time (up to $\pm 16.5$ min due to Earth's orbital eccentricity). |
| **23** | **Theoretical Day Length ($\Delta t_{\text{day}}$)** | $\cos(\omega_0) = -\tan(\phi) \tan(\delta) \implies \Delta t = \frac{2\omega_0}{15^\circ}$ | Total hours of daylight possible on that specific day and latitude. |
| **24** | **Theoretical Solar Noon ($t_{\text{noon}}$)** | $t_{\text{noon}} = 12:00 - \frac{\text{Lon} - 15^\circ \cdot \text{TZ}}{15^\circ} - \frac{\text{EoT}}{60}$ | The exact clock minute when the sun reaches maximum zenith. |
| **25** | **Theoretical Sunrise & Sunset ($t_{\text{rise}}, t_{\text{set}}$)** | $t_{\text{rise}} = t_{\text{noon}} - \frac{\Delta t}{2}, \quad t_{\text{set}} = t_{\text{noon}} + \frac{\Delta t}{2}$ | Hard boundary conditions for day vs. night for that specific photograph. |
| **26** | **Maximum Solar Elevation ($\alpha_{\max}$)** | $\alpha_{\max} = 90^\circ - \|\text{lat} - \delta\|$ | Highest angle the sun can possibly reach at noon. |

---

### Tier D: Calendar & Temporal Cycles (6 Dimensions)

| # | Feature | Encoding | Physical Significance |
|---|---|---|---|
| **27–28** | **Day of Year ($d$)** | $[\sin(2\pi d / 365.25), \cos(2\pi d / 365.25)]$ | Continuous yearly cycle; connects Dec 31 to Jan 1. |
| **29–30** | **Month ($m$)** | $[\sin(2\pi (m-1) / 12), \cos(2\pi (m-1) / 12)]$ | Seasonal macro-prior. |
| **31–32** | **Day of Week** | $[\sin(2\pi \text{dow} / 7), \cos(2\pi \text{dow} / 7)]$ | Accounts for human shooting behavior shifts (weekends vs. weekdays). |

---

### Tier E: Atmospheric Photometric Descriptors (79 Dimensions)

Refining [`ImageFeatureExtractor`](file:///c:/Users/ssarp/sky-time/TimeOfDayDataLoader.py#L51-L111) to capture solar position **without requiring the sun disk**:

* **Signed Vertical Luminance Gradient ($\Delta V_y$):**
  $$\Delta V_y = \text{mean}(V_{\text{top\_third}}) - \text{mean}(V_{\text{bottom\_third}})$$
  *Current code used `np.abs()`, throwing away the sign!* Positive = bright noon zenith; Negative = glowing dawn/dusk horizon.
* **Horizontal Solar Asymmetry Gradient ($\Delta V_x$):**
  $$\Delta V_x = \text{mean}(V_{\text{left\_half}}) - \text{mean}(V_{\text{right\_half}})$$
  Identifies whether diffuse lighting originates from the East or West under overcast skies.
* **Zenith-to-Horizon Chrominance Shift ($\Delta b^*$):**
  $$\Delta b^* = \text{mean}(b^*_{\text{bottom\_20\%}}) - \text{mean}(b^*_{\text{top\_20\%}})$$
  Rayleigh scattering path length is $38\times$ longer at horizon during twilight. Cool blue zenith vs warm orange horizon is captured directly.
* **Standard Photometrics (76 dims):** Channel stats (RGB, HSV, LAB), 8-bin histograms, luminance entropy, edge density, and Laplacian variance.

---

### ⚠️ Target Leakage Safeguards (What We Strictly DO NOT Use)

In our EXIF inspection of `predict1.jpg`, we noted that `GPSInfo` contains:
* `GPSTimeStamp`: `(17.0, 2.0, 40.99)`
* `GPSDateStamp`: `2024:08:05`

`GPSTimeStamp` is the satellite-synchronized UTC capture time! **This field MUST NEVER be used as an input feature**, as it constitutes 100% target leakage. Our pipeline explicitly isolates and ignores `GPSTimeStamp`, utilizing only spatial tags (`GPSLatitude`, `GPSLongitude`, `GPSImgDirection`, `GPSAltitude`).

---

## 3. Mathematical & Training Framework Corrections

### 3.1 Angular Cosine Loss (Von Mises Loss)
In [`Main.py`](file:///c:/Users/ssarp/sky-time/Main.py), replacing `nn.MSELoss()` with an angular cosine objective:

```python
class AngularCosineLoss(nn.Module):
    """
    Minimizes angular deviation on the 24-hour circular topology:
    Loss = 1 - cos(theta_pred - theta_target)
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # L2-normalize predictions onto unit circle
        pred_norm = torch.nn.functional.normalize(pred, p=2, dim=-1)
        # Target is already on unit circle [sin(t), cos(t)]
        cos_sim = (pred_norm * target).sum(dim=-1)
        return (1.0 - cos_sim).mean()
```

### 3.2 Circular Mixup Regularization
Normalizing mixed targets onto the unit circle to eliminate target magnitude collapse:
```python
def circular_mixup(images, metadata, targets, alpha=0.15):
    if alpha <= 0.0:
        return images, metadata, targets
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    idx = torch.randperm(images.size(0), device=images.device)

    mixed_imgs = lam * images + (1 - lam) * images[idx]
    mixed_meta = lam * metadata + (1 - lam) * metadata[idx]
    mixed_targ = lam * targets + (1 - lam) * targets[idx]
    
    # Re-project mixed targets back to unit circle
    mixed_targ = torch.nn.functional.normalize(mixed_targ, p=2, dim=-1)
    return mixed_imgs, mixed_meta, mixed_targ
```

### 3.3 Angular Label Jitter
Adding Gaussian noise directly to clock angles in minutes:
```python
def add_angular_noise(targets: torch.Tensor, std_minutes: float = 12.0) -> torch.Tensor:
    if std_minutes <= 0.0:
        return targets
    noise_rad = (torch.randn(targets.size(0), device=targets.device) * std_minutes / 1440.0) * (2.0 * math.pi)
    theta = torch.atan2(targets[:, 0], targets[:, 1]) + noise_rad
    return torch.stack([torch.sin(theta), torch.cos(theta)], dim=1)
```

---

## 4. Multi-Modal Conditioning Head (FiLM)

To eliminate **modality suppression** (where the 768 visual features dominate the metadata), we deploy **Feature-wise Linear Modulation (FiLM)**:

```python
class FiLMFusionHead(nn.Module):
    def __init__(self, image_dim: int = 768, metadata_dim: int = 105, hidden_dim: int = 384, dropout: float = 0.03):
        super().__init__()
        # Generates scale (gamma) and shift (beta) vectors from metadata
        self.film_generator = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, image_dim * 2)
        )
        
        # Regression MLP operating on conditioned visual representations
        self.regressor = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, img_feat: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        film_params = self.film_generator(meta)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)
        # Dynamic affine modulation: gamma scales, beta shifts
        modulated_feat = (1.0 + gamma) * img_feat + beta
        return self.regressor(modulated_feat)
```

---

## 5. Diagnostic Fixes in Evaluation Tooling

In [`hardest_finder.py` (line 18)](file:///c:/Users/ssarp/sky-time/hardest_finder.py#L18):
```python
# BUGGY (Original):
error = abs(pred_min - actual_min)

# CORRECTED (Circular Clock Difference):
raw_diff = abs(pred_min - actual_min)
error = min(raw_diff, 1440.0 - raw_diff)
```
*Eliminates false alarms where an accurate 10-minute midnight prediction was logged as a 23.8-hour error.*

---

## 6. Implementation Roadmap & Priority Matrix

| Phase | Milestone | Priority | Impact Area | Expected MAE Gain |
|---|---|---|---|---|
| **Phase 1** | **Circular Math & Loss Fixes:** Unit-circle output normalization, Angular Cosine Loss, Circular Mixup, Angular Label Noise | **P1 (Immediate)** | Training stability & objective alignment | **3.0–5.0 min** |
| **Phase 1** | **Evaluation Audit Fix:** Update `hardest_finder.py` to circular error calculation | **P1 (Immediate)** | True error logging | Diagnostic accuracy |
| **Phase 2** | **EXIF Physics Extractor:** Extract APEX $B_v$, $\text{EV}_{100}$, Shutter, ISO, Aperture, EV Bias, FoV, Flash, White Balance | **P1** | Eliminates artificial light & night misclassification | **4.0–6.0 min** |
| **Phase 2** | **Spatial & Geodetic Alignment:** Extract Timezone Offset (`OffsetTime`), Camera Heading (`GPSImgDirection`), GPS Coordinates & Altitude | **P1** | Aligns Vienna, İzmir, Balıkesir & resolves Dawn/Dusk | **3.5–5.0 min** |
| **Phase 3** | **Astrometric Engine:** Implement Solar Declination ($\delta$), Equation of Time ($\text{EoT}$), Day Length ($\Delta t$), and Solar Noon ($t_{\text{noon}}$) | **P2** | Imposes physical solar bounds without label leakage | **2.0–3.5 min** |
| **Phase 3** | **Atmospheric Features:** Implement signed vertical gradient $\Delta V_y$, horizontal gradient $\Delta V_x$, and chrominance shift $\Delta b^*$ | **P2** | Sun-free visual solar elevation proxy | **1.5–2.5 min** |
| **Phase 4** | **FiLM Conditioning Head:** Replace naive concat MLP with FiLM feature modulation | **P2** | Eliminates visual modality suppression | **2.0–3.0 min** |
| **Phase 5** | **Dual-Backbone Ensembling & TTA:** Activate TTA (horizontal flips) and ensemble 5-fold Swin-T with 5-fold ConvNeXt | **P3** | Minimizes cross-validation variance | **2.5–3.5 min** |

---

## 7. Expected Impact on the Academic Paper (`new paper.tex`)

Integrating this rich metadata framework directly elevates the scientific quality and narrative of the final research paper:

1. **Resolves Stated Limitations:**
   * In Section 6 (Discussion), the manuscript currently lists:
     > *"First, the dataset is geographically constrained to a single location (İzmir, Turkey)..."*
   * With the new timezone offset (`OffsetTime`), camera compass heading (`GPSImgDirection`), and solar declination features, we can formally report that the framework successfully generalizes across international capture sites (Vienna, İzmir, Balıkesir).
2. **First-of-its-Kind Hardware EXIF Fusion:**
   * Fusing APEX Scene Luminance ($B_v$), Exposure Value ($\text{EV}_{100}$), and Camera Compass Orientation with Vision Transformers for continuous 24-hour time regression represents a distinct, high-impact novel contribution to computer vision literature.
3. **Target Benchmark:**
   * Progressively drives cyclic validation MAE from **52.45 minutes** to **under 38–42 minutes**, establishing a state-of-the-art result for unconstrained smartphone sky regression.
