"""
visualize_dataset.py
====================
Scans the dataset directory and produces a comprehensive multi-page
diagnostic report for time-of-day estimation datasets.

Pages
-----
  Page 1 — Temporal Distribution
    • Linear histogram (30-min bins, colour-coded by time category)
    • Polar / radial histogram (hourly, cyclic view)
    • Cumulative distribution with quartile markers

  Page 2 — Calendar Coverage
    • Hour × Month heatmap (density)
    • Season distribution bar chart
    • Year-over-year breakdown (if dataset spans multiple years)

  Page 3 — Camera & Colour Intelligence
    • Top cameras by image count
    • Dominant sky colour palette per hour-block (night/morning/midday/afternoon/dusk)
    • Average RGB channel curves across the day

  Page 4 — Dataset Health Report
    • Coverage gaps (blind-spot hours)
    • Class-imbalance bar (images per hour, flagging sparse hours)
    • Per-hour sample count table with status indicators

Usage
-----
    python visualize_dataset.py
    python visualize_dataset.py --dir dataset_512
    python visualize_dataset.py --dir dataset_512 --out report.png
    python visualize_dataset.py --dir dataset_512 --out report.png --no-color
"""

import argparse
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as mgridspec
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config as cfg
from TimeOfDayDataLoader import TimeOfDayDataset, get_transforms


# ---------------------------------------------------------------------------
# Styling constants
# ---------------------------------------------------------------------------

PALETTE = {
    "night":     "#1a237e",
    "morning":   "#e65100",
    "midday":    "#f9a825",
    "afternoon": "#bf360c",
    "dusk":      "#4a148c",
    "bg":        "#ffffff",
    "panel":     "#f8f9fa",
    "grid":      "#e0e0e0",
    "text":      "#212529",
    "subtext":   "#6c757d",
    "accent":    "#4C8EDA",
    "good":      "#43a047",
    "warn":      "#fdd835",
    "bad":       "#e53935",
}

TIME_COLOURS = {
    "Night (00–06 / 20–24)":   PALETTE["night"],
    "Morning (06–10)":          PALETTE["morning"],
    "Midday (10–14)":           PALETTE["midday"],
    "Afternoon (14–18)":        PALETTE["afternoon"],
    "Dusk (18–20)":             PALETTE["dusk"],
}


def _slot(h: float) -> str:
    if h < 6 or h >= 20: return "Night (00–06 / 20–24)"
    if h < 10:            return "Morning (06–10)"
    if h < 14:            return "Midday (10–14)"
    if h < 18:            return "Afternoon (14–18)"
    return                       "Dusk (18–20)"


def _slot_colour(h: float) -> str:
    return TIME_COLOURS[_slot(h)]


def _fmt(m: float) -> str:
    h, mn = divmod(int(round(m)) % 1440, 60)
    return f"{h:02d}:{mn:02d}"


def _apply_theme_style(fig, axes_flat):
    fig.patch.set_facecolor(PALETTE["bg"])
    for ax in axes_flat:
        if ax is None:
            continue
        ax.set_facecolor(PALETTE["panel"])
        ax.tick_params(colors=PALETTE["text"], labelsize=8)
        ax.xaxis.label.set_color(PALETTE["text"])
        ax.yaxis.label.set_color(PALETTE["text"])
        ax.title.set_color(PALETTE["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["grid"])
        ax.grid(color=PALETTE["grid"], linestyle="--", alpha=0.5)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

class DatasetRecord:
    """Holds per-image metadata extracted from the dataset."""
    __slots__ = ("path", "time_min", "month", "year", "camera")

    def __init__(self, path, time_min, month, year, camera="Unknown"):
        self.path     = path
        self.time_min = float(time_min)
        self.month    = int(month)
        self.year     = int(year)
        self.camera   = str(camera).strip() or "Unknown"


def _collect_records(image_dir: str) -> list:
    """
    Walk the dataset and collect per-image metadata.
    Uses TimeOfDayDataset for EXIF parsing, then tries to also grab
    camera model and year via a lightweight secondary pass.
    """
    import exifread

    tf  = get_transforms(augment=False)
    ds  = TimeOfDayDataset(image_dir=image_dir, transform=tf)
    records = []

    for path, label in ds.samples:
        camera = "Unknown"
        year   = 2024
        try:
            with open(path, "rb") as f:
                tags = exifread.process_file(f, details=False)
            cam_tag = tags.get("Image Model") or tags.get("EXIF LensModel")
            if cam_tag:
                camera = str(cam_tag).strip()
            for ts_tag in ("EXIF DateTimeOriginal", "Image DateTime"):
                if ts_tag in tags:
                    ts = str(tags[ts_tag])
                    dt = datetime.strptime(ts, "%Y:%m:%d %H:%M:%S")
                    year = dt.year
                    break
        except Exception:
            pass

        records.append(DatasetRecord(
            path=path,
            time_min=label.time_min,
            month=label.month,
            year=year,
            camera=camera,
        ))

    return records


def _dominant_rgb_per_hour(records: list, n_clusters: int = 1) -> dict:
    """
    For each integer hour block, compute average dominant RGB by
    sampling a small thumbnail — fast enough for hundreds of images.
    Returns {hour: (R, G, B)} in 0-255 range.
    """
    try:
        from PIL import Image as PILImage
        from sklearn.cluster import MiniBatchKMeans
    except ImportError:
        return {}

    hour_pixels: dict = {h: [] for h in range(24)}

    for rec in records:
        h = int(rec.time_min // 60) % 24
        try:
            img = PILImage.open(rec.path).convert("RGB")
            img.thumbnail((32, 32))
            px  = np.asarray(img, dtype=np.float32).reshape(-1, 3)
            hour_pixels[h].append(px)
        except Exception:
            continue

    result = {}
    for h, px_list in hour_pixels.items():
        if not px_list:
            continue
        all_px = np.vstack(px_list)
        if len(all_px) < n_clusters:
            result[h] = tuple(all_px.mean(axis=0).astype(int))
            continue
        km = MiniBatchKMeans(n_clusters=n_clusters, n_init=3, random_state=42)
        km.fit(all_px)
        result[h] = tuple(km.cluster_centers_[0].astype(int))

    return result


# ---------------------------------------------------------------------------
# Page 1 — Temporal distribution
# ---------------------------------------------------------------------------

def _page_temporal(records: list, fig_path_base: str) -> None:
    minutes = np.array([r.time_min for r in records])
    hours   = minutes / 60.0
    N       = len(minutes)

    bins_30       = np.arange(0, 24.5, 0.5)
    counts_30, _  = np.histogram(hours, bins=bins_30)
    bar_centers   = bins_30[:-1] + 0.25

    bins_hour     = np.arange(25)
    counts_hr, _  = np.histogram(hours, bins=bins_hour)

    fig = plt.figure(figsize=(12, 6), facecolor=PALETTE["bg"])
    fig.suptitle(
        f"Temporal Distribution  |  N = {N:,} images",
        fontsize=15, fontweight="bold", color=PALETTE["text"], y=1.01,
    )
    ax1 = fig.add_subplot(111)

    # ── Panel A: linear histogram ─────────────────────────────────────────
    colours = [_slot_colour(b) for b in bar_centers]
    ax1.bar(bar_centers, counts_30, width=0.44,
            color=colours, edgecolor="none", alpha=0.88)

    # smooth envelope
    from scipy.ndimage import gaussian_filter1d
    smooth = gaussian_filter1d(counts_30.astype(float), sigma=1.5)
    ax1.plot(bar_centers, smooth, color="white", linewidth=1.2,
             alpha=0.55, linestyle="--")

    ax1.set_xlim(0, 24)
    ax1.set_xticks(range(0, 25, 2))
    ax1.set_xticklabels([f"{h:02d}h" for h in range(0, 25, 2)], rotation=45, fontsize=8)
    ax1.set_xlabel("Time of Day", fontsize=10)
    ax1.set_ylabel("Image Count", fontsize=10)
    ax1.set_title("Capture-Time Histogram (30-min bins)", fontsize=11)
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # peak annotation
    pi = counts_30.argmax()
    ax1.annotate(
        f"peak\n{_fmt(bar_centers[pi]*60)}\n({counts_30[pi]})",
        xy=(bar_centers[pi], counts_30[pi]),
        xytext=(bar_centers[pi] + 2.0, counts_30[pi] * 0.88),
        arrowprops=dict(arrowstyle="->", color=PALETTE["text"], lw=1.0),
        fontsize=8, color=PALETTE["text"],
    )

    # legend
    legend_items = [
        mpatches.Patch(color=c, label=lbl)
        for lbl, c in TIME_COLOURS.items()
    ]
    ax1.legend(handles=legend_items, fontsize=7, loc="upper left",
               framealpha=0.3, facecolor=PALETTE["panel"],
               labelcolor=PALETTE["text"])

    # stats box
    stats = (
        f"Mean   {_fmt(minutes.mean())}\n"
        f"Median {_fmt(float(np.median(minutes)))}\n"
        f"Std    {minutes.std()/60:.2f} h\n"
        f"Min    {_fmt(minutes.min())}\n"
        f"Max    {_fmt(minutes.max())}"
    )
    ax1.text(0.98, 0.97, stats, transform=ax1.transAxes, fontsize=8,
             va="top", ha="right", color=PALETTE["text"],
             bbox=dict(boxstyle="round,pad=0.4", fc=PALETTE["bg"],
                       ec=PALETTE["grid"], alpha=0.85),
             fontfamily="monospace")

    _apply_theme_style(fig, [ax1])
    fig.tight_layout()

    fig.savefig(fig_path_base + "_p1_temporal.png",
                dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  → Page 1 saved: {fig_path_base}_p1_temporal.png")


# ---------------------------------------------------------------------------
# Page 2 — Calendar coverage
# ---------------------------------------------------------------------------

def _page_calendar(records: list, fig_path_base: str) -> None:
    months  = np.array([r.month    for r in records])
    hours   = np.array([int(r.time_min // 60) % 24 for r in records])
    years   = np.array([r.year     for r in records])

    def _season(m):
        return {1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
                6:"Summer",7:"Summer",8:"Summer",9:"Autumn",
                10:"Autumn",11:"Autumn",12:"Winter"}[m]

    seasons = np.array([_season(m) for m in months])
    unique_years = sorted(set(years.tolist()))
    multi_year   = len(unique_years) > 1

    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    season_order = ["Spring","Summer","Autumn","Winter"]
    season_cols  = {"Spring": "#66bb6a", "Summer": "#ffa726",
                    "Autumn": "#ef6c00", "Winter": "#42a5f5"}

    fig = plt.figure(figsize=(12, 6), facecolor=PALETTE["bg"])
    fig.suptitle("Calendar Coverage", fontsize=15, fontweight="bold",
                 color=PALETTE["text"], y=1.01)

    # ── Heatmap: Hour × Month ─────────────────────────────────────────────
    ax_hm = fig.add_subplot(111)
    heat  = np.zeros((12, 24), dtype=int)
    for m, h in zip(months, hours):
        heat[m - 1, h] += 1

    cmap = LinearSegmentedColormap.from_list(
        "tod", [PALETTE["panel"], PALETTE["accent"], "#0b2545"], N=256
    )
    im = ax_hm.imshow(heat, aspect="auto", cmap=cmap, interpolation="nearest")
    ax_hm.set_xticks(range(24))
    ax_hm.set_xticklabels([f"{h:02d}h" for h in range(24)],
                           fontsize=7, color=PALETTE["text"])
    ax_hm.set_yticks(range(12))
    ax_hm.set_yticklabels(month_labels, fontsize=8, color=PALETTE["text"])
    ax_hm.set_xlabel("Hour of Day", fontsize=10, color=PALETTE["text"])
    ax_hm.set_title("Image Density: Hour of Day × Month", fontsize=11,
                    color=PALETTE["text"])
    ax_hm.set_facecolor(PALETTE["panel"])
    ax_hm.tick_params(colors=PALETTE["text"])

    cb = plt.colorbar(im, ax=ax_hm, pad=0.01, fraction=0.015)
    cb.set_label("Image Count", color=PALETTE["text"], fontsize=8)
    cb.ax.yaxis.set_tick_params(color=PALETTE["text"], labelcolor=PALETTE["text"])

    # annotate non-zero cells
    thresh = heat.max() * 0.55
    for r in range(12):
        for c in range(24):
            if heat[r, c] > 0:
                col = "#ffffff" if heat[r, c] > thresh else PALETTE["text"]
                ax_hm.text(c, r, str(heat[r, c]),
                           ha="center", va="center", fontsize=5.5, color=col)

    _apply_theme_style(fig, [ax_hm])
    fig.tight_layout()

    fig.savefig(fig_path_base + "_p2_calendar.png",
                dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  → Page 2 saved: {fig_path_base}_p2_calendar.png")


# ---------------------------------------------------------------------------
# Page 3 — Camera & colour intelligence
# ---------------------------------------------------------------------------

def _page_colour(records: list, fig_path_base: str) -> None:
    print("  Computing dominant colours per hour (may take a moment)…")
    hour_rgb = _dominant_rgb_per_hour(records)

    fig = plt.figure(figsize=(12, 3.5), facecolor=PALETTE["bg"])
    fig.suptitle("Dominant Sky Colour by Hour", fontsize=14,
                 fontweight="bold", color=PALETTE["text"], y=1.05)

    # ── Dominant colour palette strip ─────────────────────────────────────
    ax_pal = fig.add_subplot(111)
    ax_pal.set_facecolor(PALETTE["panel"])
    ax_pal.set_xlim(0, 24)
    ax_pal.set_ylim(0, 1)
    ax_pal.set_xticks(range(0, 25, 2))
    ax_pal.set_xticklabels([f"{h:02d}h" for h in range(0, 25, 2)],
                            fontsize=9, color=PALETTE["text"])
    ax_pal.set_yticks([])
    ax_pal.set_xlabel("Hour of Day", fontsize=11, color=PALETTE["text"])

    for h in range(24):
        if h in hour_rgb:
            r, g, b = hour_rgb[h]
            col = (r / 255, g / 255, b / 255)
            ax_pal.add_patch(mpatches.Rectangle(
                (h, 0), 1, 1, linewidth=0, facecolor=col
            ))
            # Luminance-aware text
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            txt_col = "black" if lum > 128 else "white"
            ax_pal.text(h + 0.5, 0.5, f"{h:02d}", ha="center", va="center",
                        fontsize=9, color=txt_col, fontweight="bold")
        else:
            ax_pal.add_patch(mpatches.Rectangle(
                (h, 0), 1, 1, linewidth=0,
                facecolor=PALETTE["panel"], edgecolor=PALETTE["grid"]
            ))
            ax_pal.text(h + 0.5, 0.5, "–", ha="center", va="center",
                        fontsize=9, color=PALETTE["subtext"])

    ax_pal.spines["top"].set_visible(False)
    ax_pal.spines["right"].set_visible(False)
    ax_pal.spines["left"].set_visible(False)
    ax_pal.tick_params(colors=PALETTE["text"])

    _apply_theme_style(fig, [ax_pal])
    fig.tight_layout()

    fig.savefig(fig_path_base + "_p3_colour.png",
                dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  → Page 3 saved: {fig_path_base}_p3_colour.png")


# ---------------------------------------------------------------------------
# Page 4 — Dataset health
# ---------------------------------------------------------------------------

def _page_health(records: list, fig_path_base: str) -> None:
    hours_arr = np.array([int(r.time_min // 60) % 24 for r in records])
    N = len(records)

    counts_per_hour = np.bincount(hours_arr, minlength=24)
    mean_count      = counts_per_hour.mean()
    blind_spots     = [h for h in range(24) if counts_per_hour[h] == 0]
    sparse_hours    = [h for h in range(24)
                       if 0 < counts_per_hour[h] < mean_count * 0.25]

    fig = plt.figure(figsize=(14, 5), facecolor=PALETTE["bg"])
    fig.suptitle("Dataset Health Report", fontsize=15, fontweight="bold",
                 color=PALETTE["text"], y=1.01)

    # ── Imbalance bar ─────────────────────────────────────────────────────
    ax_bal = fig.add_subplot(111)
    bar_cols = []
    for h in range(24):
        c = counts_per_hour[h]
        if c == 0:             bar_cols.append(PALETTE["bad"])
        elif c < mean_count * 0.25: bar_cols.append(PALETTE["warn"])
        else:                  bar_cols.append(PALETTE["good"])

    bars = ax_bal.bar(range(24), counts_per_hour, color=bar_cols,
                      edgecolor="none", alpha=0.90)
    ax_bal.axhline(mean_count, color=PALETTE["accent"], linewidth=1.5,
                   linestyle="--", label=f"Mean ({mean_count:.1f})")
    ax_bal.axhline(mean_count * 0.25, color=PALETTE["warn"], linewidth=1.0,
                   linestyle=":", alpha=0.7, label="Sparse threshold (25%)")

    for bar, c in zip(bars, counts_per_hour):
        if c > 0:
            ax_bal.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(counts_per_hour) * 0.01,
                        str(c), ha="center", va="bottom",
                        fontsize=6.5, color=PALETTE["text"])

    ax_bal.set_xticks(range(24))
    ax_bal.set_xticklabels([f"{h:02d}h" for h in range(24)], fontsize=8)
    ax_bal.set_xlabel("Hour of Day", fontsize=10)
    ax_bal.set_ylabel("Image Count", fontsize=10)
    ax_bal.set_title(
        f"Class Imbalance  —  "
        f"{len(blind_spots)} blind-spot hour(s), "
        f"{len(sparse_hours)} sparse hour(s)",
        fontsize=11,
    )
    legend_patches = [
        mpatches.Patch(color=PALETTE["good"], label="Healthy"),
        mpatches.Patch(color=PALETTE["warn"], label="Sparse (<25% mean)"),
        mpatches.Patch(color=PALETTE["bad"],  label="Blind spot (0 images)"),
    ]
    ax_bal.legend(handles=legend_patches + [
        plt.Line2D([0], [0], color=PALETTE["accent"], linestyle="--", label=f"Mean ({mean_count:.1f})"),
    ], fontsize=8, loc="upper right",
       framealpha=0.3, facecolor=PALETTE["panel"], labelcolor=PALETTE["text"])

    _apply_theme_style(fig, [ax_bal])
    fig.tight_layout()

    fig.savefig(fig_path_base + "_p4_health.png",
                dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  → Page 4 saved: {fig_path_base}_p4_health.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive dataset visualisation for time-of-day estimation."
    )
    parser.add_argument("--dir", default=cfg.IMAGE_DIR,
                        help="Dataset directory (default: cfg.IMAGE_DIR)")
    parser.add_argument("--out", default="dataset_report",
                        help="Output base path (pages appended automatically, "
                             "default: dataset_report)")
    parser.add_argument("--no-color", action="store_true",
                        help="Skip the colour-extraction page (faster for large datasets)")
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"ERROR: '{args.dir}' not found.")
        sys.exit(1)

    base = args.out.removesuffix(".png")

    print(f"\nScanning '{args.dir}' …")
    records = _collect_records(args.dir)
    print(f"Found {len(records):,} images with valid EXIF timestamps.\n")

    if not records:
        print("Nothing to plot.")
        sys.exit(0)

    print("Generating Page 1 — Temporal distribution…")
    _page_temporal(records, base)

    print("Generating Page 2 — Calendar coverage…")
    _page_calendar(records, base)

    if not args.no_color:
        print("Generating Page 3 — Camera & colour…")
        _page_colour(records, base)
    else:
        print("Skipping Page 3 (--no-color).")

    print("Generating Page 4 — Dataset health…")
    _page_health(records, base)

    print(f"\nDone. Four pages written with base path '{base}'.")


if __name__ == "__main__":
    main()