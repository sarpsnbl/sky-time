import os
import json
import shutil
import subprocess
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image as PILImage

# Styling constants
PALETTE = {
    "accent": "#4C8EDA",
    "text": "#212529",
    "grid": "#e0e0e0",
    "panel": "#f8f9fa",
    "bg": "#ffffff"
}

def setup_matplotlib_style():
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['text.color'] = PALETTE["text"]
    plt.rcParams['axes.labelcolor'] = PALETTE["text"]
    plt.rcParams['xtick.color'] = PALETTE["text"]
    plt.rcParams['ytick.color'] = PALETTE["text"]

def _cyclic_diff(a: float, b: float) -> float:
    """Cyclic absolute difference in minutes."""
    d = abs(a - b)
    return min(d, 1440.0 - d)

def load_python_data(file_path):
    actuals = []
    preds = []
    with open(file_path, 'r') as f:
        for line in f:
            record = json.loads(line)
            if record.get('type') == 'image':
                actuals.append(record['actual_min'] / 60.0)
                preds.append(record['pred_min'] / 60.0)
    return np.array(actuals), np.array(preds)

def generate_swint_arch():
    """Generates swint_arch.png (Figure 1 in the paper)."""
    print("Generating Figure 1: swint_arch.png...")
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 12)
    ax.axis('off')

    w, h = 4.0, 0.9

    nodes = {
        'img': (2.5, 11, 'Sky Image\n(512×512×3)', '#ffffe0'),
        'meta': (7.5, 11, 'Calendar Metadata\n(83 dims)', '#ffffe0'),
        'swin_t': (2.5, 9, 'Swin-T\nBackbone', '#f08080'),
        'pool': (2.5, 7, 'Global Average Pooling\n(768 dims)', '#add8e6'),
        'concat': (5, 5, 'Concatenate\n(851 dims)', '#d3d3d3'),
        'mlp1': (5, 3.2, 'FC(384) → LN → GELU → Dropout', '#90ee90'),
        'mlp2': (5, 1.6, 'FC(384) → LN → GELU → Dropout', '#90ee90'),
        'out': (5, 0, 'FC(2) → Regression Layer\n[sin, cos]', '#ffffe0')
    }

    for name, (x, y, text, color) in nodes.items():
        node_w = 5.5 if name in ['mlp1', 'mlp2'] else w
        box = patches.FancyBboxPatch((x - node_w/2, y - h/2), node_w, h,
                                     boxstyle="round,pad=0.1,rounding_size=0.2",
                                     edgecolor="black", facecolor=color, zorder=2)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=11, zorder=3, fontweight='bold')

    def draw_arrow(n1, n2):
        x1, y1 = nodes[n1][0], nodes[n1][1] - h/2 - 0.1
        x2, y2 = nodes[n2][0], nodes[n2][1] + h/2 + 0.1
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), 
                    arrowprops=dict(arrowstyle="->", lw=2), zorder=1)

    draw_arrow('img', 'swin_t')
    draw_arrow('swin_t', 'pool')
    draw_arrow('pool', 'concat')
    draw_arrow('meta', 'concat')
    draw_arrow('concat', 'mlp1')
    draw_arrow('mlp1', 'mlp2')
    draw_arrow('mlp2', 'out')

    plt.savefig('swint_arch.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Successfully generated swint_arch.png")

def generate_dataset_plots():
    """Generates dataset distribution plots (Figures 2 and 3 in the paper)."""
    print("Generating Figure 2 & 3: dataset_report_p1_temporal.png & dataset_report_p2_calendar.png...")
    # Invoke visualize_dataset.py directly as a subprocess to avoid any state pollution
    cmd = [sys.executable, 'visualize_dataset.py', '--dir', 'dataset_512', '--out', 'dataset_report']
    subprocess.run(cmd, check=True)
    print("Successfully generated dataset plots")

def generate_swint_fold_mae():
    """Generates swint_fold_mae.png (Figure 4 in the paper)."""
    print("Generating Figure 4: swint_fold_mae.png...")
    folds = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']
    maes = [51.98, 51.61, 55.31, 47.76, 55.60]
    mean_mae = 52.45

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(folds, maes, color=PALETTE["accent"], alpha=0.85, width=0.45, edgecolor='black', linewidth=1)
    ax.axhline(mean_mae, color='#d32f2f', linestyle='--', linewidth=1.5, label=f'Mean MAE ({mean_mae:.2f} min)')
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f} m',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Validation MAE (minutes)', fontsize=11, fontweight='bold')
    ax.set_title('Swin-T Validation MAE by Fold', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylim(0, 65)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', fontsize=10)
    fig.tight_layout()
    plt.savefig('swint_fold_mae.png', dpi=300)
    plt.close()
    print("Successfully generated swint_fold_mae.png")

def generate_swint_scatter():
    """Generates swint_scatter.png (Figure 5 in the paper)."""
    print("Generating Figure 5: swint_scatter.png...")
    actuals, preds = load_python_data('checkpoints/train_log_swin_t.jsonl')
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    
    # Clean scatter plot
    ax.scatter(actuals, preds, alpha=0.4, s=15, c='#4c72b0', edgecolors='none', label='Predictions')
    
    # Perfect diagonal line (red dashed)
    ax.plot([0, 24], [0, 24], color='#d32f2f', linestyle='--', linewidth=2.0, label='Perfect prediction')
    
    # ±1-hour margins of error (green dotted)
    ax.plot([0, 23], [1, 24], color='#388e3c', linestyle=':', linewidth=1.5, label='±1-hour margin')
    ax.plot([1, 24], [0, 23], color='#388e3c', linestyle=':', linewidth=1.5)
    
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 24)
    ax.set_xticks(np.arange(0, 25, 3))
    ax.set_yticks(np.arange(0, 25, 3))
    ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 25, 3)], fontsize=9)
    ax.set_yticklabels([f'{h:02d}:00' for h in range(0, 25, 3)], fontsize=9)
    
    ax.set_xlabel('Actual Time (hours)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Predicted Time (hours)', fontsize=11, fontweight='bold')
    ax.set_title('Swin-T Predicted vs. Actual Capture Times', fontsize=13, fontweight='bold', pad=15)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', fontsize=10)
    fig.tight_layout()
    plt.savefig('swint_scatter.png', dpi=300)
    plt.close()
    print("Successfully generated swint_scatter.png")

def generate_swint_examples():
    """Generates swint_examples.png (Figure 6 in the paper)."""
    print("Generating Figure 6: swint_examples.png...")
    # Group predictions and find representative images with very low error
    from PIL import ImageOps
    
    categories = {
        'Morning': {'min': 450, 'max': 570},
        'Midday': {'min': 660, 'max': 780},
        'Evening/Dusk': {'min': 1110, 'max': 1200},
        'Night': {'min': 1350, 'max': 1430}
    }
    
    def get_cat(m):
        if 450 <= m < 570: return 'Morning'
        if 660 <= m < 780: return 'Midday'
        if 1110 <= m < 1200: return 'Evening/Dusk'
        if 1350 <= m < 1430: return 'Night'
        return None

    records = []
    log_path = 'checkpoints/train_log_swin_t.jsonl'
    if not os.path.exists(log_path):
        print(f"Error: log file {log_path} not found.")
        return
        
    with open(log_path, 'r') as f:
        for line in f:
            rec = json.loads(line)
            if rec.get('type') == 'image':
                if os.path.exists(rec['path']):
                    err = _cyclic_diff(rec['pred_min'], rec['actual_min'])
                    cat = get_cat(rec['actual_min'])
                    if cat:
                        records.append((cat, err, rec))

    best_examples = {}
    for cat in categories.keys():
        cat_recs = [r for r in records if r[0] == cat]
        if cat_recs:
            # Sort by error ascending and pick the best one
            cat_recs.sort(key=lambda x: x[1])
            best_examples[cat] = cat_recs[0][2]
        else:
            print(f"Warning: No examples found for category {cat}")

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    
    cat_order = ['Morning', 'Midday', 'Evening/Dusk', 'Night']
    for i, cat in enumerate(cat_order):
        ax = axes[i]
        rec = best_examples.get(cat)
        if rec:
            img = PILImage.open(rec['path'])
            img = ImageOps.exif_transpose(img)
            ax.imshow(img)
            ax.axis('off')
            
            def fmt_time(m):
                h = int(m // 60) % 24
                mn = int(m % 60)
                return f"{h:02d}:{mn:02d}"
                
            actual_str = fmt_time(rec['actual_min'])
            pred_str = fmt_time(rec['pred_min'])
            err_val = _cyclic_diff(rec['pred_min'], rec['actual_min'])
            err_str = f"{err_val:.1f} min"
            
            ax.set_title(f"{cat}\nActual: {actual_str} | Pred: {pred_str}\nError: {err_str}", 
                         fontsize=12, fontweight='bold', pad=8)
        else:
            ax.axis('off')
            ax.text(0.5, 0.5, f"No image for {cat}", ha='center', va='center')
            
    fig.suptitle('Swin-T Representative Predictions under Various Conditions', fontsize=15, fontweight='bold', y=0.98)
    fig.tight_layout()
    plt.savefig('swint_examples.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Successfully generated swint_examples.png")

def main():
    setup_matplotlib_style()
    
    # Run all generator functions
    generate_swint_arch()
    generate_dataset_plots()
    generate_swint_fold_mae()
    generate_swint_scatter()
    generate_swint_examples()
    
    # Ensure copies are in figures/ directory
    os.makedirs('figures', exist_ok=True)
    generated_files = [
        'swint_arch.png',
        'dataset_report_p1_temporal.png',
        'dataset_report_p2_calendar.png',
        'swint_fold_mae.png',
        'swint_scatter.png',
        'swint_examples.png'
    ]
    for filename in generated_files:
        if os.path.exists(filename):
            shutil.copy2(filename, os.path.join('figures', filename))
            print(f"Copied {filename} to figures/")

if __name__ == '__main__':
    main()
