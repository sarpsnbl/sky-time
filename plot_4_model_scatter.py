import os
import matplotlib.pyplot as plt
import numpy as np
import json
import csv

os.makedirs('figures', exist_ok=True)

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

def load_matlab_data(file_path):
    actuals = []
    preds = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            def hhmm_to_hours(s):
                parts = s.split(':')
                return float(parts[0]) + float(parts[1]) / 60.0
            actuals.append(hhmm_to_hours(row['Actual']))
            preds.append(hhmm_to_hours(row['Guess']))
    return np.array(actuals), np.array(preds)

# Load actual prediction data
actual_swint, pred_swint = load_python_data('checkpoints/train_log_swin_t.jsonl')
actual_convnext, pred_convnext = load_python_data('checkpoints/train_log_convnext_tiny.jsonl')
actual_squeezenet, pred_squeezenet = load_matlab_data('matlab_method/results/cnn_results.csv')
actual_statml, pred_statml = load_matlab_data('matlab_method/results/ml_results.csv')

models = [
    ('Python Swin-T', actual_swint, pred_swint),
    ('Python ConvNeXt-Tiny', actual_convnext, pred_convnext),
    ('MATLAB SqueezeNet', actual_squeezenet, pred_squeezenet),
    ('MATLAB Statistical ML', actual_statml, pred_statml)
]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, (name, actuals, preds) in enumerate(models):
    ax = axes[i]
    ax.scatter(actuals, preds, alpha=0.3, s=15, c='blue', edgecolors='none')
    ax.plot([0, 24], [0, 24], 'r--', lw=2) # Perfect prediction line
    
    # Optional: plot the +/- 1 hour margin
    ax.plot([0, 23], [1, 24], 'g:', lw=1)
    ax.plot([1, 24], [0, 23], 'g:', lw=1)

    ax.set_xlim(0, 24)
    ax.set_ylim(0, 24)
    ax.set_xticks(np.arange(0, 25, 4))
    ax.set_yticks(np.arange(0, 25, 4))
    ax.set_xlabel('Actual Time (hours)')
    ax.set_ylabel('Predicted Time (hours)')
    ax.set_title(name, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)

fig.suptitle('Predicted vs Actual Time-of-Day (Cross-Model Comparison)', fontsize=16, y=1.02)
fig.tight_layout()
plt.savefig('figures/comparison_4models_scatter.png', dpi=300, bbox_inches='tight')
print("Successfully generated figures/comparison_4models_scatter.png")
