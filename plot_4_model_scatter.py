import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs('figures', exist_ok=True)

# Generate dummy data - TODO: Load actual prediction data from your logs/CSVs
np.random.seed(42)
n_samples = 500
actual_times = np.random.uniform(0, 24, n_samples)

# Simulate predictions with varying noise based on model MAEs
pred_swint = actual_times + np.random.normal(0, 52/60, n_samples)
pred_convnext = actual_times + np.random.normal(0, 55/60, n_samples)
pred_squeezenet = actual_times + np.random.normal(0, 81/60, n_samples)
pred_statml = actual_times + np.random.normal(0, 95/60, n_samples)

# Wrap around 24 hours
for arr in [pred_swint, pred_convnext, pred_squeezenet, pred_statml]:
    arr[arr < 0] += 24
    arr[arr >= 24] -= 24

models = [
    ('Python Swin-T', pred_swint),
    ('Python ConvNeXt-Tiny', pred_convnext),
    ('MATLAB SqueezeNet', pred_squeezenet),
    ('MATLAB Statistical ML', pred_statml)
]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, (name, preds) in enumerate(models):
    ax = axes[i]
    ax.scatter(actual_times, preds, alpha=0.3, s=15, c='blue', edgecolors='none')
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
