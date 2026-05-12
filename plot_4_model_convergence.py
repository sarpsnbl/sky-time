import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs('figures', exist_ok=True)

# Generate dummy convergence data
# TODO: Replace with real training logs
epochs_python = np.arange(1, 81)
epochs_matlab = np.arange(1, 31)

# Simulating validation MAE curves
val_swint = 150 * np.exp(-epochs_python / 15) + 50 + np.random.normal(0, 2, 80)
val_convnext = 150 * np.exp(-epochs_python / 12) + 55 + np.random.normal(0, 2.5, 80)
val_squeezenet = 180 * np.exp(-epochs_matlab / 8) + 80 + np.random.normal(0, 3, 30)

# Statistical ML doesn't train over epochs, so we represent it as a flat baseline
# TODO: Replace with actual Statistical ML MAE
stat_ml_baseline = 77.75

fig, ax = plt.subplots(figsize=(10, 6))

# Plot Deep Learning models
ax.plot(epochs_python, val_swint, label='Python Swin-T', color='#2ca02c', linewidth=2)
ax.plot(epochs_python, val_convnext, label='Python ConvNeXt-Tiny', color='#4c72b0', linewidth=2)
ax.plot(epochs_matlab, val_squeezenet, label='MATLAB SqueezeNet', color='#dd8452', linewidth=2)

# Extend SqueezeNet's final value to epoch 80 with a dashed line to prevent it looking abruptly cut off
ax.plot([30, 80], [val_squeezenet[-1], val_squeezenet[-1]], color='#dd8452', linestyle='--', linewidth=1.5, alpha=0.7)

# Plot Statistical ML baseline
ax.axhline(y=stat_ml_baseline, color='#c44e52', linestyle='-.', linewidth=2, label='MATLAB Statistical ML (Baseline)')

ax.set_xlabel('Epochs', fontsize=12)
ax.set_ylabel('Validation MAE (minutes)', fontsize=12)
ax.set_title('Training Convergence Comparison (4 Models)', fontsize=14, pad=15)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(fontsize=11)

# Set x-limits to the maximum epochs
ax.set_xlim(1, 80)

fig.tight_layout()
plt.savefig('figures/comparison_4models_convergence.png', dpi=300, bbox_inches='tight')
print("Successfully generated figures/comparison_4models_convergence.png")
