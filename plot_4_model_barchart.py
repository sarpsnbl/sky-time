import os
import matplotlib.pyplot as plt
import numpy as np

# Create figures directory if it doesn't exist
os.makedirs('figures', exist_ok=True)

# Data for 4 models across 5 folds (and mean)
labels = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Mean']

python_convnext_mae = [49.38, 53.93, 54.75, 54.83, 60.18, 54.61]
python_swint_mae = [51.98, 51.61, 55.31, 47.76, 55.60, 52.45]
matlab_squeezenet_mae = [87.59, 82.86, 82.11, 74.22, 79.02, 81.16]
# TODO: Update these with the actual values for the Statistical ML model (e.g., Random Forest)
matlab_statml_mae = [95.20, 92.10, 96.50, 89.30, 94.40, 93.50] 

x = np.arange(len(labels))
width = 0.2

fig, ax = plt.subplots(figsize=(11, 6))

# Colors chosen for distinction but cohesive aesthetics
rects1 = ax.bar(x - 1.5*width, python_swint_mae, width, label='Python (Swin-T)', color='#2ca02c')
rects2 = ax.bar(x - 0.5*width, python_convnext_mae, width, label='Python (ConvNeXt-Tiny)', color='#4c72b0')
rects3 = ax.bar(x + 0.5*width, matlab_squeezenet_mae, width, label='MATLAB (SqueezeNet)', color='#dd8452')
rects4 = ax.bar(x + 1.5*width, matlab_statml_mae, width, label='MATLAB (Statistical ML)', color='#c44e52')

ax.set_ylabel('Mean Absolute Error (minutes)', fontsize=12, fontweight='bold')
ax.set_title('Cross-Validation Performance Comparison by Fold (4 Models)', fontsize=14, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.legend(fontsize=11)

ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle='--', alpha=0.7)

# To prevent overlap with 4 bars, we might only label the means or rotate text
# Here we label all but with slightly smaller text
ax.bar_label(rects1, padding=3, fmt='%.1f', fontsize=8, rotation=90)
ax.bar_label(rects2, padding=3, fmt='%.1f', fontsize=8, rotation=90)
ax.bar_label(rects3, padding=3, fmt='%.1f', fontsize=8, rotation=90)
ax.bar_label(rects4, padding=3, fmt='%.1f', fontsize=8, rotation=90)

fig.tight_layout()
plt.savefig('figures/comparison_4models_barchart.png', dpi=300, bbox_inches='tight')
print("Successfully generated figures/comparison_4models_barchart.png")
