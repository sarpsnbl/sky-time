import os
import matplotlib.pyplot as plt
import numpy as np

# Create figures directory if it doesn't exist
os.makedirs('figures', exist_ok=True)

# Data from Tables 4 and 5
labels = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5', 'Mean']
python_mae = [49.38, 53.93, 54.75, 54.83, 60.18, 54.61]
matlab_mae = [87.59, 82.86, 82.11, 74.22, 79.02, 81.16]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5.5))
rects1 = ax.bar(x - width/2, python_mae, width, label='Python (ConvNeXt-Tiny)', color='#4c72b0')
rects2 = ax.bar(x + width/2, matlab_mae, width, label='MATLAB (SqueezeNet)', color='#dd8452')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Mean Absolute Error (minutes)', fontsize=12, fontweight='bold')
ax.set_title('Cross-Validation Performance Comparison by Fold', fontsize=14, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.legend(fontsize=11)

# Add gridlines behind the bars for better readability
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle='--', alpha=0.7)

# Attach a text label above each bar, displaying its height.
ax.bar_label(rects1, padding=3, fmt='%.1f', fontsize=10)
ax.bar_label(rects2, padding=3, fmt='%.1f', fontsize=10)

fig.tight_layout()
plt.savefig('figures/comparison_barchart.png', dpi=300, bbox_inches='tight')
print("Successfully generated figures/comparison_barchart.png")
