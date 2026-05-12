import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs('figures', exist_ok=True)

# Generate dummy error data (absolute errors in minutes)
# TODO: Load actual error arrays from your logs/CSVs
np.random.seed(42)
n_samples = 500

# Using exponential distributions to mimic real absolute errors (lots of small errors, long tail of large errors)
errors_swint = np.random.exponential(scale=52, size=n_samples)
errors_convnext = np.random.exponential(scale=55, size=n_samples)
errors_squeezenet = np.random.exponential(scale=81, size=n_samples)
errors_statml = np.random.exponential(scale=95, size=n_samples)

data = [errors_swint, errors_convnext, errors_squeezenet, errors_statml]
labels = ['Python\nSwin-T', 'Python\nConvNeXt', 'MATLAB\nSqueezeNet', 'MATLAB\nStat. ML']

fig, ax = plt.subplots(figsize=(10, 6))

# Create violin plot
parts = ax.violinplot(data, showmeans=True, showmedians=False, showextrema=True)

# Color customization
colors = ['#2ca02c', '#4c72b0', '#dd8452', '#c44e52']
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_edgecolor('black')
    pc.set_alpha(0.6)

# Make the lines black for contrast
for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
    vp = parts[partname]
    vp.set_edgecolor('black')
    vp.set_linewidth(1.5)

ax.set_xticks(np.arange(1, len(labels) + 1))
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel('Absolute Error (minutes)', fontsize=12, fontweight='bold')
ax.set_title('Error Distribution Comparison Across Models', fontsize=14, pad=15)
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

fig.tight_layout()
plt.savefig('figures/comparison_4models_distribution.png', dpi=300, bbox_inches='tight')
print("Successfully generated figures/comparison_4models_distribution.png")
