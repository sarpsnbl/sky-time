import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.makedirs('figures', exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 10))
ax.set_xlim(0, 10)
ax.set_ylim(-1, 12)
ax.axis('off')

w, h = 4.0, 0.9

nodes = {
    'img': (2.5, 11, 'Sky Image\n(512×512×3)', '#ffffe0'),
    'meta': (7.5, 11, 'Calendar Metadata\n(83 dims)', '#ffffe0'),
    'convnext': (2.5, 9, 'ConvNeXt-Tiny Backbone\n(Frozen until features.4)', '#f08080'),
    'pool': (2.5, 7, 'Global Average Pooling\n(768 dims)', '#add8e6'),
    'concat': (5, 5, 'Concatenate\n(851 dims)', '#d3d3d3'),
    'mlp1': (5, 3.2, 'FC(384) → LN → GELU → Dropout', '#90ee90'),
    'mlp2': (5, 1.6, 'FC(384) → LN → GELU → Dropout', '#90ee90'),
    'out': (5, 0, 'FC(2) → Regression Layer\n[sin, cos]', '#ffffe0')
}

for name, (x, y, text, color) in nodes.items():
    box = patches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                 boxstyle="round,pad=0.1,rounding_size=0.2",
                                 edgecolor="black", facecolor=color, zorder=2)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=11, zorder=3, fontweight='bold')

def draw_arrow(n1, n2):
    x1, y1 = nodes[n1][0], nodes[n1][1] - h/2 - 0.1
    x2, y2 = nodes[n2][0], nodes[n2][1] + h/2 + 0.1
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), 
                arrowprops=dict(arrowstyle="->", lw=2), zorder=1)

draw_arrow('img', 'convnext')
draw_arrow('convnext', 'pool')
draw_arrow('pool', 'concat')
draw_arrow('meta', 'concat')
draw_arrow('concat', 'mlp1')
draw_arrow('mlp1', 'mlp2')
draw_arrow('mlp2', 'out')

plt.savefig('figures/python_arch.png', dpi=300, bbox_inches='tight')
print("Successfully generated figures/python_arch.png using matplotlib")
