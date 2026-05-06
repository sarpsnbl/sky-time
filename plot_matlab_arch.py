import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.makedirs('figures', exist_ok=True)

fig, ax = plt.subplots(figsize=(9, 11))
ax.set_xlim(0, 10)
ax.set_ylim(-1, 13)
ax.axis('off')

w, h = 4.2, 0.9

nodes = {
    'img': (2.5, 12, 'Sky Image\n(224×224×3)', '#ffffe0'),
    'date': (7.5, 12, 'Date Encoding\n(4 dims)', '#ffffe0'),
    
    'squeezenet': (2.5, 10, 'SqueezeNet\n(fire1-6 frozen, fire7-9 fine-tuned)', '#f08080'),
    'img_feat': (2.5, 8, 'Image Features\n(512 dims)', '#add8e6'),
    
    'date_fc1': (7.5, 10, 'FC(32) → BN → ReLU', '#90ee90'),
    'date_fc2': (7.5, 8.5, 'FC(64) → ReLU', '#90ee90'),
    'date_feat': (7.5, 7, 'Date Features\n(64 dims)', '#add8e6'),
    
    'concat': (5, 5, 'Concatenate\n(576 dims)', '#d3d3d3'),
    
    'head1': (5, 3.2, 'FC(256) → BN → ReLU → Dropout(0.4)', '#90ee90'),
    'head2': (5, 1.6, 'FC(64) → ReLU', '#90ee90'),
    'out': (5, 0, 'FC(1) → Regression Layer\n[Fractional Hours]', '#ffffe0')
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

draw_arrow('img', 'squeezenet')
draw_arrow('squeezenet', 'img_feat')
draw_arrow('img_feat', 'concat')

draw_arrow('date', 'date_fc1')
draw_arrow('date_fc1', 'date_fc2')
draw_arrow('date_fc2', 'date_feat')
draw_arrow('date_feat', 'concat')

draw_arrow('concat', 'head1')
draw_arrow('head1', 'head2')
draw_arrow('head2', 'out')

plt.savefig('figures/matlab_arch.png', dpi=300, bbox_inches='tight')
print("Successfully generated figures/matlab_arch.png using matplotlib")
