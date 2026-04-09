import matplotlib.pyplot as plt
import numpy as np

# Only SRCNN models have PSNR values for the scatter
srcnn_psnr = [27.41, 30.09, 30.81]
srcnn_iou = [86.84, 87.28, 87.21]
srcnn_labels = ['Basic SRCNN\n(3-layer)', 'Improved SRCNN\n(5-layer)', 'Improved 3×3\n(7-layer)']

plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'
fig, ax = plt.subplots(figsize=(7, 5))

# Scatter points
colors = ['#3498db', '#e74c3c', '#2ecc71']
for i, (x, y, label) in enumerate(zip(srcnn_psnr, srcnn_iou, srcnn_labels)):
    ax.scatter(x, y, s=180, c=colors[i], zorder=5, edgecolors='white', linewidth=1.5)
    ax.annotate(label, (x, y), textcoords="offset points", xytext=(12, -8),
                fontsize=9, fontweight='bold', color=colors[i])

# Trend line
z = np.polyfit(srcnn_psnr, srcnn_iou, 1)
p = np.poly1d(z)
x_line = np.linspace(26.5, 31.5, 100)
ax.plot(x_line, p(x_line), '--', color='#bdc3c7', alpha=0.8, linewidth=1.5,
        label=f'Linear trend (slope={z[0]:.3f})')

# Highlight the paradox: higher PSNR but lower IoU
ax.annotate('', xy=(30.81, 87.21), xytext=(30.09, 87.28),
            arrowprops=dict(arrowstyle='->', color='#e67e22', lw=2, connectionstyle='arc3,rad=0.2'))
ax.text(30.55, 87.40, 'PSNR↑ IoU↓', fontsize=9, color='#e67e22', fontweight='bold', ha='center')

ax.set_xlabel('PSNR (dB)', fontsize=12)
ax.set_ylabel('IoU (%)', fontsize=12)
ax.set_xlim(26.8, 31.5)
ax.set_ylim(86.5, 87.6)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/psnr_vs_iou.png', dpi=150, bbox_inches='tight')
print("Saved to figures/psnr_vs_iou.png")
