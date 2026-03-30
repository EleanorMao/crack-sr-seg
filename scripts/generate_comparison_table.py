"""
Generate comparison table from record.md data
"""
import os
import matplotlib.pyplot as plt
import numpy as np

# Data from record.md
# SRCNN Super-Resolution Results (Test split)
srcnn_results = {
    'SRCNN (Baseline)': {
        'PSNR': 27.41,
        'SSIM': 0.9076,
    },
    'ImprovedSRCNN': {
        'PSNR': 30.09,
        'SSIM': 0.9482,
    },
    'ImprovedSRCNN all 3x3': {
        'PSNR': 30.81,
        'SSIM': 0.9545,
    },
}

# U-Net Segmentation Results (Test split)
unet_results = {
    'Bilinear + U-Net': {
        'IoU': 36.96,
        'Dice': 49.41,
        'Accuracy': 95.70,
    },
    'Bicubic + U-Net': {
        'IoU': 36.96,
        'Dice': 49.41,
        'Accuracy': 95.70,
    },
    'No Restoration + U-Net': {
        'IoU': 59.55,
        'Dice': 72.87,
        'Accuracy': 97.20,
    },
    'SRCNN + U-Net (Ours)': {
        'IoU': 86.84,
        'Dice': 92.47,
        'Accuracy': 99.44,
    },
    'ImprovedSRCNN + U-Net': {
        'IoU': 87.28,
        'Dice': 92.67,
        'Accuracy': 99.47,
    },
    'ImprovedSRCNN all 3x3 + U-Net': {
        'IoU': 87.21,
        'Dice': 92.61,
        'Accuracy': 99.47,
    },
}

# Create output directory
os.makedirs('figures', exist_ok=True)

# ============================================================
# Print SRCNN LaTeX table
# ============================================================
print("=" * 70)
print("SRCNN Super-Resolution Results - LaTeX Table")
print("=" * 70)
print("""
\\begin{table}[h]
\\centering
\\caption{Comparison of different SRCNN models on super-resolution}
\\label{tab:srcnn_comparison}
\\begin{tabular}{lcc}
\\hline
Method & PSNR (dB) & SSIM \\\\
\\hline""")
for method, metrics in srcnn_results.items():
    print(f"{method} & {metrics['PSNR']:.2f} & {metrics['SSIM']:.4f} \\\\")
print("""\\hline
\\end{tabular}
\\end{table}
""")

# ============================================================
# Print U-Net LaTeX table
# ============================================================
print("=" * 70)
print("U-Net Segmentation Results - LaTeX Table")
print("=" * 70)
print("""
\\begin{table}[h]
\\centering
\\caption{Comparison of different methods on crack segmentation}
\\label{tab:comparison}
\\begin{tabular}{lccc}
\\hline
Method & IoU (\\%) & Dice (\\%) & Accuracy (\\%) \\\\
\\hline""")
for method, metrics in unet_results.items():
    print(f"{method} & {metrics['IoU']:.2f} & {metrics['Dice']:.2f} & {metrics['Accuracy']:.2f} \\\\")
print("""\\hline
\\end{tabular}
\\end{table}
""")

# ============================================================
# Print Markdown tables
# ============================================================
print("=" * 70)
print("SRCNN Super-Resolution Results - Markdown Table")
print("=" * 70)
print("""
| Method | PSNR (dB) | SSIM |
|--------|-----------|------|""")
for method, metrics in srcnn_results.items():
    print(f"| {method} | {metrics['PSNR']:.2f} | {metrics['SSIM']:.4f} |")

print("\n" + "=" * 70)
print("U-Net Segmentation Results - Markdown Table")
print("=" * 70)
print("""
| Method | IoU (%) | Dice (%) | Accuracy (%) |
|--------|---------|----------|--------------|""")
for method, metrics in unet_results.items():
    print(f"| {method} | {metrics['IoU']:.2f} | {metrics['Dice']:.2f} | {metrics['Accuracy']:.2f} |")

# ============================================================
# Generate SRCNN comparison bar chart
# ============================================================
print("\n" + "=" * 70)
print("Generating SRCNN comparison chart...")
print("=" * 70)

srcnn_methods = list(srcnn_results.keys())
srcnn_short_names = ['SRCNN', 'Improved', 'Improved 3x3']
psnr_values = [srcnn_results[m]['PSNR'] for m in srcnn_methods]
ssim_values = [srcnn_results[m]['SSIM'] * 100 for m in srcnn_methods]  # Scale for visibility

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(srcnn_methods))
width = 0.35

bars1 = ax.bar(x - width/2, psnr_values, width, label='PSNR (dB)', color='#3498db')
bars2 = ax.bar(x + width/2, ssim_values, width, label='SSIM (x100)', color='#2ecc71')

ax.set_ylabel('Score', fontsize=14)
ax.set_title('SRCNN Model Comparison on Super-Resolution', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(srcnn_short_names, fontsize=12)
ax.legend(fontsize=12)
ax.set_ylim(0, 110)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('figures/srcnn_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: figures/srcnn_comparison.png")

# ============================================================
# Generate U-Net comparison bar chart
# ============================================================
print("\n" + "=" * 70)
print("Generating U-Net comparison chart...")
print("=" * 70)

unet_methods = list(unet_results.keys())
unet_short_names = ['Bilinear', 'Bicubic', 'No Restore', 'SRCNN', 'Improved', 'Improved 3x3']
iou_values = [unet_results[m]['IoU'] for m in unet_methods]
dice_values = [unet_results[m]['Dice'] for m in unet_methods]
acc_values = [unet_results[m]['Accuracy'] for m in unet_methods]

x = np.arange(len(unet_methods))
width = 0.25

fig, ax = plt.subplots(figsize=(16, 6))
bars1 = ax.bar(x - width, iou_values, width, label='IoU', color='#3498db')
bars2 = ax.bar(x, dice_values, width, label='Dice', color='#2ecc71')
bars3 = ax.bar(x + width, acc_values, width, label='Accuracy', color='#e74c3c')

ax.set_ylabel('Score (%)', fontsize=14)
ax.set_title('Method Comparison on Crack Segmentation', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(unet_short_names, fontsize=11, rotation=15)
ax.legend(fontsize=12)
ax.set_ylim(0, 110)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('figures/methods_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: figures/methods_comparison.png")

# ============================================================
# Generate IoU comparison chart (most important metric)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Left: IoU comparison
colors_iou = ['#95a5a6', '#95a5a6', '#e74c3c', '#27ae60', '#3498db', '#9b59b6']
bars = axes[0].bar(unet_short_names, iou_values, color=colors_iou)
axes[0].set_ylabel('IoU (%)', fontsize=14)
axes[0].set_title('IoU Comparison', fontsize=16)
axes[0].set_ylim(0, 100)
axes[0].tick_params(axis='x', rotation=15)
for bar, val in zip(bars, iou_values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')

# Right: Improvement over Bilinear baseline
baseline_iou = iou_values[0]  # Bilinear
improvements = [(iou - baseline_iou) / baseline_iou * 100 for iou in iou_values]
colors_imp = ['gray', 'gray', '#e74c3c', '#27ae60', '#3498db', '#9b59b6']
bars = axes[1].bar(unet_short_names, improvements, color=colors_imp)
axes[1].set_ylabel('Improvement over Bilinear (%)', fontsize=14)
axes[1].set_title('Relative Improvement', fontsize=16)
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[1].tick_params(axis='x', rotation=15)
for bar, val in zip(bars, improvements):
    y_pos = bar.get_height() + 1 if val >= 0 else bar.get_height() - 5
    axes[1].text(bar.get_x() + bar.get_width()/2, y_pos,
                 f'{val:+.1f}%', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/methods_comparison_detailed.png', dpi=300, bbox_inches='tight')
print("Saved: figures/methods_comparison_detailed.png")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)

best_iou_idx = np.argmax(iou_values)
best_method = unet_methods[best_iou_idx]
best_metrics = unet_results[best_method]

# Bilinear baseline
bilinear_iou = unet_results['Bilinear + U-Net']['IoU']

print(f"""
SRCNN Super-Resolution:
  - SRCNN Baseline: PSNR {srcnn_results['SRCNN (Baseline)']['PSNR']:.2f} dB, SSIM {srcnn_results['SRCNN (Baseline)']['SSIM']:.4f}
  - ImprovedSRCNN all 3x3: PSNR {srcnn_results['ImprovedSRCNN all 3x3']['PSNR']:.2f} dB, SSIM {srcnn_results['ImprovedSRCNN all 3x3']['SSIM']:.4f}
  - Improvement: +{srcnn_results['ImprovedSRCNN all 3x3']['PSNR'] - srcnn_results['SRCNN (Baseline)']['PSNR']:.2f} dB PSNR

Best U-Net Method: {best_method}
  - IoU: {best_metrics['IoU']:.2f}% (+{(best_metrics['IoU'] - bilinear_iou) / bilinear_iou * 100:.1f}% vs Bilinear)
  - Dice: {best_metrics['Dice']:.2f}%
  - Accuracy: {best_metrics['Accuracy']:.2f}%

Key Findings:
  1. Simple interpolation (Bilinear/Bicubic) achieves only {bilinear_iou:.2f}% IoU
  2. SRCNN restoration improves IoU from {bilinear_iou:.2f}% to {best_metrics['IoU']:.2f}% (+{(best_metrics['IoU'] - bilinear_iou) / bilinear_iou * 100:.1f}% relative improvement)
  3. Learning-based SR outperforms traditional interpolation by a large margin
  4. ImprovedSRCNN variants provide marginal improvements over basic SRCNN
""")
