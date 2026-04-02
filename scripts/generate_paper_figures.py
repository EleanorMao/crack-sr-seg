#!/usr/bin/env python3
"""
Generate paper figures for crack detection paper
"""

import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random


def create_sample_visualization(
    sr_dir='outputs/restored_improved/test',
    pred_dir='outputs/predictions_improved',
    mask_dir='dataset/masks',
    output_path='figures/sample_visualization.png',
    num_samples=3
):
    """Create sample visualization showing SR -> Prediction -> Ground Truth"""

    # Get list of restored images
    sr_files = sorted([f for f in os.listdir(sr_dir) if f.endswith('.png')])

    # Randomly select samples
    selected = random.sample(sr_files, min(num_samples, len(sr_files)))

    fig, axes = plt.subplots(len(selected), 4, figsize=(12, 3 * len(selected)))

    if len(selected) == 1:
        axes = axes.reshape(1, -1)

    for i, img_name in enumerate(selected):
        # Load SR image
        sr_path = os.path.join(sr_dir, img_name)
        sr_img = np.array(Image.open(sr_path).convert('RGB'))

        # Load prediction
        pred_path = os.path.join(pred_dir, img_name)
        if os.path.exists(pred_path):
            pred_img = np.array(Image.open(pred_path))
            # Normalize and apply colormap
            pred_norm = pred_img.astype(float) / 255.0
            pred_colored = plt.cm.get_cmap('Reds')(pred_norm)[:, :, :3]
            pred_colored = (pred_colored * 255).astype(np.uint8)
        else:
            pred_colored = np.zeros((256, 256, 3), dtype=np.uint8)

        # Load ground truth mask
        mask_path = os.path.join(mask_dir, img_name)
        if os.path.exists(mask_path):
            mask_img = np.array(Image.open(mask_path).convert('L'))
            mask_norm = mask_img.astype(float) / 255.0
            mask_colored = plt.cm.get_cmap('Greens')(mask_norm)[:, :, :3]
            mask_colored = (mask_colored * 255).astype(np.uint8)
        else:
            mask_colored = np.zeros((256, 256, 3), dtype=np.uint8)

        # Create overlay (prediction on SR image)
        overlay = sr_img.copy()
        pred_mask = pred_img > 127 if os.path.exists(pred_path) else np.zeros((256, 256), dtype=bool)
        if pred_mask.any():
            overlay[pred_mask] = [255, 0, 0]  # Red for prediction

        # Plot
        axes[i, 0].imshow(sr_img)
        axes[i, 0].set_title('Super-Resolution', fontsize=11)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(pred_colored)
        axes[i, 1].set_title('Crack Prediction', fontsize=11)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(mask_colored)
        axes[i, 2].set_title('Ground Truth', fontsize=11)
        axes[i, 2].axis('off')

        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title('Overlay Result', fontsize=11)
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_sr_comparison(
    sr_dirs={
        'Basic SRCNN': 'outputs/restored/test',
        'Improved SRCNN': 'outputs/restored_improved/test',
    },
    pred_dirs={
        'Basic SRCNN': 'outputs/predictions_restored',
        'Improved SRCNN': 'outputs/predictions_improved',
    },
    mask_dir='dataset/masks',
    output_path='figures/sr_comparison.png',
    num_samples=2
):
    """Create comparison between different SRCNN models"""

    # Get common images
    first_dir = list(sr_dirs.values())[0]
    common_files = set(os.listdir(first_dir))

    for sr_dir in sr_dirs.values():
        common_files = common_files & set(os.listdir(sr_dir))

    common_files = sorted([f for f in common_files if f.endswith('.png')])
    selected = random.sample(common_files, min(num_samples, len(common_files)))

    n_cols = 1 + len(sr_dirs) * 2 + 1  # SR images + predictions + GT
    fig, axes = plt.subplots(len(selected), n_cols, figsize=(2.5 * n_cols, 2.5 * len(selected)))

    if len(selected) == 1:
        axes = axes.reshape(1, -1)

    for i, img_name in enumerate(selected):
        col = 0

        # SR images and predictions
        for name, sr_dir in sr_dirs.items():
            # SR image
            sr_img = np.array(Image.open(os.path.join(sr_dir, img_name)).convert('RGB'))
            axes[i, col].imshow(sr_img)
            axes[i, col].set_title(f'{name}\n(SR)', fontsize=9)
            axes[i, col].axis('off')
            col += 1

            # Prediction
            pred_path = os.path.join(pred_dirs[name], img_name)
            if os.path.exists(pred_path):
                pred_img = np.array(Image.open(pred_path))
                pred_norm = pred_img.astype(float) / 255.0
                pred_colored = plt.cm.get_cmap('Reds')(pred_norm)[:, :, :3]
                pred_colored = (pred_colored * 255).astype(np.uint8)
                axes[i, col].imshow(pred_colored)
            else:
                axes[i, col].text(0.5, 0.5, 'N/A', ha='center')
            axes[i, col].set_title(f'{name}\n(Pred)', fontsize=9)
            axes[i, col].axis('off')
            col += 1

        # Ground Truth
        mask_path = os.path.join(mask_dir, img_name)
        if os.path.exists(mask_path):
            mask_img = np.array(Image.open(mask_path).convert('L'))
            mask_norm = mask_img.astype(float) / 255.0
            mask_colored = plt.cm.get_cmap('Greens')(mask_norm)[:, :, :3]
            mask_colored = (mask_colored * 255).astype(np.uint8)
            axes[i, col].imshow(mask_colored)
        else:
            axes[i, col].text(0.5, 0.5, 'N/A', ha='center')
        axes[i, col].set_title('Ground\nTruth', fontsize=9)
        axes[i, col].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_method_comparison(
    output_path='figures/method_comparison.png',
    num_samples=2
):
    """Create comparison: Original HR vs Bilinear vs SRCNN"""

    # Use train data since that's what we have in processed_data
    hr_dir = 'processed_data/hr_images/train'
    pred_improved = 'outputs/predictions_improved'
    mask_dir = 'dataset/masks'

    hr_files = sorted([f for f in os.listdir(hr_dir) if f.endswith('.png') or f.endswith('.jpg')])

    # Get predictions
    pred_files = set(os.listdir(pred_improved))

    # Find common files
    common_files = []
    for f in hr_files:
        if f in pred_files:
            common_files.append(f)

    selected = random.sample(common_files, min(num_samples, len(common_files)))

    fig, axes = plt.subplots(len(selected), 4, figsize=(12, 3 * len(selected)))

    if len(selected) == 1:
        axes = axes.reshape(1, -1)

    for i, img_name in enumerate(selected):
        # Original HR
        hr_img = np.array(Image.open(os.path.join(hr_dir, img_name)).convert('RGB'))
        axes[i, 0].imshow(hr_img)
        axes[i, 0].set_title('Original HR', fontsize=11)
        axes[i, 0].axis('off')

        # SR + Prediction (Improved SRCNN)
        pred_path = os.path.join(pred_improved, img_name)
        if os.path.exists(pred_path):
            pred_img = np.array(Image.open(pred_path))
            pred_norm = pred_img.astype(float) / 255.0
            pred_colored = plt.cm.get_cmap('Reds')(pred_norm)[:, :, :3]
            pred_colored = (pred_colored * 255).astype(np.uint8)
            axes[i, 1].imshow(pred_colored)
        else:
            axes[i, 1].text(0.5, 0.5, 'N/A', ha='center')
        axes[i, 1].set_title('SRCNN Prediction', fontsize=11)
        axes[i, 1].axis('off')

        # Ground Truth
        mask_path = os.path.join(mask_dir, img_name)
        if os.path.exists(mask_path):
            mask_img = np.array(Image.open(mask_path).convert('L'))
            mask_norm = mask_img.astype(float) / 255.0
            mask_colored = plt.cm.get_cmap('Greens')(mask_norm)[:, :, :3]
            mask_colored = (mask_colored * 255).astype(np.uint8)
            axes[i, 2].imshow(mask_colored)
        else:
            axes[i, 2].text(0.5, 0.5, 'N/A', ha='center')
        axes[i, 2].set_title('Ground Truth', fontsize=11)
        axes[i, 2].axis('off')

        # Overlay
        overlay = hr_img.copy()
        if os.path.exists(pred_path):
            pred_mask = np.array(Image.open(pred_path)) > 127
            overlay[pred_mask] = [255, 0, 0]
        if os.path.exists(mask_path):
            gt_mask = np.array(Image.open(mask_path).convert('L')) > 127
            overlay[gt_mask] = [0, 255, 0]
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title('Overlay (Red=Pred, Green=GT)', fontsize=10)
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate paper figures')
    parser.add_argument('--type', type=str, default='all',
                        choices=['sample', 'sr', 'method', 'all'],
                        help='Type of figure to generate')
    parser.add_argument('--num-samples', type=int, default=3,
                        help='Number of samples to show')

    args = parser.parse_args()

    os.makedirs('figures', exist_ok=True)

    if args.type in ['sample', 'all']:
        print("Generating sample visualization...")
        create_sample_visualization(num_samples=args.num_samples)

    if args.type in ['sr', 'all']:
        print("Generating SR comparison...")
        create_sr_comparison(num_samples=min(2, args.num_samples))

    if args.type in ['method', 'all']:
        print("Generating method comparison...")
        create_method_comparison(num_samples=min(2, args.num_samples))

    print("\nDone! Figures saved to figures/")
