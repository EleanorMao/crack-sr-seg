#!/usr/bin/env python3
"""
Generate better sample visualization for paper
"""
import os
import cv2
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
    # Get list of restored images
    sr_files = sorted([f for f in os.listdir(sr_dir) if f.endswith('.png')])

    # Randomly select samples
    random.seed(42)  # For reproducibility
    selected = random.sample(sr_files, min(num_samples, len(sr_files)))

    fig, axes = plt.subplots(len(selected), 4, figsize=(12, 3 * len(selected)))

    if len(selected) == 1:
        axes = axes.reshape(1, -1)

    for i, img_name in enumerate(selected):
        # Load SR image
        sr_path = os.path.join(sr_dir, img_name)
        sr_img = np.array(Image.open(sr_path).convert('RGB'))

        # Load prediction (files have _pred suffix)
        base_name = img_name.replace('.png', '')
        pred_path = os.path.join(pred_dir, base_name + '_pred.png')
        if os.path.exists(pred_path):
            pred_img = np.array(Image.open(pred_path))
            # Handle different image formats
            if len(pred_img.shape) == 3:
                if pred_img.shape[1] > 256:  # Side-by-side format
                    pred_img = pred_img[:, :256, 0]  # Take first channel of first image
                else:
                    pred_img = pred_img[:, :, 0]
        else:
            pred_img = np.zeros((256, 256), dtype=np.uint8)

        # Load ground truth mask
        mask_path = os.path.join(mask_dir, img_name)
        if os.path.exists(mask_path):
            mask_img = np.array(Image.open(mask_path).convert('L'))
            mask_img = cv2.resize(mask_img, (256, 256))
        else:
            mask_img = np.zeros((256, 256), dtype=np.uint8)

        # Create overlay (prediction on SR image)
        overlay = sr_img.copy()
        pred_mask = pred_img > 127
        overlay[pred_mask] = [0, 255, 0]  # Green for prediction

        # Plot
        axes[i, 0].imshow(sr_img)
        axes[i, 0].set_title('Super-Resolution', fontsize=11)
        axes[i, 0].axis('off')

        # Prediction: invert for better visibility (black crack on white background)
        pred_display = 255 - pred_img
        axes[i, 1].imshow(pred_display, cmap='gray', vmin=0, vmax=255)
        axes[i, 1].set_title('Crack Prediction', fontsize=11)
        axes[i, 1].axis('off')

        # Ground truth: white crack on black background
        axes[i, 2].imshow(mask_img, cmap='gray', vmin=0, vmax=255)
        axes[i, 2].set_title('Ground Truth', fontsize=11)
        axes[i, 2].axis('off')

        # Overlay
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title('Overlay Result', fontsize=11)
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    os.makedirs('figures', exist_ok=True)
    create_sample_visualization(num_samples=3)
