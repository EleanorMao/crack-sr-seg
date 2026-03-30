"""
Visualization Script for Paper
Generate all figures needed for the paper
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from matplotlib.gridspec import GridSpec
import json
import torch
from pathlib import Path
from tqdm import tqdm

from config import (
    CHECKPOINT_DIR, LR_IMAGE_DIR, HR_IMAGE_DIR,
    ENHANCED_MASK_DIR, RESTORED_DIR,
    RESTORED_DIR_IMPROVED, RESTORED_DIR_IMPROVED_BN, RESTORED_DIR_IMPROVED_3X3,
    PREDICTIONS_DIR, PREDICTIONS_DIR_IMPROVED, PREDICTIONS_DIR_IMPROVED_3X3,
    PreprocessConfig
)

# Set style for paper-quality figures
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


class TrainingVisualizer:
    """Visualize training curves from checkpoints"""

    def __init__(self, output_dir='figures'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_srcnn_training(self, checkpoint_path=None):
        """Plot SRCNN training curves"""
        if checkpoint_path is None:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, 'srcnn_best.pth')

        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        train_losses = checkpoint.get('train_losses', [])
        val_psnrs = checkpoint.get('val_psnrs', [])

        if not train_losses:
            print("No training data found in checkpoint")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Plot training loss
        ax1 = axes[0]
        epochs = range(1, len(train_losses) + 1)
        ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.set_title('SRCNN Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot validation PSNR
        ax2 = axes[1]
        if val_psnrs:
            val_epochs = range(5, len(train_losses) + 1, 5)  # Assuming VAL_EVERY=5
            if len(val_psnrs) > len(val_epochs):
                val_epochs = range(1, len(val_psnrs) + 1)
            ax2.plot(val_epochs[:len(val_psnrs)], val_psnrs, 'g-', linewidth=2,
                    label='Validation PSNR', marker='o', markersize=4)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('PSNR (dB)')
            ax2.set_title('SRCNN Validation PSNR')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'srcnn_training_curves.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

        return save_path

    def plot_unet_training(self, checkpoint_path=None):
        """Plot U-Net training curves"""
        if checkpoint_path is None:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, 'unet_best.pth')

        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        train_losses = checkpoint.get('train_losses', [])
        val_ious = checkpoint.get('val_ious', [])

        if not train_losses:
            print("No training data found in checkpoint")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Plot training loss
        ax1 = axes[0]
        epochs = range(1, len(train_losses) + 1)
        ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (BCE + Dice)')
        ax1.set_title('U-Net Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot validation IoU
        ax2 = axes[1]
        if val_ious:
            val_epochs = range(5, len(train_losses) + 1, 5)
            if len(val_ious) > len(val_epochs):
                val_epochs = range(1, len(val_ious) + 1)
            ax2.plot(val_epochs[:len(val_ious)], val_ious, 'r-', linewidth=2,
                    label='Validation IoU', marker='o', markersize=4)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('IoU')
            ax2.set_title('U-Net Validation IoU')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'unet_training_curves.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

        return save_path


class ResultsVisualizer:
    """Visualize qualitative results"""

    def __init__(self, output_dir='figures'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def create_comparison_grid(self, split='test', n_samples=6, indices=None):
        """
        Create a grid comparing: LR | SRCNN Restored | Ground Truth | Prediction

        Args:
            split: Dataset split
            n_samples: Number of samples to show
            indices: Specific indices to use (optional)
        """
        lr_dir = os.path.join(LR_IMAGE_DIR, split)
        hr_dir = os.path.join(HR_IMAGE_DIR, split)
        mask_dir = os.path.join(ENHANCED_MASK_DIR, split)
        restored_dir = os.path.join(RESTORED_DIR, split)
        pred_dir = PREDICTIONS_DIR

        # Get image files
        lr_files = sorted([f for f in os.listdir(lr_dir) if f.endswith('.png')])

        if indices is None:
            # Select evenly spaced samples
            indices = np.linspace(0, len(lr_files) - 1, n_samples, dtype=int)

        n_cols = 4  # LR, Restored, GT Mask, Prediction
        n_rows = len(indices)

        fig = plt.figure(figsize=(16, 4 * n_rows))
        gs = GridSpec(n_rows, n_cols, figure=fig, wspace=0.05, hspace=0.15)

        for row, idx in enumerate(indices):
            filename = lr_files[idx]

            # Load images
            lr_img = cv2.imread(os.path.join(lr_dir, filename))
            hr_img = cv2.imread(os.path.join(hr_dir, filename))
            gt_mask = cv2.imread(os.path.join(mask_dir, filename), cv2.IMREAD_GRAYSCALE)

            restored_path = os.path.join(restored_dir, filename)
            restored_img = cv2.imread(restored_path) if os.path.exists(restored_path) else hr_img

            pred_path = os.path.join(pred_dir, filename.replace('.png', '_pred.png'))
            pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(pred_path) else gt_mask
            # Ensure pred_mask has same size as lr_img
            if pred_mask is not None and pred_mask.shape[:2] != lr_img.shape[:2]:
                pred_mask = cv2.resize(pred_mask, (lr_img.shape[1], lr_img.shape[0]))

            # Convert BGR to RGB
            lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
            hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
            restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)

            # Plot
            images = [lr_img, restored_img, gt_mask, pred_mask]
            titles = ['LR Input', 'SRCNN Restored', 'Ground Truth', 'U-Net Prediction']

            for col, (img, title) in enumerate(zip(images, titles)):
                ax = fig.add_subplot(gs[row, col])

                if len(img.shape) == 2:
                    ax.imshow(img, cmap='gray')
                else:
                    ax.imshow(img)

                if row == 0:
                    ax.set_title(title, fontsize=12, fontweight='bold')

                ax.axis('off')

        plt.suptitle('Qualitative Results: Low-Quality Image → Super-Resolution → Crack Segmentation',
                    fontsize=14, fontweight='bold', y=1.02)

        save_path = os.path.join(self.output_dir, f'comparison_grid_{split}.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

        return save_path

    def create_overlay_visualization(self, split='test', n_samples=4):
        """Create overlay visualization showing prediction on original image"""
        lr_dir = os.path.join(LR_IMAGE_DIR, split)
        mask_dir = os.path.join(ENHANCED_MASK_DIR, split)
        pred_dir = PREDICTIONS_DIR

        lr_files = sorted([f for f in os.listdir(lr_dir) if f.endswith('.png')])
        indices = np.linspace(0, len(lr_files) - 1, n_samples, dtype=int)

        fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))

        for row, idx in enumerate(indices):
            filename = lr_files[idx]

            # Load images
            lr_img = cv2.imread(os.path.join(lr_dir, filename))
            gt_mask = cv2.imread(os.path.join(mask_dir, filename), cv2.IMREAD_GRAYSCALE)

            pred_path = os.path.join(pred_dir, filename.replace('.png', '_pred.png'))
            pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(pred_path) else gt_mask
            # Ensure pred_mask has same size as lr_img
            if pred_mask is not None and pred_mask.shape[:2] != lr_img.shape[:2]:
                pred_mask = cv2.resize(pred_mask, (lr_img.shape[1], lr_img.shape[0]))

            lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)

            # Create overlays
            gt_overlay = lr_img.copy()
            gt_overlay[gt_mask > 127] = [255, 0, 0]  # Red for GT

            pred_overlay = lr_img.copy()
            pred_overlay[pred_mask > 127] = [0, 255, 0]  # Green for prediction

            # Blend
            gt_overlay = cv2.addWeighted(lr_img, 0.7, gt_overlay, 0.3, 0)
            pred_overlay = cv2.addWeighted(lr_img, 0.7, pred_overlay, 0.3, 0)

            # Plot
            axes[row, 0].imshow(lr_img)
            axes[row, 0].set_title('Input Image')
            axes[row, 1].imshow(gt_overlay)
            axes[row, 1].set_title('Ground Truth Overlay')
            axes[row, 2].imshow(pred_overlay)
            axes[row, 2].set_title('Prediction Overlay')

            for ax in axes[row]:
                ax.axis('off')

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f'overlay_visualization_{split}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

        return save_path


class DatasetVisualizer:
    """Visualize dataset statistics"""

    def __init__(self, output_dir='figures'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_class_distribution(self, split='train'):
        """Plot crack vs background pixel distribution"""
        mask_dir = os.path.join(ENHANCED_MASK_DIR, split)

        if not os.path.exists(mask_dir):
            print(f"Mask directory not found: {mask_dir}")
            return

        mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]

        crack_ratios = []
        for filename in tqdm(mask_files, desc="Analyzing masks"):
            mask = cv2.imread(os.path.join(mask_dir, filename), cv2.IMREAD_GRAYSCALE)
            crack_pixels = np.sum(mask > 127)
            total_pixels = mask.size
            crack_ratios.append(crack_pixels / total_pixels * 100)

        crack_ratios = np.array(crack_ratios)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Histogram
        ax1 = axes[0]
        ax1.hist(crack_ratios, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.axvline(np.mean(crack_ratios), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(crack_ratios):.2f}%')
        ax1.set_xlabel('Crack Pixel Ratio (%)')
        ax1.set_ylabel('Number of Images')
        ax1.set_title('Distribution of Crack Pixel Ratio')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Pie chart (overall)
        ax2 = axes[1]
        total_crack = np.mean(crack_ratios)
        total_bg = 100 - total_crack
        sizes = [total_bg, total_crack]
        labels = [f'Background\n({total_bg:.1f}%)', f'Crack\n({total_crack:.1f}%)']
        colors = ['#3498db', '#e74c3c']
        explode = (0, 0.1)

        ax2.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='',
               shadow=True, startangle=90)
        ax2.set_title('Overall Class Distribution')

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f'class_distribution_{split}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

        # Print statistics
        print(f"\nClass Distribution Statistics ({split}):")
        print(f"  Mean crack ratio: {np.mean(crack_ratios):.2f}%")
        print(f"  Std: {np.std(crack_ratios):.2f}%")
        print(f"  Min: {np.min(crack_ratios):.2f}%")
        print(f"  Max: {np.max(crack_ratios):.2f}%")

        return save_path, crack_ratios

    def plot_dataset_splits(self):
        """Plot train/val/test split distribution"""
        splits = ['train', 'val', 'test']
        counts = []

        for split in splits:
            txt_file = f'dataset/ImageSets/{split}.txt'
            if os.path.exists(txt_file):
                with open(txt_file, 'r') as f:
                    counts.append(len(f.readlines()))
            else:
                counts.append(0)

        fig, ax = plt.subplots(figsize=(8, 6))

        colors = ['#2ecc71', '#3498db', '#e74c3c']
        bars = ax.bar(splits, counts, color=colors, edgecolor='black')

        ax.set_xlabel('Dataset Split')
        ax.set_ylabel('Number of Images')
        ax.set_title('CRACK500 Dataset Split Distribution')

        # Add count labels on bars
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                   str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax.set_ylim(0, max(counts) * 1.15)
        ax.grid(True, alpha=0.3, axis='y')

        # Add total
        total = sum(counts)
        ax.text(0.5, 0.95, f'Total: {total} images',
               transform=ax.transAxes, ha='center', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'dataset_splits.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

        return save_path


class ComparisonVisualizer:
    """Visualize comparison between methods"""

    def __init__(self, output_dir='figures'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_methods_comparison(self, results_path='results/baseline_results_test.json'):
        """Plot bar chart comparing different methods"""
        if not os.path.exists(results_path):
            print(f"Results file not found: {results_path}")
            print("Please run baseline experiments first: python scripts/run_baselines.py")
            return

        with open(results_path, 'r') as f:
            results = json.load(f)

        methods = list(results.keys())
        metrics = ['avg_iou', 'avg_dice', 'avg_accuracy']
        metric_names = ['IoU', 'Dice', 'Accuracy']

        x = np.arange(len(methods))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            values = [results[m][metric] for m in methods]
            bars = ax.bar(x + i * width, values, width, label=name)

            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Method')
        ax.set_ylabel('Score')
        ax.set_title('Comparison of Different Methods')
        ax.set_xticks(x + width)
        ax.set_xticklabels(methods)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'methods_comparison.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

        return save_path

    def plot_metrics_radar(self, results_path='results/baseline_results_test.json'):
        """Plot radar chart comparing methods"""
        if not os.path.exists(results_path):
            print(f"Results file not found: {results_path}")
            return

        with open(results_path, 'r') as f:
            results = json.load(f)

        methods = list(results.keys())
        metrics = ['avg_iou', 'avg_dice', 'avg_accuracy']
        metric_names = ['IoU', 'Dice', 'Accuracy']

        # Number of variables
        num_vars = len(metrics)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

        for idx, method in enumerate(methods):
            values = [results[method][m] for m in metrics]
            values += values[:1]  # Complete the circle

            ax.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[idx])
            ax.fill(angles, values, alpha=0.25, color=colors[idx])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.set_title('Methods Comparison (Radar Chart)', y=1.08)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'methods_radar.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Saved: {save_path}")

        return save_path


def generate_all_figures():
    """Generate all figures for the paper"""
    print("=" * 60)
    print("Generating All Figures for Paper")
    print("=" * 60)

    # 1. Training curves
    print("\n[1/6] Plotting training curves...")
    train_viz = TrainingVisualizer()
    train_viz.plot_srcnn_training()
    train_viz.plot_unet_training()

    # 2. Dataset statistics
    print("\n[2/6] Plotting dataset statistics...")
    dataset_viz = DatasetVisualizer()
    dataset_viz.plot_dataset_splits()
    dataset_viz.plot_class_distribution('train')

    # 3. Qualitative results
    print("\n[3/6] Creating comparison grids...")
    results_viz = ResultsVisualizer()
    results_viz.create_comparison_grid(split='test', n_samples=6)
    results_viz.create_overlay_visualization(split='test', n_samples=4)

    # 4. Method comparison (if results exist)
    print("\n[4/6] Creating method comparison charts...")
    comparison_viz = ComparisonVisualizer()
    comparison_viz.plot_methods_comparison()
    comparison_viz.plot_metrics_radar()

    print("\n" + "=" * 60)
    print("All figures generated! Check the 'figures/' directory")
    print("=" * 60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate visualization figures')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['all', 'training', 'dataset', 'results', 'comparison'],
                       help='What to visualize')
    parser.add_argument('--output', type=str, default='figures',
                       help='Output directory for figures')

    args = parser.parse_args()

    if args.mode == 'all':
        generate_all_figures()
    elif args.mode == 'training':
        viz = TrainingVisualizer(args.output)
        viz.plot_srcnn_training()
        viz.plot_unet_training()
    elif args.mode == 'dataset':
        viz = DatasetVisualizer(args.output)
        viz.plot_dataset_splits()
        viz.plot_class_distribution('train')
    elif args.mode == 'results':
        viz = ResultsVisualizer(args.output)
        viz.create_comparison_grid(split='test', n_samples=6)
        viz.create_overlay_visualization(split='test', n_samples=4)
    elif args.mode == 'comparison':
        viz = ComparisonVisualizer(args.output)
        viz.plot_methods_comparison()
        viz.plot_metrics_radar()
