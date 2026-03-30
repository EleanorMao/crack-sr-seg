"""
Baseline Experiments Script
Compare different image restoration methods before U-Net segmentation
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
from tqdm import tqdm

from config import (
    LR_IMAGE_DIR, HR_IMAGE_DIR, ENHANCED_MASK_DIR,
    RESTORED_DIR, RESTORED_DIR_IMPROVED, RESTORED_DIR_IMPROVED_3X3,
    PREDICTIONS_DIR, PREDICTIONS_DIR_IMPROVED, PREDICTIONS_DIR_IMPROVED_3X3,
    DEVICE
)
from unet.model import UNet, compute_iou, compute_dice_coeff, compute_pixel_accuracy


def bilinear_upsample(lr_img, target_size=(256, 256)):
    """Simple bilinear interpolation baseline"""
    return cv2.resize(lr_img, target_size, interpolation=cv2.INTER_LINEAR)


def bicubic_upsample(lr_img, target_size=(256, 256)):
    """Bicubic interpolation baseline"""
    return cv2.resize(lr_img, target_size, interpolation=cv2.INTER_CUBIC)


class BaselineTester:
    """Test different restoration baselines"""

    def __init__(self, unet_checkpoint, device=None):
        self.device = device if device else DEVICE

        # Load trained U-Net
        self.model = UNet(in_channels=3, out_channels=1, features=[64, 128, 256, 512])
        self.model.load_state_dict(
            torch.load(unet_checkpoint, map_location=self.device)['model_state_dict']
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"Loaded U-Net from: {unet_checkpoint}")

    def predict(self, img):
        """Predict mask for an image (returns logits, not sigmoid)"""
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(rgb_img.astype(np.float32) / 255.0)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img_tensor)
            # Don't apply sigmoid here - compute_iou will do it

        return output.squeeze(0).squeeze(0).cpu().numpy()

    def evaluate_split(self, split='test', restore_fn=None, restore_name='baseline'):
        """
        Evaluate on a split using specified restoration function

        Args:
            split: 'train', 'val', or 'test'
            restore_fn: Function that takes LR image and returns restored image
            restore_name: Name for saving results
        """
        lr_dir = os.path.join(LR_IMAGE_DIR, split)
        mask_dir = os.path.join(ENHANCED_MASK_DIR, split)

        # Get image list
        image_files = [f for f in os.listdir(lr_dir) if f.endswith('.png')]
        print(f"\nEvaluating {restore_name} on {split} ({len(image_files)} images)")

        total_iou = 0.0
        total_dice = 0.0
        total_acc = 0.0
        total_psnr = 0.0

        results = []

        for filename in tqdm(image_files, desc=f"Testing {restore_name}"):
            # Load LR image
            lr_path = os.path.join(lr_dir, filename)
            lr_img = cv2.imread(lr_path)

            # Load GT mask
            mask_path = os.path.join(mask_dir, filename)
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            gt_mask = (gt_mask > 127).astype(np.float32)

            # Restore image
            if restore_fn is not None:
                restored_img = restore_fn(lr_img)
            else:
                restored_img = lr_img

            # Predict mask
            pred_mask = self.predict(restored_img)

            # Compute metrics
            pred_tensor = torch.from_numpy(pred_mask).unsqueeze(0).unsqueeze(0)
            gt_tensor = torch.from_numpy(gt_mask).unsqueeze(0).unsqueeze(0)

            iou = compute_iou(pred_tensor, gt_tensor).item()
            dice = compute_dice_coeff(pred_tensor, gt_tensor).item()
            acc = compute_pixel_accuracy(pred_tensor, gt_tensor).item()

            total_iou += iou
            total_dice += dice
            total_acc += acc

            results.append({
                'filename': filename,
                'iou': iou,
                'dice': dice,
                'accuracy': acc
            })

        n = len(image_files)
        metrics = {
            'method': restore_name,
            'split': split,
            'n_samples': n,
            'avg_iou': total_iou / n,
            'avg_dice': total_dice / n,
            'avg_accuracy': total_acc / n,
            'results': results
        }

        print(f"\n{restore_name} Results on {split}:")
        print(f"  IoU: {metrics['avg_iou']:.4f}")
        print(f"  Dice: {metrics['avg_dice']:.4f}")
        print(f"  Accuracy: {metrics['avg_accuracy']:.4f}")

        return metrics


def run_all_baselines(unet_checkpoint, split='test'):
    """Run all baseline experiments"""

    tester = BaselineTester(unet_checkpoint)

    results = {}

    # Baseline 1: Bilinear interpolation
    results['bilinear'] = tester.evaluate_split(
        split=split,
        restore_fn=bilinear_upsample,
        restore_name='Bilinear'
    )

    # Baseline 2: Bicubic interpolation
    results['bicubic'] = tester.evaluate_split(
        split=split,
        restore_fn=bicubic_upsample,
        restore_name='Bicubic'
    )

    # Baseline 3: No restoration (just resize to 256x256 if needed)
    results['no_restore'] = tester.evaluate_split(
        split=split,
        restore_fn=lambda x: cv2.resize(x, (256, 256)),
        restore_name='No_Restore'
    )

    return results


def print_comparison_table(results_dict):
    """Print a formatted comparison table"""
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    print(f"{'Method':<20} {'IoU':>10} {'Dice':>10} {'Accuracy':>10}")
    print("-" * 60)

    for method, metrics in results_dict.items():
        print(f"{method:<20} {metrics['avg_iou']:>10.4f} {metrics['avg_dice']:>10.4f} {metrics['avg_accuracy']:>10.4f}")

    print("=" * 60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run baseline experiments')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/unet_original_best.pth',
                        help='Path to trained U-Net checkpoint (default: unet_original_best.pth for fair comparison)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate')

    args = parser.parse_args()

    results = run_all_baselines(args.checkpoint, args.split)
    print_comparison_table(results)

    # Save results
    import json
    output_path = f'results/baseline_results_{args.split}.json'
    os.makedirs('results', exist_ok=True)

    # Convert to serializable format
    save_results = {}
    for method, metrics in results.items():
        save_results[method] = {
            'method': metrics['method'],
            'split': metrics['split'],
            'n_samples': metrics['n_samples'],
            'avg_iou': metrics['avg_iou'],
            'avg_dice': metrics['avg_dice'],
            'avg_accuracy': metrics['avg_accuracy']
        }

    with open(output_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
