"""
Compare SRCNN vs ImprovedSRCNN

This script:
1. Tests both SRCNN models on test set (PSNR/SSIM)
2. Trains U-Net with restored images from each model
3. Compares final segmentation performance (IoU/Dice)

Usage:
    python scripts/compare_srcnn_models.py --device cuda
"""
import os
import sys
import json
import time
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from config import (
    DEVICE, CHECKPOINT_DIR,
    SRCNN_CHECKPOINT, IMPROVED_SRCNN_CHECKPOINT,
    RESTORED_DIR, LR_IMAGE_DIR, HR_IMAGE_DIR, ENHANCED_MASK_DIR,
    SRCNNConfig, UNetConfig
)
from srcnn.model import SRCNN, ImprovedSRCNN, compute_psnr, compute_ssim
from srcnn.dataset import get_test_loader
from unet.model import UNet, CombinedLoss, compute_iou, compute_dice_coeff
from unet.dataset import get_unet_loaders, UNetDataset


def test_srcnn_model(model_type, checkpoint_path, device):
    """Test SRCNN model and return PSNR/SSIM metrics"""
    print(f"\nTesting {model_type} SRCNN...")

    # Load model
    if model_type == 'improved':
        model = ImprovedSRCNN(
            num_channels=SRCNNConfig.NUM_CHANNELS,
            num_features=SRCNNConfig.NUM_FEATURES
        )
    else:
        model = SRCNN(
            num_channels=SRCNNConfig.NUM_CHANNELS,
            num_features=SRCNNConfig.NUM_FEATURES
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Test
    test_loader = get_test_loader(split='test')
    total_psnr = 0.0
    total_ssim = 0.0

    with torch.no_grad():
        for lr_imgs, hr_imgs, _ in test_loader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            outputs = model(lr_imgs)

            for i in range(outputs.size(0)):
                psnr = compute_psnr(outputs[i:i+1], hr_imgs[i:i+1])
                ssim = compute_ssim(outputs[i:i+1], hr_imgs[i:i+1])
                total_psnr += psnr.item()
                total_ssim += ssim.item()

    n = len(test_loader.dataset)
    metrics = {
        'psnr': total_psnr / n,
        'ssim': total_ssim / n
    }

    print(f"  PSNR: {metrics['psnr']:.4f} dB")
    print(f"  SSIM: {metrics['ssim']:.4f}")

    return model, metrics


def restore_and_save(model, model_type, device):
    """Restore images and save to disk"""
    import cv2
    import numpy as np
    from tqdm import tqdm

    output_dir = os.path.join(RESTORED_DIR, model_type, 'test')
    os.makedirs(output_dir, exist_ok=True)

    test_loader = get_test_loader(split='test')

    print(f"Restoring images to {output_dir}...")

    model.eval()
    with torch.no_grad():
        for lr_imgs, _, filenames in tqdm(test_loader, desc=f"Restoring ({model_type})"):
            lr_imgs = lr_imgs.to(device)
            outputs = model(lr_imgs)

            for i in range(outputs.size(0)):
                output = outputs[i].permute(1, 2, 0).cpu().numpy()
                output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
                output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

                save_path = os.path.join(output_dir, filenames[i])
                cv2.imwrite(save_path, output)

    return output_dir


def train_unet_with_restored(restored_dir, model_name, device, epochs=50, pos_weight=5.0):
    """Train U-Net with specific restored images"""
    print(f"\nTraining U-Net with {model_name} restored images...")

    # Create custom dataset paths
    import shutil

    # Temporarily copy restored images to expected location
    temp_restored_dir = RESTORED_DIR
    target_dir = os.path.join(RESTORED_DIR, 'test')

    # Backup existing
    if os.path.exists(target_dir):
        if os.path.exists(target_dir + '_backup'):
            shutil.rmtree(target_dir + '_backup')
        shutil.move(target_dir, target_dir + '_backup')

    # Copy new restored images
    shutil.copytree(restored_dir, target_dir)

    try:
        # Train U-Net
        model = UNet(
            in_channels=UNetConfig.IN_CHANNELS,
            out_channels=UNetConfig.OUT_CHANNELS,
            features=UNetConfig.FEATURES
        ).to(device)

        criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5, pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=UNetConfig.LEARNING_RATE)

        train_loader, val_loader = get_unet_loaders(
            batch_size=UNetConfig.BATCH_SIZE,
            use_restored=True
        )

        best_iou = 0.0
        best_model_state = None

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0

            for imgs, masks, _ in train_loader:
                imgs, masks = imgs.to(device), masks.to(device)

                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validate
            model.eval()
            val_iou = 0.0
            with torch.no_grad():
                for imgs, masks, _ in val_loader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    outputs = torch.sigmoid(model(imgs))
                    val_iou += compute_iou(outputs, masks).item()

            val_iou /= len(val_loader)

            if val_iou > best_iou:
                best_iou = val_iou
                best_model_state = model.state_dict().copy()

            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: Loss={train_loss/len(train_loader):.4f}, Val IoU={val_iou:.4f}")

        model.load_state_dict(best_model_state)

        # Test
        test_dataset = UNetDataset(split='test', use_restored=True)
        test_iou = 0.0
        test_dice = 0.0

        model.eval()
        with torch.no_grad():
            for i in range(len(test_dataset)):
                img, mask_gt, _ = test_dataset[i]
                img_input = img.unsqueeze(0).to(device)
                output = torch.sigmoid(model(img_input))
                mask_tensor = mask_gt.unsqueeze(0).to(device)

                test_iou += compute_iou(output, mask_tensor).item()
                test_dice += compute_dice_coeff(output, mask_tensor).item()

        n = len(test_dataset)
        test_metrics = {
            'iou': test_iou / n,
            'dice': test_dice / n
        }

        print(f"  Test IoU: {test_metrics['iou']:.4f}")
        print(f"  Test Dice: {test_metrics['dice']:.4f}")

        return test_metrics

    finally:
        # Restore backup
        if os.path.exists(target_dir + '_backup'):
            shutil.rmtree(target_dir)
            shutil.move(target_dir + '_backup', target_dir)


def main():
    parser = argparse.ArgumentParser(description='Compare SRCNN vs ImprovedSRCNN')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'])
    parser.add_argument('--epochs', type=int, default=50, help='U-Net training epochs')
    parser.add_argument('--pos-weight', type=float, default=5.0)
    parser.add_argument('--skip-train', action='store_true', help='Skip U-Net training, only test SRCNN')

    args = parser.parse_args()
    device = args.device if args.device else DEVICE

    print("=" * 60)
    print("SRCNN vs ImprovedSRCNN Comparison")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'epochs': args.epochs,
            'pos_weight': args.pos_weight,
            'device': device
        },
        'srcnn': {},
        'improved_srcnn': {}
    }

    # ==================== Test SRCNN Models ====================
    print("\n" + "=" * 60)
    print("Part 1: Testing SRCNN Models (Restoration Quality)")
    print("=" * 60)

    # Check if checkpoints exist
    if not os.path.exists(SRCNN_CHECKPOINT):
        print(f"Error: SRCNN checkpoint not found: {SRCNN_CHECKPOINT}")
        print("Please train basic SRCNN first:")
        print("  python main.py --mode train-srcnn --model-type srcnn --epochs-srcnn 100")
        return

    if not os.path.exists(IMPROVED_SRCNN_CHECKPOINT):
        print(f"Error: ImprovedSRCNN checkpoint not found: {IMPROVED_SRCNN_CHECKPOINT}")
        print("Please train improved SRCNN first:")
        print("  python main.py --mode train-srcnn --model-type improved --epochs-srcnn 100")
        return

    # Test basic SRCNN
    model_srcnn, metrics_srcnn = test_srcnn_model('basic', SRCNN_CHECKPOINT, device)
    results['srcnn']['restoration'] = metrics_srcnn

    # Test improved SRCNN
    model_improved, metrics_improved = test_srcnn_model('improved', IMPROVED_SRCNN_CHECKPOINT, device)
    results['improved_srcnn']['restoration'] = metrics_improved

    # Compare restoration quality
    print("\n" + "-" * 40)
    print("Restoration Quality Comparison:")
    print("-" * 40)
    print(f"{'Model':<20} {'PSNR (dB)':>12} {'SSIM':>12}")
    print(f"{'SRCNN':<20} {metrics_srcnn['psnr']:>12.4f} {metrics_srcnn['ssim']:>12.4f}")
    print(f"{'ImprovedSRCNN':<20} {metrics_improved['psnr']:>12.4f} {metrics_improved['ssim']:>12.4f}")

    psnr_improve = metrics_improved['psnr'] - metrics_srcnn['psnr']
    ssim_improve = metrics_improved['ssim'] - metrics_srcnn['ssim']
    print(f"\nImprovement: PSNR {psnr_improve:+.4f} dB, SSIM {ssim_improve:+.4f}")

    if args.skip_train:
        print("\nSkipping U-Net training (--skip-train)")
        print("\nResults summary:")
        print(json.dumps(results, indent=2))
        return

    # ==================== Train U-Net with Restored Images ====================
    print("\n" + "=" * 60)
    print("Part 2: Training U-Net with Restored Images")
    print("=" * 60)

    # Restore images for both models
    restored_dir_srcnn = restore_and_save(model_srcnn, 'srcnn', device)
    restored_dir_improved = restore_and_save(model_improved, 'improved', device)

    # Train U-Net with basic SRCNN restored images
    print("\n" + "-" * 40)
    print("Training U-Net with basic SRCNN restored images...")
    print("-" * 40)
    metrics_unet_srcnn = train_unet_with_restored(
        restored_dir_srcnn, 'basic SRCNN', device,
        epochs=args.epochs, pos_weight=args.pos_weight
    )
    results['srcnn']['segmentation'] = metrics_unet_srcnn

    # Train U-Net with improved SRCNN restored images
    print("\n" + "-" * 40)
    print("Training U-Net with improved SRCNN restored images...")
    print("-" * 40)
    metrics_unet_improved = train_unet_with_restored(
        restored_dir_improved, 'improved SRCNN', device,
        epochs=args.epochs, pos_weight=args.pos_weight
    )
    results['improved_srcnn']['segmentation'] = metrics_unet_improved

    # ==================== Final Summary ====================
    print("\n" + "=" * 60)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 60)

    print("\n1. Restoration Quality:")
    print(f"   {'Model':<20} {'PSNR (dB)':>12} {'SSIM':>12}")
    print(f"   {'SRCNN':<20} {metrics_srcnn['psnr']:>12.4f} {metrics_srcnn['ssim']:>12.4f}")
    print(f"   {'ImprovedSRCNN':<20} {metrics_improved['psnr']:>12.4f} {metrics_improved['ssim']:>12.4f}")

    print("\n2. Segmentation Performance:")
    print(f"   {'Model':<20} {'IoU':>12} {'Dice':>12}")
    print(f"   {'SRCNN + U-Net':<20} {metrics_unet_srcnn['iou']:>12.4f} {metrics_unet_srcnn['dice']:>12.4f}")
    print(f"   {'ImprovedSRCNN + U-Net':<20} {metrics_unet_improved['iou']:>12.4f} {metrics_unet_improved['dice']:>12.4f}")

    iou_improve = ((metrics_unet_improved['iou'] - metrics_unet_srcnn['iou']) / metrics_unet_srcnn['iou']) * 100
    print(f"\n   IoU Improvement: {iou_improve:+.2f}%")

    # Save results
    output_path = 'results/srcnn_comparison.json'
    os.makedirs('results', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
