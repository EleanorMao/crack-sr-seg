"""
Compare all SRCNN variants

This script:
1. Tests all SRCNN models on test set (PSNR/SSIM)
2. Trains U-Net with restored images from each model
3. Compares final segmentation performance (IoU/Dice)

Usage:
    python scripts/compare_srcnn_models.py --device cuda
    python scripts/compare_srcnn_models.py --models srcnn improved improved_3x3
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
    IMPROVED_SRCNN_BN_CHECKPOINT, IMPROVED_SRCNN_ALL3X3_CHECKPOINT,
    RESTORED_DIR, RESTORED_DIR_IMPROVED, RESTORED_DIR_IMPROVED_BN, RESTORED_DIR_IMPROVED_3X3,
    LR_IMAGE_DIR, HR_IMAGE_DIR, ENHANCED_MASK_DIR,
    SRCNNConfig, UNetConfig
)
from srcnn.model import SRCNN, ImprovedSRCNN, ImprovedSRCNN_BN, ImprovedSRCNN_All3x3, compute_psnr, compute_ssim
from srcnn.dataset import get_test_loader
from unet.model import UNet, CombinedLoss, compute_iou, compute_dice_coeff
from unet.dataset import get_unet_loaders, UNetDataset


# 模型配置映射
MODEL_CONFIGS = {
    'srcnn': {
        'class': SRCNN,
        'checkpoint': SRCNN_CHECKPOINT,
        'output_dir': RESTORED_DIR,
        'name': 'Basic SRCNN'
    },
    'improved': {
        'class': ImprovedSRCNN,
        'checkpoint': IMPROVED_SRCNN_CHECKPOINT,
        'output_dir': RESTORED_DIR_IMPROVED,
        'name': 'Improved SRCNN (9x9+3x3+5x5)'
    },
    'improved_bn': {
        'class': ImprovedSRCNN_BN,
        'checkpoint': IMPROVED_SRCNN_BN_CHECKPOINT,
        'output_dir': RESTORED_DIR_IMPROVED_BN,
        'name': 'Improved SRCNN + BatchNorm'
    },
    'improved_3x3': {
        'class': ImprovedSRCNN_All3x3,
        'checkpoint': IMPROVED_SRCNN_ALL3X3_CHECKPOINT,
        'output_dir': RESTORED_DIR_IMPROVED_3X3,
        'name': 'Improved SRCNN (all 3x3)'
    }
}


def test_srcnn_model(model_type, checkpoint_path, device):
    """Test SRCNN model and return PSNR/SSIM metrics"""
    print(f"\nTesting {model_type} SRCNN...")

    # 获取模型配置
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}")

    config = MODEL_CONFIGS[model_type]
    model = config['class'](
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

    # 使用模型配置中的输出目录
    config = MODEL_CONFIGS[model_type]
    output_dir = os.path.join(config['output_dir'], 'test')
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
    parser = argparse.ArgumentParser(description='Compare all SRCNN variants')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'])
    parser.add_argument('--epochs', type=int, default=50, help='U-Net training epochs')
    parser.add_argument('--pos-weight', type=float, default=5.0)
    parser.add_argument('--skip-train', action='store_true', help='Skip U-Net training, only test SRCNN')
    parser.add_argument('--models', type=str, nargs='+',
                        default=['srcnn', 'improved', 'improved_3x3'],
                        choices=['srcnn', 'improved', 'improved_bn', 'improved_3x3'],
                        help='Models to compare')

    args = parser.parse_args()
    device = args.device if args.device else DEVICE

    print("=" * 60)
    print("SRCNN Variants Comparison")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Models: {args.models}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'epochs': args.epochs,
            'pos_weight': args.pos_weight,
            'device': device,
            'models': args.models
        }
    }

    # ==================== Test SRCNN Models ====================
    print("\n" + "=" * 60)
    print("Part 1: Testing SRCNN Models (Restoration Quality)")
    print("=" * 60)

    models = {}
    restoration_metrics = {}

    for model_type in args.models:
        config = MODEL_CONFIGS[model_type]
        checkpoint_path = config['checkpoint']

        if not os.path.exists(checkpoint_path):
            print(f"[SKIP] {config['name']}: checkpoint not found at {checkpoint_path}")
            continue

        model, metrics = test_srcnn_model(model_type, checkpoint_path, device)
        models[model_type] = model
        restoration_metrics[model_type] = metrics
        results[model_type] = {'restoration': metrics}

    # Compare restoration quality
    print("\n" + "-" * 50)
    print("Restoration Quality Comparison:")
    print("-" * 50)
    print(f"{'Model':<30} {'PSNR (dB)':>12} {'SSIM':>12}")
    for model_type, metrics in restoration_metrics.items():
        name = MODEL_CONFIGS[model_type]['name']
        print(f"{name:<30} {metrics['psnr']:>12.4f} {metrics['ssim']:>12.4f}")
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

    segmentation_metrics = {}

    for model_type, model in models.items():
        restored_dir = restore_and_save(model, model_type, device)

        print("\n" + "-" * 40)
        print(f"Training U-Net with {MODEL_CONFIGS[model_type]['name']} restored images...")
        print("-" * 40)

        metrics = train_unet_with_restored(
            restored_dir, MODEL_CONFIGS[model_type]['name'], device,
            epochs=args.epochs, pos_weight=args.pos_weight
        )
        segmentation_metrics[model_type] = metrics
        results[model_type]['segmentation'] = metrics

    # ==================== Final Summary ====================
    print("\n" + "=" * 60)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 60)

    print("\n1. Restoration Quality:")
    print(f"   {'Model':<30} {'PSNR (dB)':>12} {'SSIM':>12}")
    for model_type, metrics in restoration_metrics.items():
        name = MODEL_CONFIGS[model_type]['name']
        print(f"   {name:<30} {metrics['psnr']:>12.4f} {metrics['ssim']:>12.4f}")

    print("\n2. Segmentation Performance:")
    print(f"   {'Model':<30} {'IoU':>12} {'Dice':>12}")
    for model_type, metrics in segmentation_metrics.items():
        name = MODEL_CONFIGS[model_type]['name']
        print(f"   {name + ' + U-Net':<30} {metrics['iou']:>12.4f} {metrics['dice']:>12.4f}")

    # Calculate improvement
    if len(segmentation_metrics) > 1:
        baseline_type = args.models[0]
        baseline_iou = segmentation_metrics[baseline_type]['iou']
        print(f"\n3. Improvement over {MODEL_CONFIGS[baseline_type]['name']}:")
        for model_type, metrics in segmentation_metrics.items():
            if model_type != baseline_type:
                iou_improve = ((metrics['iou'] - baseline_iou) / baseline_iou) * 100
                print(f"   {MODEL_CONFIGS[model_type]['name']}: IoU {iou_improve:+.2f}%")

    # Save results
    output_path = 'results/srcnn_comparison.json'
    os.makedirs('results', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
