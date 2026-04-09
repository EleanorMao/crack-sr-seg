"""
Ablation Study Script
Run ablation experiments to prove each component's contribution
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import cv2
import numpy as np
import torch
from tqdm import tqdm

from config import (
    CHECKPOINT_DIR, LR_IMAGE_DIR, HR_IMAGE_DIR, ENHANCED_MASK_DIR,
    RESTORED_DIR, RESTORED_DIR_IMPROVED, RESTORED_DIR_IMPROVED_BN, RESTORED_DIR_IMPROVED_3X3,
    DEVICE, SRCNNConfig, UNetConfig
)
from srcnn.model import SRCNN, ImprovedSRCNN, ImprovedSRCNN_BN, ImprovedSRCNN_All3x3, ImprovedSRCNN_5L_RF15, compute_psnr, compute_ssim
from unet.model import UNet, CombinedLoss, compute_iou, compute_dice_coeff


def evaluate_srcnn(model, split='test', device='cpu'):
    """Evaluate SRCNN model on a split"""
    lr_dir = os.path.join(LR_IMAGE_DIR, split)
    hr_dir = os.path.join(HR_IMAGE_DIR, split)

    image_files = sorted([f for f in os.listdir(lr_dir) if f.endswith('.png')])

    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0

    with torch.no_grad():
        for filename in tqdm(image_files, desc="Evaluating SRCNN"):
            lr_path = os.path.join(lr_dir, filename)
            hr_path = os.path.join(hr_dir, filename)

            lr_img = cv2.imread(lr_path)
            hr_img = cv2.imread(hr_path)

            # Preprocess
            lr_tensor = torch.from_numpy(
                cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0).to(device)

            hr_tensor = torch.from_numpy(
                cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0).to(device)

            # Inference
            output = model(lr_tensor)

            # Metrics
            psnr = compute_psnr(output, hr_tensor).item()
            ssim = compute_ssim(output, hr_tensor).item()

            total_psnr += psnr
            total_ssim += ssim

    n = len(image_files)
    return {
        'psnr': total_psnr / n,
        'ssim': total_ssim / n,
        'n_samples': n
    }


def evaluate_unet(model, split='test', use_restored=True, device='cpu'):
    """Evaluate U-Net model on a split"""
    if use_restored:
        img_dir = os.path.join(RESTORED_DIR, split)
    else:
        img_dir = os.path.join(HR_IMAGE_DIR, split)

    mask_dir = os.path.join(ENHANCED_MASK_DIR, split)

    image_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])

    model.eval()
    total_iou = 0.0
    total_dice = 0.0
    total_acc = 0.0

    with torch.no_grad():
        for filename in tqdm(image_files, desc="Evaluating U-Net"):
            img_path = os.path.join(img_dir, filename)
            mask_path = os.path.join(mask_dir, filename)

            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # Preprocess
            img_tensor = torch.from_numpy(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0).to(device)

            mask_tensor = torch.from_numpy(
                (mask > 127).astype(np.float32)
            ).unsqueeze(0).unsqueeze(0).to(device)

            # Inference
            output = torch.sigmoid(model(img_tensor))

            # Metrics
            iou = compute_iou(output, mask_tensor).item()
            dice = compute_dice_coeff(output, mask_tensor).item()

            # Pixel accuracy
            pred_binary = (output > 0.5).float()
            acc = (pred_binary == mask_tensor).float().mean().item()

            total_iou += iou
            total_dice += dice
            total_acc += acc

    n = len(image_files)
    return {
        'iou': total_iou / n,
        'dice': total_dice / n,
        'accuracy': total_acc / n,
        'n_samples': n
    }


def run_ablation_smart_degradation(results_file='results/ablation_smart_degradation.json'):
    """
    Ablation: Smart Degradation vs Random Degradation

    Requires:
    1. Train SRCNN with SMART_DEGRADATION = False, save to checkpoints/srcnn_no_smart.pth
    2. Train SRCNN with SMART_DEGRADATION = True, save to checkpoints/srcnn_best.pth
    """
    print("\n" + "=" * 60)
    print("Ablation Study: Smart Degradation")
    print("=" * 60)

    results = {}

    # Check for models
    checkpoints = {
        'Random Degradation': 'checkpoints/srcnn_no_smart.pth',
        'Smart Degradation (Ours)': 'checkpoints/srcnn_best.pth'
    }

    device = DEVICE

    for name, ckpt_path in checkpoints.items():
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] {name}: checkpoint not found at {ckpt_path}")
            print("       To run this ablation, train SRCNN with different settings")
            continue

        print(f"\n[Evaluating] {name}")
        checkpoint = torch.load(ckpt_path, map_location=device)

        # Detect model type
        if 'improved' in ckpt_path or len(checkpoint['model_state_dict']) > 10:
            model = ImprovedSRCNN().to(device)
        else:
            model = SRCNN().to(device)

        model.load_state_dict(checkpoint['model_state_dict'])

        metrics = evaluate_srcnn(model, split='test', device=device)
        results[name] = metrics

        print(f"  PSNR: {metrics['psnr']:.4f} dB")
        print(f"  SSIM: {metrics['ssim']:.4f}")

    # Save results
    os.makedirs('results', exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    return results


def run_ablation_pos_weight(results_file='results/ablation_pos_weight.json'):
    """
    Ablation: Effect of positive sample weight on U-Net

    Requires:
    - checkpoints/unet_pos1.0.pth (pos_weight=1.0)
    - checkpoints/unet_pos3.0.pth (pos_weight=3.0)
    - checkpoints/unet_best.pth (pos_weight=5.0)
    - checkpoints/unet_pos7.0.pth (pos_weight=7.0)
    """
    print("\n" + "=" * 60)
    print("Ablation Study: Positive Sample Weight")
    print("=" * 60)

    results = {}

    checkpoints = {
        'pos_weight=1.0': 'checkpoints/unet_pos1.0.pth',
        'pos_weight=3.0': 'checkpoints/unet_pos3.0.pth',
        'pos_weight=5.0 (Ours)': 'checkpoints/unet_best.pth',
        'pos_weight=7.0': 'checkpoints/unet_pos7.0.pth',
    }

    device = DEVICE

    for name, ckpt_path in checkpoints.items():
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] {name}: checkpoint not found at {ckpt_path}")
            continue

        print(f"\n[Evaluating] {name}")
        model = UNet(in_channels=3, out_channels=1, features=[64, 128, 256, 512]).to(device)
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        metrics = evaluate_unet(model, split='test', use_restored=True, device=device)
        results[name] = metrics

        print(f"  IoU: {metrics['iou']:.4f}")
        print(f"  Dice: {metrics['dice']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")

    # Save results
    os.makedirs('results', exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    return results


def run_ablation_srcnn_vs_original(results_file='results/ablation_srcnn_vs_original.json'):
    """
    Ablation: SRCNN restored images vs original HR images for U-Net training
    """
    print("\n" + "=" * 60)
    print("Ablation Study: SRCNN vs Original for U-Net")
    print("=" * 60)

    results = {}

    # Use the restored U-Net model (trained with SRCNN restored images)
    ckpt_path = 'checkpoints/unet_restored_best.pth'

    if not os.path.exists(ckpt_path):
        print(f"[ERROR] U-Net checkpoint not found: {ckpt_path}")
        return results

    device = DEVICE
    model = UNet(in_channels=3, out_channels=1, features=[64, 128, 256, 512]).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Test with original HR images
    print("\n[Evaluating] Original HR Images")
    metrics_original = evaluate_unet(model, split='test', use_restored=False, device=device)
    results['Original HR'] = metrics_original
    print(f"  IoU: {metrics_original['iou']:.4f}")
    print(f"  Dice: {metrics_original['dice']:.4f}")

    # Test with SRCNN restored images
    print("\n[Evaluating] SRCNN Restored Images")
    metrics_restored = evaluate_unet(model, split='test', use_restored=True, device=device)
    results['SRCNN Restored'] = metrics_restored
    print(f"  IoU: {metrics_restored['iou']:.4f}")
    print(f"  Dice: {metrics_restored['dice']:.4f}")

    # Calculate improvement
    iou_improvement = metrics_restored['iou'] - metrics_original['iou']
    print(f"\n[Improvement] IoU: {iou_improvement:+.4f}")

    # Save results
    os.makedirs('results', exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_file}")

    return results


def run_all_ablations():
    """Run all ablation studies"""
    print("\n" + "=" * 60)
    print("Running All Ablation Studies")
    print("=" * 60)

    all_results = {}

    # 1. Smart degradation
    try:
        all_results['smart_degradation'] = run_ablation_smart_degradation()
    except Exception as e:
        print(f"Smart degradation ablation failed: {e}")

    # 2. Positive weight
    try:
        all_results['pos_weight'] = run_ablation_pos_weight()
    except Exception as e:
        print(f"Pos weight ablation failed: {e}")

    # 3. SRCNN vs Original
    try:
        all_results['srcnn_vs_original'] = run_ablation_srcnn_vs_original()
    except Exception as e:
        print(f"SRCNN vs Original ablation failed: {e}")

    # Print summary table
    print("\n" + "=" * 60)
    print("ABLATION STUDY SUMMARY")
    print("=" * 60)

    return all_results


def print_ablation_instructions():
    """Print instructions for running ablation experiments"""
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                    ABLATION EXPERIMENT INSTRUCTIONS                       ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  Ablation 1: Smart Degradation                                           ║
║  ─────────────────────────────                                           ║
║  1. Edit config.py: SMART_DEGRADATION = False                            ║
║  2. python main.py --mode preprocess --split all                         ║
║  3. python main.py --mode train-srcnn --epochs-srcnn 100                 ║
║  4. cp checkpoints/srcnn_best.pth checkpoints/srcnn_no_smart.pth         ║
║  5. Edit config.py: SMART_DEGRADATION = True                             ║
║  6. python main.py --mode preprocess --split all                         ║
║  7. python main.py --mode train-srcnn --epochs-srcnn 100                 ║
║  8. python scripts/ablation_study.py --ablation smart                    ║
║                                                                          ║
║  Ablation 2: Positive Weight                                             ║
║  ────────────────────────                                                ║
║  for w in 1.0 3.0 5.0 7.0; do                                            ║
║      python main.py --mode train-unet --pos-weight $w --epochs-unet 50   ║
║      cp checkpoints/unet_best.pth checkpoints/unet_pos${w}.pth           ║
║  done                                                                    ║
║  python scripts/ablation_study.py --ablation pos_weight                  ║
║                                                                          ║
║  Ablation 3: SRCNN vs Original (Auto)                                    ║
║  ─────────────────────────────────────                                   ║
║  python scripts/ablation_study.py --ablation srcnn_vs_original           ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
""")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run ablation studies')
    parser.add_argument('--ablation', type=str, default='all',
                       choices=['all', 'smart', 'pos_weight', 'srcnn_vs_original', 'instructions'],
                       help='Which ablation to run')
    parser.add_argument('--split', type=str, default='test',
                       help='Dataset split to evaluate')

    args = parser.parse_args()

    if args.ablation == 'instructions':
        print_ablation_instructions()
    elif args.ablation == 'smart':
        run_ablation_smart_degradation()
    elif args.ablation == 'pos_weight':
        run_ablation_pos_weight()
    elif args.ablation == 'srcnn_vs_original':
        run_ablation_srcnn_vs_original()
    elif args.ablation == 'all':
        run_all_ablations()
