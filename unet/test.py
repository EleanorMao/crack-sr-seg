"""U-Net Testing Code"""
import os
import torch
import cv2
import numpy as np
from tqdm import tqdm

from config import (
    DEVICE, UNET_CHECKPOINT, UNET_CHECKPOINT_RESTORED, UNET_CHECKPOINT_ORIGINAL,
    PREDICTIONS_DIR, HR_IMAGE_DIR, RESTORED_DIR, ENHANCED_MASK_DIR, UNetConfig
)
from unet.model import (
    UNet, compute_iou, compute_dice_coeff, compute_pixel_accuracy
)
from unet.dataset import get_unet_test_loader, UNetDataset


class UNetTester:
    """U-Net Tester"""

    def __init__(self, checkpoint_path=None, device=None, use_restored=True):
        self.device = device if device else DEVICE
        self.use_restored = use_restored
        print(f"Using device: {self.device}")

        # Select default checkpoint based on use_restored
        if checkpoint_path is None:
            if use_restored:
                checkpoint_path = UNET_CHECKPOINT_RESTORED
            else:
                checkpoint_path = UNET_CHECKPOINT_ORIGINAL

        self.model = UNet(
            in_channels=UNetConfig.IN_CHANNELS,
            out_channels=UNetConfig.OUT_CHANNELS,
            features=UNetConfig.FEATURES
        ).to(self.device)

        self.load_checkpoint(checkpoint_path)
        self.model.eval()

    def load_checkpoint(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = UNET_CHECKPOINT

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model: {checkpoint_path}")
        if 'best_iou' in checkpoint:
            print(f"Training best IoU: {checkpoint['best_iou']:.4f}")

    def predict(self, img):
        """
        Predict mask for a single image.
        Args:
            img: Input image (numpy array, BGR, 0-255)
        Returns:
            mask: Predicted mask (numpy array, 0-255)
        """
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_tensor = torch.from_numpy(rgb_img.astype(np.float32) / 255.0)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img_tensor)
            output = torch.sigmoid(output)

        mask = output.squeeze(0).squeeze(0).cpu().numpy()
        mask = (mask * 255).astype(np.uint8)

        return mask

    def predict_binary(self, img, threshold=0.5):
        mask = self.predict(img)
        binary_mask = (mask > threshold * 255).astype(np.uint8) * 255
        return binary_mask

    def test(self, split='test', save_results=True, output_dir=None,
             use_restored=True, threshold=0.5):
        """
        Test model on a dataset split.
        Args:
            split: Test split ('test', 'val', 'train')
            save_results: Whether to save predictions
            output_dir: Output directory
            use_restored: Whether to use restored images
            threshold: Binarization threshold
        Returns:
            metrics: Dict with evaluation metrics
        """
        if output_dir is None:
            output_dir = PREDICTIONS_DIR

        if save_results:
            os.makedirs(output_dir, exist_ok=True)

        test_dataset = UNetDataset(split=split, use_restored=use_restored)

        total_iou = 0.0
        total_dice = 0.0
        total_acc = 0.0
        results = []

        print(f"\nTesting {split} dataset ({len(test_dataset)} images)")

        for i in tqdm(range(len(test_dataset)), desc="Testing"):
            img, mask_gt, filename = test_dataset[i]

            with torch.no_grad():
                img_input = img.unsqueeze(0).to(self.device)
                output = self.model(img_input)
                # Don't apply sigmoid here - compute_iou will do it

            mask_tensor = mask_gt.unsqueeze(0).to(self.device)
            iou = compute_iou(output, mask_tensor).item()
            dice = compute_dice_coeff(output, mask_tensor).item()
            acc = compute_pixel_accuracy(output, mask_tensor).item()

            total_iou += iou
            total_dice += dice
            total_acc += acc

            results.append({
                'filename': filename,
                'iou': iou,
                'dice': dice,
                'accuracy': acc
            })

            if save_results:
                pred_mask = torch.sigmoid(output).squeeze(0).squeeze(0).cpu().numpy()
                pred_mask = (pred_mask * 255).astype(np.uint8)

                vis = self._create_visualization(
                    img.cpu().numpy(),
                    mask_gt.cpu().numpy(),
                    pred_mask
                )

                save_name = os.path.splitext(filename)[0] + '_pred.png'
                save_path = os.path.join(output_dir, save_name)
                cv2.imwrite(save_path, vis)

        n = len(test_dataset)
        avg_iou = total_iou / n
        avg_dice = total_dice / n
        avg_acc = total_acc / n

        print(f"\nResults:")
        print(f"  Avg IoU: {avg_iou:.4f}")
        print(f"  Avg Dice: {avg_dice:.4f}")
        print(f"  Avg Accuracy: {avg_acc:.4f}")
        print(f"  Saved to: {output_dir}")

        return {
            'avg_iou': avg_iou,
            'avg_dice': avg_dice,
            'avg_accuracy': avg_acc,
            'results': results
        }

    def _create_visualization(self, img, gt_mask, pred_mask):
        """Create visualization: image | GT mask | prediction"""
        img = np.transpose(img, (1, 2, 0))
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        gt_mask = (gt_mask.squeeze() * 255).astype(np.uint8)

        gt_colored = cv2.applyColorMap(gt_mask, cv2.COLORMAP_JET)
        pred_colored = cv2.applyColorMap(pred_mask, cv2.COLORMAP_JET)

        vis = np.hstack([img, gt_colored, pred_colored])

        return vis

    def predict_directory(self, input_dir, output_dir, threshold=0.5):
        """Predict masks for all images in a directory."""
        os.makedirs(output_dir, exist_ok=True)

        image_files = [f for f in os.listdir(input_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        print(f"Predicting {len(image_files)} images...")

        for filename in tqdm(image_files, desc="Predicting"):
            input_path = os.path.join(input_dir, filename)

            img = cv2.imread(input_path)
            if img is None:
                print(f"Failed to read: {input_path}")
                continue

            mask = self.predict_binary(img, threshold)

            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, mask)

        print(f"Prediction complete, saved to: {output_dir}")


def test_unet(split='test', save_results=True, use_restored=True, device=None):
    tester = UNetTester(device=device)
    return tester.test(split=split, save_results=save_results, use_restored=use_restored)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test U-Net')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val', 'train'],
                        help='Test split')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'],
                        help='Compute device')
    parser.add_argument('--input-dir', type=str, default=None,
                        help='Input directory (for directory prediction)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--no-save', action='store_true', help='Do not save results')
    parser.add_argument('--use-original', action='store_true',
                        help='Use original images instead of restored')
    parser.add_argument('--threshold', type=float, default=0.5, help='Binarization threshold')

    args = parser.parse_args()

    tester = UNetTester(checkpoint_path=args.checkpoint, device=args.device)

    if args.input_dir:
        output_dir = args.output_dir or PREDICTIONS_DIR
        tester.predict_directory(args.input_dir, output_dir, args.threshold)
    else:
        output_dir = args.output_dir or PREDICTIONS_DIR
        tester.test(
            split=args.split,
            save_results=not args.no_save,
            output_dir=output_dir,
            use_restored=not args.use_original,
            threshold=args.threshold
        )
