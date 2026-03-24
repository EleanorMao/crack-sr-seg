"""SRCNN Testing Code"""
import os
import torch
import cv2
import numpy as np
from tqdm import tqdm

from config import (
    DEVICE, SRCNN_CHECKPOINT, RESTORED_DIR,
    LR_IMAGE_DIR, HR_IMAGE_DIR, SRCNNConfig
)
from srcnn.model import SRCNN, ImprovedSRCNN, compute_psnr, compute_ssim
from srcnn.dataset import get_test_loader


class SRCNNTester:
    """SRCNN Tester"""

    def __init__(self, model_type='srcnn', checkpoint_path=None, device=None):
        self.device = device if device else DEVICE
        print(f"Using device: {self.device}")

        if model_type == 'improved':
            self.model = ImprovedSRCNN(
                num_channels=SRCNNConfig.NUM_CHANNELS,
                num_features=SRCNNConfig.NUM_FEATURES
            ).to(self.device)
        else:
            self.model = SRCNN(
                num_channels=SRCNNConfig.NUM_CHANNELS,
                num_features=SRCNNConfig.NUM_FEATURES
            ).to(self.device)

        self.load_checkpoint(checkpoint_path)
        self.model.eval()

    def load_checkpoint(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = SRCNN_CHECKPOINT

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model: {checkpoint_path}")
        if 'best_psnr' in checkpoint:
            print(f"Training best PSNR: {checkpoint['best_psnr']:.4f} dB")

    def restore_image(self, lr_img):
        """
        Restore a single image.
        Args:
            lr_img: Low-quality image (numpy array, BGR, 0-255)
        Returns:
            Restored image (numpy array, BGR, 0-255)
        """
        rgb_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)

        img_tensor = torch.from_numpy(rgb_img.astype(np.float32) / 255.0)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img_tensor)

        output = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        return output

    def test(self, split='test', save_results=True, output_dir=None):
        """
        Test model on a dataset split.
        Args:
            split: Test split ('test', 'val', 'train')
            save_results: Whether to save restored images
            output_dir: Output directory
        Returns:
            metrics: Dict with PSNR and SSIM
        """
        if output_dir is None:
            # Use split-specific directory to avoid data leakage
            output_dir = os.path.join(RESTORED_DIR, split)

        if save_results:
            os.makedirs(output_dir, exist_ok=True)

        test_loader = get_test_loader(split=split)

        total_psnr = 0.0
        total_ssim = 0.0
        results = []

        print(f"\nTesting {split} dataset ({len(test_loader.dataset)} images)")

        with torch.no_grad():
            for lr_imgs, hr_imgs, filenames in tqdm(test_loader, desc="Testing"):
                lr_imgs = lr_imgs.to(self.device)
                hr_imgs = hr_imgs.to(self.device)

                outputs = self.model(lr_imgs)

                for i in range(outputs.size(0)):
                    psnr = compute_psnr(outputs[i:i+1], hr_imgs[i:i+1])
                    ssim = compute_ssim(outputs[i:i+1], hr_imgs[i:i+1])

                    total_psnr += psnr.item()
                    total_ssim += ssim.item()

                    results.append({
                        'filename': filenames[i],
                        'psnr': psnr.item(),
                        'ssim': ssim.item()
                    })

                if save_results:
                    for i in range(outputs.size(0)):
                        output = outputs[i].permute(1, 2, 0).cpu().numpy()
                        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
                        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

                        save_path = os.path.join(output_dir, filenames[i])
                        cv2.imwrite(save_path, output)

        avg_psnr = total_psnr / len(test_loader.dataset)
        avg_ssim = total_ssim / len(test_loader.dataset)

        print(f"\nResults:")
        print(f"  Avg PSNR: {avg_psnr:.4f} dB")
        print(f"  Avg SSIM: {avg_ssim:.4f}")
        print(f"  Saved to: {output_dir}")

        return {
            'avg_psnr': avg_psnr,
            'avg_ssim': avg_ssim,
            'results': results
        }

    def restore_directory(self, input_dir, output_dir):
        """Restore all images in a directory."""
        os.makedirs(output_dir, exist_ok=True)

        image_files = [f for f in os.listdir(input_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        print(f"Restoring {len(image_files)} images...")

        for filename in tqdm(image_files, desc="Restoring"):
            input_path = os.path.join(input_dir, filename)

            lr_img = cv2.imread(input_path)
            if lr_img is None:
                print(f"Failed to read: {input_path}")
                continue

            restored = self.restore_image(lr_img)

            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, restored)

        print(f"Restoration complete, saved to: {output_dir}")


def test_srcnn(model_type='srcnn', split='test', save_results=True, device=None):
    tester = SRCNNTester(model_type=model_type, device=device)
    return tester.test(split=split, save_results=save_results)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test SRCNN')
    parser.add_argument('--model', type=str, default='srcnn', choices=['srcnn', 'improved'],
                        help='Model type')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val', 'train'],
                        help='Test split')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'],
                        help='Compute device')
    parser.add_argument('--input-dir', type=str, default=None,
                        help='Input directory (for directory restoration)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--no-save', action='store_true', help='Do not save results')

    args = parser.parse_args()

    tester = SRCNNTester(
        model_type=args.model,
        checkpoint_path=args.checkpoint,
        device=args.device
    )

    if args.input_dir:
        output_dir = args.output_dir or RESTORED_DIR
        tester.restore_directory(args.input_dir, output_dir)
    else:
        output_dir = args.output_dir or RESTORED_DIR
        tester.test(
            split=args.split,
            save_results=not args.no_save,
            output_dir=output_dir
        )
