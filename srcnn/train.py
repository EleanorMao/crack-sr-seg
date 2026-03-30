"""SRCNN Training Code"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from config import DEVICE, CHECKPOINT_DIR, SRCNN_CHECKPOINT, IMPROVED_SRCNN_CHECKPOINT, IMPROVED_SRCNN_BN_CHECKPOINT, IMPROVED_SRCNN_ALL3X3_CHECKPOINT, SRCNNConfig
from srcnn.model import SRCNN, ImprovedSRCNN, ImprovedSRCNN_BN, ImprovedSRCNN_All3x3, compute_psnr, compute_ssim
from srcnn.dataset import get_srcnn_loaders


class SRCNNTrainer:
    """SRCNN Trainer"""

    # 模型类型映射
    MODEL_TYPES = {
        'srcnn': (SRCNN, 'SRCNN (baseline)'),
        'improved': (ImprovedSRCNN, 'Improved SRCNN'),
        'improved_bn': (ImprovedSRCNN_BN, 'Improved SRCNN with BatchNorm'),
        'improved_3x3': (ImprovedSRCNN_All3x3, 'Improved SRCNN all 3x3 kernels'),
    }

    # checkpoint路径映射
    CHECKPOINT_PATHS = {
        'srcnn': SRCNN_CHECKPOINT,
        'improved': IMPROVED_SRCNN_CHECKPOINT,
        'improved_bn': IMPROVED_SRCNN_BN_CHECKPOINT,
        'improved_3x3': IMPROVED_SRCNN_ALL3X3_CHECKPOINT,
    }

    def __init__(self, model_type='srcnn', device=None):
        self.device = device if device else DEVICE
        self.model_type = model_type
        print(f"Using device: {self.device}")

        # 验证模型类型
        if model_type not in self.MODEL_TYPES:
            raise ValueError(f"Unknown model_type: {model_type}. Valid options: {list(self.MODEL_TYPES.keys())}")

        # 获取checkpoint路径
        self.checkpoint_path = self.CHECKPOINT_PATHS[model_type]
        print(f"Model: {self.MODEL_TYPES[model_type][1]}")
        print(f"Checkpoint path: {self.checkpoint_path}")

        # 创建模型
        model_class = self.MODEL_TYPES[model_type][0]
        self.model = model_class(
            num_channels=SRCNNConfig.NUM_CHANNELS,
            num_features=SRCNNConfig.NUM_FEATURES
        ).to(self.device)

        self.criterion = nn.MSELoss()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=SRCNNConfig.LEARNING_RATE,
            weight_decay=SRCNNConfig.WEIGHT_DECAY
        )

        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=SRCNNConfig.LR_STEP_SIZE,
            gamma=SRCNNConfig.LR_GAMMA
        )

        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        self.train_losses = []
        self.val_psnrs = []
        self.best_psnr = 0.0

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for lr_imgs, hr_imgs, _ in pbar:
            lr_imgs = lr_imgs.to(self.device)
            hr_imgs = hr_imgs.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(lr_imgs)
            loss = self.criterion(outputs, hr_imgs)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate(self, val_loader):
        self.model.eval()
        total_psnr = 0.0
        total_ssim = 0.0

        with torch.no_grad():
            for lr_imgs, hr_imgs, _ in tqdm(val_loader, desc="Validating"):
                lr_imgs = lr_imgs.to(self.device)
                hr_imgs = hr_imgs.to(self.device)

                outputs = self.model(lr_imgs)

                for i in range(outputs.size(0)):
                    psnr = compute_psnr(outputs[i:i+1], hr_imgs[i:i+1])
                    ssim = compute_ssim(outputs[i:i+1], hr_imgs[i:i+1])
                    total_psnr += psnr.item()
                    total_ssim += ssim.item()

        avg_psnr = total_psnr / len(val_loader.dataset)
        avg_ssim = total_ssim / len(val_loader.dataset)

        self.val_psnrs.append(avg_psnr)
        return avg_psnr, avg_ssim

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_psnr': self.best_psnr,
            'train_losses': self.train_losses,
            'val_psnrs': self.val_psnrs
        }

        last_path = os.path.join(CHECKPOINT_DIR, 'srcnn_last.pth')
        torch.save(checkpoint, last_path)

        if is_best:
            torch.save(checkpoint, self.checkpoint_path)
            print(f"Saved best model (PSNR: {self.best_psnr:.4f})")

    def load_checkpoint(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_path

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.best_psnr = checkpoint.get('best_psnr', 0.0)
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_psnrs = checkpoint.get('val_psnrs', [])
            print(f"Loaded checkpoint: {checkpoint_path}")
            print(f"Best PSNR: {self.best_psnr:.4f}")
            return checkpoint.get('epoch', 0)
        return 0

    def train(self, num_epochs=None, batch_size=None, num_workers=None):
        if num_epochs is None:
            num_epochs = SRCNNConfig.NUM_EPOCHS

        train_loader, val_loader = get_srcnn_loaders(batch_size, num_workers)

        print(f"\nTraining SRCNN")
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {train_loader.batch_size}")
        print("-" * 50)

        start_epoch = 0

        for epoch in range(start_epoch + 1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 30)

            start_time = time.time()
            train_loss = self.train_epoch(train_loader, epoch)
            epoch_time = time.time() - start_time

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            print(f"Train Loss: {train_loss:.6f}")
            print(f"Learning Rate: {current_lr:.6f}")
            print(f"Time: {epoch_time:.2f}s")

            if epoch % SRCNNConfig.VAL_EVERY == 0 or epoch == num_epochs:
                val_psnr, val_ssim = self.validate(val_loader)
                print(f"Val PSNR: {val_psnr:.4f} dB")
                print(f"Val SSIM: {val_ssim:.4f}")

                is_best = val_psnr > self.best_psnr
                if is_best:
                    self.best_psnr = val_psnr

                if epoch % SRCNNConfig.SAVE_EVERY == 0 or epoch == num_epochs or is_best:
                    self.save_checkpoint(epoch, is_best)

        print("\nTraining complete!")
        print(f"Best val PSNR: {self.best_psnr:.4f} dB")

        return self.model


def train_srcnn(model_type='srcnn', epochs=None, batch_size=None, device=None):
    trainer = SRCNNTrainer(model_type=model_type, device=device)
    model = trainer.train(num_epochs=epochs, batch_size=batch_size)
    return model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train SRCNN')
    parser.add_argument('--model', type=str, default='srcnn',
                        choices=['srcnn', 'improved', 'improved_bn', 'improved_3x3'],
                        help='Model type: srcnn, improved, improved_bn (with BatchNorm), improved_3x3 (all 3x3 kernels)')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'],
                        help='Compute device')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint path to resume')

    args = parser.parse_args()

    trainer = SRCNNTrainer(model_type=args.model, device=args.device)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train(num_epochs=args.epochs, batch_size=args.batch_size)
