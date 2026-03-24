"""U-Net Training Code"""
import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from config import (
    DEVICE, CHECKPOINT_DIR, UNET_CHECKPOINT, UNetConfig
)
from unet.model import (
    UNet, CombinedLoss, DiceLoss,
    compute_iou, compute_dice_coeff, compute_pixel_accuracy
)
from unet.dataset import get_unet_loaders


class UNetTrainer:
    """U-Net Trainer"""

    def __init__(self, device=None, pos_weight=None):
        self.device = device if device else DEVICE
        print(f"Using device: {self.device}")

        if pos_weight is None:
            pos_weight = UNetConfig.POSITIVE_WEIGHT

        self.model = UNet(
            in_channels=UNetConfig.IN_CHANNELS,
            out_channels=UNetConfig.OUT_CHANNELS,
            features=UNetConfig.FEATURES
        ).to(self.device)

        self.criterion = CombinedLoss(
            bce_weight=0.5,
            dice_weight=0.5,
            pos_weight=pos_weight
        )

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=UNetConfig.LEARNING_RATE,
            weight_decay=UNetConfig.WEIGHT_DECAY
        )

        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=UNetConfig.LR_STEP_SIZE,
            gamma=UNetConfig.LR_GAMMA
        )

        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        self.train_losses = []
        self.val_ious = []
        self.best_iou = 0.0

        print(f"Positive weight: {pos_weight}")

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0.0
        total_iou = 0.0
        total_dice = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for imgs, masks, _ in pbar:
            imgs = imgs.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            loss = self.criterion(outputs, masks)

            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                batch_iou = compute_iou(outputs, masks)
                batch_dice = compute_dice_coeff(outputs, masks)

            total_loss += loss.item()
            total_iou += batch_iou.item()
            total_dice += batch_dice.item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{batch_iou.item():.4f}'
            })

        n = len(train_loader)
        avg_loss = total_loss / n
        avg_iou = total_iou / n
        avg_dice = total_dice / n

        self.train_losses.append(avg_loss)
        return avg_loss, avg_iou, avg_dice

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        total_iou = 0.0
        total_dice = 0.0
        total_acc = 0.0

        with torch.no_grad():
            for imgs, masks, _ in tqdm(val_loader, desc="Validating"):
                imgs = imgs.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(imgs)
                loss = self.criterion(outputs, masks)

                batch_iou = compute_iou(outputs, masks)
                batch_dice = compute_dice_coeff(outputs, masks)
                batch_acc = compute_pixel_accuracy(outputs, masks)

                total_loss += loss.item()
                total_iou += batch_iou.item()
                total_dice += batch_dice.item()
                total_acc += batch_acc.item()

        n = len(val_loader)
        metrics = {
            'loss': total_loss / n,
            'iou': total_iou / n,
            'dice': total_dice / n,
            'accuracy': total_acc / n
        }

        self.val_ious.append(metrics['iou'])
        return metrics

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_iou': self.best_iou,
            'train_losses': self.train_losses,
            'val_ious': self.val_ious
        }

        last_path = os.path.join(CHECKPOINT_DIR, 'unet_last.pth')
        torch.save(checkpoint, last_path)

        if is_best:
            torch.save(checkpoint, UNET_CHECKPOINT)
            print(f"Saved best model (IoU: {self.best_iou:.4f})")

    def load_checkpoint(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = UNET_CHECKPOINT

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.best_iou = checkpoint.get('best_iou', 0.0)
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_ious = checkpoint.get('val_ious', [])
            print(f"Loaded checkpoint: {checkpoint_path}")
            print(f"Best IoU: {self.best_iou:.4f}")
            return checkpoint.get('epoch', 0)
        return 0

    def train(self, num_epochs=None, batch_size=None, num_workers=None, use_restored=True):
        if num_epochs is None:
            num_epochs = UNetConfig.NUM_EPOCHS

        train_loader, val_loader = get_unet_loaders(
            batch_size=batch_size,
            num_workers=num_workers,
            use_restored=use_restored
        )

        print(f"\nTraining U-Net")
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {train_loader.batch_size}")
        print(f"Use restored images: {use_restored}")
        print("-" * 50)

        start_epoch = 0

        for epoch in range(start_epoch + 1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 30)

            start_time = time.time()
            train_loss, train_iou, train_dice = self.train_epoch(train_loader, epoch)
            epoch_time = time.time() - start_time

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Train IoU: {train_iou:.4f}")
            print(f"Train Dice: {train_dice:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            print(f"Time: {epoch_time:.2f}s")

            if epoch % UNetConfig.VAL_EVERY == 0 or epoch == num_epochs:
                val_metrics = self.validate(val_loader)
                print(f"Val Loss: {val_metrics['loss']:.4f}")
                print(f"Val IoU: {val_metrics['iou']:.4f}")
                print(f"Val Dice: {val_metrics['dice']:.4f}")
                print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")

                is_best = val_metrics['iou'] > self.best_iou
                if is_best:
                    self.best_iou = val_metrics['iou']

                if epoch % UNetConfig.SAVE_EVERY == 0 or epoch == num_epochs or is_best:
                    self.save_checkpoint(epoch, is_best)

        print("\nTraining complete!")
        print(f"Best val IoU: {self.best_iou:.4f}")

        return self.model


def train_unet(epochs=None, batch_size=None, device=None, use_restored=True, pos_weight=None):
    trainer = UNetTrainer(device=device, pos_weight=pos_weight)
    model = trainer.train(
        num_epochs=epochs,
        batch_size=batch_size,
        use_restored=use_restored
    )
    return model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train U-Net')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'],
                        help='Compute device')
    parser.add_argument('--use-original', action='store_true',
                        help='Use original images instead of restored')
    parser.add_argument('--pos-weight', type=float, default=None,
                        help='Positive sample weight')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint path to resume')

    args = parser.parse_args()

    trainer = UNetTrainer(device=args.device, pos_weight=args.pos_weight)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        use_restored=not args.use_original
    )
