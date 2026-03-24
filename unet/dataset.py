"""U-Net Dataset Loader"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import random

from config import (
    HR_IMAGE_DIR, RESTORED_DIR, ENHANCED_MASK_DIR, UNetConfig, DEVICE
)


class UNetDataset(Dataset):
    """U-Net Training Dataset"""

    def __init__(self, split='train', use_restored=True, transform=None,
                 input_dir=None, mask_dir=None):
        self.split = split
        self.transform = transform

        if input_dir is not None:
            self.input_dir = input_dir
        elif use_restored:
            self.input_dir = RESTORED_DIR
            if not os.path.exists(self.input_dir):
                print(f"Warning: Restored directory not found {self.input_dir}, using original")
                self.input_dir = os.path.join(HR_IMAGE_DIR, split)
        else:
            self.input_dir = os.path.join(HR_IMAGE_DIR, split)

        if mask_dir is not None:
            self.mask_dir = mask_dir
        else:
            self.mask_dir = os.path.join(ENHANCED_MASK_DIR, split)

        if not os.path.exists(self.input_dir):
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        if not os.path.exists(self.mask_dir):
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        self.filenames = []
        for f in os.listdir(self.input_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                mask_name = os.path.splitext(f)[0] + '.png'
                if os.path.exists(os.path.join(self.mask_dir, mask_name)):
                    self.filenames.append(f)

        print(f"Loaded {split} dataset: {len(self.filenames)} images (input: {self.input_dir})")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        img_path = os.path.join(self.input_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to read image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask_name = os.path.splitext(filename)[0] + '.png'
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)

        if self.split == 'train':
            img, mask = self._augment(img, mask)

        img = img.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        return img_tensor, mask_tensor, filename

    def _augment(self, img, mask):
        """Data augmentation"""
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)

        if random.random() > 0.5:
            img = cv2.flip(img, 0)
            mask = cv2.flip(mask, 0)

        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            img = self._rotate(img, angle)
            mask = self._rotate(mask, angle)

        if random.random() > 0.5:
            brightness = random.uniform(0.8, 1.2)
            img = np.clip(img * brightness, 0, 255).astype(np.uint8)

        return img, mask

    def _rotate(self, img, angle):
        if angle == 90:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(img, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img


class UNetTestDataset(Dataset):
    """U-Net Test Dataset"""

    def __init__(self, split='test', use_restored=True, input_dir=None):
        if input_dir is not None:
            self.input_dir = input_dir
        elif use_restored:
            self.input_dir = RESTORED_DIR
        else:
            self.input_dir = os.path.join(HR_IMAGE_DIR, split)

        if not os.path.exists(self.input_dir):
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")

        self.filenames = sorted([f for f in os.listdir(self.input_dir)
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        print(f"Loaded {split} test set: {len(self.filenames)} images")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img_path = os.path.join(self.input_dir, filename)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)

        return img_tensor, filename


def get_unet_loaders(batch_size=None, num_workers=None, use_restored=True):
    """Get training and validation data loaders"""
    if batch_size is None:
        batch_size = UNetConfig.BATCH_SIZE
    if num_workers is None:
        num_workers = UNetConfig.NUM_WORKERS

    train_dataset = UNetDataset(split='train', use_restored=use_restored)
    val_dataset = UNetDataset(split='val', use_restored=use_restored)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if DEVICE == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if DEVICE == 'cuda' else False
    )

    return train_loader, val_loader


def get_unet_test_loader(split='test', use_restored=True, batch_size=None):
    """Get test data loader"""
    if batch_size is None:
        batch_size = 1

    test_dataset = UNetTestDataset(split=split, use_restored=use_restored)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if DEVICE == 'cuda' else False
    )

    return test_loader


if __name__ == '__main__':
    train_loader, val_loader = get_unet_loaders(batch_size=4, use_restored=False)

    for img, mask, name in train_loader:
        print(f"Image shape: {img.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Filename: {name}")
        break
