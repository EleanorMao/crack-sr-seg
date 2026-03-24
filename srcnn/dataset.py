"""SRCNN Dataset Loader"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from torchvision import transforms

from config import (
    LR_IMAGE_DIR, HR_IMAGE_DIR, SRCNNConfig, DEVICE
)


class SRCNNDataset(Dataset):
    """SRCNN Training Dataset"""

    def __init__(self, split='train', transform=None):
        self.split = split
        self.transform = transform

        self.lr_dir = os.path.join(LR_IMAGE_DIR, split)
        self.hr_dir = os.path.join(HR_IMAGE_DIR, split)

        if not os.path.exists(self.lr_dir):
            raise FileNotFoundError(f"LR image directory not found: {self.lr_dir}")

        self.filenames = [f for f in os.listdir(self.lr_dir)
                         if f.endswith(('.png', '.jpg', '.jpeg'))]

        print(f"Loaded {split} dataset: {len(self.filenames)} images")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        lr_path = os.path.join(self.lr_dir, filename)
        hr_path = os.path.join(self.hr_dir, filename)

        lr_img = cv2.imread(lr_path)
        hr_img = cv2.imread(hr_path)

        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)

        lr_img = lr_img.astype(np.float32) / 255.0
        hr_img = hr_img.astype(np.float32) / 255.0

        lr_tensor = torch.from_numpy(lr_img).permute(2, 0, 1)
        hr_tensor = torch.from_numpy(hr_img).permute(2, 0, 1)

        if self.transform:
            lr_tensor = self.transform(lr_tensor)
            hr_tensor = self.transform(hr_tensor)

        return lr_tensor, hr_tensor, filename


class SRCNNTestDataset(Dataset):
    """SRCNN Test Dataset"""

    def __init__(self, split='test'):
        self.split = split

        self.lr_dir = os.path.join(LR_IMAGE_DIR, split)
        self.hr_dir = os.path.join(HR_IMAGE_DIR, split)

        if not os.path.exists(self.lr_dir):
            raise FileNotFoundError(f"LR image directory not found: {self.lr_dir}")

        self.filenames = sorted([f for f in os.listdir(self.lr_dir)
                                if f.endswith(('.png', '.jpg', '.jpeg'))])

        print(f"Loaded {split} dataset: {len(self.filenames)} images")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        lr_path = os.path.join(self.lr_dir, filename)
        hr_path = os.path.join(self.hr_dir, filename)

        lr_img = cv2.imread(lr_path)
        hr_img = cv2.imread(hr_path) if os.path.exists(hr_path) else None

        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
        if hr_img is not None:
            hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)

        lr_img = lr_img.astype(np.float32) / 255.0
        lr_tensor = torch.from_numpy(lr_img).permute(2, 0, 1)

        if hr_img is not None:
            hr_img = hr_img.astype(np.float32) / 255.0
            hr_tensor = torch.from_numpy(hr_img).permute(2, 0, 1)
        else:
            hr_tensor = torch.zeros_like(lr_tensor)

        return lr_tensor, hr_tensor, filename


def get_srcnn_loaders(batch_size=None, num_workers=None):
    """Get training and validation data loaders"""
    if batch_size is None:
        batch_size = SRCNNConfig.BATCH_SIZE
    if num_workers is None:
        num_workers = SRCNNConfig.NUM_WORKERS

    train_dataset = SRCNNDataset(split='train')
    val_dataset = SRCNNDataset(split='val')

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


def get_test_loader(split='test', batch_size=None):
    """Get test data loader"""
    if batch_size is None:
        batch_size = 1

    test_dataset = SRCNNTestDataset(split=split)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if DEVICE == 'cuda' else False
    )

    return test_loader


if __name__ == '__main__':
    train_loader, val_loader = get_srcnn_loaders(batch_size=4)

    for lr, hr, name in train_loader:
        print(f"LR shape: {lr.shape}")
        print(f"HR shape: {hr.shape}")
        print(f"Filename: {name}")
        break
