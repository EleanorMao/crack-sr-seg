"""
Image Preprocessing Module
- Degrade high-quality images to low-quality (blur, downsample, compress, combined)
- Supports mask-guided smart degradation for crack regions
"""
import os
import cv2
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil

from config import (
    IMAGE_DIR, MASK_DIR, TRAIN_TXT, VAL_TXT, TEST_TXT,
    PROCESSED_DIR, LR_IMAGE_DIR, HR_IMAGE_DIR, ENHANCED_MASK_DIR,
    PreprocessConfig, RANDOM_SEED
)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class ImageDegradation:
    """Image degradation utilities"""

    @staticmethod
    def apply_blur(img, kernel_size=None, sigma=None):
        if kernel_size is None:
            kernel_size = random.choice(range(
                PreprocessConfig.BLUR_KERNEL_RANGE[0],
                PreprocessConfig.BLUR_KERNEL_RANGE[1] + 1, 2
            ))
        if sigma is None:
            sigma = random.uniform(*PreprocessConfig.BLUR_SIGMA_RANGE)

        return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

    @staticmethod
    def _sample_odd_kernel(kernel_range):
        """Sample an odd kernel size from [min, max]."""
        k_min, k_max = kernel_range
        if k_min % 2 == 0:
            k_min += 1
        if k_max % 2 == 0:
            k_max -= 1
        if k_min > k_max:
            k_min = k_max = max(3, k_min)
        return random.choice(range(k_min, k_max + 1, 2))

    @staticmethod
    def apply_downsample(img, scale=None):
        if scale is None:
            scale = random.randint(*PreprocessConfig.DOWNSAMPLE_SCALE_RANGE)

        h, w = img.shape[:2]
        small = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def apply_jpeg_compress(img, quality=None):
        if quality is None:
            quality = random.randint(*PreprocessConfig.JPEG_QUALITY_RANGE)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', img, encode_param)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    @staticmethod
    def apply_combined(img):
        """Combined degradation: blur + downsample + compress"""
        result = img.copy()
        result = ImageDegradation.apply_blur(result)
        result = ImageDegradation.apply_downsample(result)
        result = ImageDegradation.apply_jpeg_compress(result)
        return result

    @staticmethod
    def apply_smart_degradation(img, mask):
        """
        Smart degradation with mask guidance:
        - Crack regions: strong blur
        - Background regions: light blur
        - Uses soft-edge mask blending to avoid hard edge artifacts
        """
        if mask is None:
            return ImageDegradation.degrade(img)

        crack_bin = (mask > PreprocessConfig.CRACK_MASK_THRESHOLD).astype(np.float32)
        if np.sum(crack_bin) == 0:
            return ImageDegradation.degrade(img)

        crack_kernel = ImageDegradation._sample_odd_kernel(PreprocessConfig.CRACK_BLUR_KERNEL_RANGE)
        crack_sigma = random.uniform(*PreprocessConfig.CRACK_BLUR_SIGMA_RANGE)
        crack_degraded = cv2.GaussianBlur(img, (crack_kernel, crack_kernel), crack_sigma)

        bg_kernel = ImageDegradation._sample_odd_kernel(PreprocessConfig.BG_BLUR_KERNEL_RANGE)
        bg_sigma = random.uniform(*PreprocessConfig.BG_BLUR_SIGMA_RANGE)
        bg_degraded = cv2.GaussianBlur(img, (bg_kernel, bg_kernel), bg_sigma)

        edge_kernel = PreprocessConfig.MASK_EDGE_BLUR_KERNEL
        if edge_kernel % 2 == 0:
            edge_kernel += 1
        soft_mask = cv2.GaussianBlur(crack_bin, (edge_kernel, edge_kernel), 0)
        soft_mask = np.clip(soft_mask, 0.0, 1.0).astype(np.float32)
        soft_mask_3ch = np.repeat(soft_mask[:, :, None], 3, axis=2)

        blended = crack_degraded.astype(np.float32) * soft_mask_3ch + \
                  bg_degraded.astype(np.float32) * (1.0 - soft_mask_3ch)
        return np.clip(blended, 0, 255).astype(np.uint8)

    @staticmethod
    def degrade(img, degradation_type=None):
        """Apply random degradation based on configured probabilities"""
        if degradation_type is None:
            probs = PreprocessConfig.DEGRADATION_PROBS
            r = random.random()
            cumulative = 0
            for dtype, prob in probs.items():
                cumulative += prob
                if r <= cumulative:
                    degradation_type = dtype
                    break
            else:
                degradation_type = 'blur'

        if degradation_type == 'blur':
            return ImageDegradation.apply_blur(img)
        elif degradation_type == 'downsample':
            return ImageDegradation.apply_downsample(img)
        elif degradation_type == 'compress':
            return ImageDegradation.apply_jpeg_compress(img)
        elif degradation_type == 'combined':
            return ImageDegradation.apply_combined(img)
        else:
            return ImageDegradation.apply_blur(img)


def has_crack(mask, threshold=100):
    """Check if mask contains crack (white pixel count exceeds threshold)"""
    if mask is None:
        return False
    white_pixels = np.sum(mask > 127)
    return white_pixels > threshold


def get_filename_from_path(path_str):
    """Extract filename without extension from path or filename string"""
    path_str = path_str.replace('\\', '/')
    filename = os.path.basename(path_str)
    return os.path.splitext(filename)[0]


def process_dataset(split='train'):
    """
    Process dataset split.
    Args:
        split: 'train', 'val', or 'test'
    """
    txt_file = {
        'train': TRAIN_TXT,
        'val': VAL_TXT,
        'test': TEST_TXT
    }.get(split, TRAIN_TXT)

    split_lr_dir = os.path.join(LR_IMAGE_DIR, split)
    split_hr_dir = os.path.join(HR_IMAGE_DIR, split)
    split_mask_dir = os.path.join(ENHANCED_MASK_DIR, split)

    os.makedirs(split_lr_dir, exist_ok=True)
    os.makedirs(split_hr_dir, exist_ok=True)
    os.makedirs(split_mask_dir, exist_ok=True)

    with open(txt_file, 'r') as f:
        lines = f.readlines()

    filenames = [get_filename_from_path(line.strip()) for line in lines if line.strip()]

    print(f"Processing {split} dataset: {len(filenames)} images")

    processed_count = 0
    crack_count = 0

    for filename in tqdm(filenames, desc=f"Processing {split}"):
        img_path = os.path.join(IMAGE_DIR, f"{filename}.jpg")
        mask_path = os.path.join(MASK_DIR, f"{filename}.png")

        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image: {img_path}")
            continue

        mask = None
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, PreprocessConfig.IMG_SIZE)
        if mask is not None:
            mask = cv2.resize(mask, PreprocessConfig.IMG_SIZE, interpolation=cv2.INTER_NEAREST)

        if mask is not None and PreprocessConfig.SMART_DEGRADATION:
            degraded_img = ImageDegradation.apply_smart_degradation(img, mask)
        else:
            degraded_img = ImageDegradation.degrade(img)

        hr_path = os.path.join(split_hr_dir, f"{filename}.png")
        cv2.imwrite(hr_path, img)

        lr_path = os.path.join(split_lr_dir, f"{filename}.png")
        cv2.imwrite(lr_path, degraded_img)

        if mask is not None:
            mask_save_path = os.path.join(split_mask_dir, f"{filename}.png")
            cv2.imwrite(mask_save_path, mask)

        if has_crack(mask):
            crack_count += 1

        processed_count += 1

    print(f"\n{split} processing complete:")
    print(f"  - Total images: {processed_count}")
    print(f"  - Images with cracks: {crack_count}")

    return processed_count


def preprocess_all():
    """Process all dataset splits"""
    print("=" * 50)
    print("Starting dataset preprocessing")
    print("=" * 50)

    for split in ['train', 'val', 'test']:
        process_dataset(split)

    print("\nAll datasets preprocessing complete!")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Image Preprocessing')
    parser.add_argument('--split', type=str, default='all', choices=['train', 'val', 'test', 'all'],
                        help='Which dataset split to process')

    args = parser.parse_args()

    if args.split == 'all':
        preprocess_all()
    else:
        process_dataset(args.split)
