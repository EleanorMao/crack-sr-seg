"""
Configuration file - all tunable parameters
"""
import torch

# ==================== General ====================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RANDOM_SEED = 42

# ==================== Data Paths ====================
DATASET_DIR = 'dataset'
IMAGE_DIR = f'{DATASET_DIR}/image'
MASK_DIR = f'{DATASET_DIR}/masks'

TRAIN_TXT = f'{DATASET_DIR}/ImageSets/train.txt'
VAL_TXT = f'{DATASET_DIR}/ImageSets/val.txt'
TEST_TXT = f'{DATASET_DIR}/ImageSets/test.txt'

# Output paths after preprocessing
PROCESSED_DIR = 'processed_data'
LR_IMAGE_DIR = f'{PROCESSED_DIR}/lr_images' 
HR_IMAGE_DIR = f'{PROCESSED_DIR}/hr_images'
ENHANCED_MASK_DIR = f'{PROCESSED_DIR}/enhanced_masks'

# Model checkpoint paths
CHECKPOINT_DIR = 'checkpoints'
SRCNN_CHECKPOINT = f'{CHECKPOINT_DIR}/srcnn_best.pth'
IMPROVED_SRCNN_CHECKPOINT = f'{CHECKPOINT_DIR}/improved_srcnn_best.pth'

# U-Net checkpoints (different for each training mode)
UNET_CHECKPOINT = f'{CHECKPOINT_DIR}/unet_best.pth'
UNET_CHECKPOINT_RESTORED = f'{CHECKPOINT_DIR}/unet_restored_best.pth'  # Trained with SRCNN restored images
UNET_CHECKPOINT_ORIGINAL = f'{CHECKPOINT_DIR}/unet_original_best.pth'  # Trained with original HR images

# Inference/output paths
OUTPUT_DIR = 'outputs'
RESTORED_DIR = f'{OUTPUT_DIR}/restored'  # Restored images
PREDICTIONS_DIR = f'{OUTPUT_DIR}/predictions'  # Segmentation predictions

# ==================== Preprocessing ====================
class PreprocessConfig:
    # Image size
    IMG_SIZE = (256, 256)  # Unified resize target

    # Degradation parameters
    BLUR_KERNEL_RANGE = (3, 9)  # Blur kernel size range
    BLUR_SIGMA_RANGE = (0.5, 3.0)  # Gaussian blur sigma range

    DOWNSAMPLE_SCALE_RANGE = (2, 4)  # Downsample factor range

    JPEG_QUALITY_RANGE = (30, 70)  # JPEG quality range

    # Degradation type probabilities
    DEGRADATION_PROBS = {
        'blur': 0.3,
        'downsample': 0.3,
        'compress': 0.2,
        'combined': 0.2  # Combined degradation
    }

    # Smart degradation (mask-guided regional degradation)
    SMART_DEGRADATION = True  # If True, apply stronger degradation on crack regions
    CRACK_MASK_THRESHOLD = 127  # Binarization threshold for crack mask

    # Crack region (stronger)
    CRACK_BLUR_KERNEL_RANGE = (7, 13)  # Strong blur kernel range (odd values)
    CRACK_BLUR_SIGMA_RANGE = (2.0, 4.0)  # Strong blur sigma range

    # Background region (lighter)
    BG_BLUR_KERNEL_RANGE = (3, 5)  # Light blur kernel range (odd values)
    BG_BLUR_SIGMA_RANGE = (0.3, 1.0)  # Light blur sigma range

    # Boundary smoothing for seamless blending between regions
    MASK_EDGE_BLUR_KERNEL = 9  # Gaussian smoothing kernel for soft mask edges


# ==================== SRCNN ====================
class SRCNNConfig:
    # Model parameters
    NUM_CHANNELS = 3  # Input channels (RGB)
    NUM_FEATURES = 64  # Number of feature maps
    SCALE_FACTOR = 2  # Upsampling scale factor

    # Training parameters
    BATCH_SIZE = 16
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5

    # Learning rate schedule
    LR_STEP_SIZE = 30
    LR_GAMMA = 0.1

    # Data loading
    NUM_WORKERS = 4

    # Validation / checkpointing
    SAVE_EVERY = 10  # Save every N epochs
    VAL_EVERY = 5  # Validate every N epochs


# ==================== U-Net ====================
class UNetConfig:
    # Model parameters
    IN_CHANNELS = 3  # Input channels
    OUT_CHANNELS = 1  # Output channels (binary mask)
    FEATURES = [64, 128, 256, 512]  # Feature sizes per stage

    # Training parameters
    BATCH_SIZE = 8
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5

    # Learning rate schedule
    LR_STEP_SIZE = 30
    LR_GAMMA = 0.1

    # Data loading
    NUM_WORKERS = 4

    # Validation / checkpointing
    SAVE_EVERY = 10
    VAL_EVERY = 5

    # Loss weighting (for crack regions)
    POSITIVE_WEIGHT = 5.0  # Positive class (crack) weight
