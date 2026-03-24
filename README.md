# Road Crack Restoration and Annotation System

A deep learning-based system for road crack image restoration and segmentation, consisting of two core modules:

1. **SRCNN** - Image Restoration: Restore low-quality crack images to high-resolution
2. **U-Net** - Crack Segmentation: Annotate cracks on restored images

## Project Structure

```
mark-crack/
├── config.py              # Configuration (all tunable parameters)
├── main.py                # Main entry point
├── preprocess.py          # Image preprocessing (degradation + smart degradation)
├── requirements.txt       # Python dependencies
├── dataset/               # Original dataset (CRACK500)
│   ├── image/            # Original images (.jpg)
│   ├── masks/            # Annotation masks (.png)
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
├── processed_data/        # Preprocessed data (auto-generated)
│   ├── lr_images/        # Low-quality images
│   ├── hr_images/        # High-quality images
│   └── enhanced_masks/   # Enhanced masks
├── checkpoints/           # Model checkpoints (auto-generated)
│   ├── srcnn_best.pth
│   └── unet_best.pth
├── outputs/               # Output results (auto-generated)
│   ├── restored/         # SRCNN restored images
│   └── predictions/      # U-Net segmentation predictions
├── srcnn/                 # SRCNN image restoration module
│   ├── model.py          # Model definition + PSNR/SSIM metrics
│   ├── dataset.py        # Data loader
│   ├── train.py          # Training code
│   └── test.py           # Testing code
└── unet/                  # U-Net segmentation module
    ├── model.py          # Model definition + Dice/IoU metrics
    ├── dataset.py        # Data loader
    ├── train.py          # Training code
    └── test.py           # Testing code
```

## Installation

```bash
pip install -r requirements.txt
```

Dependencies:
- torch >= 1.9.0
- torchvision >= 0.10.0
- numpy >= 1.19.0
- opencv-python >= 4.5.0
- tqdm >= 4.60.0

## Quick Start

### Full Pipeline (One Command)

```bash
python3 main.py --mode full
```

### Step-by-Step Execution

#### 1. Data Preprocessing

Degrade high-resolution images to low-quality. Supports mask-guided smart degradation (stronger on crack regions, lighter on background):

```bash
python3 main.py --mode preprocess --split all
```

#### 2. Train SRCNN (Image Restoration)

```bash
python3 main.py --mode train-srcnn --epochs 100 --model-type improved
```

Parameters:
- `--epochs`: Number of training epochs (default: 100)
- `--model-type`: Model type, `srcnn` (basic) or `improved` (deeper with BatchNorm)
- `--batch-size`: Batch size (default: 16)

#### 3. Test SRCNN (Generate Restored Images)

```bash
python3 main.py --mode test-srcnn --test-split test
```

Restored images are saved to `outputs/restored/`.

#### 4. Train U-Net (Crack Segmentation)

```bash
python3 main.py --mode train-unet --use-restored --pos-weight 5.0
```

Parameters:
- `--use-restored`: Use SRCNN restored images for training
- `--pos-weight`: Positive sample weight for balancing crack/non-crack pixels (default: 5.0)
- `--epochs`: Number of training epochs (default: 100)

#### 5. Test U-Net (Predict Cracks)

```bash
python3 main.py --mode test-unet --use-restored --threshold 0.5
```

Predictions are saved to `outputs/predictions/`.

## Configuration

All tunable parameters are in `config.py`:

### Preprocessing Config
```python
class PreprocessConfig:
    IMG_SIZE = (256, 256)           # Unified resize target
    BLUR_KERNEL_RANGE = (3, 9)      # Blur kernel size range
    DOWNSAMPLE_SCALE_RANGE = (2, 4) # Downsample factor range
    JPEG_QUALITY_RANGE = (30, 70)   # JPEG compression quality range
    SMART_DEGRADATION = True        # Enable smart degradation
```

### SRCNN Config
```python
class SRCNNConfig:
    NUM_FEATURES = 64      # Number of feature maps
    BATCH_SIZE = 16        # Batch size
    NUM_EPOCHS = 100       # Number of epochs
    LEARNING_RATE = 1e-4   # Learning rate
```

### U-Net Config
```python
class UNetConfig:
    FEATURES = [64, 128, 256, 512]  # Feature sizes per stage
    BATCH_SIZE = 8                   # Batch size
    NUM_EPOCHS = 100                 # Number of epochs
    LEARNING_RATE = 1e-4             # Learning rate
    POSITIVE_WEIGHT = 5.0            # Crack pixel weight
```

## Core Features

### 1. Image Degradation Preprocessing

Supports 4 degradation types:
- **blur**: Gaussian blur
- **downsample**: Downsample then upsample
- **compress**: JPEG compression
- **combined**: Combination of the above 3

One degradation type is randomly selected each time.

### 2. Smart Degradation (Mask-Guided)

Mask-guided regional degradation to help the model focus on crack restoration:
- Stronger blur parameters on crack regions
- Lighter blur parameters on background regions
- Soft-edge mask blending to reduce boundary artifacts

### 3. SRCNN Image Restoration

Two model options:
- **SRCNN**: Classic 3-layer convolutional network
- **ImprovedSRCNN**: Deeper network with BatchNorm

Evaluation Metrics: PSNR, SSIM

### 4. U-Net Crack Segmentation

- Standard U-Net architecture
- Combined loss function: BCE + Dice Loss
- Positive sample weighting for class imbalance

Evaluation Metrics: IoU, Dice Coefficient, Pixel Accuracy

## Command Line Arguments

```
usage: main.py [-h] [--mode MODE] [--device DEVICE] [--split SPLIT]
               [--model-type MODEL_TYPE] [--epochs-srcnn EPOCHS_SRCNN]
               [--batch-size BATCH_SIZE]
               [--resume-srcnn RESUME_SRCNN] [--checkpoint-srcnn CHECKPOINT_SRCNN]
               [--test-split TEST_SPLIT] [--output-restored OUTPUT_RESTORED]
               [--epochs-unet EPOCHS_UNET] [--batch-size-unet BATCH_SIZE_UNET]
               [--pos-weight POS_WEIGHT] [--use-restored] [--resume-unet RESUME_UNET]
               [--checkpoint-unet CHECKPOINT_UNET] [--output-predictions OUTPUT_PREDICTIONS]
               [--threshold THRESHOLD]
```

| Parameter | Description | Default |
|------------|-------------|---------|
| `--mode` | Run mode | full |
| `--device` | cuda/cpu | auto-detect |
| `--model-type` | SRCNN model type | srcnn |
| `--pos-weight` | Crack pixel weight | 5.0 |
| `--use-restored` | U-Net uses restored images | True |
| `--threshold` | Binarization threshold | 0.5 |

## Example: Using Modules Independently

```python
# Use SRCNN to restore images
from srcnn.test import SRCNNTester

tester = SRCNNTester(model_type='improved')
restored = tester.restore_image(low_quality_img)

# Use U-Net to predict cracks
from unet.test import UNetTester

tester = UNetTester()
mask = tester.predict_binary(restored_img)
```

## Dataset Format

This project uses the CRACK500 dataset with the following structure:

```
dataset/
├── image/
│   ├── xxx.jpg
│   └── ...
└── masks/
    ├── xxx.png  # Same name as image, white=crack, black=background
    └── ...
```

## License

MIT
