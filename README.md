# Road Crack Restoration and Annotation System

A deep learning-based system for road crack image restoration and segmentation, consisting of two core modules:

1. **SRCNN** - Image Restoration: Restore low-quality crack images to high-resolution
2. **U-Net** - Crack Segmentation: Annotate cracks on restored images

## Project Structure

```
crack-sr-seg/
├── config.py              # Configuration (all tunable parameters)
├── main.py                # Main entry point
├── preprocess.py          # Image preprocessing (degradation + smart degradation)
├── requirements.txt       # Python dependencies
├── dataset/               # Original dataset (CRACK500)
│   ├── image/            # Original images (.jpg)
│   └── masks/            # Annotation masks (.png)
├── processed_data/        # Preprocessed data (auto-generated)
│   ├── lr_images/        # Low-quality images
│   ├── hr_images/        # High-quality images
│   └── enhanced_masks/   # Enhanced masks
├── checkpoints/           # Model checkpoints
│   ├── srcnn_best.pth              # Basic SRCNN
│   ├── improved_srcnn_best.pth     # Improved SRCNN
│   ├── unet_original_best.pth      # U-Net (original HR)
│   ├── unet_restored_best.pth      # U-Net (basic SRCNN restored)
│   └── unet_improved_best.pth      # U-Net (improved SRCNN restored)
├── outputs/               # Output results
│   ├── restored/                  # Basic SRCNN restored images
│   ├── restored_improved/         # Improved SRCNN restored images
│   ├── predictions_original/      # U-Net predictions (original HR)
│   ├── predictions_restored/      # U-Net predictions (basic SRCNN)
│   └── predictions_improved/      # U-Net predictions (improved SRCNN)
├── srcnn/                 # SRCNN module
│   ├── model.py
│   ├── dataset.py
│   ├── train.py
│   └── test.py
└── unet/                  # U-Net module
    ├── model.py
    ├── dataset.py
    ├── train.py
    └── test.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Full Pipeline

```bash
python main.py --mode full
```

### Step-by-Step Execution

#### 1. Data Preprocessing

```bash
python main.py --mode preprocess --split all
```

#### 2. Train SRCNN

```bash
# Basic SRCNN
python main.py --mode train-srcnn --model-type srcnn --epochs-srcnn 100

# Improved SRCNN
python main.py --mode train-srcnn --model-type improved --epochs-srcnn 100
```

#### 3. Test SRCNN (Generate Restored Images)

```bash
# Basic SRCNN -> outputs/restored/
python main.py --mode test-srcnn --model-type srcnn --test-split all

# Improved SRCNN -> outputs/restored_improved/
python main.py --mode test-srcnn --model-type improved --test-split all
```

#### 4. Train U-Net

```bash
# Original HR images
python main.py --mode train-unet --input-mode original --epochs-unet 100

# Basic SRCNN restored images
python main.py --mode train-unet --input-mode restored --epochs-unet 100

# Improved SRCNN restored images
python main.py --mode train-unet --input-mode improved --epochs-unet 100
```

#### 5. Test U-Net

```bash
# Original -> outputs/predictions_original/
python main.py --mode test-unet --input-mode original --test-split test

# Basic SRCNN -> outputs/predictions_restored/
python main.py --mode test-unet --input-mode restored --test-split test

# Improved SRCNN -> outputs/predictions_improved/
python main.py --mode test-unet --input-mode improved --test-split test
```

## Model Comparison Table

| Method | SRCNN PSNR | U-Net IoU | U-Net Dice |
|--------|------------|-----------|------------|
| Original HR + U-Net | - | ~60% | ~73% |
| Basic SRCNN + U-Net | ~27 dB | ~87% | ~92% |
| Improved SRCNN + U-Net | ~30 dB | ? | ? |

## Command Line Arguments

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--mode` | Run mode (preprocess/train-srcnn/test-srcnn/train-unet/test-unet/full) | full |
| `--model-type` | SRCNN type (srcnn/improved) | srcnn |
| `--input-mode` | U-Net input (original/restored/improved) | restored |
| `--epochs-srcnn` | SRCNN epochs | 100 |
| `--epochs-unet` | U-Net epochs | 100 |
| `--pos-weight` | Crack pixel weight | 5.0 |
| `--threshold` | Binarization threshold | 0.5 |

## License

MIT
