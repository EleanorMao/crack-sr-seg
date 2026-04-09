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
│   ├── srcnn_best.pth                    # Basic SRCNN
│   ├── improved_srcnn_best.pth           # Improved SRCNN
│   ├── improved_srcnn_all3x3_best.pth    # Improved SRCNN (all 3x3 kernels)
│   ├── improved_srcnn_bn_best.pth        # Improved SRCNN (with BatchNorm)
│   ├── improved_srcnn_5l_rf15_best.pth   # Improved SRCNN (5-layer RF=15) [ablation]
│   ├── unet_original_best.pth            # U-Net (original HR)
│   ├── unet_restored_best.pth            # U-Net (basic SRCNN restored)
│   ├── unet_improved_best.pth            # U-Net (improved SRCNN restored)
│   ├── unet_improved_3x3_best.pth        # U-Net (improved_3x3 SRCNN restored)
│   └── unet_improved_5l_rf15_best.pth    # U-Net (improved_5l_rf15 SRCNN restored) [ablation]
├── outputs/               # Output results
│   ├── restored/                        # Basic SRCNN restored images
│   ├── restored_improved/               # Improved SRCNN restored images
│   ├── restored_improved_bn/            # Improved SRCNN (BatchNorm) restored images
│   ├── restored_improved_3x3/           # Improved 3x3 SRCNN restored images
│   ├── restored_improved_5l_rf15/       # Improved 5-layer RF=15 SRCNN restored images [ablation]
│   ├── predictions_original/            # U-Net predictions (original HR)
│   ├── predictions_restored/            # U-Net predictions (basic SRCNN)
│   ├── predictions_improved/            # U-Net predictions (improved SRCNN)
│   ├── predictions_improved_3x3/        # U-Net predictions (improved_3x3 SRCNN)
│   └── predictions_improved_5l_rf15/    # U-Net predictions (improved_5l_rf15 SRCNN) [ablation]
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

# Improved SRCNN (all 3x3 kernels) - Best PSNR
python main.py --mode train-srcnn --model-type improved_3x3 --epochs-srcnn 100

# Improved SRCNN (with BatchNorm)
python main.py --mode train-srcnn --model-type improved_bn --epochs-srcnn 100

# Improved SRCNN (5-layer RF=15) - Ablation
python main.py --mode train-srcnn --model-type improved_5l_rf15 --epochs-srcnn 100
```

#### 3. Test SRCNN (Generate Restored Images)

```bash
# Basic SRCNN -> outputs/restored/
python main.py --mode test-srcnn --model-type srcnn --test-split all

# Improved SRCNN -> outputs/restored_improved/
python main.py --mode test-srcnn --model-type improved --test-split all

# Improved SRCNN (BatchNorm) -> outputs/restored_improved_bn/
python main.py --mode test-srcnn --model-type improved_bn --test-split all

# Improved 3x3 SRCNN -> outputs/restored_improved_3x3/
python main.py --mode test-srcnn --model-type improved_3x3 --test-split all

# Improved 5-layer RF=15 SRCNN -> outputs/restored_improved_5l_rf15/
python main.py --mode test-srcnn --model-type improved_5l_rf15 --test-split all
```

#### 4. Train U-Net

```bash
# Original HR images
python main.py --mode train-unet --use-original --epochs-unet 100
# or: python main.py --mode train-unet --input-mode original --epochs-unet 100

# Basic SRCNN restored images
python main.py --mode train-unet --use-restored --epochs-unet 100
# or: python main.py --mode train-unet --input-mode restored --epochs-unet 100

# Improved SRCNN restored images - Best IoU (87.28%)
python main.py --mode train-unet --use-improved --epochs-unet 100
# or: python main.py --mode train-unet --input-mode improved --epochs-unet 100

# Improved 3x3 SRCNN restored images - Best PSNR (30.81 dB)
python main.py --mode train-unet --use-3x3 --epochs-unet 100
# or: python main.py --mode train-unet --input-mode improved_3x3 --epochs-unet 100

# Improved 5-layer RF=15 SRCNN restored images (Ablation)
python main.py --mode train-unet --input-mode improved_5l_rf15 --epochs-unet 100
```

#### 5. Test U-Net

```bash
# Original -> outputs/predictions_original/
python main.py --mode test-unet --use-original --test-split test

# Basic SRCNN -> outputs/predictions_restored/
python main.py --mode test-unet --use-restored --test-split test

# Improved SRCNN -> outputs/predictions_improved/
python main.py --mode test-unet --use-improved --test-split test

# Improved 3x3 SRCNN -> outputs/predictions_improved_3x3/
python main.py --mode test-unet --use-3x3 --test-split test

# Improved 5-layer RF=15 -> outputs/predictions_improved_5l_rf15/
python main.py --mode test-unet --input-mode improved_5l_rf15 --test-split test
```

## Experimental Results

### SRCNN Super-Resolution Performance

| Model | PSNR (dB) | SSIM |
|-------|-----------|------|
| Basic SRCNN | 27.41 | 0.9076 |
| Improved SRCNN (5-layer) | 30.09 | 0.9482 |
| **Improved SRCNN (7-layer, all 3x3)** | **30.81** | **0.9545** |

### U-Net Segmentation Performance

| Method | IoU (%) | Dice (%) | Accuracy (%) |
|--------|---------|----------|--------------|
| Bilinear + U-Net | 36.96 | 49.41 | 95.70 |
| Bicubic + U-Net | 36.96 | 49.41 | 95.70 |
| Original HR + U-Net | 59.55 | 72.87 | 97.20 |
| Basic SRCNN + U-Net | 86.84 | 92.47 | 99.44 |
| **Improved SRCNN + U-Net** | **87.28** | **92.67** | **99.47** |
| Improved 3x3 + U-Net | 87.21 | 92.61 | 99.47 |

### Key Findings

- **Best Segmentation**: Improved SRCNN (5-layer) + U-Net achieves **87.28% IoU**
- **Learning-based SR >> Traditional Interpolation**: +135.9% improvement over Bilinear (36.96% → 87.28%)
- **SR Quality ≠ Downstream Performance**: 7-layer 3x3 has higher PSNR (30.81 dB) but slightly lower IoU (87.21%)

## Command Line Arguments

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--mode` | Run mode (preprocess/train-srcnn/test-srcnn/train-unet/test-unet/full) | full |
| `--model-type` | SRCNN type (srcnn/improved/improved_bn/improved_3x3/improved_5l_rf15) | srcnn |
| `--input-mode` | U-Net input (original/restored/improved/improved_3x3/improved_5l_rf15) | restored |
| `--use-original` | Shortcut for `--input-mode original` | - |
| `--use-restored` | Shortcut for `--input-mode restored` | - |
| `--use-improved` | Shortcut for `--input-mode improved` | - |
| `--use-3x3` | Shortcut for `--input-mode improved_3x3` | - |
| `--epochs-srcnn` | SRCNN epochs | 100 |
| `--epochs-unet` | U-Net epochs | 100 |
| `--pos-weight` | Crack pixel weight | 5.0 |
| `--threshold` | Binarization threshold | 0.5 |

## License

MIT
