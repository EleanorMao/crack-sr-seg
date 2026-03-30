# Complete Pipeline Guide

## Overview

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Preprocess  │ -> │ SRCNN Train │ -> │ U-Net Train │ -> │ Test & Viz  │
│             │    │             │    │             │    │             │
│ Degrade HR  │    │ Restore LR  │    │ Segment     │    │ Compare     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
     Step 1            Step 2            Step 3            Step 4
```

---

## Step 1: Data Preprocessing

### 1.1 Check Original Data
```bash
ls dataset/image/ | head -5
ls dataset/masks/ | head -5
cat dataset/ImageSets/train.txt | wc -l  # Should be 2712
cat dataset/ImageSets/val.txt | wc -l    # Should be 316
cat dataset/ImageSets/test.txt | wc -l   # Should be 336
```

### 1.2 Run Preprocessing
```bash
python main.py --mode preprocess --split all
```

**Output locations:**
- `processed_data/lr_images/` - Low-quality images
- `processed_data/hr_images/` - High-quality images
- `processed_data/enhanced_masks/` - Enhanced masks

**Time:** 5-10 minutes

---

## Step 2: SRCNN Training & Testing

### 2.1 Train Basic SRCNN
```bash
python main.py --mode train-srcnn --model-type srcnn --epochs-srcnn 100
```
**Checkpoint:** `checkpoints/srcnn_best.pth`

### 2.2 Train Improved SRCNN
```bash
python main.py --mode train-srcnn --model-type improved --epochs-srcnn 100
```
**Checkpoint:** `checkpoints/improved_srcnn_best.pth`

### 2.3 Test Basic SRCNN
```bash
python main.py --mode test-srcnn --model-type srcnn --test-split all
```
**Output:** `outputs/restored/` (train/val/test subdirectories)

### 2.4 Test Improved SRCNN
```bash
python main.py --mode test-srcnn --model-type improved --test-split all
```
**Output:** `outputs/restored_improved/` (train/val/test subdirectories)

**Expected Results:**
| Model | PSNR | SSIM |
|-------|------|------|
| Basic SRCNN | ~27 dB | ~0.90 |
| Improved SRCNN | ~30 dB | ~0.95 |

---

## Step 3: U-Net Training & Testing

### 3.1 Train U-Net with Original HR Images
```bash
python main.py --mode train-unet --use-original --epochs-unet 100
# or: python main.py --mode train-unet --input-mode original --epochs-unet 100
```
**Checkpoint:** `checkpoints/unet_original_best.pth`

### 3.2 Train U-Net with Basic SRCNN Restored Images
```bash
python main.py --mode train-unet --use-restored --epochs-unet 100
# or: python main.py --mode train-unet --input-mode restored --epochs-unet 100
```
**Checkpoint:** `checkpoints/unet_restored_best.pth`

### 3.3 Train U-Net with Improved SRCNN Restored Images
```bash
python main.py --mode train-unet --use-improved --epochs-unet 100
# or: python main.py --mode train-unet --input-mode improved --epochs-unet 100
```
**Checkpoint:** `checkpoints/unet_improved_best.pth`

### 3.4 Test All U-Net Models
```bash
# Test with original HR
python main.py --mode test-unet --use-original --test-split test

# Test with basic SRCNN restored
python main.py --mode test-unet --use-restored --test-split test

# Test with improved SRCNN restored
python main.py --mode test-unet --use-improved --test-split test
```

**Outputs:**
- `outputs/predictions_original/`
- `outputs/predictions_restored/`
- `outputs/predictions_improved/`

---

## Step 4: Comparison Experiments

### 4.1 Run Baseline Comparison
```bash
python scripts/run_baselines.py --split test
```

### 4.2 Generate Comparison Table
```bash
python scripts/generate_comparison_table.py
```

### 4.3 Generate Visualizations
```bash
python scripts/visualize.py --mode all
```

**Output:** `figures/` directory with all paper figures

---

## Complete One-Command Pipeline

```bash
# Full pipeline with basic SRCNN
python main.py --mode full

# Or step-by-step (recommended for debugging)
python main.py --mode preprocess --split all
python main.py --mode train-srcnn --model-type srcnn --epochs-srcnn 100
python main.py --mode test-srcnn --model-type srcnn --test-split all
python main.py --mode train-unet --use-restored --epochs-unet 100
python main.py --mode test-unet --use-restored --test-split test
```

---

## Directory Structure Summary

```
checkpoints/
├── srcnn_best.pth              # Basic SRCNN
├── improved_srcnn_best.pth     # Improved SRCNN
├── unet_original_best.pth      # U-Net (original HR)
├── unet_restored_best.pth      # U-Net (basic SRCNN restored)
└── unet_improved_best.pth      # U-Net (improved SRCNN restored)

outputs/
├── restored/                   # Basic SRCNN restored images
│   ├── train/
│   ├── val/
│   └── test/
├── restored_improved/          # Improved SRCNN restored images
│   ├── train/
│   ├── val/
│   └── test/
├── predictions_original/       # U-Net predictions (original HR)
├── predictions_restored/       # U-Net predictions (basic SRCNN)
└── predictions_improved/       # U-Net predictions (improved SRCNN)
```

---

## Expected Final Results

| Method | IoU | Dice | Accuracy |
|--------|-----|------|----------|
| Original HR + U-Net | ~60% | ~73% | ~97% |
| Basic SRCNN + U-Net | ~87% | ~92% | ~99% |
| Improved SRCNN + U-Net | ? | ? | ? |

---

## Time Estimates

| Step | GPU (RTX 3080) | CPU |
|------|----------------|-----|
| Preprocess | 5 min | 10 min |
| SRCNN Train (100 epochs) | 30 min | 3 hours |
| SRCNN Test | 2 min | 5 min |
| U-Net Train (100 epochs) | 20 min | 2 hours |
| U-Net Test | 1 min | 3 min |
| **Total** | **~1 hour** | **~6 hours** |

---

## Checkpoint Verification

After completing the pipeline, verify all checkpoints exist:

```bash
ls -la checkpoints/
# Expected files:
# srcnn_best.pth
# improved_srcnn_best.pth
# unet_original_best.pth
# unet_restored_best.pth
# unet_improved_best.pth
```
