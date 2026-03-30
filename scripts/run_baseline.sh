#!/bin/bash
# Baseline Comparison Script
# Compares U-Net performance with/without SRCNN restoration

set -e

echo "=========================================="
echo "Baseline Comparison Experiment"
echo "=========================================="

cd "$(dirname "$0")/.."

# Check if restored images exist (now split-specific directories)
if [ ! -d "outputs/restored/train" ] || [ ! -d "outputs/restored/test" ]; then
    echo "Warning: outputs/restored/train or outputs/restored/test not found."
    echo "Please run SRCNN test first:"
    echo "  python main.py --mode test-srcnn --test-split train"
    echo "  python main.py --mode test-srcnn --test-split val"
    echo "  python main.py --mode test-srcnn --test-split test"
    echo ""
    echo "Running comparison with original images only..."
fi

# Run comparison
python scripts/baseline_comparison.py \
    --epochs 50 \
    --pos-weight 5.0 \
    "$@"

echo ""
echo "Done! Check results in: outputs/baseline_comparison/"
