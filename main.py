"""
Main Entry File
Road Crack Restoration and Annotation System

Pipeline:
1. Preprocessing: Degrade high-resolution images
2. SRCNN Training: Learn image restoration
3. SRCNN Testing: Restore low-quality images
4. U-Net Training: Learn crack segmentation
5. U-Net Testing: Predict crack annotations

Usage:
    python main.py --help
"""
import os
import argparse
import torch

from config import DEVICE
from preprocess import preprocess_all, process_dataset
from srcnn.train import train_srcnn, SRCNNTrainer
from srcnn.test import test_srcnn, SRCNNTester
from unet.train import train_unet, UNetTrainer
from unet.test import test_unet, UNetTester


def check_cuda():
    """Check CUDA availability"""
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("CUDA not available, using CPU")
        return False


def step_preprocess(args):
    print("\n" + "=" * 50)
    print("Step 1: Data Preprocessing")
    print("=" * 50)

    if args.split == 'all':
        preprocess_all()
    else:
        process_dataset(args.split)


def step_train_srcnn(args):
    print("\n" + "=" * 50)
    print("Step 2: SRCNN Training")
    print("=" * 50)

    trainer = SRCNNTrainer(
        model_type=args.model_type,
        device=args.device
    )

    if args.resume_srcnn:
        trainer.load_checkpoint(args.resume_srcnn)

    trainer.train(
        num_epochs=args.epochs_srcnn,
        batch_size=args.batch_size
    )


def step_test_srcnn(args):
    print("\n" + "=" * 50)
    print("Step 3: SRCNN Testing (Image Restoration)")
    print("=" * 50)

    tester = SRCNNTester(
        model_type=args.model_type,
        checkpoint_path=args.checkpoint_srcnn,
        device=args.device
    )

    # Process specified split(s)
    if args.test_split == 'all':
        # Process all splits for full pipeline
        for split in ['train', 'val', 'test']:
            print(f"\nProcessing {split} split...")
            tester.test(
                split=split,
                save_results=True,
                output_dir=args.output_restored
            )
    else:
        tester.test(
            split=args.test_split,
            save_results=True,
            output_dir=args.output_restored
        )


def step_train_unet(args):
    print("\n" + "=" * 50)
    print("Step 4: U-Net Training")
    print("=" * 50)

    # Determine input mode from arguments
    if hasattr(args, 'input_mode') and args.input_mode:
        input_mode = args.input_mode
    elif hasattr(args, 'use_3x3') and args.use_3x3:
        input_mode = 'improved_3x3'
    elif hasattr(args, 'use_improved') and args.use_improved:
        input_mode = 'improved'
    elif hasattr(args, 'use_original') and args.use_original:
        input_mode = 'original'
    else:
        input_mode = 'restored'

    trainer = UNetTrainer(
        device=args.device,
        pos_weight=args.pos_weight,
        input_mode=input_mode
    )

    if args.resume_unet:
        trainer.load_checkpoint(args.resume_unet)

    trainer.train(
        num_epochs=args.epochs_unet,
        batch_size=args.batch_size_unet,
        input_mode=input_mode
    )


def step_test_unet(args):
    print("\n" + "=" * 50)
    print("Step 5: U-Net Testing (Crack Segmentation)")
    print("=" * 50)

    # Determine input mode from arguments
    if hasattr(args, 'input_mode') and args.input_mode:
        input_mode = args.input_mode
    elif hasattr(args, 'use_3x3') and args.use_3x3:
        input_mode = 'improved_3x3'
    elif hasattr(args, 'use_improved') and args.use_improved:
        input_mode = 'improved'
    elif hasattr(args, 'use_original') and args.use_original:
        input_mode = 'original'
    else:
        input_mode = 'restored'

    tester = UNetTester(
        checkpoint_path=args.checkpoint_unet,
        device=args.device,
        input_mode=input_mode
    )

    tester.test(
        split=args.test_split,
        save_results=True,
        output_dir=args.output_predictions,
        input_mode=input_mode,
        threshold=args.threshold
    )


def run_full_pipeline(args):
    print("\n" + "=" * 60)
    print("Road Crack Restoration and Annotation System - Full Pipeline")
    print("=" * 60)

    check_cuda()

    # For full pipeline, process all splits for SRCNN restoration
    original_test_split = args.test_split
    args.test_split = 'all'

    step_preprocess(args)
    step_train_srcnn(args)
    step_test_srcnn(args)

    # Restore original test_split for U-Net testing
    args.test_split = original_test_split

    step_train_unet(args)
    step_test_unet(args)

    print("\n" + "=" * 60)
    print("Complete pipeline finished!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Road Crack Restoration and Annotation System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  python main.py --mode full
  python main.py --mode preprocess --split all
  python main.py --mode train-srcnn --model-type improved --epochs 100
  python main.py --mode test-srcnn --test-split test
  python main.py --mode train-unet --use-restored --pos-weight 5.0
  python main.py --mode test-unet --use-restored
        """
    )

    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'preprocess', 'train-srcnn', 'test-srcnn',
                                'train-unet', 'test-unet'],
                        help='Run mode')

    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'],
                        help='Compute device (default: cuda if available)')

    parser.add_argument('--split', type=str, default='all',
                        choices=['train', 'val', 'test', 'all'],
                        help='Dataset split to preprocess')

    parser.add_argument('--model-type', type=str, default='srcnn',
                        choices=['srcnn', 'improved', 'improved_bn', 'improved_3x3', 'improved_5l_rf15'],
                        help='SRCNN model type')
    parser.add_argument('--epochs-srcnn', type=int, default=100,
                        help='Number of epochs for SRCNN training')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for SRCNN')
    parser.add_argument('--resume-srcnn', type=str, default=None,
                        help='Checkpoint path to resume SRCNN training')
    parser.add_argument('--checkpoint-srcnn', type=str, default=None,
                        help='Checkpoint path for SRCNN testing')

    parser.add_argument('--test-split', type=str, default='test',
                        choices=['train', 'val', 'test', 'all'],
                        help='Test split (use "all" for full pipeline to process all splits)')
    parser.add_argument('--output-restored', type=str, default=None,
                        help='Output directory for restored images')

    parser.add_argument('--epochs-unet', type=int, default=100,
                        help='Number of epochs for U-Net training')
    parser.add_argument('--batch-size-unet', type=int, default=8,
                        help='Batch size for U-Net')
    parser.add_argument('--pos-weight', type=float, default=5.0,
                        help='Positive sample weight (crack pixel weight)')
    parser.add_argument('--input-mode', type=str, default='restored',
                        choices=['original', 'restored', 'improved', 'improved_3x3', 'improved_5l_rf15'],
                        help='Input mode: original (HR), restored (basic SRCNN), improved, improved_3x3, improved_5l_rf15')
    parser.add_argument('--use-original', action='store_true',
                        help='Shortcut for --input-mode original (for baseline comparison)')
    parser.add_argument('--use-restored', action='store_true',
                        help='Shortcut for --input-mode restored (basic SRCNN)')
    parser.add_argument('--use-improved', action='store_true',
                        help='Shortcut for --input-mode improved')
    parser.add_argument('--use-3x3', action='store_true',
                        help='Shortcut for --input-mode improved_3x3')
    parser.add_argument('--resume-unet', type=str, default=None,
                        help='Checkpoint path to resume U-Net training')
    parser.add_argument('--checkpoint-unet', type=str, default=None,
                        help='Checkpoint path for U-Net testing')

    parser.add_argument('--output-predictions', type=str, default=None,
                        help='Output directory for predictions')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Binarization threshold')

    args = parser.parse_args()

    # Handle shortcut arguments for input_mode
    if args.use_original:
        args.input_mode = 'original'
    elif args.use_restored:
        args.input_mode = 'restored'
    elif args.use_improved:
        args.input_mode = 'improved'
    elif args.use_3x3:
        args.input_mode = 'improved_3x3'
    # else: use the value from --input-mode (default: 'restored')

    if args.mode == 'full':
        run_full_pipeline(args)
    elif args.mode == 'preprocess':
        step_preprocess(args)
    elif args.mode == 'train-srcnn':
        step_train_srcnn(args)
    elif args.mode == 'test-srcnn':
        step_test_srcnn(args)
    elif args.mode == 'train-unet':
        step_train_unet(args)
    elif args.mode == 'test-unet':
        step_test_unet(args)


if __name__ == '__main__':
    main()
