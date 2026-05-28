"""
Hand-Raise Detection Pipeline - Main Orchestration Script

Usage:
    python main.py --synthetic    Generate synthetic data and run full pipeline
    python main.py --real         Use real videos from data/videos/ with annotations
    python main.py --eval-only    Only run evaluation and Grad-CAM on existing model
"""

import argparse
import time
from config import ensure_dirs


def run_pipeline(mode):
    ensure_dirs()
    start_time = time.time()

    print("=" * 60)
    print("HAND-RAISE DETECTION PIPELINE")
    print("=" * 60)

    # Step 1: Data preparation
    if mode == "synthetic":
        print("\n[1/5] Generating synthetic data...")
        from synthetic_data import generate_synthetic_dataset
        generate_synthetic_dataset()
    elif mode == "real":
        print("\n[1/5] Extracting frames from real videos...")
        from data_preparation import prepare_all_videos
        prepare_all_videos()
    else:
        print("\n[1/5] Skipping data preparation (eval-only mode)")

    # Step 2: Feature extraction
    if mode != "eval-only":
        print("\n[2/5] Extracting MobileNetV2 features...")
        from feature_extraction import extract_all_features
        extract_all_features()
    else:
        print("\n[2/5] Skipping feature extraction (eval-only mode)")

    # Step 3: Training
    if mode != "eval-only":
        print("\n[3/5] Training temporal averaging model...")
        from train import train_model
        train_model()
    else:
        print("\n[3/5] Skipping training (eval-only mode)")

    # Step 4: Evaluation
    print("\n[4/5] Evaluating model...")
    from evaluate import evaluate_model
    acc, prec, rec, f1 = evaluate_model()

    # Step 5: Grad-CAM
    print("\n[5/5] Generating Grad-CAM heatmaps...")
    from gradcam import generate_gradcam_heatmaps
    generate_gradcam_heatmaps(num_samples=10)

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Final accuracy: {acc:.4f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Hand-Raise Detection Pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--synthetic", action="store_true",
                       help="Generate synthetic data and run full pipeline")
    group.add_argument("--real", action="store_true",
                       help="Use real videos from data/videos/")
    group.add_argument("--eval-only", action="store_true",
                       help="Only evaluate existing model and generate heatmaps")

    args = parser.parse_args()

    if args.synthetic:
        mode = "synthetic"
    elif args.real:
        mode = "real"
    else:
        mode = "eval-only"

    run_pipeline(mode)


if __name__ == "__main__":
    main()
