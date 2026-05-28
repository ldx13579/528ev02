"""
Hand-Raise Detection Pipeline - Main Orchestration Script

Usage:
    python main.py --synthetic    Generate synthetic data and run full pipeline
    python main.py --real         Use real videos from data/videos/ with annotations
    python main.py --eval-only    Only run evaluation and Grad-CAM on existing model

Options:
    --extractor NAME    Feature extractor: mobilenetv2, resnet50, efficientnet_b0
    --no-clean          Disable data cleaning
    --outlier-method    Outlier detection method: iqr, zscore
    --norm-method       Normalization: standard, l2, minmax, none
"""

import argparse
import time
from config import ensure_dirs


def run_pipeline(mode, extractor_name="mobilenetv2", cleaning_config=None, eval_config=None):
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

    # Step 2: Feature extraction (with pluggable extractor)
    if mode != "eval-only":
        print(f"\n[2/5] Extracting features (extractor={extractor_name})...")
        from feature_extraction import extract_all_features
        extract_all_features(extractor_name=extractor_name)
    else:
        print("\n[2/5] Skipping feature extraction (eval-only mode)")

    # Step 3: Training (with data cleaning)
    if mode != "eval-only":
        print("\n[3/5] Training temporal averaging model...")
        from train import train_model
        train_model(cleaning_config=cleaning_config)
    else:
        print("\n[3/5] Skipping training (eval-only mode)")

    # Step 4: Evaluation (with extended metrics)
    print("\n[4/5] Evaluating model...")
    from evaluate import evaluate_model
    acc, prec, rec, f1 = evaluate_model(
        eval_config=eval_config,
        cleaning_config=cleaning_config
    )

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

    # Mode selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--synthetic", action="store_true",
                       help="Generate synthetic data and run full pipeline")
    group.add_argument("--real", action="store_true",
                       help="Use real videos from data/videos/")
    group.add_argument("--eval-only", action="store_true",
                       help="Only evaluate existing model and generate heatmaps")

    # Feature extractor
    parser.add_argument("--extractor", default="mobilenetv2",
                        choices=["mobilenetv2", "resnet50", "efficientnet_b0"],
                        help="Feature extractor backbone (default: mobilenetv2)")

    # Data cleaning
    parser.add_argument("--no-clean", action="store_true",
                        help="Disable data cleaning pipeline")
    parser.add_argument("--outlier-method", default="iqr",
                        choices=["iqr", "zscore"],
                        help="Outlier detection method (default: iqr)")
    parser.add_argument("--outlier-strategy", default="clip",
                        choices=["clip", "drop", "mean"],
                        help="How to handle outliers (default: clip)")
    parser.add_argument("--norm-method", default="standard",
                        choices=["standard", "l2", "minmax", "none"],
                        help="Feature normalization method (default: standard)")
    parser.add_argument("--missing-strategy", default="mean",
                        choices=["mean", "median", "zero", "drop"],
                        help="Missing value strategy (default: mean)")

    # Evaluation
    parser.add_argument("--metrics", nargs="+",
                        default=None,
                        help="Metrics to compute (default: all). "
                             "Options: accuracy precision recall f1 auc_roc auc_pr specificity mcc")

    args = parser.parse_args()

    # Build cleaning config
    cleaning_config = None
    if not args.no_clean:
        from data_cleaning import (
            CleaningConfig, OutlierMethod, OutlierStrategy, NormMethod, MissingStrategy
        )
        cleaning_config = CleaningConfig(
            missing_strategy=MissingStrategy(args.missing_strategy),
            outlier_method=OutlierMethod(args.outlier_method),
            outlier_strategy=OutlierStrategy(args.outlier_strategy),
            norm_method=NormMethod(args.norm_method),
        )

    # Build eval config
    eval_config = None
    if args.metrics:
        from evaluate import EvalConfig, MetricType
        selected = [MetricType(m) for m in args.metrics]
        eval_config = EvalConfig(metrics=selected)

    # Determine mode
    if args.synthetic:
        mode = "synthetic"
    elif args.real:
        mode = "real"
    else:
        mode = "eval-only"

    run_pipeline(mode, extractor_name=args.extractor,
                 cleaning_config=cleaning_config, eval_config=eval_config)


if __name__ == "__main__":
    main()
