"""
Hand-Raise Detection Pipeline - Main Orchestration Script

Usage:
    python main.py --synthetic    Generate synthetic data and run full pipeline
    python main.py --real         Use real videos from data/videos/ with annotations
    python main.py --eval-only    Only run evaluation on existing model
    python main.py --realtime     Run real-time prediction (webcam or video)

Options:
    --model NAME        Model type: tcn (default), temporal_avg
    --extractor NAME    Feature extractor: mobilenetv2, resnet50, efficientnet_b0
    --no-augment        Disable data augmentation
    --no-clean          Disable data cleaning
    --source PATH       Video source for realtime mode (default: webcam)
"""

import argparse
import time
from config import ensure_dirs


def run_pipeline(mode, extractor_name="mobilenetv2", cleaning_config=None,
                 eval_config=None, model_type="tcn", augment=True):
    ensure_dirs()
    start_time = time.time()

    print("=" * 60)
    print("HAND-RAISE DETECTION PIPELINE")
    print(f"  Model: {model_type} | Extractor: {extractor_name} | Augment: {augment}")
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
        print(f"\n[2/5] Extracting features (extractor={extractor_name})...")
        from feature_extraction import extract_all_features
        extract_all_features(extractor_name=extractor_name)
    else:
        print("\n[2/5] Skipping feature extraction (eval-only mode)")

    # Step 3: Training
    if mode != "eval-only":
        print(f"\n[3/5] Training {model_type} model...")
        from train import train_model
        train_model(cleaning_config=cleaning_config, model_type=model_type, augment=augment)
    else:
        print("\n[3/5] Skipping training (eval-only mode)")

    # Step 4: Evaluation
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
    print(f"Model: {model_type} | F1: {f1:.4f} | Accuracy: {acc:.4f}")
    print("=" * 60)


def run_comparison(cleaning_config=None, eval_config=None, augment=True):
    """Train and evaluate both models for comparison."""
    from train import train_model
    from evaluate import evaluate_model, ModelEvaluator, EvalConfig
    import os
    from config import MODEL_DIR

    results = {}

    for mtype in ["temporal_avg", "tcn"]:
        print(f"\n{'='*60}")
        print(f"TRAINING: {mtype.upper()}")
        print(f"{'='*60}")
        train_model(cleaning_config=cleaning_config, model_type=mtype, augment=augment)
        acc, prec, rec, f1 = evaluate_model(eval_config=eval_config,
                                            cleaning_config=cleaning_config)
        results[mtype] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"  {'Model':<15s} {'Accuracy':<10s} {'Precision':<10s} {'Recall':<10s} {'F1':<10s}")
    print(f"  {'-'*55}")
    for mtype, metrics in results.items():
        print(f"  {mtype:<15s} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
              f"{metrics['recall']:<10.4f} {metrics['f1']:<10.4f}")

    f1_improvement = results["tcn"]["f1"] - results["temporal_avg"]["f1"]
    print(f"\n  TCN F1 improvement over baseline: {f1_improvement:+.4f}")
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
                       help="Only evaluate existing model")
    group.add_argument("--realtime", action="store_true",
                       help="Run real-time sliding window prediction")
    group.add_argument("--compare", action="store_true",
                       help="Train and compare TCN vs Temporal Avg models")

    # Model
    parser.add_argument("--model", default="tcn",
                        choices=["tcn", "temporal_avg"],
                        help="Model architecture (default: tcn)")

    # Feature extractor
    parser.add_argument("--extractor", default="mobilenetv2",
                        choices=["mobilenetv2", "resnet50", "efficientnet_b0"],
                        help="Feature extractor backbone (default: mobilenetv2)")

    # Augmentation
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable data augmentation")

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
    parser.add_argument("--metrics", nargs="+", default=None,
                        help="Metrics to compute (default: all)")

    # Real-time options
    parser.add_argument("--source", default="webcam",
                        help="Video source for realtime mode (default: webcam)")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable OpenCV display in realtime mode")

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

    # Real-time mode
    if args.realtime:
        from realtime_predict import run_realtime
        run_realtime(
            video_source=args.source,
            extractor_name=args.extractor,
            show_display=not args.no_display
        )
        return

    # Comparison mode
    if args.compare:
        run_comparison(cleaning_config=cleaning_config,
                       eval_config=eval_config,
                       augment=not args.no_augment)
        return

    # Standard pipeline
    if args.synthetic:
        mode = "synthetic"
    elif args.real:
        mode = "real"
    else:
        mode = "eval-only"

    run_pipeline(mode, extractor_name=args.extractor,
                 cleaning_config=cleaning_config, eval_config=eval_config,
                 model_type=args.model, augment=not args.no_augment)


if __name__ == "__main__":
    main()
