"""
Extended Evaluation Module - Configurable Multi-Metric Assessment

Supports:
- Accuracy, Precision, Recall, F1-Score
- AUC-ROC, AUC-PR (Average Precision)
- Specificity, MCC (Matthews Correlation Coefficient)
- Per-class metrics and weighted/macro averaging
- Confusion matrix visualization
- ROC curve and Precision-Recall curve plots
- Configurable metric selection via EvalConfig
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

from config import (
    FEATURES_DIR, FRAMES_DIR, MODEL_DIR, METRICS_DIR,
    WINDOW_SIZE, BATCH_SIZE, TRAIN_SPLIT, ensure_dirs
)
from model import TemporalAvgClassifier
from train import WindowDataset
from data_cleaning import DataCleaner, CleaningConfig


class MetricType(Enum):
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    AUC_ROC = "auc_roc"
    AUC_PR = "auc_pr"
    SPECIFICITY = "specificity"
    MCC = "mcc"


@dataclass
class EvalConfig:
    """Configuration for evaluation metrics.

    Enable/disable individual metrics as needed for your task.
    """
    metrics: list = field(default_factory=lambda: [
        MetricType.ACCURACY,
        MetricType.PRECISION,
        MetricType.RECALL,
        MetricType.F1,
        MetricType.AUC_ROC,
        MetricType.AUC_PR,
        MetricType.SPECIFICITY,
        MetricType.MCC,
    ])

    # Averaging for multi-class (though we're binary here)
    average: str = "binary"

    # Plot options
    plot_confusion_matrix: bool = True
    plot_roc_curve: bool = True
    plot_pr_curve: bool = True

    # Output
    save_json: bool = True
    verbose: bool = True


class ModelEvaluator:
    """Evaluates a trained model with configurable metrics."""

    def __init__(self, config: Optional[EvalConfig] = None):
        self.config = config or EvalConfig()
        self.results = {}

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray,
                 y_scores: Optional[np.ndarray] = None) -> dict:
        """Compute all configured metrics.

        Args:
            y_true: Ground truth binary labels.
            y_pred: Predicted binary labels (after thresholding).
            y_scores: Predicted probabilities/scores (needed for AUC metrics).

        Returns:
            Dictionary of metric_name -> value.
        """
        self.results = {}

        for metric in self.config.metrics:
            value = self._compute_metric(metric, y_true, y_pred, y_scores)
            if value is not None:
                self.results[metric.value] = value

        return self.results

    def _compute_metric(self, metric: MetricType, y_true, y_pred, y_scores):
        """Compute a single metric."""
        try:
            if metric == MetricType.ACCURACY:
                return accuracy_score(y_true, y_pred)

            elif metric == MetricType.PRECISION:
                return precision_score(y_true, y_pred, zero_division=0,
                                       average=self.config.average)

            elif metric == MetricType.RECALL:
                return recall_score(y_true, y_pred, zero_division=0,
                                    average=self.config.average)

            elif metric == MetricType.F1:
                return f1_score(y_true, y_pred, zero_division=0,
                                average=self.config.average)

            elif metric == MetricType.AUC_ROC:
                if y_scores is None:
                    return None
                if len(np.unique(y_true)) < 2:
                    return None
                return roc_auc_score(y_true, y_scores)

            elif metric == MetricType.AUC_PR:
                if y_scores is None:
                    return None
                if len(np.unique(y_true)) < 2:
                    return None
                return average_precision_score(y_true, y_scores)

            elif metric == MetricType.SPECIFICITY:
                cm = confusion_matrix(y_true, y_pred)
                if cm.shape == (2, 2):
                    tn, fp = cm[0, 0], cm[0, 1]
                    return tn / (tn + fp) if (tn + fp) > 0 else 0.0
                return None

            elif metric == MetricType.MCC:
                return matthews_corrcoef(y_true, y_pred)

        except Exception as e:
            if self.config.verbose:
                print(f"  Warning: Could not compute {metric.value}: {e}")
            return None

    def print_results(self):
        """Print results to console in formatted table."""
        if not self.results:
            print("No results to display. Run evaluate() first.")
            return

        print("=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        for name, value in self.results.items():
            print(f"  {name:<15s}: {value:.4f}")
        print("=" * 50)

    def save_results(self, output_dir: Optional[str] = None):
        """Save results to JSON file."""
        output_dir = output_dir or METRICS_DIR
        os.makedirs(output_dir, exist_ok=True)

        if self.config.save_json:
            json_path = os.path.join(output_dir, "metrics.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2)

        # Also save as plain text
        txt_path = os.path.join(output_dir, "metrics.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for name, value in self.results.items():
                f.write(f"{name}: {value:.4f}\n")

    def plot_all(self, y_true, y_pred, y_scores=None, output_dir=None):
        """Generate all configured plots."""
        output_dir = output_dir or METRICS_DIR
        os.makedirs(output_dir, exist_ok=True)

        if self.config.plot_confusion_matrix:
            self._plot_confusion_matrix(y_true, y_pred, output_dir)

        if self.config.plot_roc_curve and y_scores is not None:
            self._plot_roc_curve(y_true, y_scores, output_dir)

        if self.config.plot_pr_curve and y_scores is not None:
            self._plot_pr_curve(y_true, y_scores, output_dir)

    def _plot_confusion_matrix(self, y_true, y_pred, output_dir):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title("Confusion Matrix")
        plt.colorbar(im, ax=ax)

        classes = ["No Hand-Raise", "Hand-Raise"]
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
        plt.close()

    def _plot_roc_curve(self, y_true, y_scores, output_dir):
        if len(np.unique(y_true)) < 2:
            return

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_val = roc_auc_score(y_true, y_scores)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {auc_val:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150)
        plt.close()

    def _plot_pr_curve(self, y_true, y_scores, output_dir):
        if len(np.unique(y_true)) < 2:
            return

        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(recall, precision, color='green', lw=2,
                label=f'PR curve (AP = {ap:.4f})')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "pr_curve.png"), dpi=150)
        plt.close()


def evaluate_model(eval_config: Optional[EvalConfig] = None,
                   cleaning_config: Optional[CleaningConfig] = None):
    """Evaluate the trained model on the validation set with configurable metrics.

    Args:
        eval_config: Evaluation configuration (which metrics, which plots).
        cleaning_config: Data cleaning configuration to apply before evaluation.

    Returns:
        Tuple of (results_dict, y_true, y_pred, y_scores)
    """
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if eval_config is None:
        eval_config = EvalConfig()

    # Load model
    model = TemporalAvgClassifier().to(device)
    model_path = os.path.join(MODEL_DIR, "best_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Load validation data
    video_ids = sorted([f.replace(".npy", "") for f in os.listdir(FEATURES_DIR)
                        if f.endswith(".npy")])

    all_features = []
    all_labels = []

    for video_id in video_ids:
        feat_path = os.path.join(FEATURES_DIR, f"{video_id}.npy")
        label_path = os.path.join(FRAMES_DIR, video_id, "labels.npy")
        if os.path.exists(label_path):
            all_features.append(np.load(feat_path))
            all_labels.append(np.load(label_path))

    n_train = int(len(all_features) * TRAIN_SPLIT)
    val_features = all_features[n_train:]
    val_labels = all_labels[n_train:]

    # Apply data cleaning
    if cleaning_config is None:
        cleaning_config = CleaningConfig(verbose=False)

    cleaner = DataCleaner(cleaning_config)
    train_combined = np.concatenate(all_features[:n_train], axis=0)
    cleaner.fit(train_combined)

    cleaned_val_features = []
    cleaned_val_labels = []
    for feat, lab in zip(val_features, val_labels):
        clean_feat, clean_lab = cleaner.transform(feat, lab)
        cleaned_val_features.append(clean_feat)
        cleaned_val_labels.append(clean_lab)

    val_dataset = WindowDataset(cleaned_val_features, cleaned_val_labels)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Predict
    all_preds = []
    all_scores = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            scores = torch.sigmoid(outputs).cpu().numpy()
            preds = (scores > 0.5).astype(float)
            all_scores.extend(scores)
            all_preds.extend(preds)
            all_targets.extend(batch_y.numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_scores = np.array(all_scores)

    # Evaluate
    evaluator = ModelEvaluator(eval_config)
    results = evaluator.evaluate(y_true, y_pred, y_scores)
    evaluator.print_results()

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred,
                                target_names=["No Hand-Raise", "Hand-Raise"],
                                zero_division=0))

    # Save results and plots
    evaluator.save_results()
    evaluator.plot_all(y_true, y_pred, y_scores)

    print(f"\nOutputs saved to {METRICS_DIR}")

    # Return the primary metric (accuracy) for backward compatibility
    acc = results.get("accuracy", 0.0)
    prec = results.get("precision", 0.0)
    rec = results.get("recall", 0.0)
    f1 = results.get("f1", 0.0)
    return acc, prec, rec, f1


if __name__ == "__main__":
    evaluate_model()
