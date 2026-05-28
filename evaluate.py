import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
from config import (
    FEATURES_DIR, FRAMES_DIR, MODEL_DIR, METRICS_DIR,
    WINDOW_SIZE, BATCH_SIZE, TRAIN_SPLIT, ensure_dirs
)
from model import TemporalAvgClassifier
from train import WindowDataset


def evaluate_model():
    """Evaluate the trained model on the validation set."""
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    val_dataset = WindowDataset(val_features, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Predict
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            preds = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(batch_y.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Metrics
    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)

    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print()
    print("Classification Report:")
    print(classification_report(all_targets, all_preds,
                                target_names=["No Hand-Raise", "Hand-Raise"],
                                zero_division=0))

    # Confusion matrix plot
    cm = confusion_matrix(all_targets, all_preds)
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

    cm_path = os.path.join(METRICS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    # Save metrics to text file
    metrics_path = os.path.join(METRICS_DIR, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall:    {rec:.4f}\n")
        f.write(f"F1-Score:  {f1:.4f}\n")
        f.write(f"\nConfusion Matrix:\n{cm}\n")
        f.write(f"\n{classification_report(all_targets, all_preds, target_names=classes, zero_division=0)}")

    print(f"Metrics saved to {metrics_path}")
    return acc, prec, rec, f1


if __name__ == "__main__":
    evaluate_model()
