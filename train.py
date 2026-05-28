import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional

from config import (
    FEATURES_DIR, FRAMES_DIR, WINDOW_SIZE, MODEL_DIR,
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, TRAIN_SPLIT,
    AUGMENT, ensure_dirs
)
from model import TCNClassifier, TemporalAvgClassifier
from data_cleaning import DataCleaner, CleaningConfig


class WindowDataset(Dataset):
    """Dataset of sliding windows over frame features with majority-vote labels."""

    def __init__(self, features_list, labels_list, window_size=WINDOW_SIZE,
                 augment=False):
        self.windows = []
        self.window_labels = []
        self.augment = augment

        for features, labels in zip(features_list, labels_list):
            num_frames = min(len(features), len(labels))
            for start in range(num_frames - window_size + 1):
                window_feat = features[start:start + window_size]
                window_lab = labels[start:start + window_size]
                label = 1 if window_lab.sum() > window_size / 2 else 0
                self.windows.append(window_feat)
                self.window_labels.append(label)

        self.windows = np.array(self.windows, dtype=np.float32)
        self.window_labels = np.array(self.window_labels, dtype=np.float32)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = self.windows[idx].copy()
        y = self.window_labels[idx]

        if self.augment:
            x = self._apply_augmentation(x)

        return torch.tensor(x), torch.tensor(y)

    def _apply_augmentation(self, x):
        # Random temporal flip (reverse the sequence)
        if np.random.random() < 0.5:
            x = x[::-1].copy()

        # Brightness simulation: scale features by a random factor
        if np.random.random() < 0.5:
            scale = np.random.uniform(0.8, 1.2)
            x = x * scale

        # Random Gaussian noise
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.02, x.shape).astype(np.float32)
            x = x + noise

        # Random feature dropout (zero out some features)
        if np.random.random() < 0.3:
            mask = np.random.binomial(1, 0.95, x.shape).astype(np.float32)
            x = x * mask

        return x


def load_data(cleaning_config: Optional[CleaningConfig] = None):
    """Load all features and labels, apply data cleaning, split by video into train/val."""
    video_ids = sorted([f.replace(".npy", "") for f in os.listdir(FEATURES_DIR)
                        if f.endswith(".npy")])

    all_features = []
    all_labels = []

    for video_id in video_ids:
        feat_path = os.path.join(FEATURES_DIR, f"{video_id}.npy")
        label_path = os.path.join(FRAMES_DIR, video_id, "labels.npy")

        if not os.path.exists(label_path):
            print(f"  Skipping {video_id}: no labels found")
            continue

        features = np.load(feat_path)
        labels = np.load(label_path)
        all_features.append(features)
        all_labels.append(labels)

    # Split by video
    n_train = int(len(all_features) * TRAIN_SPLIT)
    train_features = all_features[:n_train]
    train_labels = all_labels[:n_train]
    val_features = all_features[n_train:]
    val_labels = all_labels[n_train:]

    # Apply data cleaning per video
    if cleaning_config is None:
        cleaning_config = CleaningConfig()

    cleaner = DataCleaner(cleaning_config)

    train_combined = np.concatenate(train_features, axis=0)
    print(f"  Fitting data cleaner on {len(train_combined)} training samples...")
    cleaner.fit(train_combined)

    print("  Cleaning training data...")
    cleaned_train_features = []
    cleaned_train_labels = []
    for feat, lab in zip(train_features, train_labels):
        clean_feat, clean_lab = cleaner.transform(feat, lab)
        cleaned_train_features.append(clean_feat)
        cleaned_train_labels.append(clean_lab)

    print("  Cleaning validation data...")
    cleaned_val_features = []
    cleaned_val_labels = []
    for feat, lab in zip(val_features, val_labels):
        clean_feat, clean_lab = cleaner.transform(feat, lab)
        cleaned_val_features.append(clean_feat)
        cleaned_val_labels.append(clean_lab)

    return cleaned_train_features, cleaned_train_labels, cleaned_val_features, cleaned_val_labels


def train_model(cleaning_config: Optional[CleaningConfig] = None,
                feature_dim: Optional[int] = None,
                model_type: str = "tcn",
                augment: bool = AUGMENT):
    """Train the hand-raise detection model.

    Args:
        cleaning_config: Data cleaning configuration. None uses defaults.
        feature_dim: Feature dimension (auto-detected from data if None).
        model_type: "tcn" for TCN or "temporal_avg" for baseline.
        augment: Whether to apply data augmentation during training.
    """
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    print(f"Model: {model_type} | Window: {WINDOW_SIZE} | Augment: {augment}")

    train_features, train_labels, val_features, val_labels = load_data(cleaning_config)

    train_dataset = WindowDataset(train_features, train_labels, augment=augment)
    val_dataset = WindowDataset(val_features, val_labels, augment=False)

    print(f"Train windows: {len(train_dataset)}, Val windows: {len(val_dataset)}")
    if len(train_dataset) > 0:
        print(f"Train positive rate: {train_dataset.window_labels.mean():.3f}")
    if len(val_dataset) > 0:
        print(f"Val positive rate: {val_dataset.window_labels.mean():.3f}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=len(train_dataset) > BATCH_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Auto-detect feature dim from data
    if feature_dim is None:
        feature_dim = train_dataset.windows.shape[2] if len(train_dataset) > 0 else 1280

    # Create model
    if model_type == "tcn":
        model = TCNClassifier(feature_dim=feature_dim).to(device)
    else:
        model = TemporalAvgClassifier(feature_dim=feature_dim).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    best_val_loss = float("inf")
    best_val_f1 = 0.0
    patience = 10
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_x.size(0)

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_tp = 0
        val_fp = 0
        val_fn = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item() * batch_x.size(0)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_x.size(0)

                val_tp += ((preds == 1) & (batch_y == 1)).sum().item()
                val_fp += ((preds == 1) & (batch_y == 0)).sum().item()
                val_fn += ((preds == 0) & (batch_y == 1)).sum().item()

        train_loss /= max(train_total, 1)
        val_loss /= max(val_total, 1)
        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        val_precision = val_tp / max(val_tp + val_fp, 1)
        val_recall = val_tp / max(val_tp + val_fn, 1)
        val_f1 = 2 * val_precision * val_recall / max(val_precision + val_recall, 1e-8)

        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

        # Save best model by validation F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_loss = val_loss
            patience_counter = 0
            save_path = os.path.join(MODEL_DIR, "best_model.pth")
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_type": model_type,
                "feature_dim": feature_dim,
                "window_size": WINDOW_SIZE,
                "val_f1": val_f1,
                "epoch": epoch + 1,
            }, save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"Best validation F1: {best_val_f1:.4f} (loss: {best_val_loss:.4f})")
    print(f"Model saved to {os.path.join(MODEL_DIR, 'best_model.pth')}")
    return model


if __name__ == "__main__":
    train_model()
