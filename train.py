import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from config import (
    FEATURES_DIR, FRAMES_DIR, WINDOW_SIZE, MODEL_DIR,
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, TRAIN_SPLIT, ensure_dirs
)
from model import TemporalAvgClassifier


class WindowDataset(Dataset):
    """Dataset of sliding windows over frame features with majority-vote labels."""

    def __init__(self, features_list, labels_list, window_size=WINDOW_SIZE):
        self.windows = []
        self.window_labels = []

        for features, labels in zip(features_list, labels_list):
            num_frames = min(len(features), len(labels))
            for start in range(num_frames - window_size + 1):
                window_feat = features[start:start + window_size]
                window_lab = labels[start:start + window_size]
                # Majority vote label
                label = 1 if window_lab.sum() > window_size / 2 else 0
                self.windows.append(window_feat)
                self.window_labels.append(label)

        self.windows = np.array(self.windows, dtype=np.float32)
        self.window_labels = np.array(self.window_labels, dtype=np.float32)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return torch.tensor(self.windows[idx]), torch.tensor(self.window_labels[idx])


def load_data():
    """Load all features and labels, split by video into train/val."""
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

    return train_features, train_labels, val_features, val_labels


def train_model():
    """Train the temporal averaging classifier."""
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    train_features, train_labels, val_features, val_labels = load_data()

    train_dataset = WindowDataset(train_features, train_labels)
    val_dataset = WindowDataset(val_features, val_labels)

    print(f"Train windows: {len(train_dataset)}, Val windows: {len(val_dataset)}")
    print(f"Train positive rate: {train_dataset.window_labels.mean():.3f}")
    print(f"Val positive rate: {val_dataset.window_labels.mean():.3f}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = TemporalAvgClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    patience = 5
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
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_x.size(0)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

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

        train_loss /= train_total
        val_loss /= val_total
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to {os.path.join(MODEL_DIR, 'best_model.pth')}")
    return model


if __name__ == "__main__":
    train_model()
