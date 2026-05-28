import os
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from config import FRAMES_DIR, FEATURES_DIR, IMAGE_SIZE, FEATURE_DIM, ensure_dirs


def get_mobilenet_extractor(device):
    """Load MobileNetV2 with the final classification layer removed."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    # Remove the classifier, keep features + global avg pool
    # model.features -> conv layers, model.classifier -> FC layers
    # We want: features -> adaptive_avg_pool -> flatten -> 1280-dim
    model.classifier = torch.nn.Identity()
    model = model.to(device)
    model.eval()
    return model


def get_transform():
    """Standard ImageNet preprocessing for MobileNetV2."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def extract_features_for_video(video_id, model, transform, device, batch_size=32):
    """Extract 1280-dim features for all frames of a given video."""
    frames_dir = os.path.join(FRAMES_DIR, video_id)
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])

    if not frame_files:
        print(f"  No frames found for {video_id}")
        return None

    all_features = []

    for batch_start in range(0, len(frame_files), batch_size):
        batch_files = frame_files[batch_start:batch_start + batch_size]
        batch_images = []

        for fname in batch_files:
            img_path = os.path.join(frames_dir, fname)
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img)
            batch_images.append(img_tensor)

        batch_tensor = torch.stack(batch_images).to(device)

        with torch.no_grad():
            features = model(batch_tensor)  # (batch, 1280)

        all_features.append(features.cpu().numpy())

    all_features = np.concatenate(all_features, axis=0)
    assert all_features.shape[1] == FEATURE_DIM, \
        f"Expected {FEATURE_DIM} features, got {all_features.shape[1]}"

    return all_features


def extract_all_features():
    """Extract features for all videos in the frames directory."""
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = get_mobilenet_extractor(device)
    transform = get_transform()

    video_dirs = sorted([d for d in os.listdir(FRAMES_DIR)
                         if os.path.isdir(os.path.join(FRAMES_DIR, d))])

    print(f"Extracting features for {len(video_dirs)} videos...")

    for video_id in video_dirs:
        print(f"  Processing {video_id}...")
        features = extract_features_for_video(video_id, model, transform, device)

        if features is not None:
            output_path = os.path.join(FEATURES_DIR, f"{video_id}.npy")
            np.save(output_path, features)
            print(f"    Saved {features.shape[0]} feature vectors ({features.shape})")

    print("Feature extraction complete.")


if __name__ == "__main__":
    extract_all_features()
