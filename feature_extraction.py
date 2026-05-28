"""
Feature Extraction Module - Pluggable Interface with Dependency Injection

Provides a base class `FeatureExtractor` that defines the contract.
Concrete implementations (MobileNetV2, ResNet50, EfficientNet) can be
swapped via dependency injection without changing downstream code.

Usage:
    # Default (MobileNetV2)
    extractor = create_extractor("mobilenetv2")

    # Or inject a custom one
    extractor = create_extractor("resnet50")

    # Use in pipeline
    extract_all_features(extractor=extractor)
"""

import os
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image

from config import FRAMES_DIR, FEATURES_DIR, IMAGE_SIZE, ensure_dirs


class FeatureExtractor(ABC):
    """Abstract base class for all feature extractors.

    Subclass this to add new backbone networks. The pipeline only depends
    on this interface, enabling algorithm-level pluggable replacement.
    """

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Dimension of the output feature vector."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for logging."""
        ...

    @abstractmethod
    def extract_batch(self, images: list) -> np.ndarray:
        """Extract features from a batch of PIL Images.

        Args:
            images: List of PIL.Image.Image in RGB mode.

        Returns:
            np.ndarray of shape (batch_size, feature_dim)
        """
        ...

    def extract_single(self, image) -> np.ndarray:
        """Extract features from a single PIL Image."""
        return self.extract_batch([image])[0]


class MobileNetV2Extractor(FeatureExtractor):
    """MobileNetV2 backbone — outputs 1280-dim feature vectors."""

    def __init__(self, device: Optional[torch.device] = None):
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier = torch.nn.Identity()
        model = model.to(self._device)
        model.eval()
        self._model = model
        self._transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @property
    def feature_dim(self) -> int:
        return 1280

    @property
    def name(self) -> str:
        return "MobileNetV2"

    def extract_batch(self, images: list) -> np.ndarray:
        tensors = [self._transform(img) for img in images]
        batch = torch.stack(tensors).to(self._device)
        with torch.no_grad():
            features = self._model(batch)
        return features.cpu().numpy()


class ResNet50Extractor(FeatureExtractor):
    """ResNet50 backbone — outputs 2048-dim feature vectors."""

    def __init__(self, device: Optional[torch.device] = None):
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Remove final FC, keep avgpool output
        model.fc = torch.nn.Identity()
        model = model.to(self._device)
        model.eval()
        self._model = model
        self._transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @property
    def feature_dim(self) -> int:
        return 2048

    @property
    def name(self) -> str:
        return "ResNet50"

    def extract_batch(self, images: list) -> np.ndarray:
        tensors = [self._transform(img) for img in images]
        batch = torch.stack(tensors).to(self._device)
        with torch.no_grad():
            features = self._model(batch)
        return features.cpu().numpy()


class EfficientNetB0Extractor(FeatureExtractor):
    """EfficientNet-B0 backbone — outputs 1280-dim feature vectors."""

    def __init__(self, device: Optional[torch.device] = None):
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier = torch.nn.Identity()
        model = model.to(self._device)
        model.eval()
        self._model = model
        self._transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @property
    def feature_dim(self) -> int:
        return 1280

    @property
    def name(self) -> str:
        return "EfficientNet-B0"

    def extract_batch(self, images: list) -> np.ndarray:
        tensors = [self._transform(img) for img in images]
        batch = torch.stack(tensors).to(self._device)
        with torch.no_grad():
            features = self._model(batch)
        return features.cpu().numpy()


# --- Registry & Factory ---

_EXTRACTOR_REGISTRY = {
    "mobilenetv2": MobileNetV2Extractor,
    "resnet50": ResNet50Extractor,
    "efficientnet_b0": EfficientNetB0Extractor,
}


def list_extractors() -> list:
    """List all registered extractor names."""
    return list(_EXTRACTOR_REGISTRY.keys())


def register_extractor(name: str, cls):
    """Register a custom FeatureExtractor implementation."""
    if not issubclass(cls, FeatureExtractor):
        raise TypeError(f"{cls} must be a subclass of FeatureExtractor")
    _EXTRACTOR_REGISTRY[name] = cls


def create_extractor(name: str = "mobilenetv2", **kwargs) -> FeatureExtractor:
    """Factory function — creates a feature extractor by name.

    This is the dependency injection point. Pass any registered name
    to swap the backbone without changing pipeline code.
    """
    if name not in _EXTRACTOR_REGISTRY:
        available = ", ".join(_EXTRACTOR_REGISTRY.keys())
        raise ValueError(f"Unknown extractor '{name}'. Available: {available}")
    return _EXTRACTOR_REGISTRY[name](**kwargs)


# --- Pipeline Integration ---

def extract_features_for_video(video_id: str, extractor: FeatureExtractor,
                               batch_size: int = 32) -> Optional[np.ndarray]:
    """Extract features for all frames of a given video using the injected extractor."""
    frames_dir = os.path.join(FRAMES_DIR, video_id)
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])

    if not frame_files:
        print(f"  No frames found for {video_id}")
        return None

    all_features = []

    for batch_start in range(0, len(frame_files), batch_size):
        batch_files = frame_files[batch_start:batch_start + batch_size]
        images = []

        for fname in batch_files:
            img_path = os.path.join(frames_dir, fname)
            img = Image.open(img_path).convert("RGB")
            images.append(img)

        features = extractor.extract_batch(images)
        all_features.append(features)

    all_features = np.concatenate(all_features, axis=0)
    return all_features


def extract_all_features(extractor: Optional[FeatureExtractor] = None,
                         extractor_name: str = "mobilenetv2"):
    """Extract features for all videos using the specified extractor.

    Args:
        extractor: Pre-built extractor instance (dependency injection).
                   If None, creates one from extractor_name.
        extractor_name: Name to look up in registry if extractor is None.
    """
    ensure_dirs()

    if extractor is None:
        extractor = create_extractor(extractor_name)

    print(f"Using extractor: {extractor.name} (output dim={extractor.feature_dim})")

    video_dirs = sorted([d for d in os.listdir(FRAMES_DIR)
                         if os.path.isdir(os.path.join(FRAMES_DIR, d))])

    print(f"Extracting features for {len(video_dirs)} videos...")

    for video_id in video_dirs:
        print(f"  Processing {video_id}...")
        features = extract_features_for_video(video_id, extractor)

        if features is not None:
            output_path = os.path.join(FEATURES_DIR, f"{video_id}.npy")
            np.save(output_path, features)
            print(f"    Saved {features.shape[0]} vectors, dim={features.shape[1]}")

    print("Feature extraction complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--extractor", default="mobilenetv2",
                        choices=list_extractors(),
                        help="Feature extractor backbone to use")
    args = parser.parse_args()
    extract_all_features(extractor_name=args.extractor)
