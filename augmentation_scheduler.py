"""
Augmentation Strategy Scheduler

Dynamically adjusts data augmentation intensity and combination based on:
- Current training epoch (warm-up -> peak -> cool-down)
- Loss trajectory (increase augmentation when overfitting detected)
- Configurable phase boundaries and intensity curves
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AugmentationConfig:
    """Per-augmentation parameters with intensity range."""
    name: str
    base_prob: float        # base probability of applying this augmentation
    min_intensity: float    # minimum strength parameter
    max_intensity: float    # maximum strength parameter
    enabled: bool = True


@dataclass
class SchedulerConfig:
    """Configuration for the augmentation scheduler."""
    # Phase boundaries (fraction of total epochs)
    warmup_fraction: float = 0.1    # ramp up augmentation
    peak_fraction: float = 0.6      # full augmentation intensity
    cooldown_fraction: float = 0.3  # reduce augmentation for fine-tuning

    # Overfitting response
    overfit_window: int = 3         # epochs to look back for overfitting detection
    overfit_threshold: float = 0.05 # val_loss - train_loss gap to trigger response
    overfit_boost: float = 1.5      # multiplier on augmentation when overfitting

    # Overall intensity bounds
    min_global_intensity: float = 0.2
    max_global_intensity: float = 1.0


class AugmentationScheduler:
    """Dynamically adjusts augmentation strength during training."""

    def __init__(self, total_epochs: int, config: Optional[SchedulerConfig] = None):
        self.total_epochs = total_epochs
        self.config = config or SchedulerConfig()

        self.augmentations = [
            AugmentationConfig("temporal_flip", base_prob=0.5,
                               min_intensity=0.0, max_intensity=1.0),
            AugmentationConfig("brightness", base_prob=0.5,
                               min_intensity=0.05, max_intensity=0.25),
            AugmentationConfig("noise", base_prob=0.3,
                               min_intensity=0.005, max_intensity=0.04),
            AugmentationConfig("feature_dropout", base_prob=0.3,
                               min_intensity=0.02, max_intensity=0.1),
        ]

        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.current_intensity = self.config.min_global_intensity
        self.current_epoch = 0
        self._overfit_active = False

    def _compute_phase_intensity(self, epoch: int) -> float:
        """Compute base intensity from training phase (warmup/peak/cooldown)."""
        progress = epoch / max(self.total_epochs - 1, 1)
        warmup_end = self.config.warmup_fraction
        peak_end = warmup_end + self.config.peak_fraction

        if progress < warmup_end:
            # Linear ramp-up
            phase_progress = progress / max(warmup_end, 1e-8)
            return self.config.min_global_intensity + \
                   (self.config.max_global_intensity - self.config.min_global_intensity) * phase_progress
        elif progress < peak_end:
            # Peak intensity
            return self.config.max_global_intensity
        else:
            # Cosine cooldown
            cooldown_progress = (progress - peak_end) / max(self.config.cooldown_fraction, 1e-8)
            cooldown_progress = min(cooldown_progress, 1.0)
            return self.config.min_global_intensity + \
                   (self.config.max_global_intensity - self.config.min_global_intensity) * \
                   (1 + np.cos(np.pi * cooldown_progress)) / 2

    def _detect_overfitting(self) -> bool:
        """Detect overfitting by comparing train/val loss trends."""
        if len(self.val_losses) < self.config.overfit_window:
            return False

        recent_val = self.val_losses[-self.config.overfit_window:]
        recent_train = self.train_losses[-self.config.overfit_window:]

        avg_gap = np.mean([v - t for v, t in zip(recent_val, recent_train)])
        val_increasing = all(
            recent_val[i] > recent_val[i - 1]
            for i in range(1, len(recent_val))
        )

        return avg_gap > self.config.overfit_threshold or val_increasing

    def step(self, epoch: int, train_loss: float, val_loss: float):
        """Update scheduler state after each epoch."""
        self.current_epoch = epoch
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        # Compute phase-based intensity
        base_intensity = self._compute_phase_intensity(epoch)

        # Apply overfitting boost
        self._overfit_active = self._detect_overfitting()
        if self._overfit_active:
            base_intensity = min(
                base_intensity * self.config.overfit_boost,
                self.config.max_global_intensity
            )

        self.current_intensity = np.clip(
            base_intensity,
            self.config.min_global_intensity,
            self.config.max_global_intensity
        )

    def get_augmentation_params(self) -> dict:
        """Get current augmentation parameters scaled by intensity.

        Returns dict mapping augmentation name to (probability, intensity_value).
        """
        params = {}
        for aug in self.augmentations:
            if not aug.enabled:
                continue
            prob = aug.base_prob * self.current_intensity
            intensity = aug.min_intensity + \
                        (aug.max_intensity - aug.min_intensity) * self.current_intensity
            params[aug.name] = {"prob": prob, "intensity": intensity}
        return params

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply augmentations with current scheduled parameters."""
        params = self.get_augmentation_params()

        # Temporal flip
        if "temporal_flip" in params:
            p = params["temporal_flip"]
            if np.random.random() < p["prob"]:
                x = x[::-1].copy()

        # Brightness simulation
        if "brightness" in params:
            p = params["brightness"]
            if np.random.random() < p["prob"]:
                scale = np.random.uniform(1.0 - p["intensity"], 1.0 + p["intensity"])
                x = x * scale

        # Gaussian noise
        if "noise" in params:
            p = params["noise"]
            if np.random.random() < p["prob"]:
                noise = np.random.normal(0, p["intensity"], x.shape).astype(np.float32)
                x = x + noise

        # Feature dropout
        if "feature_dropout" in params:
            p = params["feature_dropout"]
            if np.random.random() < p["prob"]:
                keep_rate = 1.0 - p["intensity"]
                mask = np.random.binomial(1, keep_rate, x.shape).astype(np.float32)
                x = x * mask

        return x

    def get_status(self) -> str:
        """Return human-readable status string."""
        phase = self._get_phase_name()
        overfit_str = " [OVERFIT-BOOST]" if self._overfit_active else ""
        return f"Phase: {phase} | Intensity: {self.current_intensity:.3f}{overfit_str}"

    def _get_phase_name(self) -> str:
        progress = self.current_epoch / max(self.total_epochs - 1, 1)
        warmup_end = self.config.warmup_fraction
        peak_end = warmup_end + self.config.peak_fraction
        if progress < warmup_end:
            return "warmup"
        elif progress < peak_end:
            return "peak"
        else:
            return "cooldown"
