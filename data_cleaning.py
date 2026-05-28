"""
Data Cleaning Module - Anomaly Detection and Preprocessing

Provides configurable strategies for:
- Missing value detection and imputation (mean, median, zero, drop)
- Outlier detection and handling (IQR, Z-score, clip, drop)
- Feature normalization (L2, min-max, standard)
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


class MissingStrategy(Enum):
    MEAN = "mean"
    MEDIAN = "median"
    ZERO = "zero"
    DROP = "drop"


class OutlierMethod(Enum):
    IQR = "iqr"
    ZSCORE = "zscore"


class OutlierStrategy(Enum):
    CLIP = "clip"
    DROP = "drop"
    MEAN = "mean"


class NormMethod(Enum):
    L2 = "l2"
    MINMAX = "minmax"
    STANDARD = "standard"
    NONE = "none"


@dataclass
class CleaningConfig:
    """Configuration for the data cleaning pipeline."""
    # Missing value handling
    missing_strategy: MissingStrategy = MissingStrategy.MEAN

    # Outlier detection
    outlier_method: OutlierMethod = OutlierMethod.IQR
    outlier_strategy: OutlierStrategy = OutlierStrategy.CLIP
    iqr_multiplier: float = 1.5
    zscore_threshold: float = 3.0

    # Normalization
    norm_method: NormMethod = NormMethod.STANDARD

    # Reporting
    verbose: bool = True


class DataCleaner:
    """Cleans extracted feature vectors by handling missing values,
    detecting/treating outliers, and normalizing features."""

    def __init__(self, config: Optional[CleaningConfig] = None):
        self.config = config or CleaningConfig()
        self._fit_params = {}

    def fit(self, features: np.ndarray) -> "DataCleaner":
        """Compute statistics needed for cleaning (call on training data)."""
        clean = self._handle_missing(features, fit=True)
        self._compute_outlier_bounds(clean)
        clean = self._handle_outliers(clean)
        self._compute_norm_params(clean)
        return self

    def transform(self, features: np.ndarray, labels: Optional[np.ndarray] = None):
        """Apply cleaning pipeline. Returns (cleaned_features, cleaned_labels) or just features."""
        original_count = len(features)
        drop_mask = np.ones(len(features), dtype=bool)

        # Step 1: Missing value handling
        features, missing_drop = self._handle_missing(features, fit=False, return_mask=True)
        drop_mask &= missing_drop

        # Step 2: Outlier handling
        features, outlier_drop = self._handle_outliers(features, return_mask=True)
        drop_mask &= outlier_drop

        # Step 3: Apply drop mask
        features = features[drop_mask]
        if labels is not None:
            labels = labels[drop_mask]

        # Step 4: Normalization
        features = self._normalize(features)

        if self.config.verbose:
            dropped = original_count - len(features)
            if dropped > 0:
                print(f"    Cleaned: dropped {dropped}/{original_count} samples "
                      f"({dropped/original_count*100:.1f}%)")

        if labels is not None:
            return features, labels
        return features

    def fit_transform(self, features: np.ndarray, labels: Optional[np.ndarray] = None):
        """Fit on data and transform in one step."""
        self.fit(features)
        return self.transform(features, labels)

    # --- Missing Value Handling ---

    def _handle_missing(self, features, fit=False, return_mask=False):
        """Detect and handle missing values (NaN, Inf)."""
        is_invalid = ~np.isfinite(features)
        has_invalid = is_invalid.any(axis=1)
        keep_mask = np.ones(len(features), dtype=bool)

        if not is_invalid.any():
            if return_mask:
                return features.copy(), keep_mask
            return features.copy()

        if self.config.verbose and is_invalid.any():
            n_invalid = has_invalid.sum()
            print(f"    Found {n_invalid} samples with missing/infinite values")

        result = features.copy()

        if self.config.missing_strategy == MissingStrategy.DROP:
            keep_mask = ~has_invalid
            result[is_invalid] = 0  # placeholder, will be dropped
        elif self.config.missing_strategy == MissingStrategy.ZERO:
            result[is_invalid] = 0.0
        elif self.config.missing_strategy == MissingStrategy.MEAN:
            if fit:
                self._fit_params["col_means"] = np.nanmean(features, axis=0)
            col_means = self._fit_params.get("col_means", np.nanmean(features, axis=0))
            for col in range(features.shape[1]):
                mask = is_invalid[:, col]
                result[mask, col] = col_means[col]
        elif self.config.missing_strategy == MissingStrategy.MEDIAN:
            if fit:
                self._fit_params["col_medians"] = np.nanmedian(features, axis=0)
            col_medians = self._fit_params.get("col_medians", np.nanmedian(features, axis=0))
            for col in range(features.shape[1]):
                mask = is_invalid[:, col]
                result[mask, col] = col_medians[col]

        if return_mask:
            return result, keep_mask
        return result

    # --- Outlier Detection and Handling ---

    def _compute_outlier_bounds(self, features: np.ndarray):
        """Compute outlier boundaries from training data."""
        if self.config.outlier_method == OutlierMethod.IQR:
            q1 = np.percentile(features, 25, axis=0)
            q3 = np.percentile(features, 75, axis=0)
            iqr = q3 - q1
            k = self.config.iqr_multiplier
            self._fit_params["lower_bound"] = q1 - k * iqr
            self._fit_params["upper_bound"] = q3 + k * iqr
        elif self.config.outlier_method == OutlierMethod.ZSCORE:
            self._fit_params["outlier_mean"] = features.mean(axis=0)
            self._fit_params["outlier_std"] = features.std(axis=0) + 1e-8

    def _detect_outliers(self, features: np.ndarray) -> np.ndarray:
        """Returns boolean mask of shape (n_samples, n_features) where True = outlier."""
        if self.config.outlier_method == OutlierMethod.IQR:
            lower = self._fit_params["lower_bound"]
            upper = self._fit_params["upper_bound"]
            return (features < lower) | (features > upper)
        elif self.config.outlier_method == OutlierMethod.ZSCORE:
            mean = self._fit_params["outlier_mean"]
            std = self._fit_params["outlier_std"]
            z_scores = np.abs((features - mean) / std)
            return z_scores > self.config.zscore_threshold
        return np.zeros_like(features, dtype=bool)

    def _handle_outliers(self, features: np.ndarray, return_mask=False):
        """Detect and handle outlier values."""
        outlier_mask = self._detect_outliers(features)
        has_outlier = outlier_mask.any(axis=1)
        keep_mask = np.ones(len(features), dtype=bool)
        result = features.copy()

        if self.config.verbose and outlier_mask.any():
            n_outlier_samples = has_outlier.sum()
            n_outlier_values = outlier_mask.sum()
            print(f"    Outliers: {n_outlier_samples} samples affected, "
                  f"{n_outlier_values} values total "
                  f"({self.config.outlier_method.value}, "
                  f"strategy={self.config.outlier_strategy.value})")

        if self.config.outlier_strategy == OutlierStrategy.DROP:
            keep_mask = ~has_outlier
        elif self.config.outlier_strategy == OutlierStrategy.CLIP:
            lower = self._fit_params.get("lower_bound")
            upper = self._fit_params.get("upper_bound")
            if lower is not None and upper is not None:
                result = np.clip(result, lower, upper)
            else:
                mean = self._fit_params["outlier_mean"]
                std = self._fit_params["outlier_std"]
                threshold = self.config.zscore_threshold
                lower = mean - threshold * std
                upper = mean + threshold * std
                result = np.clip(result, lower, upper)
        elif self.config.outlier_strategy == OutlierStrategy.MEAN:
            col_means = features.mean(axis=0)
            for col in range(features.shape[1]):
                mask = outlier_mask[:, col]
                result[mask, col] = col_means[col]

        if return_mask:
            return result, keep_mask
        return result

    # --- Normalization ---

    def _compute_norm_params(self, features: np.ndarray):
        """Compute normalization parameters from training data."""
        if self.config.norm_method == NormMethod.STANDARD:
            self._fit_params["norm_mean"] = features.mean(axis=0)
            self._fit_params["norm_std"] = features.std(axis=0) + 1e-8
        elif self.config.norm_method == NormMethod.MINMAX:
            self._fit_params["norm_min"] = features.min(axis=0)
            self._fit_params["norm_max"] = features.max(axis=0)

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """Apply normalization."""
        if self.config.norm_method == NormMethod.NONE:
            return features

        result = features.copy()

        if self.config.norm_method == NormMethod.STANDARD:
            mean = self._fit_params["norm_mean"]
            std = self._fit_params["norm_std"]
            result = (result - mean) / std
        elif self.config.norm_method == NormMethod.MINMAX:
            fmin = self._fit_params["norm_min"]
            fmax = self._fit_params["norm_max"]
            denom = fmax - fmin
            denom[denom == 0] = 1.0
            result = (result - fmin) / denom
        elif self.config.norm_method == NormMethod.L2:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            result = result / norms

        return result

    def get_stats(self) -> dict:
        """Return fitted statistics for inspection."""
        return dict(self._fit_params)
