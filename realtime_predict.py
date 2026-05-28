"""
Real-time Sliding Window Predictor for Hand-Raise Detection

Features:
- Configurable output interval with dynamic frequency adjustment
- Priority-aware ring buffer with overflow warnings
- Supports webcam, video file, and pre-extracted feature inputs
"""

import time
import logging
import numpy as np
import torch
import cv2
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Callable

from config import (
    WINDOW_SIZE, FEATURE_DIM, FPS, MODEL_DIR,
    PREDICTION_INTERVAL, IMAGE_SIZE, ensure_dirs
)
from model import TCNClassifier, TemporalAvgClassifier
from evaluate import load_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configurable Prediction Interval
# ---------------------------------------------------------------------------

@dataclass
class IntervalConfig:
    """Configurable prediction interval parameters."""
    base_interval: float = PREDICTION_INTERVAL  # default 0.5s
    min_interval: float = 0.1                   # fastest allowed interval
    max_interval: float = 5.0                   # slowest allowed interval
    adaptive: bool = True                       # enable dynamic adjustment

    # Adaptive adjustment parameters
    activity_speedup: float = 0.5   # multiplier when high activity detected
    idle_slowdown: float = 2.0      # multiplier when scene is idle
    score_threshold_high: float = 0.7  # score above this = high activity
    score_threshold_low: float = 0.3   # score below this = low activity
    smoothing_window: int = 5       # number of recent scores to smooth over


class AdaptiveInterval:
    """Dynamically adjusts prediction frequency based on scene activity."""

    def __init__(self, config: Optional[IntervalConfig] = None):
        self.config = config or IntervalConfig()
        self.current_interval = self.config.base_interval
        self.recent_scores: List[float] = []

    @property
    def interval(self) -> float:
        return self.current_interval

    def update(self, score: float):
        """Update interval based on latest prediction score."""
        self.recent_scores.append(score)
        if len(self.recent_scores) > self.config.smoothing_window:
            self.recent_scores.pop(0)

        if not self.config.adaptive:
            return

        avg_score = np.mean(self.recent_scores)

        if avg_score > self.config.score_threshold_high:
            # High activity: predict more frequently
            target = self.config.base_interval * self.config.activity_speedup
        elif avg_score < self.config.score_threshold_low:
            # Low activity: predict less frequently to save compute
            target = self.config.base_interval * self.config.idle_slowdown
        else:
            target = self.config.base_interval

        # Smooth transition (exponential moving average)
        alpha = 0.3
        self.current_interval = self.current_interval * (1 - alpha) + target * alpha
        self.current_interval = np.clip(
            self.current_interval,
            self.config.min_interval,
            self.config.max_interval
        )

    def set_interval(self, interval: float):
        """Manually override the prediction interval."""
        self.current_interval = np.clip(
            interval,
            self.config.min_interval,
            self.config.max_interval
        )

    def reset(self):
        """Reset to base interval."""
        self.current_interval = self.config.base_interval
        self.recent_scores.clear()


# ---------------------------------------------------------------------------
# Priority Ring Buffer with Overflow Warning
# ---------------------------------------------------------------------------

@dataclass
class FrameEntry:
    """A frame entry in the priority buffer."""
    feature: np.ndarray
    timestamp: float
    priority: float         # higher = more important to retain
    is_critical: bool       # critical frames trigger overflow warnings


class PriorityRingBuffer:
    """Ring buffer with priority-based discard and overflow alerting.

    When the buffer is full and a new frame arrives:
    - If the new frame has higher priority than the lowest-priority frame,
      the lowest-priority frame is discarded.
    - If discarding would remove a critical frame, an overflow warning is emitted.
    - Standard (non-priority) mode: behaves like a normal ring buffer (FIFO).
    """

    def __init__(self, capacity: int = WINDOW_SIZE,
                 overflow_callback: Optional[Callable[[str], None]] = None):
        self.capacity = capacity
        self.entries: List[FrameEntry] = []
        self.overflow_callback = overflow_callback or self._default_overflow_handler
        self.overflow_count = 0
        self.critical_discard_count = 0
        self.total_frames_processed = 0

    def _default_overflow_handler(self, message: str):
        logger.warning(message)

    def _compute_priority(self, feature: np.ndarray, timestamp: float) -> float:
        """Compute frame priority based on feature characteristics.

        High-motion or high-variance frames are more informative.
        """
        variance = np.var(feature)
        magnitude = np.linalg.norm(feature)
        # Combine variance and magnitude as priority signal
        return float(variance * 0.6 + magnitude * 0.4)

    def _is_critical(self, feature: np.ndarray, priority: float) -> bool:
        """Determine if a frame is critical (should not be discarded lightly).

        Frames with extreme priority (very high activity) are marked critical.
        """
        if len(self.entries) < 2:
            return False
        priorities = [e.priority for e in self.entries]
        mean_p = np.mean(priorities)
        std_p = np.std(priorities) + 1e-8
        # Critical if priority is >2 std above mean
        return priority > mean_p + 2 * std_p

    def append(self, feature: np.ndarray, timestamp: float = 0.0,
               force_critical: bool = False):
        """Add a frame to the buffer with priority scoring.

        Args:
            feature: The feature vector for this frame.
            timestamp: Frame timestamp.
            force_critical: If True, mark frame as critical regardless of score.
        """
        self.total_frames_processed += 1
        priority = self._compute_priority(feature, timestamp)
        is_critical = force_critical or self._is_critical(feature, priority)

        entry = FrameEntry(
            feature=feature,
            timestamp=timestamp,
            priority=priority,
            is_critical=is_critical
        )

        if len(self.entries) < self.capacity:
            self.entries.append(entry)
        else:
            # Buffer full: find lowest-priority entry to discard
            min_idx = self._find_discard_candidate()
            discarded = self.entries[min_idx]

            if discarded.is_critical:
                self.critical_discard_count += 1
                self.overflow_callback(
                    f"[OVERFLOW WARNING] Discarding critical frame "
                    f"(timestamp={discarded.timestamp:.2f}s, priority={discarded.priority:.4f}). "
                    f"Total critical discards: {self.critical_discard_count}"
                )

            self.overflow_count += 1
            self.entries.pop(min_idx)
            self.entries.append(entry)

    def _find_discard_candidate(self) -> int:
        """Find the index of the lowest-priority non-critical frame to discard.

        If all frames are critical, discard the oldest critical frame (FIFO fallback).
        """
        non_critical = [(i, e) for i, e in enumerate(self.entries) if not e.is_critical]

        if non_critical:
            # Discard lowest-priority non-critical frame
            return min(non_critical, key=lambda x: x[1].priority)[0]
        else:
            # All critical: fallback to oldest (index 0)
            return 0

    def get_window(self) -> Optional[np.ndarray]:
        """Get the current buffer contents as a numpy array.

        Returns None if buffer is not full.
        """
        if len(self.entries) < self.capacity:
            return None
        # Sort by timestamp to maintain temporal order
        sorted_entries = sorted(self.entries, key=lambda e: e.timestamp)
        return np.array([e.feature for e in sorted_entries], dtype=np.float32)

    def get_fill_level(self) -> Tuple[int, int]:
        """Return (current_count, capacity)."""
        return len(self.entries), self.capacity

    def get_stats(self) -> dict:
        """Return buffer statistics."""
        critical_count = sum(1 for e in self.entries if e.is_critical)
        return {
            "fill": len(self.entries),
            "capacity": self.capacity,
            "total_processed": self.total_frames_processed,
            "overflow_count": self.overflow_count,
            "critical_in_buffer": critical_count,
            "critical_discards": self.critical_discard_count,
        }

    def clear(self):
        """Clear the buffer."""
        self.entries.clear()

    def __len__(self):
        return len(self.entries)


# ---------------------------------------------------------------------------
# Real-time Predictor (with configurable interval + priority buffer)
# ---------------------------------------------------------------------------

class RealtimePredictor:
    """Sliding window predictor with adaptive interval and priority buffer."""

    def __init__(self, model_path=None, extractor_name="mobilenetv2",
                 threshold=0.5, interval_config: Optional[IntervalConfig] = None,
                 use_priority_buffer: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.use_priority_buffer = use_priority_buffer

        # Load model
        self.model, self.model_type = load_model(self.device, model_path)
        self.model.eval()
        print(f"Loaded {self.model_type} model on {self.device}")

        # Load feature extractor
        from feature_extraction import create_extractor
        self.extractor = create_extractor(extractor_name)
        print(f"Feature extractor: {extractor_name}")

        # Adaptive interval controller
        self.interval_ctrl = AdaptiveInterval(interval_config)

        # Priority buffer or simple buffer
        if use_priority_buffer:
            self.buffer = PriorityRingBuffer(
                capacity=WINDOW_SIZE,
                overflow_callback=self._on_overflow
            )
        else:
            import collections
            self.buffer = collections.deque(maxlen=WINDOW_SIZE)

        self.last_prediction_time = 0.0
        self._overflow_warnings: List[str] = []
        self._frame_timestamp = 0.0

    def _on_overflow(self, message: str):
        """Handle overflow warnings from priority buffer."""
        self._overflow_warnings.append(message)
        print(f"  {message}")

    def _extract_feature(self, frame):
        """Extract feature vector from a single BGR frame."""
        from PIL import Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        features = self.extractor.extract([pil_img])
        return features[0]

    def _predict(self) -> Tuple[Optional[bool], float]:
        """Run prediction on the current buffer."""
        if self.use_priority_buffer:
            window = self.buffer.get_window()
            if window is None:
                return None, 0.0
        else:
            if len(self.buffer) < WINDOW_SIZE:
                return None, 0.0
            window = np.array(list(self.buffer), dtype=np.float32)

        x = torch.tensor(window).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logit = self.model(x)
            score = torch.sigmoid(logit).item()

        is_hand_raise = score > self.threshold
        return is_hand_raise, score

    def set_interval(self, interval: float):
        """Manually set prediction interval (seconds)."""
        self.interval_ctrl.set_interval(interval)

    def get_interval(self) -> float:
        """Get current prediction interval."""
        return self.interval_ctrl.interval

    def process_frame(self, frame, timestamp: float = None):
        """Add a frame to the buffer. Returns prediction if interval elapsed.

        Args:
            frame: BGR image (numpy array)
            timestamp: Optional explicit timestamp (auto-increments if None)

        Returns:
            (is_hand_raise or None, confidence_score)
        """
        if timestamp is None:
            self._frame_timestamp += 1.0 / FPS
            timestamp = self._frame_timestamp

        feature = self._extract_feature(frame)

        if self.use_priority_buffer:
            self.buffer.append(feature, timestamp=timestamp)
        else:
            self.buffer.append(feature)

        current_time = time.time()
        if current_time - self.last_prediction_time >= self.interval_ctrl.interval:
            self.last_prediction_time = current_time
            result, score = self._predict()
            if result is not None:
                self.interval_ctrl.update(score)
            return result, score
        return None, 0.0

    def run_on_video(self, video_path, show_display=True):
        """Run real-time prediction on a video file."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = int(video_fps / FPS)
        frame_count = 0
        prediction_count = 0

        print(f"\nProcessing video: {video_path}")
        print(f"Video FPS: {video_fps:.1f}, Sampling every {frame_interval} frames (target {FPS} FPS)")
        print(f"Base interval: {self.interval_ctrl.config.base_interval}s "
              f"(adaptive={'ON' if self.interval_ctrl.config.adaptive else 'OFF'})")
        print(f"Buffer: {'priority' if self.use_priority_buffer else 'FIFO'} "
              f"(capacity={WINDOW_SIZE})")
        print("-" * 60)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % frame_interval != 0:
                    continue

                timestamp = frame_count / video_fps
                result, score = self.process_frame(frame, timestamp=timestamp)

                if result is not None:
                    prediction_count += 1
                    status = "HAND RAISED" if result else "no hand raise"
                    interval_str = f"interval={self.interval_ctrl.interval:.2f}s"
                    print(f"  [{timestamp:6.1f}s] {status} "
                          f"(conf: {score:.3f}, {interval_str})")

                    if show_display:
                        display = frame.copy()
                        color = (0, 0, 255) if result else (0, 255, 0)
                        label = f"{'HAND RAISED' if result else 'No Raise'} ({score:.2f})"
                        cv2.putText(display, label, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                        interval_label = f"Interval: {self.interval_ctrl.interval:.2f}s"
                        cv2.putText(display, interval_label, (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                        cv2.imshow("Hand-Raise Detection", display)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

        finally:
            cap.release()
            if show_display:
                cv2.destroyAllWindows()

        print(f"\nDone. Processed {frame_count} frames, made {prediction_count} predictions.")
        if self.use_priority_buffer:
            stats = self.buffer.get_stats()
            print(f"Buffer stats: {stats}")
        if self._overflow_warnings:
            print(f"Overflow warnings: {len(self._overflow_warnings)}")

    def run_on_webcam(self):
        """Run real-time prediction on webcam input."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("Cannot open webcam")

        frame_count = 0
        frame_interval = max(1, int(30 / FPS))

        print("\nWebcam hand-raise detection started. Press 'q' to quit.")
        print(f"Base interval: {self.interval_ctrl.config.base_interval}s | "
              f"Buffer: {'priority' if self.use_priority_buffer else 'FIFO'} "
              f"(capacity={WINDOW_SIZE})")
        print("Press '+'/'-' to manually adjust interval")
        print("-" * 60)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % frame_interval != 0:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('+') or key == ord('='):
                        self.interval_ctrl.set_interval(self.interval_ctrl.interval + 0.1)
                        print(f"  Interval -> {self.interval_ctrl.interval:.2f}s")
                    elif key == ord('-'):
                        self.interval_ctrl.set_interval(self.interval_ctrl.interval - 0.1)
                        print(f"  Interval -> {self.interval_ctrl.interval:.2f}s")
                    continue

                timestamp = frame_count / 30.0
                result, score = self.process_frame(frame, timestamp=timestamp)

                display = frame.copy()
                if result is not None:
                    color = (0, 0, 255) if result else (0, 255, 0)
                    label = f"{'HAND RAISED' if result else 'No Raise'} ({score:.2f})"
                else:
                    color = (128, 128, 128)
                    fill, cap_size = (len(self.buffer), WINDOW_SIZE) if not self.use_priority_buffer \
                        else self.buffer.get_fill_level()
                    label = f"Buffering... ({fill}/{cap_size})"

                cv2.putText(display, label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                info = f"Interval: {self.interval_ctrl.interval:.2f}s"
                cv2.putText(display, info, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.imshow("Hand-Raise Detection (Webcam)", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('+') or key == ord('='):
                    self.interval_ctrl.set_interval(self.interval_ctrl.interval + 0.1)
                elif key == ord('-'):
                    self.interval_ctrl.set_interval(self.interval_ctrl.interval - 0.1)

        finally:
            cap.release()
            cv2.destroyAllWindows()

        print("Webcam detection stopped.")
        if self.use_priority_buffer:
            print(f"Buffer stats: {self.buffer.get_stats()}")

    def run_on_features(self, features_array, timestamps=None):
        """Run sliding window prediction on pre-extracted features.

        Args:
            features_array: numpy array of shape (num_frames, feature_dim)
            timestamps: optional array of timestamps per frame

        Returns:
            List of (timestamp, is_hand_raise, score) tuples
        """
        predictions = []

        if timestamps is None:
            timestamps = np.arange(len(features_array)) / FPS

        last_pred_ts = -self.interval_ctrl.interval
        for i in range(len(features_array)):
            ts = timestamps[i]

            if self.use_priority_buffer:
                self.buffer.append(features_array[i], timestamp=ts)
                ready = self.buffer.get_window() is not None
            else:
                self.buffer.append(features_array[i])
                ready = len(self.buffer) == WINDOW_SIZE

            if ready and (ts - last_pred_ts) >= self.interval_ctrl.interval:
                result, score = self._predict()
                self.interval_ctrl.update(score)
                predictions.append((ts, result, score))
                last_pred_ts = ts

        return predictions


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_realtime(video_source="webcam", model_path=None, extractor_name="mobilenetv2",
                 show_display=True, interval=None, adaptive=True,
                 use_priority_buffer=True):
    """Entry point for real-time prediction.

    Args:
        video_source: "webcam", or path to a video file
        model_path: Path to model checkpoint (None uses default)
        extractor_name: Feature extractor backbone name
        show_display: Whether to show OpenCV window
        interval: Override base prediction interval (seconds)
        adaptive: Enable adaptive interval adjustment
        use_priority_buffer: Use priority-aware buffer vs simple FIFO
    """
    ensure_dirs()

    interval_config = IntervalConfig(adaptive=adaptive)
    if interval is not None:
        interval_config.base_interval = interval

    predictor = RealtimePredictor(
        model_path=model_path,
        extractor_name=extractor_name,
        interval_config=interval_config,
        use_priority_buffer=use_priority_buffer
    )

    if video_source == "webcam":
        predictor.run_on_webcam()
    else:
        predictor.run_on_video(video_source, show_display=show_display)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Real-time Hand-Raise Detection")
    parser.add_argument("--source", default="webcam",
                        help="Video source: 'webcam' or path to video file")
    parser.add_argument("--model", default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--extractor", default="mobilenetv2",
                        help="Feature extractor backbone")
    parser.add_argument("--interval", type=float, default=None,
                        help="Prediction interval in seconds (default: 0.5)")
    parser.add_argument("--no-adaptive", action="store_true",
                        help="Disable adaptive interval adjustment")
    parser.add_argument("--no-priority", action="store_true",
                        help="Use simple FIFO buffer instead of priority buffer")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable OpenCV display window")
    args = parser.parse_args()

    run_realtime(
        video_source=args.source,
        model_path=args.model,
        extractor_name=args.extractor,
        show_display=not args.no_display,
        interval=args.interval,
        adaptive=not args.no_adaptive,
        use_priority_buffer=not args.no_priority
    )
