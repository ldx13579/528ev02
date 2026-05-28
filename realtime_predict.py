"""
Real-time Sliding Window Predictor for Hand-Raise Detection

Processes video input with a sliding window approach:
- Extracts features from frames in real-time
- Maintains a buffer of the last WINDOW_SIZE (30) frames
- Outputs prediction every PREDICTION_INTERVAL (0.5s)
- Supports webcam, video file, and pre-extracted feature inputs
"""

import time
import collections
import numpy as np
import torch
import cv2

from config import (
    WINDOW_SIZE, FEATURE_DIM, FPS, MODEL_DIR,
    PREDICTION_INTERVAL, IMAGE_SIZE, ensure_dirs
)
from model import TCNClassifier, TemporalAvgClassifier
from evaluate import load_model


class RealtimePredictor:
    """Sliding window predictor for real-time hand-raise detection."""

    def __init__(self, model_path=None, extractor_name="mobilenetv2", threshold=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold

        # Load model
        self.model, self.model_type = load_model(self.device, model_path)
        self.model.eval()
        print(f"Loaded {self.model_type} model on {self.device}")

        # Load feature extractor
        from feature_extraction import create_extractor
        self.extractor = create_extractor(extractor_name)
        print(f"Feature extractor: {extractor_name}")

        # Feature buffer: ring buffer of size WINDOW_SIZE
        self.buffer = collections.deque(maxlen=WINDOW_SIZE)
        self.last_prediction_time = 0.0
        self.prediction_interval = PREDICTION_INTERVAL

    def _extract_feature(self, frame):
        """Extract feature vector from a single BGR frame."""
        from PIL import Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        features = self.extractor.extract([pil_img])
        return features[0]

    def _predict(self):
        """Run prediction on the current buffer."""
        if len(self.buffer) < WINDOW_SIZE:
            return None, 0.0

        window = np.array(list(self.buffer), dtype=np.float32)
        x = torch.tensor(window).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logit = self.model(x)
            score = torch.sigmoid(logit).item()

        is_hand_raise = score > self.threshold
        return is_hand_raise, score

    def process_frame(self, frame):
        """Add a frame to the buffer. Returns prediction if interval elapsed."""
        feature = self._extract_feature(frame)
        self.buffer.append(feature)

        current_time = time.time()
        if current_time - self.last_prediction_time >= self.prediction_interval:
            self.last_prediction_time = current_time
            return self._predict()
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
        print(f"Prediction interval: {self.prediction_interval}s")
        print(f"Window size: {WINDOW_SIZE} frames")
        print("-" * 50)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % frame_interval != 0:
                    continue

                result, score = self.process_frame(frame)

                if result is not None:
                    prediction_count += 1
                    status = "HAND RAISED" if result else "no hand raise"
                    timestamp = frame_count / video_fps
                    print(f"  [{timestamp:6.1f}s] {status} (confidence: {score:.3f})")

                    if show_display:
                        display = frame.copy()
                        color = (0, 0, 255) if result else (0, 255, 0)
                        label = f"{'HAND RAISED' if result else 'No Raise'} ({score:.2f})"
                        cv2.putText(display, label, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                        cv2.imshow("Hand-Raise Detection", display)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

        finally:
            cap.release()
            if show_display:
                cv2.destroyAllWindows()

        print(f"\nDone. Processed {frame_count} frames, made {prediction_count} predictions.")

    def run_on_webcam(self):
        """Run real-time prediction on webcam input."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("Cannot open webcam")

        frame_count = 0
        frame_interval = max(1, int(30 / FPS))

        print("\nWebcam hand-raise detection started. Press 'q' to quit.")
        print(f"Prediction every {self.prediction_interval}s | Window: {WINDOW_SIZE} frames")
        print("-" * 50)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % frame_interval != 0:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                result, score = self.process_frame(frame)

                display = frame.copy()
                if result is not None:
                    color = (0, 0, 255) if result else (0, 255, 0)
                    label = f"{'HAND RAISED' if result else 'No Raise'} ({score:.2f})"
                else:
                    color = (128, 128, 128)
                    label = f"Buffering... ({len(self.buffer)}/{WINDOW_SIZE})"

                cv2.putText(display, label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                cv2.imshow("Hand-Raise Detection (Webcam)", display)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

        print("Webcam detection stopped.")

    def run_on_features(self, features_array):
        """Run sliding window prediction on pre-extracted features.

        Args:
            features_array: numpy array of shape (num_frames, feature_dim)

        Returns:
            List of (timestamp, is_hand_raise, score) tuples
        """
        predictions = []
        frames_per_interval = max(1, int(FPS * self.prediction_interval))

        for i in range(len(features_array)):
            self.buffer.append(features_array[i])

            if (i + 1) % frames_per_interval == 0 and len(self.buffer) == WINDOW_SIZE:
                result, score = self._predict()
                timestamp = (i + 1) / FPS
                predictions.append((timestamp, result, score))

        return predictions


def run_realtime(video_source="webcam", model_path=None, extractor_name="mobilenetv2",
                 show_display=True):
    """Entry point for real-time prediction.

    Args:
        video_source: "webcam", or path to a video file
        model_path: Path to model checkpoint (None uses default)
        extractor_name: Feature extractor backbone name
        show_display: Whether to show OpenCV window
    """
    ensure_dirs()
    predictor = RealtimePredictor(
        model_path=model_path,
        extractor_name=extractor_name
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
    parser.add_argument("--no-display", action="store_true",
                        help="Disable OpenCV display window")
    args = parser.parse_args()

    run_realtime(
        video_source=args.source,
        model_path=args.model,
        extractor_name=args.extractor,
        show_display=not args.no_display
    )
