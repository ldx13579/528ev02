import os
import json
import numpy as np
import cv2
from config import (
    FRAMES_DIR, ANNOTATIONS_DIR, IMAGE_SIZE, FPS,
    NUM_SYNTHETIC_VIDEOS, FRAMES_PER_VIDEO, ensure_dirs
)


def imwrite_unicode(path, img):
    """cv2.imwrite wrapper that handles non-ASCII paths on Windows."""
    _, buf = cv2.imencode('.jpg', img)
    buf.tofile(path)


def generate_synthetic_frame(is_hand_raise, rng):
    """Generate a synthetic 224x224 frame.

    Hand-raise frames have a vertical rectangle in the upper portion (simulating a raised arm).
    No-raise frames have only background elements in the lower portion.
    """
    frame = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

    # Random background color
    bg_color = rng.integers(30, 80, size=3).tolist()
    frame[:] = bg_color

    # Add some "classroom" texture (random rectangles in lower half)
    for _ in range(rng.integers(2, 5)):
        x1 = rng.integers(0, IMAGE_SIZE - 30)
        y1 = rng.integers(IMAGE_SIZE // 2, IMAGE_SIZE - 20)
        x2 = x1 + rng.integers(20, 60)
        y2 = y1 + rng.integers(10, 40)
        color = rng.integers(60, 150, size=3).tolist()
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)

    # Add "person" body (rectangle in lower-center area)
    body_x = rng.integers(IMAGE_SIZE // 4, IMAGE_SIZE // 2)
    body_w = rng.integers(40, 60)
    body_y = IMAGE_SIZE // 2 + rng.integers(0, 30)
    body_h = IMAGE_SIZE - body_y - 10
    body_color = rng.integers(100, 200, size=3).tolist()
    cv2.rectangle(frame, (body_x, body_y), (body_x + body_w, body_y + body_h), body_color, -1)

    # Head
    head_cx = body_x + body_w // 2
    head_cy = body_y - 15
    head_color = [int(c + 30) for c in body_color[:2]] + [body_color[2]]
    cv2.circle(frame, (head_cx, head_cy), 12, head_color, -1)

    if is_hand_raise:
        # Draw raised arm: a vertical rectangle from body to upper region
        arm_x = body_x + body_w - 5 + rng.integers(-5, 10)
        arm_y_top = rng.integers(10, 50)  # reaches into upper portion
        arm_y_bottom = body_y + 10
        arm_w = rng.integers(8, 15)
        arm_color = [min(255, c + 50) for c in body_color]
        cv2.rectangle(frame, (arm_x, arm_y_top), (arm_x + arm_w, arm_y_bottom), arm_color, -1)

        # Hand at top of arm
        hand_color = [min(255, c + 80) for c in body_color]
        cv2.circle(frame, (arm_x + arm_w // 2, arm_y_top), 10, hand_color, -1)

    # Add noise
    noise = rng.integers(-10, 10, size=frame.shape, dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return frame


def generate_synthetic_dataset():
    """Generate NUM_SYNTHETIC_VIDEOS synthetic videos with annotations."""
    ensure_dirs()
    rng = np.random.default_rng(42)

    print("Generating synthetic dataset...")

    for vid_idx in range(NUM_SYNTHETIC_VIDEOS):
        video_id = f"synthetic_{vid_idx:02d}"
        frames_dir = os.path.join(FRAMES_DIR, video_id)
        os.makedirs(frames_dir, exist_ok=True)

        # Generate random hand-raise intervals (1-3 per video)
        num_intervals = rng.integers(1, 4)
        intervals = []
        used_times = set()

        for _ in range(num_intervals):
            for attempt in range(20):
                start = round(rng.uniform(1.0, 25.0), 1)
                duration = round(rng.uniform(2.0, 6.0), 1)
                end = min(start + duration, 29.0)
                # Check no overlap
                overlap = False
                for existing in intervals:
                    if start < existing["end"] and end > existing["start"]:
                        overlap = True
                        break
                if not overlap:
                    intervals.append({"start": start, "end": end})
                    break

        intervals.sort(key=lambda x: x["start"])

        # Generate frame labels
        labels = np.zeros(FRAMES_PER_VIDEO, dtype=np.int64)
        for interval in intervals:
            start_frame = int(interval["start"] * FPS)
            end_frame = int(interval["end"] * FPS)
            labels[start_frame:min(end_frame, FRAMES_PER_VIDEO)] = 1

        # Generate frames
        for frame_idx in range(FRAMES_PER_VIDEO):
            is_raise = labels[frame_idx] == 1
            frame = generate_synthetic_frame(is_raise, rng)
            frame_path = os.path.join(frames_dir, f"frame_{frame_idx:05d}.jpg")
            imwrite_unicode(frame_path, frame)

        # Save labels
        np.save(os.path.join(frames_dir, "labels.npy"), labels)

        # Save annotations
        annotation_path = os.path.join(ANNOTATIONS_DIR, f"{video_id}.json")
        with open(annotation_path, "w", encoding="utf-8") as f:
            json.dump(intervals, f, indent=2)

        pos_count = labels.sum()
        print(f"  {video_id}: {FRAMES_PER_VIDEO} frames, "
              f"{pos_count} positive ({pos_count/FRAMES_PER_VIDEO*100:.1f}%), "
              f"{len(intervals)} intervals")

    print(f"Done. Generated {NUM_SYNTHETIC_VIDEOS} synthetic videos.")


if __name__ == "__main__":
    generate_synthetic_dataset()
