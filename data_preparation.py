import os
import json
import cv2
import numpy as np
from config import FPS, FRAMES_DIR, VIDEOS_DIR, ANNOTATIONS_DIR


def imwrite_unicode(path, img):
    """cv2.imwrite wrapper that handles non-ASCII paths on Windows."""
    _, buf = cv2.imencode('.jpg', img)
    buf.tofile(path)


def extract_frames(video_path, output_dir, fps=FPS):
    """Extract frames from a video at the specified FPS."""
    os.makedirs(output_dir, exist_ok=True)
    # Use stream-based open for non-ASCII paths on Windows
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        import tempfile, shutil
        # Fallback: copy to temp path if path has non-ASCII chars
        tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        tmp.close()
        shutil.copy2(video_path, tmp.name)
        cap = cv2.VideoCapture(tmp.name)

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        cap.release()
        print(f"  Warning: Cannot read {video_path}")
        return 0
    frame_interval = int(round(video_fps / fps))

    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{saved_count:05d}.jpg")
            imwrite_unicode(frame_path, frame)
            saved_count += 1
        frame_idx += 1

    cap.release()
    print(f"  Extracted {saved_count} frames from {os.path.basename(video_path)}")
    return saved_count


def load_annotations(annotation_path):
    """Load hand-raise time intervals from a JSON file.

    Expected format: [{"start": 2.0, "end": 5.5}, ...]
    """
    with open(annotation_path, "r", encoding="utf-8") as f:
        intervals = json.load(f)
    return intervals


def generate_frame_labels(num_frames, intervals, fps=FPS):
    """Generate per-frame binary labels based on annotation intervals."""
    labels = np.zeros(num_frames, dtype=np.int64)
    for interval in intervals:
        start_frame = int(interval["start"] * fps)
        end_frame = int(interval["end"] * fps)
        start_frame = max(0, start_frame)
        end_frame = min(num_frames, end_frame)
        labels[start_frame:end_frame] = 1
    return labels


def prepare_all_videos():
    """Extract frames from all videos and generate labels."""
    video_files = sorted([f for f in os.listdir(VIDEOS_DIR)
                          if f.endswith(('.mp4', '.avi', '.mov'))])

    all_labels = {}

    for video_file in video_files:
        video_id = os.path.splitext(video_file)[0]
        video_path = os.path.join(VIDEOS_DIR, video_file)
        frames_output = os.path.join(FRAMES_DIR, video_id)

        print(f"Processing {video_file}...")
        num_frames = extract_frames(video_path, frames_output)

        annotation_file = os.path.join(ANNOTATIONS_DIR, f"{video_id}.json")
        if os.path.exists(annotation_file):
            intervals = load_annotations(annotation_file)
            labels = generate_frame_labels(num_frames, intervals)
        else:
            print(f"  Warning: No annotation for {video_id}, defaulting to all-zero labels")
            labels = np.zeros(num_frames, dtype=np.int64)

        labels_path = os.path.join(frames_output, "labels.npy")
        np.save(labels_path, labels)
        all_labels[video_id] = labels
        print(f"  Labels: {labels.sum()} positive / {len(labels)} total frames")

    return all_labels


if __name__ == "__main__":
    from config import ensure_dirs
    ensure_dirs()
    prepare_all_videos()
