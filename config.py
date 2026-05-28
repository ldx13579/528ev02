import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Frame extraction
FPS = 5
VIDEO_DURATION = 30  # seconds per video
FRAMES_PER_VIDEO = FPS * VIDEO_DURATION  # 150

# Model
WINDOW_SIZE = 10  # consecutive frames for temporal averaging
FEATURE_DIM = 1280  # MobileNetV2 output dimension
IMAGE_SIZE = 224

# Training
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
TRAIN_SPLIT = 0.8

# Paths
DATA_DIR = os.path.join(BASE_DIR, "data")
VIDEOS_DIR = os.path.join(DATA_DIR, "videos")
ANNOTATIONS_DIR = os.path.join(DATA_DIR, "annotations")
FRAMES_DIR = os.path.join(DATA_DIR, "frames")
FEATURES_DIR = os.path.join(DATA_DIR, "features")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "model")
HEATMAPS_DIR = os.path.join(OUTPUT_DIR, "heatmaps")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")

# Synthetic data
NUM_SYNTHETIC_VIDEOS = 10


def ensure_dirs():
    for d in [VIDEOS_DIR, ANNOTATIONS_DIR, FRAMES_DIR, FEATURES_DIR,
              MODEL_DIR, HEATMAPS_DIR, METRICS_DIR]:
        os.makedirs(d, exist_ok=True)
