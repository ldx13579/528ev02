import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Frame extraction
FPS = 5
VIDEO_DURATION = 30  # seconds per video
FRAMES_PER_VIDEO = FPS * VIDEO_DURATION  # 150

# Model
WINDOW_SIZE = 30  # consecutive frames for TCN input
FEATURE_DIM = 1280  # MobileNetV2 output dimension
IMAGE_SIZE = 224

# TCN Architecture
TCN_NUM_CHANNELS = [128, 128, 64]  # channels per residual block
TCN_KERNEL_SIZE = 3  # convolution kernel size
TCN_DROPOUT = 0.3  # dropout rate in TCN blocks

# Training
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
NUM_EPOCHS = 50
TRAIN_SPLIT = 0.8
AUGMENT = True  # enable data augmentation

# Real-time inference
PREDICTION_INTERVAL = 0.5  # seconds between predictions
SLIDING_WINDOW_STRIDE = 1  # frames to advance per prediction step

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
NUM_SYNTHETIC_VIDEOS = 30


def ensure_dirs():
    for d in [VIDEOS_DIR, ANNOTATIONS_DIR, FRAMES_DIR, FEATURES_DIR,
              MODEL_DIR, HEATMAPS_DIR, METRICS_DIR]:
        os.makedirs(d, exist_ok=True)
