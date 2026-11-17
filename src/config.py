
"""Central configuration for the three-class chest X-ray experiment."""

from pathlib import Path

# Project root directory
ROOT = Path(__file__).resolve().parents[1]

# Dataset configuration
DATA_ROOT = ROOT / "data" / "lung_xray"
SPLIT_ROOT = ROOT / "data" / "lung_xray_split_clean"

# Experiment output directory
OUTPUT_ROOT = ROOT / "outputs" / "experiments"

# Classification classes
CLASS_NAMES = [
    "Lung_Opacity",
    "Normal",
    "Viral Pneumonia"
]

NUM_CLASSES = len(CLASS_NAMES)

# Image preprocessing
IMAGE_SIZE = 224

# Training configuration
BATCH_SIZE = 8
EPOCHS = 2
LEARNING_RATE = 0.0001

# Reproducibility
SEED = 42
NUM_WORKERS = 0

# Supported comparison models
MODEL_NAMES = ["resnet50", "densenet121"]
