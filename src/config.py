"""Central configuration for the three-class chest X-ray experiment."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "lung_xray"
SPLIT_ROOT = ROOT / "data" / "lung_xray_split"
OUTPUT_ROOT = ROOT / "outputs" / "week_2"

CLASS_NAMES = ["Lung_Opacity", "Normal", "Viral Pneumonia"]
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 2
LEARNING_RATE = 1e-4
SEED = 42
NUM_WORKERS = 0
