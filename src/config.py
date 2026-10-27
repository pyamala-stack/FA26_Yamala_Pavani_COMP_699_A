"""Central place for paths and label names used by the initial research scripts."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "chest_xray"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "week_1"
LABELS = {"NORMAL": 0, "PNEUMONIA": 1}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
