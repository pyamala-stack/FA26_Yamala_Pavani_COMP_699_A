# Research source code plan

## Directory tree

```text
src/
├── __init__.py
├── config.py
├── data/
│   ├── __init__.py
│   ├── inspect_dataset.py
│   └── validate_images.py
└── analysis/
    └── class_distribution.py
requirements.txt
```

## Dataset layout

Download the public Kaggle Chest X-Ray Images (Pneumonia) dataset and extract it to `data/chest_xray/`. The scripts expect folders named `train`, `val`, and `test`, each containing `NORMAL` and `PNEUMONIA` folders. If your downloaded copy has a different structure, update the path arguments or adapt the script after inspecting the actual files.

Dataset page: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Do not commit the dataset itself to GitHub.

## Setup

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Install dependencies:
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run in order from the repository root

```bash
python src/data/inspect_dataset.py --data-root data/chest_xray --output-dir outputs/week_1
python src/data/validate_images.py --manifest outputs/week_1/image_manifest.csv --output-dir outputs/week_1
python src/analysis/class_distribution.py --manifest outputs/week_1/image_manifest.csv --output-dir outputs/week_1
```

## Expected outputs

- `image_manifest.csv`: one row per discovered image.
- `dataset_summary.txt`: counts and missing expected folders.
- `image_validation_report.txt`: number of checked and unreadable files.
- `invalid_images.csv`: details for images that failed validation.
- `class_distribution.csv`: counts per split and class.
- `class_distribution.png`: bar chart of class counts.

These are generated outputs. Review them and report the actual results; do not make up counts or claim successful execution before running the scripts.

## Scope

These scripts establish a Week 1 computational foundation. They do not train ResNet-50 or DenseNet-121. Week 2 can add data loaders, preprocessing tests, a pretrained ResNet-50 baseline, metrics, then the second model and Grad-CAM comparisons.
