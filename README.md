# Comparative Study of Explainable AI Models for Chest X-ray Classification

## Project overview

This project compares two convolutional neural network models, **ResNet-50** and **DenseNet-121**, for classifying chest X-ray images into three categories:

1. `Lung_Opacity`
2. `Normal`
3. `Viral Pneumonia`

The project evaluates classification performance and uses **Grad-CAM** to create visual explanations of model predictions. It also compares the predictions made by the two models and identifies images where their predictions differ.

Grad-CAM heatmaps show which image regions influenced a model's output. They are research visualizations only. They do not prove that a model is using medically correct evidence and must not be treated as a clinical decision tool.

## Research question

How do ResNet-50 and DenseNet-121 compare in three-class chest X-ray classification, and how do their Grad-CAM visual explanations differ on the same test images?

## Current project features

- Three-class chest X-ray dataset loading.
- Consistent image resizing and ImageNet normalization.
- Training and validation workflow for ResNet-50 and DenseNet-121.
- Frozen pretrained feature backbone with classifier-layer training.
- Best-checkpoint saving based on validation loss.
- Test-set evaluation with accuracy, macro precision, macro recall, macro F1, confusion matrix, per-class report, and macro one-vs-rest ROC-AUC.
- CSV export of individual test predictions and class probabilities.
- Side-by-side prediction comparison on matching image paths.
- Separate list of model disagreements.
- Grad-CAM heatmaps for selected test images, with actual class, predicted class, confidence, and correctness.
- Dataset audit for unreadable images, dimensions, and exact duplicate content across splits.
- Reproducible duplicate-free split preparation that refuses to overwrite a non-empty destination.
- **Added feature:** class-wise error analysis from saved test predictions. It reports support, correct and incorrect counts, per-class recall/error rate, actual/predicted class counts, and individual misclassified image records.

## Technology

- Python
- PyTorch and Torchvision
- NumPy
- scikit-learn
- Matplotlib
- Pillow

The supplied source archive does not include a `requirements.txt`, dataset, model checkpoints, or existing `outputs/` files. Use the package versions from your working environment or create a requirements file after confirming the versions that successfully run on your computer.

## Project structure

```text
project/
├── data/
│   └── lung_xray_split_clean/
│       ├── train/
│       │   ├── Lung_Opacity/
│       │   ├── Normal/
│       │   └── Viral Pneumonia/
│       ├── val/
│       │   ├── Lung_Opacity/
│       │   ├── Normal/
│       │   └── Viral Pneumonia/
│       └── test/
│           ├── Lung_Opacity/
│           ├── Normal/
│           └── Viral Pneumonia/
├── outputs/
│   ├── experiments/
│   └── model_comparison/
├── src/
│   ├── config.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── metrics.py
│   ├── compare.py
│   ├── explain.py
│   ├── error_analysis.py
│   ├── data/
│   │   ├── audit.py
│   │   ├── clean_resplit.py
│   │   ├── dataset.py
│   │   └── prepare_splits.py
│   └── tests/
│       ├── test_metrics.py
│       └── test_error_analysis.py
└── README.md
```

## Dataset layout

The configured split directory is:

```text
data/lung_xray_split_clean/
```

Each split must contain all three class folders:

```text
train/
val/
test/
```

The folder names must match the class names in `src/config.py` exactly. The dataset loader reads common image formats (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`) recursively.

**Important validity note:** The dataset audit can detect exact duplicate image files across splits, but it cannot establish patient-level separation when patient identifiers are unavailable. Do not describe the split as patient-independent unless patient-level metadata has been checked.

## Environment setup

Open a terminal in the project root and create a virtual environment:

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the libraries used by the source:

```powershell
pip install torch torchvision numpy scikit-learn matplotlib pillow
```

If PyTorch installation fails or the installed Torch/Torchvision versions are incompatible, use the official PyTorch installation selector for the correct command for your computer.

## Configuration

The main settings are in `src/config.py`:

- `SPLIT_ROOT`: clean dataset split directory.
- `OUTPUT_ROOT`: experiment output directory.
- `CLASS_NAMES`: the fixed three-class order.
- `IMAGE_SIZE`: image size used by the models.
- `BATCH_SIZE`: training batch size.
- `EPOCHS`: training epochs.
- `LEARNING_RATE`: Adam learning rate.
- `SEED`: random seed.
- `MODEL_NAMES`: supported model names.

Keep the class order consistent across training, evaluation, checkpoints, and Grad-CAM. The checkpoint loader checks the saved class names before generating explanations.

## Run the project

Run commands from the project root.

### 1. Audit the existing dataset

```powershell
python -m src.data.audit
```

Optional custom data location:

```powershell
python -m src.data.audit --data-root "data/lung_xray_split_clean" --output "outputs/data_audit.json"
```

Review unreadable image counts, split counts, dimensions, and duplicate warnings before training.

### 2. Train ResNet-50

```powershell
python -m src.train --model resnet50
```

### 3. Train DenseNet-121

```powershell
python -m src.train --model densenet121
```

The training script saves the best checkpoint, training history, experiment configuration, and experiment summary under:

```text
outputs/experiments/resnet50/
outputs/experiments/densenet121/
```

The default training code freezes the pretrained feature backbone and trains the final classifier. It uses the configured seed and saves the checkpoint with the lowest validation loss.

### 4. Evaluate each model

```powershell
python -m src.evaluate --model resnet50
python -m src.evaluate --model densenet121
```

Each evaluation creates:

- `test_metrics.json`
- `test_predictions.csv`

inside the corresponding model output directory.

### 5. Compare the model predictions

```powershell
python -m src.compare
```

The comparison uses the image paths shared by both prediction files and checks that their actual labels match. It creates:

- `prediction_comparison.csv`
- `model_disagreements.csv`
- `comparison_summary.json`
- `prediction_distribution.png`

The agreement rate is descriptive. Agreement does not prove that either model is correct.

### 6. Generate Grad-CAM explanations

Generate Grad-CAM for the first six sorted test images:

```powershell
python -m src.explain --model resnet50 --checkpoint outputs/experiments/resnet50/best_model.pt --count 6
python -m src.explain --model densenet121 --checkpoint outputs/experiments/densenet121/best_model.pt --count 6
```

Generate Grad-CAM only for images where the models disagree:

```powershell
python -m src.explain --model resnet50 --checkpoint outputs/experiments/resnet50/best_model.pt --count 12 --disagreements
python -m src.explain --model densenet121 --checkpoint outputs/experiments/densenet121/best_model.pt --count 12 --disagreements
```

The disagreement option expects:

```text
outputs/model_comparison/model_disagreements.csv
```

Grad-CAM images and result files are saved under:

```text
outputs/experiments/resnet50/gradcam/
outputs/experiments/densenet121/gradcam/
```

Each folder contains heatmap images, `gradcam_results.csv`, and `gradcam_summary.json`.

### 7. Run the added class-wise error analysis

Run this after evaluating a model:

```powershell
python -m src.error_analysis --model resnet50
python -m src.error_analysis --model densenet121
```

The command reads the existing `test_predictions.csv` files and writes:

```text
outputs/experiments/error_analysis/
├── resnet50_error_analysis.json
├── resnet50_class_error_summary.csv
├── resnet50_misclassified_predictions.csv
├── densenet121_error_analysis.json
├── densenet121_class_error_summary.csv
└── densenet121_misclassified_predictions.csv
```

The class summary reports:

- Number of actual test images in each class (`support`).
- Correct and incorrect predictions for each actual class.
- Per-class recall (shown as `accuracy_recall`).
- Per-class error rate.
- Actual and predicted class counts.

The misclassified-predictions CSV keeps the image path, actual class, predicted class, and probability columns from the original evaluation file. This makes it possible to inspect individual errors without retraining the models.

## Tests

Run the unit tests from the project root:

```powershell
python -m unittest discover -s src/tests -v
```

The tests check the existing metric calculations and the added class-wise error-analysis calculations. A passing unit test confirms the tested code behavior; it does not validate model quality or medical usefulness.

## Existing experiment results

The following values were shared from the project's current test runs. They are recorded here as the project's reported results, not independently re-created by this README.

| Metric | ResNet-50 | DenseNet-121 |
|---|---:|---:|
| Test images | 523 | 523 |
| Accuracy | 82.03% | 77.25% |
| Macro precision | 0.8187 | 0.7721 |
| Macro recall | 0.8266 | 0.7771 |
| Macro F1 | 0.8196 | 0.7728 |
| Macro OVR ROC-AUC | 0.9330 | 0.9153 |

The models agreed on 436 of 523 test images (83.37%) and disagreed on 87 (16.63%). These values should be regenerated if the dataset, checkpoints, preprocessing, or experiment settings change.

## Known limitations

1. Patient identifiers were not available in the supplied dataset structure, so patient-level split independence has not been confirmed.
2. The experiments were run on CPU in the reported environment; training is therefore slower than a suitable GPU run.
3. The current Grad-CAM review uses selected examples. A small selection cannot represent the full test set.
4. Grad-CAM highlights model sensitivity, not verified medical reasoning.
5. Performance on this dataset does not establish clinical readiness or performance on external hospitals and populations.
6. The supplied archive contains source files only. Dataset files, checkpoints, generated outputs, and dependency lock files were not included for independent reproduction.

## Safe GitHub update guidance

This update adds `src/error_analysis.py`, its unit test, and this README. It does not require replacing the existing training, evaluation, comparison, or Grad-CAM modules.

Before uploading:

1. Keep a local backup or download a ZIP of the current working repository.
2. Add the new files without deleting or renaming existing files.
3. Upload `README.md` at the repository root.
4. Run the unit tests and then run the new analysis against the existing prediction CSVs.
5. Review the generated CSV/JSON files before committing them.
6. Do not upload private patient information, credentials, virtual environments, or large datasets.

## Repository

https://github.com/pyamala-stack/FA26_Yamala_Pavani_COMP_699_A
