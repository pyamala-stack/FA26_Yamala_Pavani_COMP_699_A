# Comparative Study of Explainable AI Models for Chest X-ray Classification

## Project Overview

This project compares two pretrained deep learning models, **ResNet-50**
and **DenseNet-121**, for classifying chest X-ray images into three
categories:

-   `Lung_Opacity`
-   `Normal`
-   `Viral Pneumonia`

The project studies both classification performance and model
explanations. Grad-CAM is used to create heat maps that highlight image
regions influencing a model's prediction. The project also aims to
examine examples where ResNet-50 and DenseNet-121 make different
predictions.

The models are evaluated using accuracy, macro-averaged precision,
macro-averaged recall, macro-averaged F1-score, a confusion matrix, and
macro one-versus-rest ROC-AUC when it can be calculated.

Grad-CAM visualizations are treated as explanations of model behavior.
They are not proof that a model focuses on medically correct features or
that it is suitable for clinical use.

## Current Project Status

**Status: Week 2 implementation and initial data preparation completed;
model experiments are still in progress.**

The following work has been completed and checked:

-   The project dataset was downloaded and its three class folders were
    confirmed.
-   A script created training, validation, and test splits using random
    seed `42`.
-   The prepared dataset contains **3,475 images**.
-   A PowerShell folder check confirmed the image counts in all nine
    class/split folders.
-   The required Python libraries imported successfully.
-   PyTorch reported version `2.15.0+cpu` and Torchvision reported
    version `0.29.2+cpu`.
-   CUDA was not available in the current environment, so training is
    running on the CPU.
-   The ResNet-50 training command was started, and the pretrained
    weights downloaded successfully.

The following work is **not yet confirmed as complete**:

-   ResNet-50 training completion and its epoch-level results.
-   DenseNet-121 training.
-   Final test-set evaluation and comparison of the two models.
-   Grad-CAM generation and review.
-   A successful run of the automated unit tests.

No model accuracy or other classification performance values are
reported here because completed training and evaluation results have not
yet been confirmed. Update this section when those experiments finish.

## Dataset

### Dataset Source

This project currently uses the Kaggle **Lung X-Ray Image** dataset:

https://www.kaggle.com/datasets/fatemehmehrparvar/lung-disease

The dataset used in this project has three classes: `Lung_Opacity`,
`Normal`, and `Viral Pneumonia`.

**Dataset note:** This is not the two-class pediatric `NORMAL` /
`PNEUMONIA` dataset described by Kermany et al. (2018). Therefore, the
Kermany et al. paper is not identified as the source of the current
three-class dataset. Check the Kaggle dataset page for its current
license, attribution requirements, and usage terms before redistributing
or publishing any images.

### Original Dataset Folder Structure

After extracting the downloaded dataset, locate the folder that directly
contains these three class folders:

``` text
Lung X-Ray Image/
├── Lung_Opacity/
├── Normal/
└── Viral Pneumonia/
```

The extracted files may be nested inside more than one folder. The
`--source` argument must point to the directory directly containing the
three class folders.

### Prepared Dataset Folder Structure

The preparation script creates the following structure inside the
project:

``` text
data/
└── lung_xray_split/
    ├── train/
    │   ├── Lung_Opacity/
    │   ├── Normal/
    │   └── Viral Pneumonia/
    ├── val/
    │   ├── Lung_Opacity/
    │   ├── Normal/
    │   └── Viral Pneumonia/
    └── test/
        ├── Lung_Opacity/
        ├── Normal/
        └── Viral Pneumonia/
```

### Current Dataset Counts

  Class                Training   Validation   Testing       Total
  ----------------- ----------- ------------ --------- -----------
  Lung_Opacity              787          168       170       1,125
  Normal                    875          187       188       1,250
  Viral Pneumonia           770          165       165       1,100
  **Total**           **2,432**      **520**   **523**   **3,475**

These are the counts from the current prepared split. They may differ if
another dataset copy or a new split is used.

### Prepare the Dataset

Run this command from the project root directory. Change the source path
if the dataset is stored somewhere else:

``` powershell
python -m src.data.prepare_splits --source "C:\path\to\Lung X-Ray Image\Lung X-Ray Image" --destination data/lung_xray_split --seed 42
```

The source directory must directly contain `Lung_Opacity`, `Normal`, and
`Viral Pneumonia`.

The script copies the images into the prepared split folders. If the
destination folder already exists and you need to create the split
again, first review and rename or remove the old generated split. Do not
rerun the command into an existing destination without checking it,
because old files may remain.

The current split is random at the image level, not patient level. If
patient identifiers are available, a patient-level split should be
considered to reduce the possibility of images from the same patient
appearing in more than one split.

## Technologies and Tools

-   Python
-   PyTorch
-   Torchvision
-   NumPy
-   Pillow
-   Scikit-learn
-   Matplotlib
-   Grad-CAM implementation in the project source code
-   Visual Studio Code
-   Git and GitHub

The current environment was checked with PyTorch `2.15.0+cpu` and
Torchvision `0.29.2+cpu`. The project currently uses CPU execution
because CUDA is unavailable.

## Project Structure

The main source-code structure is:

``` text
Project/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── metrics.py
│   ├── explain.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── prepare_splits.py
│   │   └── dataset.py
│   └── tests/
│       ├── __init__.py
│       └── test_metrics.py
├── data/
│   └── lung_xray_split/
├── docs/
│   └── step_2/
│       └── week_2/
│           └── method_and_baseline.pdf
├── requirements.txt
└── README.md
```

Additional folders for experiment runners, baseline implementations, or
result visualizations should only be added when corresponding code is
implemented. The two models in this study are the comparison models; a
separate baseline model should not be claimed unless one is actually
implemented and evaluated.

## Setup Instructions

### 1. Clone the repository

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

Replace the placeholders with your actual GitHub repository URL and
folder name.

### 2. Create a virtual environment

On Windows PowerShell:

``` powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

``` bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
python -m pip install -r requirements.txt
```

### 4. Prepare the dataset

Download and extract the three-class dataset. Follow the folder
structure and preparation instructions in the Dataset section above.

## Running the Experiments

Run commands from the project root directory.

### Train ResNet-50

``` powershell
python -m src.train --model resnet50
```

### Train DenseNet-121

``` powershell
python -m src.train --model densenet121
```

The current initial configuration uses an image size of `224 × 224`,
batch size `8`, two epochs, learning rate `0.0001`, Adam optimizer,
cross-entropy loss, and random seed `42`. Two epochs are an initial
pipeline setting and may not be sufficient for a final research
comparison.

### Evaluate the Models and Generate Explanations

The project includes `src/evaluate.py` for test-set evaluation and
`src/explain.py` for Grad-CAM visualizations. Check each script's
command-line options before running it, and use the saved model
checkpoints produced by training. Record the actual commands and output
paths after confirming them in the current implementation.

### Run Automated Tests

``` powershell
python -m unittest discover -s src/tests -v
```

A test file is present at `src/tests/test_metrics.py`. The test suite
should be run and its result recorded; the presence of the test file
alone does not mean the tests have passed.

## Data Processing

The current implementation is designed to:

-   Load images from the three class folders.
-   Resize images to `224 × 224` pixels.
-   Prepare images as three-channel inputs for the pretrained models.
-   Normalize images using ImageNet mean and standard deviation.
-   Apply the configured horizontal-flip augmentation to training
    images.
-   Keep training, validation, and test data separate.

The validation set is used during development and model checkpoint
selection. The test set is reserved for final evaluation and should not
be used to tune the models.

## Research Workflow

The planned experiment follows these steps:

1.  Load and preprocess the chest X-ray images.
2.  Train ResNet-50 using the training set and validation set.
3.  Train DenseNet-121 using the same data split and training settings.
4.  Evaluate both trained models on the same held-out test set.
5.  Calculate accuracy, macro precision, macro recall, macro F1-score,
    confusion matrices, and ROC-AUC when calculable.
6.  Generate Grad-CAM visualizations for selected test images.
7.  Compare the visual explanations and identify examples where the
    models make different predictions.
8.  Record the results, limitations, and observations.

## Expected Outputs

After the experiments are completed, the project is expected to produce:

-   Trained model checkpoints for ResNet-50 and DenseNet-121.
-   Training and validation history.
-   Test-set evaluation metrics.
-   Confusion matrices.
-   Prediction records for test images.
-   Grad-CAM visualizations.
-   A comparison of model predictions and selected disagreement cases.

The actual output locations should be recorded after confirming where
the current scripts save their files.

## Limitations and Reproducibility Notes

-   The current environment uses CPU-only PyTorch, so training may take
    longer than GPU training.
-   The current two-epoch configuration is intended for an initial
    implementation check, not a final performance claim.
-   The current split is image-level. Patient-level separation should be
    considered if patient identifiers can be obtained.
-   The dataset classes have different image counts. Macro-averaged
    metrics are included to give each class equal weight in the summary.
-   Grad-CAM highlights image regions that influence model output, but
    it does not establish medical correctness or clinical reliability.
-   Record the dataset version, code changes, parameters, and actual
    experiment outputs so that the work can be reproduced.

## Project Status Summary

At the current Week 2 stage, the dataset has been prepared and verified,
the Python environment has passed the import check, and the source code
for model training, evaluation, metrics, and Grad-CAM is in place.
ResNet-50 training has been launched and its pretrained weights have
downloaded, but the training completion and model metrics have not yet
been confirmed. DenseNet-121 training and the final comparison remain to
be completed.

**Update this README when training, testing, and Grad-CAM runs finish.
Only add measured results after they have been produced by the code.**
