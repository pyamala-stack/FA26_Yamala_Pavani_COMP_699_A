# Comparative Study of Explainable AI Models for Pneumonia Detection from Chest X-rays

## Project Purpose

This project compares two pretrained deep learning models, ResNet50 and DenseNet121, for detecting pneumonia from chest X-ray images. The goal is to compare their classification performance and understand how they explain their predictions using Grad-CAM visualizations.

The project will evaluate both models using accuracy, precision, recall, F1-score, and ROC-AUC. Grad-CAM will help visualize the image regions that influence each model's predictions. The project will also examine cases where the two models make different predictions.

## Technologies Used

- Python
- PyTorch and Torchvision
- NumPy and Pandas
- Scikit-learn
- Matplotlib
- Grad-CAM for model explainability

## Dataset Instructions

This project uses a publicly available chest X-ray dataset containing images labelled as **NORMAL** or **PNEUMONIA**. The dataset is associated with the research by Kermany et al. (2018).

Download the dataset from the source provided with the research dataset and extract it locally. Arrange the image folders in the following structure:

```text
data/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

Keep the dataset files on your local computer rather than uploading the complete dataset to GitHub. Follow the dataset's original terms and conditions when using or sharing the images.

## Setup Instructions

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd explainable-pneumonia-detection
```

Replace `<YOUR_GITHUB_REPOSITORY_URL>` with the URL of your GitHub repository.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` file should list the packages and versions required by the project.

### 4. Prepare the dataset

Download and extract the chest X-ray dataset. Place it in the `data/chest_xray/` directory using the folder structure shown above.

## How to Run the Code

Run the preliminary experiment using:

```bash
python src/pilot.py
```

The pilot script is intended to test the initial image-processing and model-inference workflow. The exact command and expected output should be updated if the implementation uses different filenames or arguments.

After the pilot runs successfully, record its actual output and save any generated sample predictions or Grad-CAM visualizations in the project's `results/` directory.

## Research Approach

The project will follow these steps:

1. Load and preprocess the chest X-ray images.
2. Set up the ResNet50 and DenseNet121 models.
3. Evaluate both models using the same test dataset.
4. Calculate accuracy, precision, recall, F1-score, and ROC-AUC.
5. Generate Grad-CAM visualizations to examine the image regions influencing predictions.
6. Compare model predictions and investigate cases where the models disagree.
7. Document the results, limitations, and observations.

## Expected Project Output

The project is expected to produce model evaluation metrics, prediction comparisons, and Grad-CAM visualizations. These outputs will provide evidence for comparing the models' performance and explanations.

The results will be reported only after the experiments have been run. Grad-CAM visualizations will be treated as explanations of model behavior, not as proof that the models use medically correct features.
