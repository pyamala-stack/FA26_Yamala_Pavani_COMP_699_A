# Dataset Instructions

## Dataset Overview

This project uses a publicly available chest X-ray dataset to study pneumonia detection using deep learning. The images are classified into two categories: NORMAL and PNEUMONIA.

The dataset is used to train and evaluate ResNet50 and DenseNet121 models and to generate Grad-CAM visualizations.

## Dataset Source

The project is based on the pediatric chest X-ray dataset described by Kermany et al. (2018).

Research article: https://doi.org/10.1016/j.cell.2018.02.010

Download the dataset from the source associated with the research article or its publicly available dataset distribution. Follow the dataset's usage terms.

## Folder Structure

After downloading and extracting the dataset, arrange the files as follows:

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

Place each chest X-ray image in the appropriate category folder.

- `train`: Images used to train the model.
- `val`: Images used to evaluate the model during development.
- `test`: Images reserved for final model evaluation.
- `NORMAL`: Images labelled as normal.
- `PNEUMONIA`: Images labelled as pneumonia.

## Local Dataset Location

The application expects the dataset at:

```text
data/chest_xray/
```

If you store the dataset in a different location, enter the correct folder path in the application's Dataset folder field.

## Data Handling

The application resizes images to 224 × 224 pixels, converts them to three-channel images, and normalizes them before passing them to the models.

The dataset should retain its original labels. Keep the training, validation, and test sets separate to avoid using test images during model training.
