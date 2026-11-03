# Dataset Instructions

## Dataset Overview

This project compares two deep learning models, ResNet-50 and
DenseNet-121, for classifying chest X-ray images into three categories:

-   `Lung_Opacity`
-   `Normal`
-   `Viral Pneumonia`

The same dataset split is used for both models so their results can be
compared under the same conditions. The project also uses Grad-CAM to
create visual explanations of model predictions.

## Dataset Source

The project currently uses the Kaggle dataset **Lung X-Ray Image**:

https://www.kaggle.com/datasets/fatemehmehrparvar/lung-disease

The downloaded dataset contains the three class folders listed above.
Follow the dataset provider's terms of use and check the dataset page
for its current license and attribution requirements.

**Important:** This is a three-class dataset. It is not the two-class
`NORMAL` / `PNEUMONIA` pediatric dataset described by Kermany et
al. (2018). The Kermany et al. article should not be listed as the
source for this dataset unless that specific dataset is used.

## Original Dataset Folder Structure

After downloading and extracting the dataset, locate the folder that
directly contains the three class folders:

``` text
Lung X-Ray Image/
├── Lung_Opacity/
├── Normal/
└── Viral Pneumonia/
```

The folder may be nested inside one or more extraction folders. When
running the preparation script, set `--source` to the directory that
directly contains `Lung_Opacity`, `Normal`, and `Viral Pneumonia`.

## Prepared Dataset Folder Structure

The project preparation script creates the training, validation, and
test folders under:

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

The current split was created with random seed `42` using an approximate
70% training, 15% validation, and 15% test split.

### Current Image Counts

  Class                Training   Validation   Testing       Total
  ----------------- ----------- ------------ --------- -----------
  Lung_Opacity              787          168       170       1,125
  Normal                    875          187       188       1,250
  Viral Pneumonia           770          165       165       1,100
  **Total**           **2,432**      **520**   **523**   **3,475**

These counts describe the dataset split prepared for the current project
run. If you download a different version of the dataset or create a new
split, the counts may change.

## Preparing the Dataset

Run the preparation command from the project root directory. Replace the
source path with the location on your computer that directly contains
the three class folders.

Example for the current Windows setup:

``` powershell
python -m src.data.prepare_splits --source "C:\Users\hello\Desktop\Pavani\archive\Lung X-Ray Image\Lung X-Ray Image" --destination data/lung_xray_split --seed 42
```

The script copies images into the `train`, `val`, and `test` folders. It
does not require the original downloaded dataset to be stored inside the
project directory.

If `data/lung_xray_split` already exists and you need to create the
split again, first review and safely remove or rename the old generated
split. Do not rerun the command into an existing destination without
checking it, because old files may remain.

## Local Dataset Location

The training and evaluation scripts use the prepared dataset at:

``` text
data/lung_xray_split/
```

Keep the original downloaded dataset in a separate location if desired.
If the source or destination is different, provide the correct paths
when running the preparation script.

## Data Handling

The current implementation:

-   Loads images from the three class folders.
-   Resizes images to `224 × 224` pixels.
-   Converts images to three color channels as needed by the model
    pipeline.
-   Normalizes image values using ImageNet mean and standard deviation.
-   Applies the configured horizontal-flip augmentation to training
    images only.
-   Keeps training, validation, and test folders separate.

The validation set is used during model development and checkpoint
selection. The test set is reserved for final evaluation and should not
be used to tune the models.

## Reproducibility and Limitations

-   The split script uses random seed `42` so the image-level split can
    be reproduced when using the same source files and script.
-   The current split is image-level, not patient-level. If patient
    identifiers are available, a patient-level split should be
    considered to reduce the risk of images from the same patient
    appearing in more than one split.
-   The classes have slightly different image counts. The project
    therefore reports macro-averaged precision, recall, and F1-score in
    addition to accuracy.
-   The dataset's patient-level provenance and license should be checked
    on the provider's page before final publication or redistribution.
-   Do not include the dataset images in the Git repository unless the
    dataset license and course requirements clearly allow it. The source
    code and this README are sufficient to explain how to obtain and
    prepare the data.

## Related Research

The current project dataset is the Kaggle **Lung X-Ray Image** dataset
linked above. The following paper describes a different pediatric chest
X-ray dataset with `NORMAL` and `PNEUMONIA` labels; it is not the source
identified for the current three-class dataset:

Kermany, D. S., Goldbaum, M., Cai, W., et al. (2018). Identifying
medical diagnoses and treatable diseases by image-based deep learning.
*Cell, 172*(5), 1122--1131.e9.
https://doi.org/10.1016/j.cell.2018.02.010
