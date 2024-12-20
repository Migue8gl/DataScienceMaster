# Image Classification Data Processing and Model Training

This document explains the usage of the `process_data.py` and `model_train_eval.py` scripts to preprocess image data, extract features, and train a model for classification tasks. Follow the steps below to use these scripts effectively.

## Requirements

Ensure you have the following dependencies installed:

* Python 3.8 or later
* numpy
* polars
* opencv-python
* scikit-learn
* seaborn

Install these dependencies using pip:
```bash
pip install -r requirements.txt
```

## File Overview

### 1. process_data.py

This script processes image datasets by resizing images, optionally extracting features using HOG or LBP descriptors, and saving the data in Parquet format for efficient storage and retrieval.

#### Usage:
```bash
python process_data.py -d <dataset_path> -l <labels> -s <image_size> -m <mode>
```

#### Arguments:

* `-d, --data-path`: Path to the dataset directory. Default: `./data`
* `-l, --labels`: List of label subdirectories. Example: `Cat Dog`
* `-s, --image-size`: Image size (MxM) to resize images. Default: 28
* `-m, --mode`: Processing mode. Options: `lbp`, `hog`, or `raw` (default: `raw`)

#### Examples:

Process a dataset with HOG descriptor:
```bash
python process_data.py -d ./data -l Cat Dog -s 64 -m hog
```
This command processes the dataset located in `./data` with subdirectories `Cat` and `Dog`, resizes images to 64x64, applies HOG descriptor, and saves the processed data as a Parquet file.

Process a dataset with raw images:
```bash
python process_data.py -d ./data -l Car Bike -s 128 -m raw
```
This command resizes images to 128x128 without applying any feature descriptor.

Process a dataset with LBP descriptor:
```bash
python process_data.py -d ./data -l Apple Orange -s 32 -m lbp
```
This command applies LBP descriptor to the images resized to 32x32.

### 2. model_train_eval.py

This script loads the processed dataset, trains an SVM classifier, and evaluates its performance. It can also perform hyperparameter searches for HOG or LBP descriptors and the SVM.

#### Usage:
```bash
python model_train_eval.py -d <data_file> -l <limit> -hs -desc <descriptor>
```

#### Arguments:

* `-d, --data-file`: Path to the processed dataset in Parquet format
* `-l, --limit`: Number of samples to load from the dataset. Default: 1000
* `-hs, --hyperparameter-search`: Enable hyperparameter search for the descriptor
* `-desc, --descriptor`: Descriptor type for hyperparameter search (`lbp` or `hog`)

#### Examples:

Train a model with HOG descriptor and hyperparameter search:
```bash
python model_train_eval.py -d dataset_raw_cat_dog_64.parquet -l 1000 -hs -desc hog
```
This command loads the processed dataset `dataset_raw_cat_dog_64.parquet`, trains an SVM with hyperparameter search for the HOG descriptor, and evaluates its performance. Its must be the raw one so the function can apply different HOG transformations.

Train a model with LBP descriptor:
```bash
python model_train_eval.py -d dataset_raw_apple_orange_32.parquet -l 500 -desc lbp
```
This command trains an SVM using features extracted with the LBP descriptor. Its must be the raw one so the function can apply different LBP transformations.

Train a model with raw images:
```bash
python model_train_eval.py -d dataset_raw_car_bike_128.parquet -l 2000
```
This command trains an SVM and search hyperparameters only for SVM.

## Workflow

1. Preprocess the Dataset:
   * Use `process_data.py` to preprocess the dataset and save it in Parquet format.
   ```bash
   python process_data.py -d ./data -l Cat Dog -s 64 -m hog
   ```

2. Train and Evaluate the Model:
   * Use `model_train_eval.py` to train and evaluate the model.
   ```bash
   python model_train_eval.py -d dataset_raw_cat_dog_64.parquet -l 1000 -hs -desc hog
   ```

## Output

* **Processed Data**: A Parquet file containing image features and labels.
* **Model Evaluation**:
  * Cross-validation accuracy and test accuracy
  * Saved model and evaluation plots in the results directory

For further details, refer to the docstrings within each script.
