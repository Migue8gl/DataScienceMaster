import argparse
import os
import textwrap
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
import time


def create_hog_descriptor(hog_params: Optional[Dict] = None) -> cv2.HOGDescriptor:
    """Create and return a HOG descriptor with specified parameters."""
    if hog_params is None:
        return None
    return cv2.HOGDescriptor(
        _winSize=hog_params["winSize"],
        _blockSize=hog_params["blockSize"],
        _blockStride=hog_params["blockStride"],
        _cellSize=hog_params["cellSize"],
        _nbins=hog_params.get("nbins", 9),
        _derivAperture=hog_params.get("derivAperture", 1),
        _winSigma=hog_params.get("winSigma", -1.0),
        _histogramNormType=hog_params.get("histogramNormType", 0),
        _L2HysThreshold=hog_params.get("L2HysThreshold", 0.2),
        _gammaCorrection=hog_params.get("gammaCorrection", 1),
        _nlevels=hog_params.get("nlevels", 64),
        _signedGradient=hog_params.get("signedGradients", True),
    )


def extract_hog_features(
    img: np.ndarray, hog_descriptor: cv2.HOGDescriptor
) -> np.ndarray:
    """Extract HOG features from an image using a HOG descriptor."""
    return hog_descriptor.compute(img).flatten()


def process_image(
    file_path: str,
    image_size: Tuple[int, int],
    hog_descriptor: Optional[cv2.HOGDescriptor] = None,
) -> np.ndarray:
    """Load and preprocess an image, returning its HOG features."""
    img = cv2.imread(file_path)
    if img is None or img.size == 0:
        raise ValueError(f"Invalid image at {file_path}")
    img = cv2.resize(img, image_size)
    result = img
    if hog_descriptor is not None:
        result = extract_hog_features(img, hog_descriptor)
    return result


def read_dataset(
    path: str,
    labels: List[str],
    image_size: Tuple[int, int],
    hog_params: Dict,
    parquet_file: str = "dataset.parquet",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads images from a dataset path, extracts descriptor features, and saves the dataset as a Parquet file.
    Reloads the dataset if the Parquet file already exists using Polars.

    Args:
        path (str): Path to the dataset directory.
        labels (List[str]): List of label subdirectories.
        image_size (Tuple[int, int]): Target image size for processing.
        hog_params (Dict): Parameters for HOG descriptor.
        parquet_file (str): Path to save or load the dataset in Parquet format.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Processed data and corresponding labels.
    """
    if os.path.exists(parquet_file):
        print(f"Loading dataset from {parquet_file} using Polars...")
        df = pl.read_parquet(parquet_file)
        data = np.vstack(df["features"].to_numpy())
        data_labels = df["label"].to_numpy()
        return data, data_labels

    if not os.path.exists(path):
        raise FileNotFoundError("Dataset path does not exist.")

    data, data_labels = [], []
    hog_descriptor = create_hog_descriptor(hog_params)

    for label_idx, label_name in enumerate(labels):
        label_dir = os.path.join(path, label_name)
        if not os.path.exists(label_dir):
            raise FileNotFoundError(
                f"Data not found for label '{label_name}' at {label_dir}."
            )
        for filename in os.listdir(label_dir):
            file_path = os.path.join(label_dir, filename)
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                try:
                    features = process_image(file_path, image_size, hog_descriptor)
                    data.append(features)
                    data_labels.append(label_idx)
                except ValueError:
                    print(f"Invalid image removed: {file_path}")
                    os.remove(file_path)

    # Convert to NumPy arrays
    data = np.array(data)
    data_labels = np.array(data_labels)

    # Shuffle
    p = np.random.permutation(data.shape[0])
    data, data_labels = data[p], data_labels[p]

    # Save as Parquet file using Polars
    print(f"Saving processed dataset to {parquet_file} using Polars...")
    df = pl.DataFrame({"features": data.tolist(), "label": data_labels.tolist()})
    df.write_parquet(parquet_file)

    return data, data_labels


def train_and_evaluate_model(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> float:
    """Train an SVM model and evaluate its accuracy."""
    model = SVC(kernel="rbf")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)


def random_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    image_size: Tuple[int, int],
    search_hog_parameters: bool = False,
) -> Tuple[Dict, Dict, SVC]:
    """Perform Randomized Search for HOG and SVM parameters."""
    svm_param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10],
        "gamma": [1e-3, 1e-2, 1e-1, 1],
        "kernel": ["rbf", "linear"],
    }

    best_model = None
    best_svm_params = None
    best_score = 0

    start_time = time.time()

    if search_hog_parameters:
        hog_param_grid = {
            "nbins": [6, 9, 12],
            "winSigma": [0.5, 1.0, 2.0, 5.0],
            "L2HysThreshold": [0.1, 0.2, 0.3, 0.4],
            "signedGradients": [True, False],
            "gammaCorrection": [0, 1],
        }
        best_hog_params = None
        for nbins in hog_param_grid["nbins"]:
            for winSigma in hog_param_grid["winSigma"]:
                for L2HysThreshold in hog_param_grid["L2HysThreshold"]:
                    for signedGradients in hog_param_grid["signedGradients"]:
                        for gammaCorrection in hog_param_grid["gammaCorrection"]:
                            hog_params = {
                                "winSize": (image_size[0] // 2, image_size[1] // 2),
                                "blockSize": (image_size[0] // 2, image_size[1] // 2),
                                "blockStride": (image_size[0] // 4, image_size[1] // 4),
                                "cellSize": (image_size[0] // 2, image_size[1] // 2),
                                "nbins": nbins,
                                "winSigma": winSigma,
                                "L2HysThreshold": L2HysThreshold,
                                "signedGradients": signedGradients,
                                "gammaCorrection": gammaCorrection,
                            }
                            X_train_transformed = np.array(
                                [
                                    extract_hog_features(
                                        x.reshape(image_size[0], image_size[1], -1),
                                        create_hog_descriptor(hog_params),
                                    )
                                    for x in X_train
                                ]
                            )
                            random_search = RandomizedSearchCV(
                                SVC(),
                                svm_param_grid,
                                n_iter=10,
                                scoring="accuracy",
                                cv=5,
                            )
                            random_search.fit(X_train_transformed, y_train)
                            if random_search.best_score_ > best_score:
                                best_score = random_search.best_score_
                                best_hog_params = hog_params
                                best_svm_params = random_search.best_params_
                                best_model = random_search.best_estimator_
    else:
        random_search = RandomizedSearchCV(
            SVC(),
            svm_param_grid,
            n_iter=30,
            scoring="accuracy",
            cv=3,
        )
        random_search.fit(X_train, y_train)
        if random_search.best_score_ > best_score:
            best_score = random_search.best_score_
            if search_hog_parameters:
                best_hog_params = hog_params
            best_svm_params = random_search.best_params_
            best_model = random_search.best_estimator_

    diff_time = time.time() - start_time
    if search_hog_parameters:
        print(f"Best HOG Parameters: {best_hog_params}")
    print(f"Best SVM Parameters: {best_svm_params}")
    print(f"Best Cross-Validation Accuracy: {best_score:.2f}")
    print(f"Hyperparameter Search Time: {diff_time:.2f} secs")

    return (
        best_svm_params,
        best_model,
        best_hog_params if search_hog_parameters else None,
    )


def main(args: argparse.Namespace):
    """Main function."""
    image_size = tuple(map(int, args.image_size.split(",")))
    hog_params = {
        "winSize": (image_size[0] // 2, image_size[1] // 2),
        "blockSize": (image_size[0] // 2, image_size[1] // 2),
        "blockStride": (image_size[0] // 4, image_size[1] // 4),
        "cellSize": (image_size[0] // 2, image_size[1] // 2),
        "nbins": 9,
        "derivAperture": 1,
        "winSigma": -1.0,
        "histogramNormType": 0,
        "L2HysThreshold": 0.2,
        "gammaCorrection": 1,
        "nlevels": 64,
        "signedGradients": True,
    }

    X, y = read_dataset(
        args.data_path,
        args.labels,
        image_size,
        hog_params if not args.search_hog else None,
    )

    print(
        f"Loaded dataset: {min(X.shape[0], args.limit)} images, each resized to {image_size}."
    )
    print(f"Labels: {args.labels}")
    print(f"Labels encoded: {np.unique(y)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X[: args.limit], y[: args.limit], test_size=0.2, random_state=42
    )
    print("Dataset split into training and test sets.")

    # accuracy = train_and_evaluate_model(X_train, y_train, X_test, y_test)
    # print(f"Model accuracy: {accuracy:.2f}")

    models_params, best_svc, hog_params = random_search(
        X_train, y_train, image_size, args.search_hog
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """Prepare, preprocess and play with a dataset of images.
            Default dataset used is https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset"""
        )
    )
    parser.add_argument(
        "-d", "--data_path", type=str, default="./data", help="Path to the dataset."
    )
    parser.add_argument(
        "-l",
        "--labels",
        type=str,
        nargs="+",
        default=["Cat", "Dog"],
        help="Labels for the dataset.",
    )
    parser.add_argument(
        "-s",
        "--image_size",
        type=str,
        default="128,128",
        help="Image size (width,height) to resize.",
    )
    parser.add_argument(
        "-li", "--limit", type=int, default=1000, help="Total number of images to load."
    )
    parser.add_argument(
        "-sh",
        "--search_hog",
        action="store_true",
        help="Whether to search best hyperparameters for HOG.",
    )
    args = parser.parse_args()
    main(args)
