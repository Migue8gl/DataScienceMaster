import argparse
import os
import textwrap
from typing import List, Tuple, Dict, Optional, Union

import cv2
import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
import time
from LBPDescriptor import LBPDescriptor


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
    search_descriptor: Optional[str] = None,
    descriptor: Optional[str] = None,
) -> Tuple[Dict, Dict, SVC]:
    """Perform Randomized Search for HOG/LBP and SVM parameters."""
    svm_param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10],
        "gamma": [1e-3, 1e-2, 1e-1, 1],
        "kernel": ["rbf", "linear"],
    }

    best_model = None
    best_svm_params = None
    best_score = 0

    start_time = time.time()

    if search_descriptor == "hog":
        hog_param_grid = {
            "nbins": [6, 9, 12],
            "winSigma": [0.5, 1.0, 2.0, 5.0],
            "L2HysThreshold": [0.1, 0.2, 0.3, 0.4],
            "signedGradients": [True, False],
            "gammaCorrection": [0, 1],
        }
        best_descriptor_params = None
        for nbins in hog_param_grid["nbins"]:
            for winSigma in hog_param_grid["winSigma"]:
                for L2HysThreshold in hog_param_grid["L2HysThreshold"]:
                    for signedGradients in hog_param_grid["signedGradients"]:
                        for gammaCorrection in hog_param_grid["gammaCorrection"]:
                            X = X_train.copy()
                            hog_params = {
                                "winSize": (image_size[0], image_size[1]),
                                "blockSize": (image_size[0] // 2, image_size[1] // 2),
                                "blockStride": (image_size[0] // 4, image_size[1] // 4),
                                "cellSize": (image_size[0] // 4, image_size[1] // 4),
                                "nbins": nbins,
                                "winSigma": winSigma,
                                "L2HysThreshold": L2HysThreshold,
                                "signedGradients": signedGradients,
                                "gammaCorrection": gammaCorrection,
                            }
                            descriptor = create_descriptor(
                                hog_params, "hog", image_size
                            )
                            X_train_transformed = np.array(
                                [
                                    descriptor.compute(
                                        x.reshape(image_size[0], image_size[1], -1)
                                    ).flatten()
                                    for x in X
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
                                best_descriptor_params = hog_params
                                best_svm_params = random_search.best_params_
                                best_model = random_search.best_estimator_
    elif search_descriptor == "lbp":
        lbp_param_grid = {"radius": [1, 2, 3, 4], "n_neighbors": [8, 16, 32, 64]}
        best_descriptor_params = None
        for radius in lbp_param_grid["radius"]:
            for n_neighbors in lbp_param_grid["n_neighbors"]:
                X = X_train.copy()
                lbp_params = {
                    "radius": radius,
                    "n_neighbors": n_neighbors,
                }
                descriptor = create_descriptor(lbp_params, "lbp", image_size)
                X_train_transformed = np.array(
                    [
                        descriptor.compute(
                            x.reshape(image_size[0], image_size[1], -1)
                        ).flatten()
                        for x in X
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
                    best_descriptor_params = lbp_params
                    best_svm_params = random_search.best_params_
                    best_model = random_search.best_estimator_
    else:
        if len(X_train.shape) > 2 and search_descriptor is None:
            raise Exception(
                "Parquet file contains raw images. You should specify the descriptor type to use."
            )
        elif len(X_train.shape) > 2:
            descriptor = create_descriptor({}, search_descriptor, image_size)
            X_train = np.array([descriptor.compute(x) for x in X_train])
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
            best_svm_params = random_search.best_params_
            best_model = random_search.best_estimator_

    diff_time = time.time() - start_time
    if search_descriptor is not None:
        print(
            f"Best {"HOG" if search_descriptor == "hog" else "LBP"} Parameters: {best_descriptor_params}"
        )
    print(f"Best SVM Parameters: {best_svm_params}")
    print(f"Best Cross-Validation Accuracy: {best_score:.2f}")
    print(f"Hyperparameter Search Time: {diff_time:.2f} secs")

    return (
        best_svm_params,
        best_model,
        best_descriptor_params if search_descriptor is not None else None,
    )


def test(args: argparse.Namespace):
    import matplotlib.pyplot as plt
    import seaborn as sns

    image_size = tuple(map(int, args.image_size.split(",")))
    X, y = read_dataset(
        args.data_path,
        args.labels,
        image_size,
        None,
        "dataset_raw.parquet",
    )

    img = X[67]

    start = time.time()
    lbp = LBPDescriptor(radius=2, n_neighbors=16)
    hist = lbp.compute(img)
    print(hist.shape)
    print(f"Time for one image LBP computation: {time.time() - start:.2f} secs")

    # Visualization
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))

    # Original image
    axs[0].imshow(img, cmap="gray")
    axs[0].set_title("Original Image", fontsize=14)
    axs[0].axis("off")

    axs[1].imshow(hist, cmap="gray")
    axs[1].set_title("LBP Image", fontsize=14)
    axs[1].axis("off")

    plt.tight_layout()
    plt.savefig("lbp_visualization.png")


def create_descriptor(
    params: Optional[Dict] = None,
    descriptor_type: str = "hog",
    image_size: Optional[Tuple[int, int]] = None,
) -> Union[LBPDescriptor, cv2.HOGDescriptor, None]:
    """Create and return a LBP or HOG descriptor with specified parameters."""
    if params is None:
        return None
    if descriptor_type == "hog":
        return cv2.HOGDescriptor(
            _winSize=params.get("winSize", (image_size[0] // 2, image_size[1] // 2)),
            _blockSize=params.get(
                "blockSize", (image_size[0] // 2, image_size[1] // 2)
            ),
            _blockStride=params.get(
                "blockStride", (image_size[0] // 4, image_size[1] // 4)
            ),
            _cellSize=params.get("cellSize", (image_size[0] // 4, image_size[1] // 4)),
            _nbins=params.get("nbins", 9),
            _derivAperture=params.get("derivAperture", 1),
            _winSigma=params.get("winSigma", -1.0),
            _histogramNormType=params.get("histogramNormType", 0),
            _L2HysThreshold=params.get("L2HysThreshold", 0.2),
            _gammaCorrection=params.get("gammaCorrection", 1),
            _nlevels=params.get("nlevels", 64),
            _signedGradient=params.get("signedGradients", True),
        )
    elif descriptor_type == "lbp":
        return LBPDescriptor(
            radius=params.get("radius", 2),
            n_neighbors=params.get("n_neighbors", 8),
        )
    else:
        raise ValueError(f"No descriptor defined with {descriptor_type} name.")


def process_image(
    file_path: str,
    image_size: Tuple[int, int],
    descriptor: Optional[Union[cv2.HOGDescriptor, LBPDescriptor]] = None,
    raw_data: bool = False,
) -> np.ndarray:
    """
    Load and preprocess an image, returning its features or raw data.

    Args:
        file_path (str): Path to the image file
        image_size (Tuple[int, int]): Target image size
        descriptor (Optional): Feature descriptor to use
        raw_data (bool): If True, return raw pixel data instead of features

    Returns:
        np.ndarray: Processed image data
    """
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # One channel
    img = img.astype(np.uint8)
    if img is None or img.size == 0:
        raise ValueError(f"Invalid image at {file_path}")
    img = cv2.resize(img, image_size)

    if raw_data:
        return img

    result = img
    if descriptor is not None:
        result = descriptor.compute(img).flatten()
    return result


def read_dataset(
    path: str,
    labels: List[str],
    image_size: Tuple[int, int],
    descriptor_params: Optional[Dict] = None,
    parquet_file: Optional[str] = None,
    raw_data: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads images from a dataset path, extracts descriptor features or returns raw data.

    Args:
        path (str): Path to the dataset directory.
        labels (List[str]): List of label subdirectories.
        image_size (Tuple[int, int]): Target image size for processing.
        descriptor_params (Optional[Dict]): Parameters for descriptor.
        parquet_file (Optional[str]): Path to save or load the dataset in Parquet format.
        raw_data (bool): If True, return raw pixel data instead of features.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Processed data and corresponding labels.
    """
    # If Parquet file exists and not using raw data, load from file
    if parquet_file and os.path.exists(parquet_file) and not raw_data:
        print(f"Loading dataset from {parquet_file} using Polars.")
        df = pl.read_parquet(parquet_file)
        data = np.array(df["features"].to_list())
        original_shape = df["shape"][0]  # Retrieve the original shape
        data = data.reshape(original_shape)
        data_labels = df["label"].to_numpy()
        return data, data_labels

    if not os.path.exists(path):
        raise FileNotFoundError("Dataset path does not exist.")

    data, data_labels = [], []
    descriptor = None

    # Create descriptor if not raw data and descriptor params provided
    if not raw_data and descriptor_params:
        descriptor_type = descriptor_params.get("type", "hog")
        descriptor_config = descriptor_params.get("config", {})
        descriptor = create_descriptor(descriptor_config, descriptor_type, image_size)

    for label_idx, label_name in enumerate(labels):
        label_dir = os.path.join(path, label_name)
        if not os.path.exists(label_dir):
            raise FileNotFoundError(
                f"Data not found for label '{label_name}' at {label_dir}."
            )
        for filename in os.listdir(label_dir):
            file_path = os.path.join(label_dir, filename)
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                features = process_image(
                    file_path, image_size, descriptor, raw_data=raw_data
                )
                data.append(features)
                data_labels.append(label_idx)

    # Convert to NumPy arrays
    data = np.array(data)
    data_labels = np.array(data_labels)

    # Shuffle
    p = np.random.permutation(data.shape[0])
    data, data_labels = data[p], data_labels[p]

    # Save as Parquet file using Polars if not raw data
    if parquet_file:
        print(f"Saving processed dataset to {parquet_file} using Polars.")
        df = pl.DataFrame(
            {
                "features": data.tolist(),
                "label": data_labels.tolist(),
                "shape": [tuple(data.shape)] * len(data),
            }
        )
        df.write_parquet(parquet_file)

    return data, data_labels


def main(args: argparse.Namespace):
    """Main function with enhanced data reading capabilities."""
    image_size = tuple(map(int, args.image_size.split(",")))

    # Descriptor configuration
    descriptor_params = None
    if args.descriptor:
        descriptor_params = {"type": args.descriptor.lower(), "config": {}}

        if args.descriptor.lower() == "hog":
            descriptor_params["config"] = {
                "winSize": (image_size[0] // 2, image_size[1] // 2),
                "blockSize": (image_size[0] // 2, image_size[1] // 2),
                "blockStride": (image_size[0] // 4, image_size[1] // 4),
                "cellSize": (image_size[0] // 4, image_size[1] // 4),
                "nbins": 9,
                "derivAperture": 1,
                "winSigma": -1.0,
                "histogramNormType": 0,
                "L2HysThreshold": 0.2,
                "gammaCorrection": 1,
                "nlevels": 64,
                "signedGradients": True,
            }
        elif args.descriptor.lower() == "lbp":
            descriptor_params["config"] = {"radius": 1, "n_neighbors": 8}

    # Determine if we're using raw data or descriptor-transformed data
    raw_data = args.raw_data if args.raw_data is not None else False
    
    # Parquet file naming strategy
    parquet_file = (
        "dataset_raw.parquet"
        if raw_data
        else f"dataset_{args.descriptor.lower() if args.descriptor else 'default'}.parquet"
    )
    
    # Read dataset
    X, y = read_dataset(
        args.data_path,
        args.labels,
        image_size,
        descriptor_params,
        parquet_file,
        raw_data,
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

    # Optional hyperparameter search for descriptors
    if args.descriptor_search and args.descriptor:
        models_params, best_svc, descriptor_params = random_search(
            X_train, y_train, image_size, args.descriptor.lower()
        )
    else:
        models_params, best_svc, descriptor_params = random_search(
            X_train,
            y_train,
            image_size,
            None,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """Prepare, preprocess and play with a dataset of images.
            Default dataset used is https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset"""
        )
    )
    parser.add_argument(
        "-d", "--data-path", type=str, default="./data", help="Path to the dataset."
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
        "--image-size",
        type=str,
        default="28,28",
        help="Image size (width,height) to resize.",
    )
    parser.add_argument(
        "-li", "--limit", type=int, default=1000, help="Total number of images to load."
    )
    parser.add_argument(
        "-desc",
        "--descriptor",
        choices=["HOG", "LBP", None],
        default=None,
        help="Use HOG, LBP or none for feature extraction.",
    )
    parser.add_argument(
        "-raw",
        "--raw-data",
        action="store_true",
        help="Use raw pixel data instead of descriptor features.",
    )
    parser.add_argument(
        "-ds",
        "--descriptor-search",
        action="store_true",
        help="Perform hyperparameter search for descriptor.",
    )

    args = parser.parse_args()
    main(args)
