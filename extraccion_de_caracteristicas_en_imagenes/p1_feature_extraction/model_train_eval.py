import argparse
import textwrap
import time
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import SVC

from process_data import read_dataset
from utils import (
    create_descriptor,
    save_evaluation_plots,
    save_model,
    save_search_results,
)


def random_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    descriptor_name: str,
    descriptor_hyperparameter_search: bool = False,
) -> Tuple[Dict, SVC, Dict]:
    """
    Perform a Randomized Search for optimizing parameters of an image descriptor
    (e.g., HOG or LBP) and an SVM classifier. Assumes the feature extraction process is compatible with the provided `image_size`.

    Args:
        X_train (np.ndarray): Training data, where each row represents an image and each column represents a feature.
        y_train (np.ndarray): Training labels corresponding to the data in `X_train`.
        descriptor_name (str): Name of the image descriptor to use. Supported options may include "HOG" or "LBP".
        descriptor_hyperparameter_search (bool, optional): If True, performs hyperparameter search for the chosen image descriptor
                                                        in addition to optimizing SVM parameters. Defaults to False.
    Returns:
        Tuple[Dict, Dict, SVC]:
            best_descriptor_params (Dict): Best hyperparameters found for the image descriptor, if applicable.
            best_svm_params (Dict): Best hyperparameters found for the SVM classifier.
            best_svm_model (SVC): The SVM model trained with the best parameters.
    """
    svm_param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10],
        "gamma": [1e-3, 1e-2, 1e-1, 1],
        "kernel": ["rbf", "linear"],
    }

    best_model = None
    best_svm_params = None
    best_score = 0

    image_size = X_train.shape[1]

    start_time = time.time()

    if descriptor_hyperparameter_search and descriptor_name == "hog":
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
                            hog_params = {
                                "winSize": (image_size, image_size),
                                "blockSize": (image_size // 2, image_size // 2),
                                "blockStride": (image_size // 4, image_size // 4),
                                "cellSize": (image_size // 4, image_size // 4),
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
                                    descriptor.compute(x.astype(np.uint8)).flatten()
                                    for x in X_train
                                ]
                            )
                            random_search = RandomizedSearchCV(
                                SVC(probability=True),
                                svm_param_grid,
                                n_iter=20,
                                scoring="accuracy",
                                cv=3,
                            )
                            random_search.fit(X_train_transformed, y_train)
                            if random_search.best_score_ > best_score:
                                best_score = random_search.best_score_
                                best_descriptor_params = hog_params
                                best_svm_params = random_search.best_params_
                                best_model = random_search.best_estimator_
    elif descriptor_hyperparameter_search and descriptor_name == "lbp":
        lbp_param_grid = {"radius": [1, 2, 3, 4], "n_neighbors": [8, 16, 32, 64]}
        best_descriptor_params = None
        for radius in lbp_param_grid["radius"]:
            for n_neighbors in lbp_param_grid["n_neighbors"]:
                lbp_params = {
                    "radius": radius,
                    "n_neighbors": n_neighbors,
                }
                descriptor = create_descriptor(lbp_params, "lbp", image_size)
                X_train_transformed = np.array(
                    [descriptor.compute(x.astype(np.uint8)).flatten() for x in X_train]
                )
                random_search = RandomizedSearchCV(
                    SVC(probability=True),
                    svm_param_grid,
                    n_iter=20,
                    scoring="accuracy",
                    cv=3,
                )
                random_search.fit(X_train_transformed, y_train)
                if random_search.best_score_ > best_score:
                    best_score = random_search.best_score_
                    best_descriptor_params = lbp_params
                    best_svm_params = random_search.best_params_
                    best_model = random_search.best_estimator_
    else:
        if len(X_train.shape) > 2 and descriptor_name is None:
            raise Exception(
                "Parquet file contains raw images. You should specify the descriptor type to use."
            )
        elif len(X_train.shape) > 2:
            descriptor = create_descriptor({}, descriptor_name, image_size)
            X_train = np.array(
                [descriptor.compute(x.astype(np.uint8)) for x in X_train]
            )
        random_search = RandomizedSearchCV(
            SVC(probability=True),
            svm_param_grid,
            n_iter=20,
            scoring="accuracy",
            cv=3,
        )
        random_search.fit(X_train, y_train)
        if random_search.best_score_ > best_score:
            best_score = random_search.best_score_
            best_svm_params = random_search.best_params_
            best_model = random_search.best_estimator_

    diff_time = time.time() - start_time
    if descriptor_hyperparameter_search:
        print(
            f"Best {"HOG" if descriptor_name == "hog" else "LBP"} Parameters: {best_descriptor_params}"
        )
    print(f"Best SVM Parameters: {best_svm_params}")
    print(f"Best Cross-Validation Accuracy: {best_score:.2f}")
    print(f"Hyperparameter Search Time: {diff_time:.2f} secs")

    save_search_results(
        svm_params=best_svm_params,
        best_score=best_score,
        execution_time=diff_time,
        descriptor_name=descriptor_name if descriptor_hyperparameter_search else None,
        descriptor_params=best_descriptor_params
        if descriptor_hyperparameter_search
        else None,
    )
    save_model(
        model=best_model,
        model_score=best_score,
        svm_params=best_svm_params,
        descriptor_name=descriptor_name if descriptor_hyperparameter_search else None,
        descriptor_params=best_descriptor_params
        if descriptor_hyperparameter_search
        else None,
    )

    return (
        best_svm_params,
        best_model,
        best_descriptor_params if descriptor_hyperparameter_search else None,
    )


def main(args: argparse.Namespace):
    """Main functionality"""
    # Read data from parquet
    X, y = read_dataset(args.data_file)

    # Extract the descriptor from dataset file name
    descriptor_name = args.data_file.split("_")[1]

    print(f"Loaded dataset: {min(X.shape[0], args.limit)} with images.")

    X_train, X_test, y_train, y_test = train_test_split(
        X[: args.limit], y[: args.limit], test_size=0.2
    )
    print("Dataset split into training and test sets.")

    # Optional hyperparameter search for descriptors
    if args.hyperparameter_search and args.descriptor is not None:
        models_params, best_svc, descriptor_params = random_search(
            X_train,
            y_train,
            args.descriptor,
            args.hyperparameter_search,
        )
    else:
        models_params, best_svc, descriptor_params = random_search(
            X_train,
            y_train,
            descriptor_name,
        )

    # Get predictions on test set
    if (len(X_test.shape) > 2 and descriptor_name != "raw") or (
        len(X_test.shape) > 2 and args.descriptor is not None
    ):
        # If we have raw images, we need to transform them using the descriptor
        descriptor = create_descriptor(
            descriptor_params if descriptor_params else {},
            descriptor_name if args.descriptor is None else args.descriptor,
            X_test.shape[1],
        )
        X_test_transformed = np.array(
            [descriptor.compute(x.astype(np.uint8)).flatten() for x in X_test]
        )
        y_pred = best_svc.predict(X_test_transformed)
        y_pred_proba = best_svc.predict_proba(X_test_transformed)[:, 1]
    else:
        # If we already have features, use them directly
        y_pred = best_svc.predict(X_test)
        y_pred_proba = best_svc.predict_proba(X_test)[:, 1]

    # Generate model name based on parameters
    model_name = f"svm_{descriptor_name}_{'with_desc_search' if args.hyperparameter_search else 'base'}"
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Test acurracy: {test_acc:.2f}")

    # Save evaluation plots and metrics
    save_evaluation_plots(y_test, y_pred, y_pred_proba, model_name)

    print("\nEvaluation complete. Results saved in 'results' directory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """Prepare, preprocess and play with a dataset of images.
            Default dataset used is https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset"""
        )
    )
    parser.add_argument(
        "-d",
        "--data-file",
        type=str,
        default="",
        help="Dataset file path in parquet format.",
    )
    parser.add_argument(
        "-l", "--limit", type=int, default=1000, help="Total number of images to load."
    )
    parser.add_argument(
        "-hs",
        "--hyperparameter-search",
        action="store_true",
        help="Perform hyperparameter search for descriptor (lbp or hog).",
    )
    parser.add_argument(
        "-desc",
        "--descriptor",
        type=str,
        choices=["lbp", "hog"],
        default=None,
        help="When doing hyperparameter search for a descriptor it should be specified.",
    )

    args = parser.parse_args()
    main(args)
