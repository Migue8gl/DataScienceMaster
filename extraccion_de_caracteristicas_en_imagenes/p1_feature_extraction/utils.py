import csv
import json
import os
from datetime import datetime
from typing import Dict, Optional, Union

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

from LBPDescriptor import LBPDescriptor


def create_descriptor(
    params: Optional[Dict] = None,
    descriptor_type: str = "hog",
    image_size: int = None,
) -> Union[LBPDescriptor, cv2.HOGDescriptor, None]:
    """
    Create and return a LBP or HOG descriptor with specified parameters.

    Args:
        params (Optional[Dict], optional): Dictionary of parameters to configure the descriptor. If None, default parameters will be used. Defaults to None.
        descriptor_type (str, optional): Type of descriptor to create. Can be "hog" for Histogram of Oriented Gradients or "lbp" for Local Binary Pattern. Defaults to "hog".
        image_size (int, optional): The size of the image, used to determine descriptor window and cell sizes. Required if descriptor_type is "hog". Defaults to None.
    Returns:
        Union[LBPDescriptor, cv2.HOGDescriptor, None]:
            LBPDescriptor: A Local Binary Pattern (LBP) descriptor, if descriptor_type is "lbp".
            cv2.HOGDescriptor: A Histogram of Oriented Gradients (HOG) descriptor, if descriptor_type is "hog".
            None: If no descriptor is created due to missing parameters.
    """
    if params is None:
        return None
    if descriptor_type == "hog":
        return cv2.HOGDescriptor(
            _winSize=params.get("winSize", (image_size // 2, image_size // 2)),
            _blockSize=params.get("blockSize", (image_size // 2, image_size // 2)),
            _blockStride=params.get("blockStride", (image_size // 4, image_size // 4)),
            _cellSize=params.get("cellSize", (image_size // 4, image_size // 4)),
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


def save_search_results(
    svm_params: Dict,
    best_score: float,
    execution_time: float,
    descriptor_name: Optional[str] = None,
    descriptor_params: Optional[Dict] = None,
    output_dir: str = "results",
):
    """
    Save the hyperparameter search results in CSV format.

    Args:
        svm_params (Dict): Best parameters found for the SVM model
        best_score (float): Best cross-validation accuracy score
        execution_time (float): Time taken for hyperparameter search in seconds
        descriptor_name (Optional[str]): Name of the descriptor used (if any)
        descriptor_params (Optional[Dict]): Best parameters found for the descriptor (if any)
        output_dir (str): Directory to save the results file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)

    # Prepare CSV
    csv_filename = os.path.join(output_dir, "search_results.csv")
    file_exists = os.path.isfile(csv_filename)

    with open(csv_filename, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            header = ["timestamp", "best_score", "execution_time_seconds"]
            svm_param_headers = [f"svm_{k}" for k in svm_params.keys()]
            header.extend(svm_param_headers)

            if descriptor_name and descriptor_params:
                header.append("descriptor_name")
                descriptor_param_headers = [
                    f"descriptor_{k}" for k in descriptor_params.keys()
                ]
                header.extend(descriptor_param_headers)

            writer.writerow(header)

        # Write values
        row = [timestamp, best_score, execution_time]
        row.extend(svm_params.values())

        if descriptor_name and descriptor_params:
            row.append(descriptor_name)
            row.extend(descriptor_params.values())

        writer.writerow(row)

    print(f"Results appended to: {csv_filename}")


def save_model(
    model: BaseEstimator,
    model_score: float,
    svm_params: Dict,
    output_dir: str = "models",
    descriptor_name: Optional[str] = None,
    descriptor_params: Optional[Dict] = None,
):
    """
    Save the trained model along with its metadata.

    Args:
        model: Trained sklearn model to save
        model_score (float): Model's best score
        svm_params (Dict): SVM parameters used
        output_dir (str): Directory to save model and metadata
        descriptor_name (Optional[str]): Name of the descriptor used (if any)
        descriptor_params (Optional[Dict]): Parameters of the descriptor (if any)
    """
    # Create timestamp for unique model directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"svm_model_{timestamp}"

    # Create model directory
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    try:
        # Save the model
        model_path = os.path.join(model_dir, "model.joblib")
        joblib.dump(model, model_path)

        # Prepare metadata
        metadata = {
            "timestamp": timestamp,
            "model_type": "SVM",
            "best_score": model_score,
            "svm_parameters": svm_params,
        }

        if descriptor_name and descriptor_params:
            metadata["descriptor"] = {
                "name": descriptor_name,
                "parameters": descriptor_params,
            }

        # Save metadata
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"Model and metadata saved successfully in: {model_dir}")
        print(f"Model file: {model_path}")
        print(f"Metadata file: {metadata_path}")

    except Exception as e:
        print(f"Error saving model: {str(e)}")
        raise


def load_model(model_dir: str) -> tuple[BaseEstimator, Dict]:
    """
    Load a saved model and its metadata.

    Args:
        model_dir (str): Directory containing the saved model and metadata

    Returns:
        tuple: (loaded_model, metadata)
    """
    try:
        # Load model
        model_path = os.path.join(model_dir, "model.joblib")
        model = joblib.load(model_path)

        # Load metadata
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        return model, metadata

    except Exception as e:
        print(f"Error loading model from {model_dir}: {str(e)}")
        raise


def save_evaluation_plots(
    y_true, y_pred, y_pred_proba, model_name, results_dir="results"
):
    """
    Save evaluation plots using Seaborn styling and advanced visualization techniques.

    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        y_pred_proba (np.ndarray): Prediction probabilities
        model_name (str): Name of the model (used for saving files)
        results_dir (str): Directory to save results
    """
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    # Set Seaborn style
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    # 1. ROC Curve with confidence intervals
    plt.figure(figsize=(10, 6))
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        color=sns.color_palette()[0],
        lw=2,
        label=f"ROC curve (AUC = {roc_auc:.2f})",
    )
    plt.plot(
        [0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random classifier"
    )
    plt.fill_between(fpr, tpr, alpha=0.2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve with Confidence Region", fontsize=14, pad=20)
    plt.legend(loc="lower right")
    sns.despine()
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, f"{model_name}_roc_curve.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 2. Enhanced Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(
        cm_normalized,
        annot=cm,  # Show raw counts
        fmt="d",
        cmap="crest",
        square=True,
        cbar_kws={"label": "Normalized Frequency"},
        annot_kws={"size": 12},
    )

    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title(
        "Confusion Matrix\n(numbers show raw counts, colors show percentages)",
        fontsize=14,
        pad=20,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, f"{model_name}_confusion_matrix.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 3. Precision-Recall Curve
    plt.figure(figsize=(10, 6))
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)

    plt.plot(
        recall,
        precision,
        color=sns.color_palette()[2],
        lw=2,
        label=f"PR curve (AUC = {pr_auc:.2f})",
    )
    plt.axhline(
        y=sum(y_true) / len(y_true),
        color="gray",
        linestyle="--",
        label="Random classifier",
    )
    plt.fill_between(recall, precision, alpha=0.2)
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curve", fontsize=14, pad=20)
    plt.legend(loc="lower left")
    sns.despine()
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, f"{model_name}_precision_recall.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # 4. Distribution of Prediction Probabilities
    plt.figure(figsize=(12, 6))
    sns.histplot(
        data=pl.DataFrame({"Probability": y_pred_proba, "True Class": y_true}),
        x="Probability",
        hue="True Class",
        bins=50,
        multiple="layer",
        alpha=0.6,
    )
    plt.xlabel("Prediction Probability", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title("Distribution of Prediction Probabilities by Class", fontsize=14, pad=20)
    sns.despine()
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_dir, f"{model_name}_prob_distribution.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Save metrics to text file with enhanced formatting
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "ROC AUC": roc_auc,
        "PR AUC": pr_auc,
    }

    with open(os.path.join(results_dir, f"{model_name}_metrics.txt"), "w") as f:
        f.write("═══════════════════════════════\n")
        f.write("     Classification Metrics     \n")
        f.write("═══════════════════════════════\n\n")
        for metric_name, value in metrics.items():
            f.write(f"{metric_name:.<20} {value:.4f}\n")
        f.write("\n═══════════════════════════════\n")
        f.write("    Classification Report    \n")
        f.write("═══════════════════════════════\n\n")
        f.write(classification_report(y_true, y_pred))
