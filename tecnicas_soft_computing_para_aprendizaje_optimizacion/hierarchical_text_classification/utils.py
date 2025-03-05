import json
import os
from datetime import datetime
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn import metrics
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    label: str = None,
    model: str = "lstm",
    figsize: tuple = (16, 10),
) -> plt.Figure:
    """Generate confusion matrix plot with Seaborn"""
    cm = metrics.confusion_matrix(y_true, y_pred)
    classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
    fig = plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"img/confusion_matrix_{label}_{model}.png")
    return fig


def evaluate(
    clf: BaseEstimator, X_test, y_test, label: Optional[str] = None, **save_kwargs
):
    if isinstance(clf, torch.nn.Module):
        device = next(clf.parameters()).device
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

        with torch.no_grad():
            if label:
                pred = clf(X_test_tensor, label)
            else:
                pred = clf(X_test_tensor)

        pred = pred.argmax(dim=1).cpu().numpy()

    else:
        if label:
            pred = clf.predict(X_test, label)
        else:
            pred = clf.predict(X_test)

    # Calculate evaluation metrics
    acc = metrics.accuracy_score(y_test, pred)
    f1 = metrics.f1_score(y_test, pred, average="weighted", zero_division=0)
    rec = metrics.recall_score(y_test, pred, average="weighted", zero_division=0)
    prec = metrics.precision_score(y_test, pred, average="weighted", zero_division=0)

    # Print the evaluation metrics
    print(f"Accuracy: {100 * acc:.1f}%")
    print(f"Precision: {100 * prec:.1f}%")
    print(f"Recall: {100 * rec:.1f}%")
    print(f"F1-score: {100 * f1:.1f}%")

    store_results(
        **save_kwargs,
        metrics={"accuracy": acc, "f1": f1, "recall": rec, "precision": prec},
    )

    return pred


def create_pipeline_nb() -> Pipeline:
    """Create base ML pipeline"""
    return Pipeline(
        [
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", MultinomialNB()),
        ]
    )


def store_results(
    name: str,
    metrics: Dict[str, float],
    category: Optional[str] = None,
    include_timestamp: bool = True,
) -> str:
    os.makedirs("results", exist_ok=True)

    # Format filename
    timestamp = (
        f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}" if include_timestamp else ""
    )
    category_str = f"_{category}" if category else ""
    base_filename = f"results_{name}{category_str}{timestamp}"

    # Save metrics
    metrics_path = os.path.join("results", f"{base_filename}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Results saved to {os.path.join('results', base_filename)}*")
    return os.path.join("results", base_filename)
