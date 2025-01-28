import os
import sys
from datetime import datetime
from typing import List

import joblib
import matplotlib.pyplot as plt
from scipy.stats import randint
import polars as pl
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier, plot_tree
from utils import (
    build_structure,
    evaluate,
    perform_hyperparameter_search,
    save_feature_importance_plot,
    calculate_feature_importance,
)


def save_tree_visualization(
    model: BaseEstimator, feature_names: List[str], filename: str
) -> None:
    """
    Save a visualization of the trained Decision Tree.

    Args:
        pipeline (Pipeline): Trained pipeline.
        feature_names (pl.Index): Names of the features used in training.
        filename (str): Path to save the plot.
    """
    plt.figure(figsize=(20, 10))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=["Normal", "Anomaly"],
        filled=True,
        rounded=True,
    )
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def main():
    random_state = 42
    metric = "precision"

    train_df = pl.read_csv("data/train.csv")
    test_df = pl.read_csv("data/test.csv")

    # Prepare training data
    X_train = train_df.drop("is_anomaly")
    y_train = train_df["is_anomaly"]

    # Build and cross-validate the model
    model = DecisionTreeClassifier(class_weight="balanced", random_state=random_state)

    # Perform hyperparameter tuning
    param_grid = {
        "max_depth": randint(2, 20),
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 10),
        "criterion": ["gini", "entropy"],
    }
    best_score, best_params = perform_hyperparameter_search(
        model, X_train, y_train, metric, param_grid
    )
    print(f"Best {metric} score: {best_score:.4f}")
    print(f"Best parameters: {best_params}")

    # Fit the model with the best parameters
    model.set_params(**best_params)
    model.fit(X_train.drop("fold"), y_train)

    # Feature importance
    feature_importance = calculate_feature_importance(model, X_train)
    save_feature_importance_plot(feature_importance, "img/feature_importance.png")

    # Tree visualization
    save_tree_visualization(model, X_train.columns, "img/tree_visualization.svg")

    # Train evaluation
    train_metrics = evaluate(train_df.drop("fold"), model)

    # Test data evaluation
    test_metrics = evaluate(test_df, model, prefix="test_")

    # Prepare metadata
    model_filename = f"decision_tree_{datetime.now().strftime('%Y-%m-%d-%H:%M')}"
    metadata = {
        "best_score": best_score,
        "best_params": best_params,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "model_filename": model_filename,
    }

    model_with_metadata = {"model": model, "metadata": metadata}
    # Save the model
    joblib.dump(model_with_metadata, os.path.join("models", model_filename))
    print(f"Model and metadata saved as {model_filename}")


if __name__ == "__main__":
    paths = [
        "./MineriaMetroPT-3/DecisionTree&XGBoost/src",
        "./DecisionTree&XGBoost/src",
        "./src",
    ]

    file_path = ""
    for path in paths:
        if os.path.exists(path):
            file_path = os.path.dirname(path)
            break
    if file_path:
        sys.path.append(file_path)
        os.chdir(file_path)
    build_structure()
    main()
