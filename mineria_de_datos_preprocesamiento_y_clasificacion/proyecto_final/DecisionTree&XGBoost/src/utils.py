import os
import shutil
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier


def get_cv_iterable(
    folds: list,
    fold_column: str,
    train: pl.DataFrame,
):
    """
    Generates cross-validation (CV) train-test splits for each fold based on a specified fold column in the dataframe.

    Args:
        folds (list): A list of fold identifiers. Each fold corresponds to a subset of the dataset to be used as the test set in each iteration.
        fold_column (str): The name of the column in the `train` DataFrame that contains the fold identifiers.
        train (pl.DataFrame): The training dataset containing the data and fold identifiers.

    Yields:
        tuple: A tuple of two elements:
            - train_indexes (Index): The indices of the training set for the current fold.
            - test_indexes (Index): The indices of the test set for the current fold.
    """

    train_pd = train.to_pandas()
    for fold in folds:
        test_indexes = train_pd[train_pd[fold_column] == fold].index.to_list()
        train_indexes = train_pd[train_pd[fold_column] != fold].index.to_list()
        yield (train_indexes, test_indexes)


def plot_roc_curve(y_true: pl.Series, y_prob: pl.DataFrame, filename: str) -> None:
    """
    Plot the ROC curve and save it to a file.

    Args:
        y_true (pl.Series): True labels.
        y_prob (pl.DataFrame): Predicted probabilities.
        filename (str): Path to save the plot.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_confusion_matrix(y_true: pl.Series, y_pred: pl.Series, filename: str) -> None:
    """
    Plot the confusion matrix using Seaborn's heatmap and save it to a file.

    Args:
        y_true (pl.Series): True labels.
        y_pred (pl.Series): Predicted labels.
        filename (str): Path to save the plot.
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def build_structure():
    """
    Creates the necessary project folder structure ('data', 'img', 'doc', 'src') if they do not exist.
    Moves files from the current directory into the appropriate folders based on their extensions:
    - .csv files to 'data'
    - .png and .jpg files to 'img'
    - .pdf files to 'doc'
    - .ipynb and .py files to 'src'
    """
    folders = ["data", "img", "doc", "src", "models"]

    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created folder: {folder}")
        else:
            print(f"Folder already exists: {folder}")

    current_dir = os.getcwd()

    for filename in os.listdir(current_dir):
        if os.path.isdir(filename):
            continue

        file_path = os.path.join(current_dir, filename)

        if filename.endswith(".csv"):
            shutil.move(file_path, os.path.join(current_dir, "data", filename))
            print(f"Moved {filename} to data/")

        elif filename.endswith(".png") or filename.endswith(".jpg"):
            shutil.move(file_path, os.path.join(current_dir, "img", filename))
            print(f"Moved {filename} to img/")

        elif filename.endswith(".pdf"):
            shutil.move(file_path, os.path.join(current_dir, "doc", filename))
            print(f"Moved {filename} to doc/")

        elif filename.endswith(".ipynb") or filename.endswith(".py"):
            shutil.move(file_path, os.path.join(current_dir, "src", filename))
            print(f"Moved {filename} to src/")


def evaluate_model(
    y_true: pl.Series, y_pred: pl.Series, type_model: str, prefix: str
) -> Dict[str, float]:
    """
    Evaluate the model performance using various metrics and return the classification report.

    Args:
        y_true (pl.Series): True labels.
        y_pred (pl.Series): Predicted labels.
        type_model (str)): Type of model.

    Returns:
        Dict[str, float]: Dictionary containing evaluation metrics and classification report.
    """
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
    }

    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    report_df.to_csv(f"data/{prefix}{type_model}_classification_report.csv")

    return metrics


def perform_cross_validation(
    model: Union[BaseEstimator, XGBClassifier],
    X: pl.DataFrame,
    y: pl.Series,
    metric: str,
) -> None:
    """
    Perform cross-validation and compute metrics for each fold.

    Args:
        model (BaseEstimator): The model to evaluate.
        X (pl.DataFrame): Features for training, including a "fold" column.
        y (pl.Series): Target variable.
        metric (str): Score metric to use.
    """
    folds = X["fold"].unique()
    fold_metrics: pl.List[Dict[str, Union[int, float]]] = []
    X = X.with_row_index("index")

    for fold, (train_indexes, test_indexes) in enumerate(
        get_cv_iterable(folds, "fold", X)
    ):
        X_train, X_test = (
            X.filter(pl.col("index").is_in(train_indexes)),
            X.filter(pl.col("index").is_in(test_indexes)),
        )
        y_train, y_test = (
            y[train_indexes],
            y[test_indexes],
        )

        # Train the model on the current training fold
        model.fit(X_train.drop("fold", "index"), y_train)

        # Predict and evaluate on the validation fold
        predictions = model.predict(X_test.drop("fold", "index"))

        if metric == "accuracy":
            score = accuracy_score(y_test, predictions)
        elif metric == "precision":
            score = precision_score(y_test, predictions)
        elif metric == "recall":
            score = recall_score(y_test, predictions)
        else:
            score = f1_score(y_test, predictions)

        fold_metrics.append({"fold": fold, metric: score})

    # Calculate average metrics
    avg_score = sum([aux_metric[metric] for aux_metric in fold_metrics]) / len(
        fold_metrics
    )
    fold_metrics.append({"fold": "average", metric: avg_score})

    # Save metrics to CSV
    metrics_df = pd.DataFrame(fold_metrics)
    type_model = "xgboost" if isinstance(model, XGBClassifier) else "decision_tree"
    metrics_df.to_csv(f"data/{type_model}_cross_validation_metrics.csv", index=False)

    print("Cross-validation metrics saved to cross_validation_cv_metrics.csv")
    print(f"Average {metric} score: {avg_score:.4f}")


def evaluate(
    df: pl.DataFrame, model: Union[BaseEstimator, XGBClassifier], prefix: str = ""
):
    X = df.drop("is_anomaly")
    y = df["is_anomaly"]

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    type_model = "xgboost" if isinstance(model, XGBClassifier) else "decision_tree"

    metrics = evaluate_model(y, y_pred, type_model, prefix)
    print(f"{prefix.replace('_', ' - ').upper()}Evaluation Metrics:", metrics)

    # Save ROC plot
    plot_roc_curve(y, y_prob, f"img/{prefix}{type_model}_roc_curve_decision_tree.png")

    # Save confusion matrix plot
    plot_confusion_matrix(
        y, y_pred, f"img/{prefix}{type_model}_confusion_matrix_decision_tree.png"
    )

    return metrics


def get_xgboost_space():
    return {
        "max_depth": hp.quniform("max_depth", 3, 18, 1),
        "gamma": hp.uniform("gamma", 1, 9),
        "reg_alpha": hp.quniform("reg_alpha", 40, 180, 1),
        "reg_lambda": hp.uniform("reg_lambda", 0, 1),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
        "min_child_weight": hp.quniform("min_child_weight", 0, 10, 1),
        "n_estimators": hp.quniform("n_estimators", 1, 140, 2),
        "seed": 0,
    }


def get_xgboost_objective_func(
    space: Dict[str, Any], X_train, y_train, folds, fold_column
):
    def objective(params):
        params["max_depth"] = int(params["max_depth"])
        params["gamma"] = int(params["gamma"])
        params["reg_lambda"] = int(params["reg_lambda"])
        params["n_estimators"] = int(params["n_estimators"])
        params["reg_alpha"] = int(params["reg_alpha"])
        params["min_child_weight"] = int(params["min_child_weight"])

        fold_scores = []

        for train_indexes, test_indexes in get_cv_iterable(folds, fold_column, X_train):
            # Split the data
            X_train_fold, X_test_fold = X_train[train_indexes], X_train[test_indexes]
            y_train_fold, y_test_fold = y_train[train_indexes], y_train[test_indexes]

            clf = XGBClassifier(**params)
            clf.fit(
                X_train_fold,
                y_train_fold,
                eval_set=[(X_train_fold, y_train_fold), (X_test_fold, y_test_fold)],
                verbose=False,
            )

            pred = clf.predict(X_test_fold)
            accuracy = accuracy_score(y_test_fold, pred > 0.5)
            fold_scores.append(accuracy)

        # Compute the average score across all folds
        avg_score = np.mean(fold_scores)
        print("Average CV Score:", avg_score)
        return {"loss": -avg_score, "status": STATUS_OK}

    return objective


def bayesian_optimization(
    space: Dict[str, Any], objective: callable
) -> Any | dict | None:
    trials = Trials()
    best = fmin(
        fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials
    )

    print("Best hyperparameters:", best)
    return best


def perform_hyperparameter_search(
    model: BaseEstimator,
    X: pl.DataFrame,
    y: pl.Series,
    metric: str,
    param_grid: Dict[str, Any],
) -> Tuple[float, Dict[str, object]]:
    """
    Perform hyperparameter search using RandomizedSearchCV.

    Args:
        model (BaseEstimator): The model to optimize.
        X (pl.DataFrame): Features for training.
        y (pl.Series): Target variable.
        metric (str): Score metric to use.
        param_grid (Dict[str, Any]): Parameters to search.

    Returns:
        Tuple[float, Dict[str, object]]: Best score and best parameters.
    """
    
    folds = X["fold"].unique()

    search = RandomizedSearchCV(
        model,
        param_grid,
        cv=get_cv_iterable(folds, "fold", X),
        scoring=metric,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X.drop("fold"), y)
    return search.best_score_, search.best_params_


def calculate_feature_importance(model: Union[BaseEstimator, XGBClassifier], X: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate feature importance from a trained Decision Tree.

    Args:
        model (BaseEstimator): Trained pipeline.
        X (pl.DataFrame): Features used in training.

    Returns:
        pl.DataFrame: Feature importance sorted in descending order.
    """
    feature_names = [col for col in X.columns if col != "fold"]
    importances = model.feature_importances_
    importance_df = pl.DataFrame({"feature": feature_names, "importance": importances})

    return importance_df.sort("importance", descending=True)


def save_feature_importance_plot(
    feature_importance: pl.DataFrame, filename: str
) -> None:
    """
    Save a bar plot of the top 10 most important features.

    Args:
        feature_importance (pl.DataFrame): DataFrame containing feature importance.
        filename (str): Path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(10), x="importance", y="feature")
    plt.title("Top 10 Most Important Features")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()