import os
import sys
from datetime import datetime
from scipy.stats import randint, uniform
import joblib
import polars as pl
from utils import build_structure, calculate_feature_importance, evaluate, perform_hyperparameter_search, save_feature_importance_plot
from xgboost.sklearn import XGBClassifier


def main():
    random_state = 42
    metric = "precision"

    train_df = pl.read_csv("data/train.csv")
    test_df = pl.read_csv("data/test.csv")

    # Prepare training data
    X_train = train_df.drop("is_anomaly")
    y_train = train_df["is_anomaly"]

    """
    # Perform bayesian optimization
    space = get_xgboost_space()
    folds = X_train["fold"].unique()
    objective = get_xgboost_objective_func(space, X_train, y_train, folds, "fold")

    # Run Bayesian optimization
    best_params = bayesian_optimization(space, objective)

    # Train model with best params
    best_params = {key: int(best_params[key]) for key in best_params.keys()}
    model = XGBClassifier(
        objective="binary:logistic", random_state=random_state, **best_params
    )
    """

    model = XGBClassifier(random_state=random_state)

    # Perform hyperparameter tuning
    param_grid = {
        "n_estimators": randint(50, 500),
        "max_depth": randint(3, 10),
        "learning_rate": uniform(0.01, 0.3),
        "subsample": uniform(0.5, 0.5),
        "colsample_bytree": uniform(0.5, 0.5),
        "gamma": uniform(0, 5),
        "min_child_weight": randint(1, 10),
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
    save_feature_importance_plot(feature_importance, "img/feature_importance_xgboost.png")


    # Train evaluation
    train_metrics = evaluate(train_df.drop("fold"), model)

    # Test data evaluation
    test_metrics = evaluate(test_df, model, prefix="test_")

    # Prepare metadata
    model_filename = f"xgboost_{datetime.now().strftime('%Y-%m-%d-%H:%M')}"
    metadata = {
        "best_score": [],
        "best_params": [],
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
