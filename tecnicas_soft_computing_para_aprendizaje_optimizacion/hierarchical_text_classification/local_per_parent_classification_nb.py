import warnings

warnings.filterwarnings("ignore")

from typing import Any, List, Optional, Tuple

import numpy as np
import polars as pl
from sklearn import clone

from utils import create_pipeline_nb, evaluate, plot_confusion_matrix

seed = np.random.randint(0, 1000)


class HierarchicalModel:
    def __init__(self, hierarchy_levels: List[Any], hierarchy_labels: List[str]):
        """
        Initialize with hierarchical classification levels
        hierarchy_levels: List where first element is root classifier,
                        subsequent elements are dicts of classifiers
        """
        self.hierarchy = hierarchy_levels
        self.hierarchy_labels = hierarchy_labels

    def predict(
        self, X: np.ndarray[str], level_label: Optional[str] = None
    ) -> np.ndarray:
        """Make hierarchical predictions"""
        current_pred = self.hierarchy[0].predict(X)
        if level_label is None:
            level_label = self.hierarchy_labels[-1]

        for index, level in enumerate(self.hierarchy[1:]):
            next_pred = []
            if self.hierarchy_labels[index].lower() == level_label.lower():
                break
            for i, pred in enumerate(current_pred):
                clf = level[pred.lower()]
                next_pred.append(clf.predict([X[i]])[0])
            current_pred = next_pred

        return np.array(current_pred)


def train_hierarchy(
    df: pl.DataFrame, hierarchy: List[Tuple[str, str]]
) -> HierarchicalModel:
    """
    Train hierarchical model
    hierarchy: List of tuples specifying (filter_column, target_column)
    """
    classifiers = []
    current_df = df

    for level, (filter_col, target_col) in enumerate(hierarchy):
        if level == 0:  # Root level
            X, y = current_df["text"].to_numpy(), current_df[target_col].to_numpy()
            clf = clone(create_pipeline_nb()).fit(X, y)
            classifiers.append(clf)
        else:
            level_classifiers = {}
            parent_categories = current_df[filter_col].unique().to_numpy()

            for category in parent_categories:
                filtered_df = current_df.filter(pl.col(filter_col) == category)
                X = filtered_df["text"].to_numpy()
                y = filtered_df[target_col].to_numpy()

                if len(np.unique(y)) > 1:  # Only train if multiple classes exist
                    clf = clone(create_pipeline_nb()).fit(X, y)
                    level_classifiers[category.lower()] = clf

            classifiers.append(level_classifiers)
            current_df = current_df.filter(pl.col(filter_col).is_in(parent_categories))

    return HierarchicalModel(classifiers, [category[1] for category in hierarchy])


def main():
    # Example usage
    df = pl.read_csv("data/train.csv")

    # Define hierarchy: [(parent_column, target_column)]

    # Train hierarchical model
    categories = df.select(pl.exclude("text")).columns
    hierarchy = [(None, categories[0])] + [
        (categories[i], categories[i + 1]) for i in range(0, len(categories) - 1)
    ]
    model = train_hierarchy(df, hierarchy)
    test_df = pl.read_csv("data/test.csv")

    for category in df.select(pl.exclude("text")).columns:
        X_test = test_df["text"].to_numpy()
        y_test = test_df[category].to_numpy()
        # Make predictions and plot
        save_kwargs = {"name": "nb_per_parent", "category": category}
        evaluate(model, X_test, y_test, category, **save_kwargs)
        preds = model.predict(X_test, category)
        plot_confusion_matrix(
            y_test, preds, f"{category}confusion matrix", category, "nb_per_parent"
        )


if __name__ == "__main__":
    main()
