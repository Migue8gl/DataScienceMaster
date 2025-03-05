import warnings

warnings.filterwarnings("ignore")

import numpy as np
import polars as pl

from utils import create_pipeline_nb, evaluate, plot_confusion_matrix

seed = np.random.randint(0, 1000)


def main():
    pipeline = create_pipeline_nb()

    df = pl.read_csv("data/train.csv")
    test_df = pl.read_csv("data/test.csv")

    for category in df.select(pl.exclude("text")).columns:
        X_test = test_df["text"].to_numpy()
        y_test = test_df[category].to_numpy()
        X_train = df["text"].to_numpy()
        y_train = df[category].to_numpy()
        _ = pipeline.fit(X_train, y_train)
        save_kwargs = {"name": "nb_per_level", "category": category}
        final_preds = evaluate(pipeline, X_test, y_test, **save_kwargs)

        plot_confusion_matrix(
            y_test, final_preds, f"{category} confusion matrix", category, "nb_per_level"
        )


if __name__ == "__main__":
    main()
