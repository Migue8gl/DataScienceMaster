import random
from typing import Any, List, Optional, Tuple

import polars as pl


def preprocess(
    path: str,
    hierarchy: Optional[List[Tuple[Any, str]]] = None,
    input_variable: Optional[str] = None,
    test_ratio: float = 0.2,
    seed: Optional[int] = None,
):
    """
    Perform hierarchical stratified train-test split on a CSV dataset and preprocess the original data.

    Args:
        path: Path to input CSV file
        hierarchy: Optional list of tuples defining hierarchical levels
        input_variable: Optional name of primary input variable
        test_ratio: Proportion of data to use for testing (default 0.2)
        seed: Random seed for reproducibility
    """
    if seed is None:
        seed = random.randint(0, 10000)

    df = pl.read_csv(path)

    selected_columns = [input_variable] + [
        f"cat{i + 1}" for i, (_, col) in enumerate(hierarchy[: len(hierarchy)])
    ]

    if hierarchy:
        alias_columns = [
            pl.col(col).alias(f"cat{i + 1}")
            for i, (_, col) in enumerate(hierarchy[: len(hierarchy)])
        ]
        df = df.with_columns(alias_columns)

    df = df.select(selected_columns)
    df = df.select([pl.col(col).alias(col.lower()) for col in df.columns])
    df = df.with_row_index()
    sample_size = int(test_ratio * df.shape[0])

    test_df = df.sample(n=sample_size, with_replacement=False, seed=seed)
    train_df = df.filter(~df["index"].is_in(test_df["index"]))

    train_df = train_df.drop("index")
    test_df = test_df.drop("index")
    df = df.drop("index")

    df.write_csv("data/data_preprocessed.csv")
    train_df.write_csv("data/train.csv")
    test_df.write_csv("data/test.csv")

    return train_df, test_df


if __name__ == "__main__":
    # Hierarchical levels (optional)
    
    hierarchy = [
        (None, "domain"),  # Root level
        ("domain", "subfield"),  # Second level
        ("subfield", "specialization"),  # Third level
    ]
    path = "data/data_synthetic.csv"
    input_variable = "text"
    
    """
    hierarchy = [
        (None, "Cat1"),  # Root level
        ("Cat1", "Cat2"),  # Second level
        ("Cat2", "Cat3"),  # Third level
    ]
    path = "data/data_kaggle.csv"
    input_variable = "Text"""

    train_df, test_df = preprocess(
        path, hierarchy=hierarchy, input_variable=input_variable, test_ratio=0.2
    )
