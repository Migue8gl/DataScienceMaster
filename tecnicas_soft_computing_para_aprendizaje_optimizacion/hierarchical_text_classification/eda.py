import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns


def check_hierarchy(df: pl.DataFrame):
    cat_non_exclusive = (
        df.group_by("cat2")
        .agg(pl.col("cat1").n_unique().alias("cat1_count"))
        .filter(pl.col("cat1_count") > 1)
    )

    if cat_non_exclusive.is_empty():
        print("All cat2 are exclusive of their cat1")
    else:
        print("Some cat2 are not exclusive of some cat1")
        print(cat_non_exclusive)

    categories = df.select(pl.exclude("text")).columns
    unique_cat1 = df.unique("cat1").select(pl.col("cat1")).n_unique()
    print(f"Total unique cat1: {unique_cat1}")

    for i in range(1, len(categories) - 1):
        cat_non_exclusive = (
            df.group_by(categories[i + 1])
            .agg(pl.col(categories[i]).n_unique().alias(f"{categories[i]}_count"))
            .filter(pl.col(f"{categories[i]}_count") > 1)
        )

        if cat_non_exclusive.is_empty():
            print(f"All {categories[i + 1]} are exclusive of it {categories[i]}")
        else:
            print(f"Some {categories[i + 1]} are not exclusive of it {categories[i]}")
            print(cat_non_exclusive)


def check_unique_categories(df: pl.DataFrame):
    # How many unique categories in each level
    categories = df.select(pl.exclude("text")).columns
    unique_cat1 = df.unique("cat1").select(pl.col("cat1")).n_unique()
    print(f"Total unique domains: {unique_cat1}")

    father_category = "cat1"
    for category in categories[1:]:
        uniques = (
            df.group_by(father_category).n_unique().select([father_category, category])
        )
        print(f"\nUnique subfields per {father_category}:")
        print(uniques)
        father_category = category


def items_per_category(df: pl.DataFrame):
    for category in df.select(pl.exclude("text")).columns:
        category_dist = (
            df.group_by(category).n_unique().select([pl.col("text"), pl.col(category)])
        )
        plt.figure(figsize=(10, 6))
        sns.barplot(category_dist, x=category, y="text")
        plt.ylabel("Text count")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f"img/{category}_dist.png")


def main():
    # Leemos el csv
    df = pl.read_csv("data/data_preprocessed.csv")

    check_unique_categories(df)
    check_hierarchy(df)
    items_per_category(df)


if __name__ == "__main__":
    main()
