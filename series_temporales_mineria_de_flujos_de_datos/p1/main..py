import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller


def apply_rolling_mean(
    df: pd.DataFrame, column: str, window: int = 7, center: bool = True
) -> pd.DataFrame:
    """
    Applies a rolling mean to the specified column in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the time series data.
    column (str): The column to apply the rolling mean to.
    window (int): The size of the rolling window (default: 7 days).
    center (bool): Whether to center the window (default: True).

    Returns:
    pd.DataFrame: DataFrame with an additional column for the rolling mean.
    """
    result_df = df.copy()
    result_df[f"{column}_Rolling_Mean"] = (
        result_df[column].rolling(window=window, center=center).mean()
    )
    return result_df


def plot_before_after_rolling_mean(
    df: pd.DataFrame,
    original_column: str,
    rolling_column: str,
    time_range: np.ndarray = None,
    figsize: tuple = (14, 7),
    title: str = None,
    filename: str = None,
) -> None:
    """
    Plots the original data and the rolling mean for comparison.

    Parameters:
    df (pd.DataFrame): DataFrame containing both original and rolling mean data.
    original_column (str): Name of the column with original data.
    rolling_column (str): Name of the column with rolling mean data.
    time_range (np.ndarray): Optional array with [start_date, end_date] for filtering.
    figsize (tuple): Figure size as (width, height) in inches.
    title (str): Plot title. If None, a default title will be generated.
    filename (str): If provided, the plot will be saved to this filename.
    """
    plot_df = df.copy()

    if time_range is not None:
        start_date = pd.to_datetime(time_range[0])
        end_date = pd.to_datetime(time_range[1])
        plot_df = plot_df[plot_df.index.to_series().between(start_date, end_date)]

    plt.figure(figsize=figsize)

    plt.plot(
        plot_df.index,
        plot_df[original_column],
        label=f"Original {original_column}",
        color="blue",
        alpha=0.5,
        linewidth=1,
    )

    plt.plot(
        plot_df.index,
        plot_df[rolling_column],
        label=f"Rolling Mean ({rolling_column})",
        color="red",
        linewidth=2,
    )

    if title is None:
        window_size = (
            rolling_column.split("_")[-2] if "Rolling" in rolling_column else "N"
        )
        title = f"{original_column} vs {window_size}-Day Rolling Mean"

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")


def split_train_test(df: pd.DataFrame, test_year: int = 2021):
    """
    Splits the dataset into training and testing sets based on the given year.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the time series data.
    test_year (int): The year to be used for testing.

    Returns:
    pd.DataFrame, pd.DataFrame: Training and testing DataFrames.
    """
    df = df.copy()
    df["Year"] = df.index.year
    train_df = df[df["Year"] < test_year].drop(columns=["Year"])
    test_df = df[df["Year"] >= test_year].drop(columns=["Year"])
    return train_df, test_df


def temperature_plot(time_range: np.ndarray, df: pd.DataFrame, filename: str):
    """
    Plots temperature data over a specified time range.

    Parameters:
    time_range (np.ndarray): Array of datetime strings for filtering.
    df (pd.DataFrame): DataFrame containing the temperature data.
    filename (str): Filename to save the plot.
    """
    start_date = pd.to_datetime(time_range[0])
    end_date = pd.to_datetime(time_range[1])
    filtered_df = df[df["DateTime"].between(start_date, end_date)]

    plt.figure(figsize=(12, 6))
    plt.plot(
        filtered_df["DateTime"],
        filtered_df["Temperature (ºC)"],
        label="Temperature (ºC)",
        color="blue",
    )
    plt.title(f"Temperature from {start_date} to {end_date}")
    plt.xlabel("DateTime")
    plt.ylabel("Temperature (ºC)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename)


def get_data(verbose: bool = False) -> pd.DataFrame:
    """
    Loads and preprocesses the data from the CSV file.

    Parameters:
    verbose (bool): If True, prints detailed information about the dataset.

    Returns:
    pd.DataFrame: Preprocessed DataFrame.
    """
    df = pd.read_csv("oikolab.csv")
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df.set_index("DateTime", inplace=True)
    if df.duplicated().sum() > 0:
        print(f"Found {df.duplicated().sum()} duplicates. Removing them.")
        df = df.drop_duplicates()
    if verbose:
        print(df.info())
        print(df.head())
    return df


def plot_ts_decomposition(ts: np.ndarray):
    decomposition = seasonal_decompose(ts, model="additive", period=12)
    fig = decomposition.plot()
    fig.set_size_inches(10, 8) 
    plt.suptitle(
        "Time Series Decomposition", fontsize=16, y=0.95
    )  #
    plt.tight_layout(rect=[0, 0, 1, 0.96])  
    plt.savefig("time_series_decomposition.png")


def check_stationary(ts: np.ndarray):
    result = adfuller(ts)
    print("Resultados de la prueba ADF:")
    print(f"Estadístico ADF: {result[0]}")
    print(f"p-valor: {result[1]}")
    print(f"Valores críticos: {result[4]}")
    if result[1] > 0.05:
        print("La serie probablemente no es estacionaria.")
    else:
        print("La serie probablemente es estacionaria.")


def main():
    df = get_data(verbose=False)

    # ----------------- Decomposition of the ts ----------------- #

    ts = df["Temperature (ºC)"].to_numpy()
    plot_ts_decomposition(ts)

    # ----------------- Check tendency ----------------- #

    df_with_rolling = apply_rolling_mean(df, "Temperature (ºC)", window=400)

    plot_before_after_rolling_mean(
        df_with_rolling,
        "Temperature (ºC)",
        "Temperature (ºC)_Rolling_Mean",
        time_range=np.array([str(df.index.min()), str(df.index.max())]),
        title="Training Data: Original vs 7-Day Rolling Mean",
        filename="temperature_train_with_rolling.jpg",
    )

    # ----------------- Check stationary ----------------- #

    plot_acf(ts, lags=40)
    plt.suptitle(
        "Time Series Decomposition"
    )  
    plt.savefig("ts_autocorrelation.png")

    # ----------------- Train test ----------------- #

    train_df, test_df = split_train_test(df)
    print(f"Train Data: {train_df.index.min()} to {train_df.index.max()}")
    print(f"Test Data: {test_df.index.min()} to {test_df.index.max()}")

    train_time_range = np.array([str(train_df.index.min()), str(train_df.index.max())])
    temperature_plot(train_time_range, train_df.reset_index(), "temperature_train.jpg")

    test_time_range = np.array([str(test_df.index.min()), str(test_df.index.max())])
    temperature_plot(test_time_range, test_df.reset_index(), "temperature_test.jpg")


if __name__ == "__main__":
    main()
