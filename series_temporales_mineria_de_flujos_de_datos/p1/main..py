from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller


def apply_rolling_mean(
    df: pd.DataFrame, column: str, window: int = 7, center: bool = True
) -> pd.DataFrame:
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
    df = df.copy()
    df["Year"] = df.index.year
    train_df = df[df["Year"] < test_year].drop(columns=["Year"])
    test_df = df[df["Year"] >= test_year].drop(columns=["Year"])
    return train_df, test_df


def temperature_plot(time_range: np.ndarray, df: pd.DataFrame, filename: str):
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
    df = pd.read_csv("oikolab.csv")
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df.set_index("DateTime", inplace=True)
    df_monthly = df.resample("ME").mean()
    df_monthly = df_monthly[~df_monthly.index.duplicated(keep="first")]
    # Verificar NaNs
    if df_monthly.isnull().sum().sum() > 0:
        df_monthly = df_monthly.interpolate()
    if verbose:
        print(df_monthly.info())
    return df_monthly


def plot_ts_decomposition(ts: np.ndarray):
    decomposition = seasonal_decompose(ts, model="additive", period=12)
    fig = decomposition.plot()
    fig.set_size_inches(10, 8)
    plt.suptitle("Time Series Decomposition", fontsize=16, y=0.95)  #
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("time_series_decomposition.png")


def check_stationary(ts: pd.Series):
    result = adfuller(ts)
    print("Resultados de la prueba ADF:")
    print(f"Estadístico ADF: {result[0]}")
    print(f"p-valor: {result[1]}")
    print(f"Valores críticos: {result[4]}")
    if result[1] > 0.05:
        print("La serie probablemente no es estacionaria.")
    else:
        print("La serie probablemente es estacionaria.")


def get_best_arima_model(ts, max_order=3):
    best_aic = np.inf
    best_order = None

    for p, d, q in product(
        range(max_order + 1), range(max_order + 1), range(max_order + 1)
    ):
        try:
            model = ARIMA(ts, order=(p, d, q))
            results = model.fit()

            if results.aic < best_aic:
                best_aic = results.aic
                best_order = (p, d, q)
        except:
            continue

    print(f"Mejor orden ARIMA: {best_order}, AIC: {best_aic}")
    return best_order, best_aic


def remove_seasonality(t, x_ts, seasonality_period=12):
    season = np.zeros(seasonality_period)
    for i in range(seasonality_period):
        season[i] = np.mean(x_ts[i::seasonality_period])

    num_seasons = int(np.ceil(len(x_ts) / seasonality_period))
    tiled_season = np.tile(season, num_seasons)[: len(x_ts)]

    plt.figure(figsize=(10, 4))
    plt.plot(season)
    plt.title("Modelo de Estacionalidad")
    plt.savefig("seasonality_model.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(t, x_ts, label="Serie Temporal")
    plt.plot(t, tiled_season, label="Modelo de Estacionalidad", linestyle="--")
    plt.title("Serie Temporal con Modelo de Estacionalidad")
    plt.savefig("time_series_with_seasonality.png")
    plt.close()

    x_ts_no_season = x_ts - tiled_season

    plt.figure(figsize=(12, 6))
    plt.plot(t, x_ts_no_season)
    plt.title("Serie Temporal sin Estacionalidad")
    plt.savefig("time_series_no_seasonality.png")
    plt.close()

    return x_ts_no_season, season, tiled_season


def predict_arima(train_data, test_data, order):
    model = ARIMA(train_data, order=order)
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=len(test_data))
    return predictions


def plot_predictions(train_data, test_data, predictions, title="Predicciones ARIMA"):
    plt.figure(figsize=(12, 6))
    plt.plot(train_data.index, train_data, label="Datos de entrenamiento", color="blue")
    plt.plot(test_data.index, test_data, label="Datos de prueba", color="green")
    plt.plot(
        test_data.index, predictions, label="Predicciones", color="red", linestyle="--"
    )
    plt.title(title)
    plt.xlabel("Fecha")
    plt.ylabel("Temperatura (ºC)")
    plt.legend()
    plt.grid(True)
    plt.savefig("arima_predictions.png")
    plt.close()


def main():
    df = get_data(verbose=False)

    # ----------------- Decomposition of the ts ----------------- #

    ts = df["Temperature (ºC)"].to_numpy()
    plot_ts_decomposition(ts)

    # ----------------- Check tendency ----------------- #

    df_with_rolling = apply_rolling_mean(df, "Temperature (ºC)", window=12)

    plot_before_after_rolling_mean(
        df_with_rolling,
        "Temperature (ºC)",
        "Temperature (ºC)_Rolling_Mean",
        time_range=np.array([str(df.index.min()), str(df.index.max())]),
        title="Training Data: Original vs 12-Day Rolling Mean",
        filename="temperature_train_with_rolling.jpg",
    )

    # ----------------- Check stationary ----------------- #

    # check_stationary(ts)
    plot_acf(ts, lags=12)
    plt.suptitle("Time Series Decomposition")
    plt.savefig("ts_autocorrelation.png")

    # ----------------- Check seasonality ----------------- #

    check_stationary(ts)
    t = np.array(range(len(ts)))
    ts_no_season, _, _ = remove_seasonality(t, ts, seasonality_period=12)
    check_stationary(ts_no_season)

    # ----------------- Best ARIMA ----------------- #
    
    best_order, best_aic = get_best_arima_model(ts_no_season)
    print(f"Mejor orden ARIMA encontrado: {best_order}")

    # ----------------- Train test ----------------- #
    train_df, test_df = split_train_test(df)
    print(f"Train Data: {train_df.index.min()} to {train_df.index.max()}")
    print(f"Test Data: {test_df.index.min()} to {test_df.index.max()}")

    train_time_range = np.array([str(train_df.index.min()), str(train_df.index.max())])
    temperature_plot(train_time_range, train_df.reset_index(), "temperature_train.jpg")

    test_time_range = np.array([str(test_df.index.min()), str(test_df.index.max())])
    temperature_plot(test_time_range, test_df.reset_index(), "temperature_test.jpg")

    # ----------------- Predicción ----------------- #

    predictions = predict_arima(train_df["Temperature (ºC)"], test_df["Temperature (ºC)"], best_order)
    plot_predictions(train_df["Temperature (ºC)"], test_df["Temperature (ºC)"], predictions)

if __name__ == "__main__":
    main()
