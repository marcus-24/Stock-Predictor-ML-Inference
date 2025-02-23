import pandas as pd
import yfinance as yf
import tensorflow as tf

DATA_COLS = ["Open", "High", "Low", "Close"]


def process_data(
    ticker: str, start_time: str, end_time: str = None, data_cols: list[str] = DATA_COLS
) -> pd.DataFrame:
    return (
        yf.Ticker(ticker).history(interval="1h", start=start_time, end=end_time)
        # .loc[:, data_cols]
        # .reset_index()  # pop out datetime index
    )  # make each row a json object


def feature_engineering(df: pd.DataFrame, win_size: int, n_future: int) -> pd.DataFrame:
    # https://medium.com/aimonks/improving-stock-price-forecasting-by-feature-engineering-8a5d0be2be96
    _df = df.copy()

    trans_df = pd.DataFrame(
        {
            "daily_var": (_df["High"] - _df["Low"]) / (_df["Open"]),
            "sev_day_sma": _df["Close"].rolling(win_size).mean(),
            "sev_day_std": _df["Close"].rolling(win_size).std(),
            "daily_return": _df["Close"].diff(),
            "sma_2std_pos": _df["Close"].rolling(win_size).mean()
            + 2 * _df["Close"].rolling(win_size).std(),
            "sma_2std_neg": _df["Close"].rolling(win_size).mean()
            - 2 * _df["Close"].rolling(win_size).std(),
            "high_close": (_df["High"] - _df["Close"]) / _df["Open"],
            "low_open": (_df["Low"] - _df["Open"]) / _df["Open"],
            "cumul_return": _df["Close"] - _df["Close"].iloc[0],
            "label": _df["Close"].shift(n_future),
        }
    ).dropna()
    return trans_df


def add_dimension_to_element(
    feature: tf.Tensor, label: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    feature_expanded = tf.expand_dims(feature, axis=0)
    label_expanded = tf.expand_dims(label, axis=0)
    return feature_expanded, label_expanded


def df_to_dataset(
    df: pd.DataFrame,
    batch_size: int,
    label_col: str = "label",
) -> tf.data.Dataset:
    _df = df.copy()
    labels = _df.pop(label_col)

    return (
        tf.data.Dataset.from_tensor_slices((_df, labels))
        .map(add_dimension_to_element)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
