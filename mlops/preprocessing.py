import pandas as pd
import yfinance as yf

DATA_COLS = ["Open", "High", "Low", "Close"]


def process_data(
    ticker: str, start_date: str, end_date: str, data_cols: list[str] = DATA_COLS
) -> pd.DataFrame:
    return (
        yf.Ticker(ticker)
        .history(interval="1d", start=start_date, end=end_date)
        .loc[:, data_cols]
        # .reset_index()  # pop out datetime index
    )  # make each row a json object
