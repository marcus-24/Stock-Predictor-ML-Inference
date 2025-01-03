from flask import Blueprint, jsonify, Response
import yfinance as yf
import pandas as pd
import random
import json
import datetime


pred_blueprint = Blueprint("pred", __name__, url_prefix="/prediction")


def add_noise(
    df: pd.DataFrame, columns: list[str] = ["Open", "High", "Low", "Close"]
) -> pd.DataFrame:
    df[columns] = df[columns].apply(lambda x: x + random.random() * 10)
    return df


def shift_time(df: pd.DataFrame, shift_days: int) -> pd.DataFrame:
    df.index = df.index + pd.Timedelta(days=shift_days)
    return df


@pred_blueprint.route("/ticker/<ticker>/days/<int:days>")
def get_prediction(ticker: str, days: int) -> Response:
    # TODO: apply actual model later
    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=days)
    json_data = (
        yf.Ticker(ticker)
        .history(start=str(start_date), end=str(today))
        .pipe(add_noise)
        .pipe(shift_time, shift_days=days)
        .reset_index()  # pop out datetime index
        .to_json(orient="records", date_format="iso")
    )  # make each row a json object

    dict_json = json.loads(json_data)

    return jsonify(dict_json)
