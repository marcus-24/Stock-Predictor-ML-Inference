# standard imports
from flask import Blueprint, jsonify, Response, current_app
import numpy as np
import yfinance as yf
from dateutil.relativedelta import relativedelta
from hopsworks.project import Project
from hsfs.feature_store import FeatureStore
from datetime import date
import pandas as pd
import os
from dotenv import load_dotenv

# local imports
from mlops import load_model
from myfeatures.dates import financial_date_correction

load_dotenv(override=True)  # load environment variables
SYMBOL = os.getenv("SYMBOL")
START_DATE = date.today()


pred_blueprint = Blueprint("pred", __name__, url_prefix="/prediction")

# TODO: Make these vars global between here and the training pipeline


def _format_prediction_times(query_date: date, n_preds: int) -> list[str]:
    """Format time stamps for predictions relative to the query date from user

    Args:
        query_date (date): query date from request
        n_preds (int): number of predictions to ahead of the query date

    Returns:
        list[str]: list of prediction timestamps
    """

    # TODO: Find a way to use a accumulate type of function to simplify this
    pred_dates = [None] * n_preds
    for idx in range(n_preds):
        if idx == 0:
            pred_dates[idx] = financial_date_correction(
                query_date + relativedelta(days=1), direction="forward"
            )
        else:
            pred_dates[idx] = financial_date_correction(
                pred_dates[idx - 1] + relativedelta(days=1), direction="forward"
            )

    return [
        dt.isoformat() for dt in pred_dates
    ]  # needs to be isoformat for front end to read


def _prepend_latest_stock(
    pred_df: pd.DataFrame, symbol: str, start_date: date
) -> pd.DataFrame:
    data_df = (
        yf.Ticker(symbol)
        .history(start=start_date, interval="1d")
        .reset_index()  # pop out datetime index
        .assign(Date=lambda x: [dt.strftime("%Y-%m-%d") for dt in x["Date"]])
        .loc[:, ["Close", "Date"]]
    )

    return pd.concat([data_df, pred_df], ignore_index=True, axis=0)


def format_predictions(predictions: np.ndarray, query_date: date) -> pd.DataFrame:
    """format predictions from the trained to be sent to the front-end

    Args:
        predictions (np.ndarray): raw predictions from the tensorflow model
        query_date (date): query date from request

    Returns:
        Response: formatted predictions
    """
    if predictions.ndim == 3:
        predictions = predictions.flatten()

    n_preds = predictions.shape[0]

    pred_dates = _format_prediction_times(query_date, n_preds)
    return pd.DataFrame({"Close": predictions, "Date": pred_dates})


@pred_blueprint.route(
    "/ticker",
    defaults={"symbol": SYMBOL, "start_date": START_DATE},
)
@pred_blueprint.route("ticker/<string:symbol>/start-date/<start_date>")
def predict(symbol: str, start_date: str) -> Response:
    # TODO: select model for given stock and query features based on time
    hopsworks_project: Project = current_app.config["HOPSWORKS_PROJECT"]
    env_name = current_app.config["ENV_NAME"]
    fs: FeatureStore = hopsworks_project.get_feature_store(
        name="stock_predictor_featurestore"
    )

    # TODO: Replace line below with hopsworks query
    df = (
        fs.get_feature_group(name=f"stock_features_{env_name}")
        .read()
        .set_index("date")
        .sort_index()
        .tail(1)
    )
    query_date = df.index[0]

    X = np.expand_dims(df.values, axis=0)

    model = load_model()
    pred: np.ndarray = model.predict(X, batch_size=X.shape[0])

    pred_df = format_predictions(pred, query_date)

    return jsonify(
        _prepend_latest_stock(pred_df, symbol, start_date).to_dict(orient="records")
    )
