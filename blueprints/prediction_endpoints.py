# standard imports
from flask import Blueprint, jsonify, Response, current_app
import numpy as np
from dateutil.relativedelta import relativedelta
from hopsworks.project import Project
from hsfs.feature_store import FeatureStore
from datetime import date
import pandas as pd

# local imports
from mlops import load_model
from myfeatures.dates import financial_date_correction


pred_blueprint = Blueprint("pred", __name__)

# TODO: Make these vars global between here and the training pipeline


def _format_prediction_times(query_date: date, n_preds: int) -> list[date]:
    """Format time stamps for predictions relative to the query date from user

    Args:
        query_date (date): query date from request
        n_preds (int): number of predictions to ahead of the query date

    Returns:
        list[date]: list of prediction timestamps
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

    return pred_dates


def format_predictions(predictions: np.ndarray, query_date: date) -> Response:
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
    df = pd.DataFrame({"predictions": predictions, "date": pred_dates})
    return jsonify(df.to_dict(orient="records"))


@pred_blueprint.route("/predict")
def predict() -> Response:

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

    return format_predictions(pred, query_date)
