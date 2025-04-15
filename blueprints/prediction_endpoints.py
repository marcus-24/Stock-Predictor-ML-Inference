# standard imports
from flask import Blueprint, jsonify, Response
import numpy as np
from dateutil.relativedelta import relativedelta
import json
import hopsworks
from hsfs.feature_store import FeatureStore, Query
from datetime import date
import pandas as pd

# local imports
from mlops import load_model
from configs.loadsettings import HopsworksSettings
from myfeatures.dates import financial_date_correction

hopsworks_settings = HopsworksSettings()
PROJECT = hopsworks.login(
    api_key_value=hopsworks_settings.HOPSWORKS_KEY.get_secret_value()
)

pred_blueprint = Blueprint("pred", __name__)

# TODO: Make these vars global between here and the training pipeline


def format_predictions(predictions: np.ndarray, query_date: date) -> pd.DataFrame:
    if predictions.ndim == 3:
        predictions = predictions.flatten()
    n_preds = predictions.shape[0]
    pred_dates = [
        financial_date_correction(query_date + relativedelta(days=idx + 1))
        for idx in range(n_preds)
    ]

    return pd.DataFrame({"predictions": predictions, "date": pred_dates})


@pred_blueprint.route("/predict")
def predict() -> Response:

    fs: FeatureStore = PROJECT.get_feature_store(name="stock_predictor_featurestore")

    # TODO: Replace line below with hopsworks query
    df = (
        fs.get_feature_group(name="stock_features")
        .read()
        .set_index("date")
        .sort_index()
        .tail(1)
    )
    query_date = df.index[0]

    X = np.expand_dims(df.values, axis=0)

    model = load_model()
    pred: np.ndarray = model.predict(X, batch_size=X.shape[0])

    formatted_preds = format_predictions(pred, query_date)
    return jsonify(formatted_preds.to_dict(orient="records"))
