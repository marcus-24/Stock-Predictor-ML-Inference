# standard imports
from flask import Blueprint, jsonify, Response
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # turn off one DNN custom operations
import keras
from keras import models
import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta

# local imports
from .cache import cache
from mlops.preprocessing import process_data
from configs.loadsettings import HuggingFaceSettings


pred_blueprint = Blueprint("pred", __name__)

MODEL_URL = HuggingFaceSettings().MODEL_URL

# TODO: Make these vars global between here and the training pipeline
N_PAST = 10
TICKER = "AAPL"
DATA_COLS = ["Open", "High", "Low", "Close"]

END_DATE = date.today()
START_DATE = END_DATE - relativedelta(days=N_PAST)


def load_model(url: str = MODEL_URL) -> models.Sequential:
    cached_model = cache.get("my_model")
    if not cached_model:
        # Data not in cache, fetch it from the model
        model = keras.saving.load_model(url)
        cache.set("my_model", model, timeout=600)  # Cache for 600 seconds
    else:
        model = cached_model

    return model


@pred_blueprint.route(
    "/predict", defaults={"start_date": START_DATE, "end_date": END_DATE}
)
@pred_blueprint.route("/predict/start_date/<start_date>/end_date/<end_date>")
def predict(start_date: date, end_date: date) -> Response:
    df = process_data(TICKER, start_date, end_date)
    X = np.expand_dims(
        df.to_numpy(), axis=0
    )  # convert to numpy and add batch dim for model input shape
    model = load_model()
    pred: np.ndarray = model.predict(X)

    return jsonify(pred.tolist())
