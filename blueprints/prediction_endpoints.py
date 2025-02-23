# standard imports
from flask import Blueprint, jsonify, Response
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

# local imports
from mlops.preprocessing import process_data
from mlops import load_model

pred_blueprint = Blueprint("pred", __name__)

# TODO: Make these vars global between here and the training pipeline
N_PAST = 7
TICKER = "AAPL"
DATA_COLS = ["Open", "High", "Low", "Close"]

START_TIME = datetime.now() - relativedelta(hours=7)
END_TIME = None


@pred_blueprint.route(
    "/predict", defaults={"start_time": START_TIME, "end_time": END_TIME}
)
@pred_blueprint.route("/predict/start_time/<start_time>/end_time/<end_time>")
def predict(start_time: datetime, end_time: datetime) -> Response:
    df = process_data(TICKER, start_time, end_time)
    X = np.expand_dims(
        df.mean().to_numpy(), axis=(0, 1)
    )  # convert to numpy and add batch dim for model input shape
    print(X)
    model = load_model()
    pred: np.ndarray = model.predict(X, batch_size=X.shape[0])

    return jsonify(pred.tolist())
