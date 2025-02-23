# standard imports
from evidently import ColumnMapping
from evidently.ui.workspace.cloud import CloudWorkspace
from evidently.tests import TestValueMAE
from evidently.test_suite import TestSuite
import pandas as pd
import tensorflow as tf
from datetime import datetime
from dateutil.relativedelta import relativedelta
from flask import Blueprint, jsonify, Response
from datasets import load_dataset

# local imports
from configs.loadsettings import EvidentlySettings, HuggingFaceSettings
from mlops.preprocessing import process_data, feature_engineering, df_to_dataset
from mlops.retrain import trigger_ml_model_retrain
from mlops import load_model

# TODO: time these calls using flask scheduler
EV_SETTINGS = EvidentlySettings()

TICKER = "AAPL"
N_PAST = 20
N_FUTURE = 1
# TODO: Make sure this range only contains validation and test data
START_TIME = datetime(month=2, day=17, year=2025, hour=9)
END_TIME = START_TIME + relativedelta(days=4)


DATA_REPO = HuggingFaceSettings().DATA_REPO
PROJECT_ID = EV_SETTINGS.EVIDENTLY_PROJECT_ID.get_secret_value()

monitor_blueprint = Blueprint("monitor", __name__)

COL_MAPPING = ColumnMapping(
    numerical_features=["Open", "High", "Low", "Close"],
    prediction="predictions",
    target="label",
)

MONITOR_TESTS = TestSuite(tests=[TestValueMAE(lt=10)])

ws = CloudWorkspace(
    token=EV_SETTINGS.EVIDENTLY_TOKEN.get_secret_value(),
    url="https://app.evidently.cloud",
)


@monitor_blueprint.route("/data_drift")
def data_drift_detection() -> Response:
    """Get historic predictions"""
    hf_dataset = load_dataset(DATA_REPO)
    historical_df = pd.DataFrame(hf_dataset["test"]).reset_index(drop=True)

    """Get production (current) predictions"""
    current_df = process_data(TICKER, start_time=START_TIME, end_time=END_TIME)
    transformed_df = feature_engineering(current_df, win_size=7, n_future=N_FUTURE)
    X = df_to_dataset(transformed_df, batch_size=1).map(lambda x, _: x)

    model = load_model()
    predictions = model.predict(X)
    transformed_df["predictions"] = tf.reshape(predictions, [-1])

    monitor_df = transformed_df[["predictions", "label"]].join(current_df, how="inner")
    print(monitor_df.head())

    """Format current data"""

    MONITOR_TESTS.run(
        reference_data=historical_df,
        current_data=monitor_df,
        column_mapping=COL_MAPPING,
    )
    test_summary = MONITOR_TESTS.as_dict()
    failed_tests = [
        test for test in test_summary["tests"] if test["status"].lower() == "fail"
    ]
    if failed_tests:  # if any test fail, trigger model retraining
        trigger_ml_model_retrain()
    ws.add_test_suite(PROJECT_ID, MONITOR_TESTS)
    return jsonify(failed_tests)
