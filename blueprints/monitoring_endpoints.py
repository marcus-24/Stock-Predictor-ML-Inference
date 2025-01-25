# standard imports
from evidently.ui.workspace.cloud import CloudWorkspace
from evidently.test_preset import DataDriftTestPreset
from evidently.test_suite import TestSuite
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta
from flask import Blueprint, jsonify, Response
from datasets import load_dataset

# local imports
from configs.loadsettings import EvidentlySettings, HuggingFaceSettings
from mlops.preprocessing import process_data
from mlops.retrain import trigger_ml_model_retrain

# TODO: time these calls using flask scheduler
# TODO: Use github api to retrigger training pipeline from python
EV_SETTINGS = EvidentlySettings()

TICKER = "AAPL"
N_PAST = 10
END_DATE = date.today()
START_DATE = END_DATE - relativedelta(days=N_PAST)
DATA_REPO = HuggingFaceSettings().DATA_REPO
PROJECT_ID = EV_SETTINGS.EVIDENTLY_PROJECT_ID.get_secret_value()


ws = CloudWorkspace(
    token=EV_SETTINGS.EVIDENTLY_TOKEN.get_secret_value(),
    url="https://app.evidently.cloud",
)

data_drift = TestSuite(
    tests=[
        DataDriftTestPreset(stattest="z", stattest_threshold=0.05),
    ]
)

monitor_blueprint = Blueprint("monitor", __name__)


@monitor_blueprint.route("/data_drift")
def data_drift_detection() -> Response:
    current_df = process_data(
        TICKER, start_date=START_DATE, end_date=END_DATE
    ).reset_index(drop=True)
    hf_dataset = load_dataset(DATA_REPO)
    historical_df = pd.DataFrame(hf_dataset["train"]).reset_index(drop=True)
    data_drift.run(reference_data=historical_df, current_data=current_df)
    test_summary = data_drift.as_dict()
    failed_tests = [
        test for test in test_summary["tests"] if test["status"].lower() == "fail"
    ]
    if failed_tests:
        trigger_ml_model_retrain()
    # ws.add_test_suite(PROJECT_ID, data_drift, include_data=True)
    return jsonify(failed_tests)
