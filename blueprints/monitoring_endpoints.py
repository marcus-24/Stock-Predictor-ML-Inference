# standard imports
from evidently.ui.workspace.cloud import CloudWorkspace
from evidently.report import Report
from evidently.metric_preset import DataQualityPreset, DataDriftPreset
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta
from flask import Blueprint, jsonify, Response
from datasets import load_dataset

# local imports
from configs.loadsettings import EvidentlySettings, HuggingFaceSettings
from preprocessing.defaults import process_data

# TODO: time these calls using flask scheduler
# TODO: Use github api to retrigger training pipeline from python

TICKER = "AAPL"
N_PAST = 10
END_DATE = date.today()
START_DATE = END_DATE - relativedelta(days=N_PAST)
DATA_REPO = HuggingFaceSettings().DATA_REPO

ev_settings = EvidentlySettings()
ws = CloudWorkspace(
    token=ev_settings.EVIDENTLY_TOKEN.get_secret_value(),
    url="https://app.evidently.cloud",
)
project = ws.get_project(ev_settings.EVIDENTLY_PROJECT_ID.get_secret_value())

data_report = Report(
    metrics=[
        DataDriftPreset(stattest="psi", stattest_threshold="0.3"),
        DataQualityPreset(),
    ],
)

monitor_blueprint = Blueprint("monitor", __name__)


@monitor_blueprint.route("/data_drift")
def data_drift_detection() -> Response:
    current_df = process_data(
        TICKER, start_date=START_DATE, end_date=END_DATE
    ).reset_index()
    hf_dataset = load_dataset(DATA_REPO)
    historical_df = pd.DataFrame(hf_dataset["train"])
    data_report.run(reference_data=historical_df, current_data=current_df)
    ws.add_report(ev_settings.EVIDENTLY_PROJECT_ID.get_secret_value(), data_report)
    return jsonify({"message": "monitoring complete"})
