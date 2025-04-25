from evidently import Dataset, DataDefinition, Report
from evidently.core.datasets import Regression
from evidently.ui.workspace import CloudWorkspace
from evidently.metrics import MAE
from evidently.presets import DataDriftPreset
from keras import models
import hopsworks
from hsfs.feature_store import FeatureStore
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from mlops.retrain import trigger_ml_model_retrain
from mlops import load_model
from configs.loadsettings import EvidentlySettings, HopsworksSettings, AppSettings

load_dotenv(override=True)

HW_SETTINGS = HopsworksSettings()
EV_SETTINGS = EvidentlySettings()
ENV_NAME = AppSettings().ENV_NAME

N_PREDS = 7
LABEL_COLS = [f"label_time_{idx + 1}" for idx in range(N_PREDS)]
PRED_COLS = [f"pred_time_{idx + 1}" for idx in range(N_PREDS)]
REG_NAMES = [
    "default",
    "default_1",
    "default_2",
    "default_3",
    "default_4",
    "default_5",
    "default_6",
]

# TODO: Make sure this range only contains validation and test data

HW_PROJECT = hopsworks.login(api_key_value=HW_SETTINGS.HOPSWORKS_KEY.get_secret_value())


PROJECT_ID = EV_SETTINGS.EVIDENTLY_PROJECT_ID.get_secret_value()

DATA_DEFINITION = DataDefinition(
    numerical_columns=[
        "daily_var",
        "sev_day_sma",
        "sev_day_std",
        "daily_return",
        "sma_2std_pos",
        "sma_2std_neg",
        "high_close",
        "low_open",
        "cumul_return",
    ],
    datetime_columns=["date"],
    regression=[
        Regression(name=reg_name, target=label_col, prediction=pred_col)
        for reg_name, label_col, pred_col in zip(REG_NAMES, LABEL_COLS, PRED_COLS)
    ],
)

MONITOR_TESTS = Report([MAE(lt=10), DataDriftPreset(threshold=10)], include_tests=True)

ws = CloudWorkspace(
    token=EV_SETTINGS.EVIDENTLY_TOKEN.get_secret_value(),
    url="https://app.evidently.cloud",
)


def _get_historical_dataset(
    feature_store: FeatureStore,
    data_definition: DataDefinition = DATA_DEFINITION,
    env_name: str = ENV_NAME,
) -> Dataset:

    fg = feature_store.get_feature_group(f"trained_model_monitor_baseline_{env_name}")
    df = fg.read()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by="date")
    df.columns = [
        col.replace(f"stock_labels_{ENV_NAME}_", "") for col in df.columns
    ]  # TODO: find out why hopsworks adds prefix
    return Dataset.from_pandas(df, data_definition=data_definition)


def _get_current_dataset(
    feature_store: FeatureStore,
    model: models.Sequential,
    data_definition: DataDefinition = DATA_DEFINITION,
    env_name: str = ENV_NAME,
    pred_cols: list[str] = PRED_COLS,
) -> Dataset:

    labels_fg = feature_store.get_feature_group(f"stock_labels_{env_name}")
    labels_df = labels_fg.read().set_index("date").sort_index().tail(10)

    features_fg = feature_store.get_feature_group(f"stock_features_{env_name}")
    features_df = features_fg.read().set_index("date").sort_index().loc[labels_df.index]

    X = np.expand_dims(features_df.to_numpy(), axis=1)
    pred: np.ndarray = model.predict(X)

    merged_df = features_df.join(labels_df, how="inner")
    merged_df[pred_cols] = pred.squeeze(axis=1)
    return Dataset.from_pandas(
        merged_df.reset_index(names="date"), data_definition=data_definition
    )


def data_drift_detection() -> None:
    """Load historical data"""
    feature_store: FeatureStore = HW_PROJECT.get_feature_store()

    historical_ds = _get_historical_dataset(feature_store)
    model = load_model()

    current_ds = _get_current_dataset(feature_store, model=model)

    """Format current data"""

    my_eval = MONITOR_TESTS.run(current_data=current_ds, reference_data=historical_ds)
    test_summary = my_eval.dict()
    failed_tests = [
        test for test in test_summary["tests"] if test["status"].lower() == "fail"
    ]
    if failed_tests:  # if any test fail, trigger model retraining
        if ENV_NAME == "dev":  # TODO: create logic for dev
            print("fake retraining")
        else:
            trigger_ml_model_retrain()
    ws.add_run(PROJECT_ID, my_eval, include_data=False)


if __name__ == "__main__":
    data_drift_detection()
