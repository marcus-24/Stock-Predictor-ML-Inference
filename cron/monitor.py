from evidently import ColumnMapping
from evidently.ui.workspace.cloud import CloudWorkspace
from evidently.tests import TestValueMAE
from evidently.test_suite import TestSuite

from mlops.retrain import trigger_ml_model_retrain
from mlops import load_model
from configs.loadsettings import EvidentlySettings


# TODO: time these calls using flask scheduler
EV_SETTINGS = EvidentlySettings()
# TODO: Make sure this range only contains validation and test data


PROJECT_ID = EV_SETTINGS.EVIDENTLY_PROJECT_ID.get_secret_value()

COL_MAPPING = ColumnMapping(
    numerical_features=["Open", "High", "Low", "Close"],
    prediction="predictions",
    target="label",
    task="regression",
)

MONITOR_TESTS = TestSuite(tests=[TestValueMAE(lt=10)])

ws = CloudWorkspace(
    token=EV_SETTINGS.EVIDENTLY_TOKEN.get_secret_value(),
    url="https://app.evidently.cloud",
)


def data_drift_detection() -> None:

    # TODO: Update feature store to keep historical data used during model training
    model = load_model()
    predictions = model.predict(X)

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
