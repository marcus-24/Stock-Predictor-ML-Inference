# standard imports
from flask import Flask, Blueprint
from flask_cors import CORS
from flask_apscheduler import APScheduler
from dotenv import load_dotenv

# local imports
from blueprints import data_blueprint, pred_blueprint
from cron.monitor import data_drift_detection
from mlops.get_models import cache
from configs.loadsettings import AppSettings

# TODO: Monitor is going to be flask scheduler and prediction will stay as standard endpoint (will change based on current time)

load_dotenv(override=True)  # load environment variables

app = Flask(__name__)


def setup_cors_extension(my_app: Flask, frontend_url: str) -> Flask:
    """_summary_

    Args:
        my_app (Flask): _description_
        frontend_url (str): _description_

    Returns:
        Flask: _description_
    """
    CORS(
        my_app, origins=[frontend_url, frontend_url]
    )  # TODO: Find out why you have to copy url twice in list

    return my_app


def register_blueprints(my_app: Flask, blueprints: list[Blueprint]) -> Flask:
    """_summary_

    Args:
        my_app (Flask): _description_
        blueprints (list[Blueprint]): _description_

    Returns:
        Flask: _description_
    """
    for blueprint in blueprints:
        my_app.register_blueprint(blueprint)

    return my_app


def setup_scheduler(my_app: Flask, my_scheduler: APScheduler) -> Flask:
    """_summary_

    Args:
        my_app (Flask): _description_
        my_scheduler (APScheduler): _description_

    Returns:
        Flask: _description_
    """
    my_scheduler.init_app(my_app)
    my_scheduler.add_job(
        id="drift_detection",
        func=data_drift_detection,
        trigger="cron",
        hour="13",
        day_of_week="mon-fri",
    )
    my_scheduler.start()
    return my_app


@app.route("/")
def hello_world() -> str:

    return "Hello world"


if __name__ == "__main__":
    app_settings = AppSettings()
    DEBUG = app_settings.ENV == "development"

    app = register_blueprints(app, blueprints=[data_blueprint, pred_blueprint])
    app = setup_cors_extension(app, frontend_url=app_settings.FRONTEND_URL)

    cache.init_app(app, config={"CACHE_TYPE": "simple"})

    # scheduler = APScheduler()

    # app = setup_scheduler(app, scheduler)

    app.run(debug=DEBUG)
