# standard imports
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv

# local imports
from blueprints import data_blueprint, pred_blueprint
from blueprints.cache import cache
from configs.loadsettings import AppSettings

load_dotenv()  # load environment variables
app_settings = AppSettings()

app = Flask(__name__)

app.register_blueprint(data_blueprint)
app.register_blueprint(pred_blueprint)

CORS(
    app, origins=[app_settings.FRONTEND_URL, app_settings.FRONTEND_URL]
)  # TODO: Find out why you have to copy url twice in list

cache.init_app(app, config={"CACHE_TYPE": "simple"})


@app.route("/")
def hello_world() -> str:

    return "Hello world"


if __name__ == "__main__":
    debug = True if app_settings.ENV == "development" else False
    app.run(debug=debug)
