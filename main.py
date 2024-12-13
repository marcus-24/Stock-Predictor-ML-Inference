from flask import Flask
from flask_cors import CORS
from endpoints import data_blueprint, pred_blueprint
from dotenv import load_dotenv
import os

load_dotenv()
FRONTEND_URL = os.getenv("FRONTEND_URL")

app = Flask(__name__)
app.register_blueprint(data_blueprint)
app.register_blueprint(pred_blueprint)

CORS(
    app, origins=[FRONTEND_URL, FRONTEND_URL]
)  # TODO: Find out why you have to copy url twice in list


@app.route("/")
def hello_world() -> str:
    return "Hello world"


if __name__ == "__main__":
    app.run(debug=True)
