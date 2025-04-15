from flask import Blueprint, jsonify, Response
from datetime import datetime
import yfinance as yf
import json

data_blueprint = Blueprint("data", __name__, url_prefix="/data")

current_time = datetime.now()
START_TIME = datetime(month=1, day=1, year=current_time.year)
END_TIME = current_time
SYMBOL = "AAPL"


@data_blueprint.route(
    "/ticker",
    defaults={"symbol": SYMBOL, "start_time": START_TIME, "end_time": END_TIME},
)
@data_blueprint.route("/ticker/<string:symbol>/range/<start_time>/<end_time>")
def get_stock_prices(symbol: str, start_time: str, end_time: str) -> Response:
    df = (
        yf.Ticker(symbol)
        .history(start=start_time, end=end_time, interval="1h")
        .reset_index()  # pop out datetime index
        .to_json(orient="records", date_format="iso")
    )  # make each row a json object

    dict_json = json.loads(df)  # convert json to python dictionary

    return jsonify(dict_json)
