from flask import Blueprint, jsonify, Response
from datetime import date
from dateutil.relativedelta import relativedelta
import yfinance as yf
import json
import os
from dotenv import load_dotenv

load_dotenv(override=True)  # load environment variables

data_blueprint = Blueprint("data", __name__, url_prefix="/data")

# TODO: Apply financial time correction
current_date = date.today()
START_DATE = date(month=1, day=1, year=current_date.year)
END_DATE = current_date + relativedelta(days=1)

SYMBOL = os.getenv("SYMBOL")


@data_blueprint.route(
    "/ticker",
    defaults={"symbol": SYMBOL, "start_date": START_DATE, "end_date": END_DATE},
)
@data_blueprint.route("/ticker/<string:symbol>/range/<start_date>/<end_date>")
def get_stock_prices(symbol: str, start_date: str, end_date: str) -> Response:
    df = (
        yf.Ticker(symbol)
        .history(start=start_date, end=end_date, interval="1d")
        .reset_index()  # pop out datetime index
        .to_json(orient="records", date_format="iso")
    )  # make each row a json object

    dict_json = json.loads(df)  # convert json to python dictionary

    return jsonify(dict_json)
