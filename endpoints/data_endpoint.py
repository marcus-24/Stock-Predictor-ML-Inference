from flask import Blueprint, jsonify, Response
import yfinance as yf
import json

data_blueprint  = Blueprint('data', __name__, url_prefix='/data')

@data_blueprint.route('/ticker/<ticker>/range/<start_date>/<end_date>')
def get_stock_prices(ticker: str, 
                     start_date: str,
                     end_date: str) -> Response:
    df = (yf.Ticker(ticker)
            .history(start=start_date, 
                     end=end_date)
            .reset_index())
    
    df_json = df.to_json(orient='records')  # make each row a json object
    dict_json = json.loads(df_json)  # convert json to python dictionary

    return jsonify(dict_json)

