import requests
import json
from . import functions


session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=2000, pool_maxsize=2000)
session.mount("https://", adapter)


def send_request(exchange, pair, timeframe):
    if exchange.lower() == "binance spot":
        url = (
            "https://api.binance.com/api/v3/klines?symbol="
            + pair
            + "&interval="
            + timeframe
            + "&limit="
            + str(20)
        )

        response = session.get(url)
        soup = ""

        if response.status_code == 502:
            return soup

        if response.status_code != 200:
            try:
                retry = 1
                while (
                    response.status_code != 200
                    and response.status_code != 404
                    and retry > 0
                ):
                    response = session.get(url=url)
                    retry -= 1
            except Exception as e:
                functions.log_error(e, "price url response in request handler")

        if response.status_code == 200:
            soup = json.loads(response.text)
            try:
                for i, _ in enumerate(soup):
                    if type(soup[i]) is list:
                        soup[i] = soup[i][0:6]
                        soup[i].extend([pair, exchange, timeframe])
                    else:
                        print(soup, url)
            except Exception as e:
                functions.log_error(e, "getting symbol info in request handler")
                soup = []
            return soup

        return soup


def getMarketData(args):
    r = requests.get(
        url="https://api.binance.com/api/v3/klines?symbol="
        + args["symbol"]
        + "&interval="
        + args["timeframe"]
        + "&limit="
        + str(50)
    )
    json = r.json()
    try:
        for i, _ in enumerate(json):
            if type(json[i]) is list:
                json[i] = json[i][0:6]
                json[i].extend([args["symbol"], args["exchange"], args["timeframe"]])
            else:
                print(
                    json,
                    "https://api.binance.com/api/v3/klines?symbol="
                    + args["symbol"]
                    + "&interval="
                    + args["timeframe"],
                )

    except Exception as e:
        print(e)
        print("can't get symbol info:", args["symbol"])
        json = []

    return json
