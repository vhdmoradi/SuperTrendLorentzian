import os
import psycopg2
from datetime import datetime
import json
from dotenv import load_dotenv
import requests
import json
from multiprocessing.dummy import Pool as ThreadPool
from itertools import chain
import pandas as pd
from requests.adapters import HTTPAdapter
import time
import asyncio
import aiohttp

load_dotenv()
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=2000, pool_maxsize=2000)
session.mount("https://", adapter)


def db_connect(db_name):
    if db_name == "main_db":
        try:
            db_connection = psycopg2.connect(
                database=os.getenv("MAIN_DB_NAME"),
                user=os.getenv("MAIN_DB_USER"),
                password=os.getenv("MAIN_DB_PASS"),
                host=os.getenv("MAIN_DB_SERVER"),
                port=os.getenv("MAIN_DB_PORT"),
            )

            db_connection.autocommit = True
            db_cursor = db_connection.cursor()
        except Exception as e:
            print(e)

    return db_connection, db_cursor


def log_error(error_obj, error_from):
    # first connect to the db:
    db_connection, db_cursor = db_connect("main_db")
    try:
        # rest of the function
        description = str(error_obj)
        current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        query = "INSERT INTO public.error_log (error_from, error_text, created_at) VALUES (%s, %s, %s)"
        values = (error_from, description, current_timestamp)
        db_cursor.execute(query, values)

    except Exception as e:
        error_message = str(e)
        current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        query = "INSERT INTO public.error_log (error_from, description, created_at) VALUES (%s, %s, %s)"
        values = (error_from, error_message, current_timestamp)
        print(f"query: {query}")
        db_cursor.execute(query, values)

    finally:
        db_cursor.close()
        db_connection.close()


async def get_market_data(session, args):
    if args["exchange"].lower() == "binance futures":
        url = (
            "https://fapi.binance.com/fapi/v1/klines/?symbol="
            + args["symbol"]
            + "&interval="
            + args["timeframe"]
            + "&limit="
            + str(1500)
        )

        async with session.get(url) as response:
            soup = ""
            if response.status == 502:
                return soup

            if response.status != 200:
                try:
                    retry = 1
                    while (
                        response.status != 200 and response.status != 404 and retry > 0
                    ):
                        async with session.get(url=url) as response:
                            retry -= 1
                except Exception as e:
                    log_error(e, "price url response in request handler")

            if response.status == 200:
                soup = await response.json()
                try:
                    for i, _ in enumerate(soup):
                        if type(soup[i]) is list:
                            soup[i] = soup[i][0:6]
                            soup[i].extend(
                                [args["symbol"], args["exchange"], args["timeframe"]]
                            )
                        else:
                            print(soup, url)
                except Exception as e:
                    log_error(e, "getting symbol info in request handler")
                    soup = []

        return soup


async def get_prices():
    async with aiohttp.ClientSession() as session:
        prices_df = pd.DataFrame(
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "symbol",
                "exchange",
                "timeframe",
            ]
        )

        try:
            timeframes = os.getenv("TIMEFRAMES").split(",")
            binance_symbols = os.getenv("SYMBOLS").split(",")
            binance_symbols = [symbol + "USDT" for symbol in binance_symbols]
        except Exception as e:
            log_error(e, "getting env variables in main")

        args = []
        for symbol in binance_symbols:
            for timeframe in timeframes:
                args.append(
                    {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "exchange": "binance futures",
                    }
                )

        tasks = [get_market_data(session, arg) for arg in args]
        binance_prices = await asyncio.gather(*tasks)

        # Convert the results to a DataFrame
        prices_df = pd.DataFrame(
            list(chain(*binance_prices)),
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "symbol",
                "exchange",
                "timeframe",
            ],
        )
        prices_df = prices_df.astype(
            {
                "open": "float64",
                "high": "float64",
                "low": "float64",
                "close": "float64",
                "volume": "float64",
            }
        )
    # # Group by timeframe and sort within each group by timestamp
    # grouped = prices_df.groupby("timeframe")
    # # Sort the data within each group by timestamp (ascending order)
    # prices_cleaned = grouped.apply(lambda x: x.sort_values(by="timestamp")).reset_index(
    #     drop=True
    # )
    # # Drop the last row from each group
    # prices_df = (
    #     prices_cleaned.groupby("timeframe")
    #     .apply(lambda x: x.iloc[:-1])
    #     .reset_index(drop=True)
    # )
    return prices_df


# asyncio.run(get_prices())
