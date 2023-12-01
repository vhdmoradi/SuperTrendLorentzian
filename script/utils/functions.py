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
            + str(225)
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


def entry_signal(spt_data, lc_data, symbol):
    # Entry signal:
    # For enrtry signal, we will also consider the last candle that has not yet been closed
    # For entry signal: for long positions -> first the spt_signal must change from 0 to 1
    # Then at the same canlde, or one before, or one after, the lc_data.df["isBearishChange"] should also be true.

    # for short positions -> first the spt_signal must change from 1 to 0
    # Then at the same canlde, or one before, or one after, the lc_data.df["isBullishChange"] should also be true.

    spt_signal_long = False
    lc_signal_long = False
    spt_signal_short = False
    lc_signal_short = False

    spt_signal_long = (
        spt_data["os"].iloc[-1] > spt_data["os"].iloc[-2]
        or spt_data["os"].iloc[-2] > spt_data["os"].iloc[-3]
    )
    lc_signal_long = (
        lc_data["isBearishChange"].iloc[-1] or lc_data["isBearishChange"].iloc[-2]
    )

    spt_signal_short = (
        spt_data["os"].iloc[-1] < spt_data["os"].iloc[-2]
        or spt_data["os"].iloc[-2] < spt_data["os"].iloc[-3]
    )
    lc_signal_short = (
        lc_data["isBullishChange"].iloc[-1] or lc_data["isBullishChange"].iloc[-2]
    )

    # For generating a long signal, both conditions must be True
    strategy_long_signal = spt_signal_long and lc_signal_long

    # For generating a short signal, both conditions must be True
    strategy_short_signal = spt_signal_short and lc_signal_short

    longshort = None
    if strategy_long_signal:
        longshort == "long"
    elif strategy_short_signal:
        longshort = "short"

    # If no signal, just return
    if not strategy_long_signal and not strategy_short_signal:
        return 0, 0, longshort

    # If there is a signal, first it must be saved in the db
    timeframe = lc_data["timeframe"].iloc[-1]
    created_at = lc_data["timestamp"].iloc[-1] / 1000
    exchange = "Binance Futures"
    entryexit = "entry"
    entry_price = lc_data["close"].iloc[-1]
    exit_price = 0.99 * entry_price
    db_connection, db_cursor = db_connect("main_db")
    try:
        insert_signal_query = "INSERT INTO public.alerts(timeframe, TO_TIMESTAMP(created_at), exchange, entryexit, entry_price, exit_price, symbol) VALUES (%s, %s, %s, %s, %s, %s, %s)"
        values = (
            timeframe,
            created_at,
            exchange,
            entryexit,
            entry_price,
            exit_price,
            symbol,
        )
        db_cursor.execute(insert_signal_query, values)
    except Exception as e:
        log_error(e, "saving entry signal into db")
    finally:
        db_cursor.close()
        db_connection.close()

    return 1, created_at


def exit_signal(lc_data, exit_price, symbol, longshort):
    if longshort == "long":
        # exit signal:
        # for long positions -> for stop loss signal, we will monitor the price to be above 99% of entry_price. (so entry_price - 0.01 * entry_price will be our stop loss)
        # for long positions -> for tp target signal, we will monitor the lc.df["isBearishChange"] value; if True, we will produce the tp signal.
        sl = tp = False
        # stop loss signal:
        if lc_data["close"].iloc[-2] < exit_price:
            sl = True

        # take profit signal
        if lc_data["isBearishChange"].iloc[-2]:
            tp = True

        # if no signal, just return
        if not (sl or tp):
            return 0, 0, 0

        created_at = lc_data["timestamp"].iloc[-1] / 1000
        exit_type = "sl" if sl else ("tp" if tp else None)

        return 1, exit_type, created_at
    elif longshort == "short":
        # exit signal:
        # for short positions -> for stop loss signal, we will monitor the price to be above 99% of entry_price. (so entry_price + 0.01 * entry_price will be our stop loss)
        # for short positions -> for tp target signal, we will monitor the lc.df["isBullishRate"] value; if True, we will produce the tp signal.

        sl = tp = False
        # stop loss signal:
        if lc_data["close"].iloc[-2] > exit_price:
            sl = True

        # take profit signal
        if lc_data["isBullishRate"].iloc[-2]:
            tp = True

        # if no signal, just return
        if not (sl or tp):
            return 0, 0, 0

        created_at = lc_data["timestamp"].iloc[-1] / 1000
        exit_type = "sl" if sl else ("tp" if tp else None)

        return 1, exit_type, created_at
    return 0, 0, 0


def exit_short_signal(lc_data, exit_price, symbol):
    # exit signal:
    # for short positions -> for stop loss signal, we will monitor the price to be above 99% of entry_price. (so entry_price + 0.01 * entry_price will be our stop loss)
    # for short positions -> for tp target signal, we will monitor the lc.df["isBullishRate"] value; if True, we will produce the tp signal.

    sl = tp = False
    # stop loss signal:
    if lc_data["close"].iloc[-2] >= exit_price:
        sl = True

    # take profit signal
    if lc_data["isBullishRate"].iloc[-2]:
        tp = True

    # if no signal, just return
    if not (sl or tp):
        return 0, 0, 0

    created_at = lc_data["timestamp"].iloc[-1] / 1000
    exit_type = "sl" if sl else ("tp" if tp else None)

    return 1, exit_type, created_at


def insert_signal_db(
    timeframe,
    created_at,
    exchange,
    entryexit,
    entry_price,
    exit_price,
    exit_type,
    symbol,
    message_id,
):
    if len(str(created_at)) == 13:
        created_at = created_at // 1000

    db_connection, db_cursor = db_connect("main_db")
    try:
        insert_signal_query = "INSERT INTO public.alerts(timeframe, TO_TIMESTAMP(created_at), exchange, entryexit, entry_price, exit_price, exit_type, symbol, message_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
        values = (
            timeframe,
            created_at,
            exchange,
            entryexit,
            entry_price,
            exit_price,
            exit_type,
            symbol,
            message_id,
        )
        db_cursor.execute(insert_signal_query, values)
    except Exception as e:
        log_error(e, "saving signal into db")
    finally:
        db_cursor.close()
        db_connection.close()
