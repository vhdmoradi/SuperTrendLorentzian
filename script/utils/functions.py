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
import pandas_ta as ta
import numpy as np
import pandas as pd
from numpy import nan as npNaN

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
    lc_signal_long = not np.isnan(lc_data["startLongTrade"].iloc[-1]) or not np.isnan(
        lc_data["startLongTrade"].iloc[-2]
    )
    # lc_signal_long = not np.isnan(lc_data["startLongTrade"].iloc[-1])

    spt_signal_short = (
        spt_data["os"].iloc[-1] < spt_data["os"].iloc[-2]
        or spt_data["os"].iloc[-2] < spt_data["os"].iloc[-3]
    )
    lc_signal_short = not np.isnan(lc_data["startShortTrade"].iloc[-1]) or not np.isnan(
        lc_data["startShortTrade"].iloc[-2]
    )

    # lc_signal_short = not np.isnan(lc_data["startShortTrade"].iloc[-1])

    # For generating a long signal, both conditions must be True
    strategy_long_signal = spt_signal_long and lc_signal_long
    # strategy_long_signal = lc_signal_long

    # For generating a short signal, both conditions must be True
    strategy_short_signal = spt_signal_short and lc_signal_short
    # strategy_short_signal = lc_signal_short

    longshort = None
    if strategy_long_signal:
        longshort = "long"
    elif strategy_short_signal:
        longshort = "short"

    # If no signal, just return
    if not strategy_long_signal and not strategy_short_signal:
        return 0, 0, longshort

    created_at = int(lc_data["timestamp"].iloc[-1] / 1000)

    return 1, created_at, longshort


def exit_signal(lc_data, exit_price, symbol, longshort):
    if longshort == "long":
        # exit signal:
        # for long positions -> for stop loss signal, we will monitor the price to be above 99% of entry_price. (so entry_price - 0.01 * entry_price will be our stop loss)
        # for long positions -> for tp target signal, we will monitor the lc.df["isBearishChange"] value; if True, we will produce the tp signal.
        sl = tp = False
        # stop loss signal:
        if lc_data["close"].iloc[-1] < exit_price:
            sl = True

        # take profit signal
        if lc_data["signal"].iloc[-2] == -1 and lc_data["signal"].iloc[-3] == 1:
            tp = True

        # if no signal, just return
        if not (sl or tp):
            return 0, 0, 0, 0

        created_at = lc_data["timestamp"].iloc[-1] / 1000
        exit_price_from_signal = lc_data["close"].iloc[-2]
        sltp = "sl" if sl else ("tp" if tp else None)

        return 1, sltp, created_at, exit_price_from_signal
    elif longshort == "short":
        # exit signal:
        # for short positions -> for stop loss signal, we will monitor the price to be above 99% of entry_price. (so entry_price + 0.01 * entry_price will be our stop loss)
        # for short positions -> for tp target signal, we will monitor the lc.df["isBullishRate"] value; if True, we will produce the tp signal.

        sl = tp = False
        # stop loss signal:
        if lc_data["close"].iloc[-2] > exit_price:
            sl = True

        # take profit signal
        if lc_data["signal"].iloc[-2] == 1 and lc_data["signal"].iloc[-3] == -1:
            tp = True

        # if no signal, just return
        if not (sl or tp):
            return 0, 0, 0, 0

        created_at = lc_data["timestamp"].iloc[-1] / 1000
        exit_price_from_signal = lc_data["close"].iloc[-2]

        sltp = "sl" if sl else ("tp" if tp else None)

        return 1, sltp, created_at, exit_price_from_signal
    return 0, 0, 0, 0


def insert_signal_db(
    timeframe,
    created_at,
    exchange,
    entryexit,
    entry_price,
    exit_price,
    symbol,
    signal_message_id,
    longshort,
    sltp=None,
):
    if len(str(created_at)) == 13:
        created_at = int(created_at // 1000)

    db_connection, db_cursor = db_connect("main_db")
    try:
        insert_signal_query = "INSERT INTO public.alerts(timeframe, created_at, exchange, entryexit, entry_price, exit_price, sltp, symbol, signal_message_id, longshort) VALUES (%s, TO_TIMESTAMP(%s), %s, %s, %s, %s, %s, %s, %s, %s)"
        values = (
            timeframe,
            created_at,
            exchange,
            entryexit,
            entry_price,
            exit_price,
            sltp,
            symbol,
            signal_message_id,
            longshort,
        )
        db_cursor.execute(insert_signal_query, values)
    except Exception as e:
        log_error(e, "saving signal into db")
    finally:
        db_cursor.close()
        db_connection.close()


def supertrend(prices, length, factor):
    multiplier = factor
    m = prices["close"].size
    dir_, trend = [1] * m, [0] * m
    long, short = [npNaN] * m, [npNaN] * m
    hl2_ = (prices["high"] + prices["low"]) / 2
    matr = multiplier * atr(prices=prices, length=length)
    upperband = hl2_ + matr
    lowerband = hl2_ - matr
    for i in range(1, m):
        if prices["close"].iloc[i] > upperband.iloc[i - 1]:
            dir_[i] = 1
        elif prices["close"].iloc[i] < lowerband.iloc[i - 1]:
            dir_[i] = -1
        else:
            dir_[i] = dir_[i - 1]
            if dir_[i] > 0 and lowerband.iloc[i] < lowerband.iloc[i - 1]:
                lowerband.iloc[i] = lowerband.iloc[i - 1]
            if dir_[i] < 0 and upperband.iloc[i] > upperband.iloc[i - 1]:
                upperband.iloc[i] = upperband.iloc[i - 1]
        if dir_[i] > 0:
            trend[i] = long[i] = lowerband.iloc[i]
        else:
            trend[i] = short[i] = upperband.iloc[i]

    df = pd.DataFrame(
        {
            "supertrend": trend,
            "direction": dir_,
            "upper": long,
            "lower": short,
        },
        index=prices["close"].index,
    )
    return df


def atr(prices, length=14, smoothing="RMA"):
    """
    Calculates the Average True Range (ATR) indicator for a Pandas DataFrame.

    Args:
      prices: A Pandas DataFrame with columns: "high", "low", "close".
      length: The number of periods for the rolling mean (default=14).
      smoothing: The method for smoothing (default="SMA").
                       Options: "RMA", "SMA", "EMA".

    Returns:
      A Pandas Series with the ATR values.
    """

    # Check if necessary columns exist and are not empty
    if not all(col in prices.columns for col in ("high", "low", "close")):
        raise ValueError(
            "Required columns (high, low, close) missing from the DataFrame"
        )
    if prices["high"].empty or prices["low"].empty or prices["close"].empty:
        raise ValueError("One or more columns (high, low, close) are empty")

    # Calculate the True Range (TR)
    tr = pd.DataFrame(index=prices.index)
    tr["h_l"] = prices["high"] - prices["low"]
    tr["h_pc"] = abs(prices["high"] - prices["close"].shift(1))
    tr["l_pc"] = abs(prices["low"] - prices["close"].shift(1))
    tr["true_range"] = tr.max(axis=1)

    # Calculate the ATR using the specified smoothing method
    if smoothing == "RMA":
        atr = rma(tr[["true_range"]], length=length)
    elif smoothing == "SMA":
        atr = tr["true_range"].rolling(window=length, min_periods=1).mean()
    elif smoothing == "EMA":
        atr = tr["true_range"].ewm(span=length, min_periods=1, adjust=True).mean()

    else:
        raise ValueError("Invalid smoothing method. Options: RMA, SMA, EMA, WMA")

    return atr


def rma(df, length=14):
    """
    Calculates the Recursive Moving Average (RMA) for a Pandas DataFrame with a single column.

    Args:
      df: A Pandas DataFrame with a single column to calculate the RMA.
      length: The number of periods for the RMA (default=14).

    Returns:
      A Pandas Series with the RMA values.
    """

    # Check if the DataFrame has a single column
    if len(df.columns) != 1:
        raise ValueError("DataFrame must have exactly one column for RMA calculation")
    column = df.columns[0]

    # Check if the necessary column is not empty
    if df[column].empty:
        raise ValueError(f"{column} column is empty")

    # Calculate the RMA using vectorized operations
    alpha = 1 / length
    rma_values = np.zeros(len(df))

    for i in range(len(df)):
        if i == 0:
            rma_values[i] = df[column].iloc[i]
        else:
            rma_values[i] = alpha * df[column].iloc[i] + (1 - alpha) * rma_values[i - 1]

    rma = pd.Series(rma_values, index=df.index, name=column + "_RMA")

    return rma
