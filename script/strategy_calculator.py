from script.utils import functions
from script.telegram_bot import send_message
import asyncio
from indicators.supertrend_luxalgo import supertrend_ai
import os
from dotenv import load_dotenv
import talib as ta
from indicators.LorentzianClassification.Classifier import LorentzianClassification
import multiprocessing
import time
import concurrent.futures

load_dotenv()
loop = asyncio.get_event_loop()


async def main():
    # First we start by getting prices
    prices = await functions.get_prices()

    binance_symbols = os.getenv("SYMBOLS").split(",")
    binance_symbols = [symbol + "USDT" for symbol in binance_symbols]
    spt_args = []
    lc_args = []

    for symbol_pair in binance_symbols:
        # We separate prices for each symbol
        prices_for_symbol = prices.query("symbol == @symbol_pair").reset_index(
            drop=True
        )
        spt_arg_dic = {
            "prices": prices_for_symbol,
            "length": os.getenv("ST_ATR_LENGTH"),
            "minMult": os.getenv("ST_MIN_MULT"),
            "maxMult": os.getenv("ST_MAX_MULT"),
            "step": os.getenv("ST_STEP"),
            "perfAlpha": os.getenv("ST_PERFORMANCE_MEMORY"),
            "fromCluster": os.getenv("ST_FROM_CLUSTER"),
            "maxIter": os.getenv("ST_MAX_ITER"),
            "maxData": os.getenv("ST_MAX_DATA"),
        }
        lc_args.append(prices_for_symbol)
        spt_args.append(spt_arg_dic)

    start_time = time.time()
    print("Process started...")

    # We use ProcessPoolExecutor for parallel processing
    with concurrent.futures.ProcessPoolExecutor() as executor:
        spt_results = [executor.submit(supertrend_ai, arg) for arg in spt_args]
        lc_results = [executor.submit(LorentzianClassification, arg) for arg in lc_args]

    spt_result_dic = {}
    # Collect results for supertrend_ai
    for result in concurrent.futures.as_completed(spt_results):
        spt_result = result.result()
        symbol = spt_result["symbol"].iloc[-1]
        spt_result_dic[symbol] = spt_result
    lc_result_dic = {}
    # Collect results for LorentzianClassification
    for result in concurrent.futures.as_completed(lc_results):
        lc_result = result.result()
        symbol = lc_result.df["symbol"].iloc[-1]
        lc_result_dic[symbol] = lc_result

    # Now that we have the results for both indicators, we can search for signals
    # We call entry_long_signal() function to give use the result of true, and timestamp, or 0, 0
    # But before calling the entry_buy_function, we should only send those symbols that does not have an existing entry signal, without a corresponding exit signal.
    # getting alarms from db:
    db_connection, db_cursor = functions.db_connect("main_db")

    try:
        alert_query = """
            SELECT entryexit, symbol, exit_price, signal_message_id, longshort, entry_price
            FROM public.alerts
            WHERE (symbol, created_at) IN (
                SELECT symbol, MAX(created_at) AS max_created_at
                FROM public.alerts
                GROUP BY symbol
            )"""

        # Execute the query
        db_cursor.execute(alert_query)

        # Fetch all results
        results = db_cursor.fetchall()

        # Create a dictionary with symbols as keys and a dictionary of entryexit and exit_price as values
        latest_alerts = {
            symbol: {
                "entryexit": entryexit,
                "exit_price": exit_price,
                "signal_message_id": signal_message_id,
                "longshort": longshort,
                "entry_price": entry_price,
            }
            for entryexit, symbol, exit_price, signal_message_id, longshort, entry_price in results
        }

    except Exception as e:
        functions.log_error(e, "getting latest signals for symbols")

    finally:
        # Close the database connection
        db_cursor.close()
        db_connection.close()

    # Now, we will only consider those symbols whose latest alarm is 'exit'
    for symbol_pair in binance_symbols:
        spt_result_for_symbol = spt_result_dic[symbol_pair]
        lc_result_for_symbol = lc_result_dic[symbol_pair]

        # We will check the symbols with the latest entryexit alarm value with 'entry' for exit signals from exit_long_signal() function
        # for exit strategy, we either need the lorentzien line to change direction (tp) or the price to be less than 99% of the entry price

        if (
            symbol_pair in latest_alerts
            and latest_alerts[symbol_pair]["entryexit"] == "entry"
        ):
            entry_price = latest_alerts[symbol_pair]["entry_price"]
            exit_price = latest_alerts[symbol_pair]["exit_price"]
            longshort = latest_alerts[symbol_pair]["longshort"]
            is_signal, exit_type, timestamp = functions.exit_signal(
                lc_result_for_symbol.df.tail(10), exit_price, symbol_pair, longshort
            )
            price_diff = abs(entry_price - exit_price) / exit_price * 100
            # if there is a signal, first send the telegram message
            # it is an exit signal, so first we should retrieve the signal_message_id of the entry signal
            if is_signal:
                entry_signal_message_id = (
                    latest_alerts[symbol_pair]["signal_message_id"] or None
                )
                singal_message_id = loop.run_until_complete(
                    send_message.send_telegram_message(
                        signal_time=timestamp,
                        entryexit="exit",
                        symbol=symbol_pair,
                        signal_message_id=entry_signal_message_id,
                        longshort=longshort,
                        exit_type=exit_type,
                        price_diff=price_diff,
                    )
                )

                # after sending message and getting singal_message_id, now we enter the signal in db
                functions.insert_signal_db(
                    timeframe=lc_result_for_symbol.df["timeframe"].iloc[-1],
                    created_at=lc_result_for_symbol.df["timestamp"].iloc[-1],
                    exchange="Binance Futures",
                    entryexit="exit",
                    entry_price=None,
                    exit_price=lc_result_for_symbol.df["close"].iloc[-2],
                    exit_type=exit_type,
                    symbol=symbol_pair,
                    message_id=singal_message_id,
                )

            continue
        is_signal, timestamp, longshort = functions.entry_signal(
            spt_result_for_symbol.tail(10),
            lc_result_for_symbol.df.tail(10),
            symbol_pair,
        )
        # If the is_signal is true, we will send the signal to telegram bot
        if is_signal:
            entry_price = lc_result_for_symbol.df["close"].iloc[-1]
            exit_price = (
                0.99 * entry_price if longshort == "long" else 1.01 * entry_price
            )
            singal_message_id = loop.run_until_complete(
                send_message.send_telegram_message(
                    signal_time=timestamp,
                    entryexit="exit",
                    symbol=symbol_pair,
                    longshort=longshort,
                    exit_price=exit_price,
                )
            )
            # after sending message and getting singal_message_id, now we enter the signal in db

            functions.insert_signal_db(
                timeframe=lc_result_for_symbol.df["timeframe"].iloc[-1],
                created_at=lc_result_for_symbol.df["timestamp"].iloc[-1],
                exchange="Binance Futures",
                entryexit="entry",
                entry_price=entry_price,
                exit_price=exit_price,
                longshort=longshort,
                symbol=symbol_pair,
                message_id=singal_message_id,
            )
    duration = time.time() - start_time
    print(f"Duration of supertrend_luxalgo: {duration} seconds")


if __name__ == "__main__":
    asyncio.run(main())
