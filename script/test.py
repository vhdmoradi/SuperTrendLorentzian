from script.utils import functions
from indicators.supertrend_luxalgo import supertrend_ai
import os
import asyncio
from dotenv import load_dotenv
import pandas_ta as ta
from script.telegram_bot import send_message

load_dotenv()


async def main():
    object = send_message.send_telegram_message(
        signal_time=1,
        entryexit="exit",
        symbol="btc",
        longshort="short",
        exit_price=10,
    )
    prices = await functions.get_prices()
    symbol_pair = "BTCUSDT"
    prices_for_symbol = prices.query("symbol == @symbol_pair").reset_index(drop=True)
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
    atr = ta.atr(
        high=prices["high"], low=prices["low"], close=prices["close"], length=111
    )
    print(atr)
    # spt_result = functions.supertrend(prices=prices, length=111, factor=5)
    # print(spt_result["supertrend"])


if __name__ == "__main__":
    asyncio.run(main())
