from script.utils import functions
from indicators.supertrend_luxalgo import supertrend_ai
import os
import asyncio
from dotenv import load_dotenv
import pandas_ta as ta
from script.telegram_bot import send_message
import pandas as pd

load_dotenv()
pd.set_option("display.max_columns", None)


async def main():
    prices = await functions.get_prices()
    symbol_pair = "BTCUSDT"
    prices_for_symbol = prices.query("symbol == @symbol_pair").reset_index(drop=True)

    ema = ta.ema(close=prices_for_symbol["close"], length=9)
    print(ema)
    # spt_result = functions.supertrend(prices=prices, length=111, factor=5)
    # print(spt_result["supertrend"])


if __name__ == "__main__":
    asyncio.run(main())
