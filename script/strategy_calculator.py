from script.utils import functions
import asyncio
from indicators.supertrend_luxalgo import supertrend_ai
import os
from dotenv import load_dotenv
import talib as ta
from indicators.LorentzianClassification.Classifier import LorentzianClassification
import multiprocessing
import time

load_dotenv()


# entry signal:
# for enrtry signal, we will also consider the last candle that has not yet been closed
# for entry signal, first the spt_signal must change from 0 to 1
# then at the same canlde, or one before, or one after, the lc.df["isNewBuySignal"] should also be true.
# after this, we will save the signal in db, and send the signal in telegram to users
# the close price of the candle in the moment of signal, will be considered as the entry_price
# exit signal:
# for stop loss signal, we will monitor the price to be above 99% of entry_price. (so entry_price -0.01 * entry_price will be our stop loss)
# for tp target signal, we will monitor the lc.df["isBearishChange"] value; if True, we will produce the tp signal.

# first we get the prices df for all the symbols
prices = asyncio.run(functions.get_prices())

# then we should separate the prices based on the symbol (assuming there would be only one timeframe, 1h)
binance_symbols = os.getenv("SYMBOLS").split(",")
binance_symbols = [symbol + "USDT" for symbol in binance_symbols]
args = []
for symbol_pair in binance_symbols:
    prices_for_symbol = prices.query("symbol == @symbol_pair").reset_index(drop=True)
    arg_dic = {
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
    args.append(arg_dic)
start_time = time.time()
print("Process started...")
with multiprocessing.Pool() as pool:
    pool.map(supertrend_ai, args)
duration = time.time() - start_time
print(f"Duration {duration} seconds")
# spt_signals = supertrend_ai(
#     prices=prices,
#     length=os.getenv("ST_ATR_LENGTH"),
#     minMult=os.getenv("ST_MIN_MULT"),
#     maxMult=os.getenv("ST_MAX_MULT"),
#     step=os.getenv("ST_STEP"),
#     perfAlpha=os.getenv("ST_PERFORMANCE_MEMORY"),
#     fromCluster=os.getenv("ST_FROM_CLUSTER"),
#     maxIter=os.getenv("ST_MAX_ITER"),
#     maxData=os.getenv("ST_MAX_DATA"),
# )


# print(spt_signals)

# lc = LorentzianClassification(prices)
# print(lc.df[["isBearishChange", "timestamp", "isNewBuySignal", "symbol"]])


# def calculate_signals(prices,spt_signals, lc.df):
#     # first stage: get the last signal for this
#     pass
