from script.utils import functions
import asyncio
from indicators.supertrend_luxalgo import supertrend_ai
import os
from dotenv import load_dotenv
import talib as ta
from indicators.LorentzianClassification.Classifier import LorentzianClassification

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

prices = asyncio.run(functions.get_prices())

spt_signals = supertrend_ai(
    prices=prices,
    length=os.getenv("ST_ATR_LENGTH"),
    minMult=os.getenv("ST_MIN_MULT"),
    maxMult=os.getenv("ST_MAX_MULT"),
    step=os.getenv("ST_STEP"),
    perfAlpha=os.getenv("ST_PERFORMANCE_MEMORY"),
    fromCluster=os.getenv("ST_FROM_CLUSTER"),
    maxIter=os.getenv("ST_MAX_ITER"),
    maxData=os.getenv("ST_MAX_DATA"),
)
# print(spt_signals)

# lc = LorentzianClassification(prices)
# print(lc.df[["isBearishChange", "timestamp", "isNewBuySignal", "symbol"]])


# def calculate_signals(prices,spt_signals, lc.df):
#     # first stage: get the last signal for this
#     pass
