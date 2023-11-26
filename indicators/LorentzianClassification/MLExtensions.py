import math
import numpy as np
import pandas as pd
import talib as ta
from sklearn.preprocessing import MinMaxScaler


# ==========================
# ==== Helper Functions ====
# ==========================


# @function Rescales a source value with an unbounded range to a target range.
# @param src <series float> The input series
# @param min <float> The minimum value of the unbounded range
# @param max <float> The maximum value of the unbounded range
# @returns <series float> The normalized series
def normalize(src: pd.Series, min, max) -> pd.Series:
    scaler = MinMaxScaler(feature_range=(min, max)).set_output(transform="pandas")
    return min + (max - min) * scaler.fit_transform(pd.DataFrame({'data': src}))['data']


# @function Rescales a source value with a bounded range to anther bounded range
# @param src <series float> The input series
# @param oldMin <float> The minimum value of the range to rescale from
# @param oldMax <float> The maximum value of the range to rescale from
# @param newMin <float> The minimum value of the range to rescale to
# @param newMax <float> The maximum value of the range to rescale to 
# @returns <series float> The rescaled series
def rescale(src: pd.Series, old_min, old_max, new_min, new_max) -> pd.Series:
    rescaled_value = new_min + (new_max - new_min) * (src - old_min) / max(old_max - old_min, 10e-10)
    return rescaled_value


def RMA(df: pd.Series, len: int) -> pd.Series:
    rma = df.copy()
    rma.iloc[:len] = rma.rolling(len).mean().iloc[:len]
    rma = rma.ewm(alpha=(1.0/len),adjust=False).mean()
    return rma


# @function Returns the normalized RSI ideal for use in ML algorithms.
# @param src <series float> The input series (i.e., the result of the RSI calculation).
# @param n1 <int> The length of the RSI.
# @param n2 <int> The smoothing length of the RSI.
# @returns signal <series float> The normalized RSI.
def n_rsi(src: pd.Series, n1, n2) -> pd.Series:
    return rescale(ta.RSI(src, n1), 0, 100, 0, 1)


# @function Returns the normalized CCI ideal for use in ML algorithms.
# @param src <series float> The input series (i.e., the result of the CCI calculation).
# @param n1 <int> The length of the CCI.
# @param n2 <int> The smoothing length of the CCI.
# @returns signal <series float> The normalized CCI.
def n_cci(highSrc: pd.Series, lowSrc: pd.Series, closeSrc: pd.Series, n1, n2) -> pd.Series:
    return normalize(ta.CCI(highSrc, lowSrc, closeSrc, n1), 0, 1)


# @function Returns the normalized WaveTrend Classic series ideal for use in ML algorithms.
# @param src <series float> The input series (i.e., the result of the WaveTrend Classic calculation).
# @param paramA <int> The first smoothing length for WaveTrend Classic.
# @param paramB <int> The second smoothing length for the WaveTrend Classic.
# @param transformLength <int> The length of the transform.
# @returns signal <series float> The normalized WaveTrend Classic series.
def n_wt(src: pd.Series, n1=10, n2=11) -> pd.Series:
    ema1 = ta.EMA(src, n1)
    ema2 = ta.EMA(abs(src - ema1), n1)
    ci = (src - ema1) / (0.015 * ema2)
    wt1 = ta.EMA(ci, n2)  # tci
    wt2 = ta.SMA(wt1, 4)
    return normalize(wt1 - wt2, 0, 1)


# @function Returns the normalized ADX ideal for use in ML algorithms.
# @param highSrc <series float> The input series for the high price.
# @param lowSrc <series float> The input series for the low price.
# @param closeSrc <series float> The input series for the close price.
# @param n1 <int> The length of the ADX.
def n_adx(highSrc: pd.Series, lowSrc: pd.Series, closeSrc: pd.Series, n1) -> pd.Series:
    return rescale(ta.ADX(highSrc, lowSrc, closeSrc, n1), 0, 100, 0, 1)
    # TODO: Replicate ADX logic from jdehorty


# =================
# ==== Filters ====
# =================

# @regime_filter
# @param src <series float> The source series.
# @param threshold <float> The threshold.
# @param useRegimeFilter <bool> Whether to use the regime filter.
# @returns <bool> Boolean indicating whether or not to let the signal pass through the filter.
def regime_filter(src: pd.Series, high: pd.Series, low: pd.Series, useRegimeFilter, threshold):
    if not useRegimeFilter: return pd.Series(True, index=src.index)
    value1 = [0.0] * src.index.size
    value2 = [0.0] * src.index.size
    klmf = [0.0] * src.index.size
    absCurveSlope = pd.Series([0.0])
    filter = pd.Series(False, index=src.index)
    for i in range(src.size):
        if (high[i] - low[i]) == 0:
            filter[i] = False
            continue
        value1[i] = 0.2 * (src[i] - src[i - 1 if i >= 1 else 0]) + 0.8 * value1[i - 1 if i >= 1 else 0]
        value2[i] = 0.1 * (high[i] - low[i]) + 0.8 * value2[i - 1 if i >= 1 else 0]
        omega = abs(value1[i] / value2[i])
        alpha = (-(omega ** 2) + math.sqrt((omega ** 4) + 16 * (omega ** 2))) / 8 
        klmf[i] = alpha * src[i] + (1 - alpha) * klmf[i - 1 if i >= 1 else 0]
        absCurveSlope[i] = abs(klmf[i] - klmf[i - 1 if i >= 1 else 0])
        exponentialAverageAbsCurveSlope = 1.0 * ta.EMA(absCurveSlope, 200)[i]
        normalized_slope_decline = (absCurveSlope[i] - exponentialAverageAbsCurveSlope) / exponentialAverageAbsCurveSlope
        filter[i] = normalized_slope_decline >= threshold
    return filter


# @function filter_adx
# @param src <series float> The source series.
# @param length <int> The length of the ADX.
# @param adxThreshold <int> The ADX threshold.
# @param useAdxFilter <bool> Whether to use the ADX filter.
# @returns <series float> The ADX.
def filter_adx(src: pd.Series, high: pd.Series, low: pd.Series, adxThreshold, useAdxFilter, length=14):
    if not useAdxFilter: return pd.Series(True, index=src.index)
    tr = np.max(np.max(high - low, np.abs(high - src.shift(1))), np.abs(low - src.shift(1)))
    directionalMovementPlus = np.max(high - high.shift(1), 0) if (high - high.shift(1)) > (low.shift(1) - low) else 0
    negMovement = low.shift(1, fill_value=0) - np.max(low.shift(1) - low, 0) if low > (high - high.shift(1)) else 0
    trSmooth = pd.Series(0.0, src.size())
    trSmooth = trSmooth.shift(1) - trSmooth.shift(1) / length + tr
    smoothDirectionalMovementPlus = pd.Series(0.0, src.size())
    smoothDirectionalMovementPlus = smoothDirectionalMovementPlus.shift(1) - smoothDirectionalMovementPlus.shift(1) / length + directionalMovementPlus
    smoothnegMovement = pd.Series(0.0, src.size())
    smoothnegMovement = smoothnegMovement.shift(1) - smoothnegMovement.shift(1) / length + negMovement
    diPositive = smoothDirectionalMovementPlus / trSmooth * 100
    diNegative = smoothnegMovement / trSmooth * 100
    dx = np.abs(diPositive - diNegative) / (diPositive + diNegative) * 100
    adx = RMA(dx, length)
    return (adx > adxThreshold)


# @function filter_volatility
# @param minLength <int> The minimum length of the ATR.
# @param maxLength <int> The maximum length of the ATR.
# @param useVolatilityFilter <bool> Whether to use the volatility filter.
# @returns <bool> Boolean indicating whether or not to let the signal pass through the filter.
def filter_volatility(high, low, close, useVolatilityFilter, minLength=1, maxLength=10):
    if not useVolatilityFilter: return pd.Series(True, index=close.index)
    recentAtr = ta.ATR(high, low, close, minLength)
    historicalAtr = ta.ATR(high, low, close, maxLength)
    return (recentAtr > historicalAtr)
