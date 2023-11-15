import pandas as pd
import talib


class Supertrend:
    def __init__(self, upper, lower, output, perf, factor, trend):
        self.upper = upper
        self.lower = lower
        self.output = output
        self.perf = perf
        self.factor = factor
        self.trend = trend


# Supertrend instances stored in a list
holder = []

# List of factors
factors = []


def supertrend_ai(
    length=111,
    minMult=1,
    maxMult=5,
    step=0.5,
    perfAlpha=10,
    fromCluster="Best",
    maxIter=1000,
    maxData=10000,
):
    # Populate Supertrend type array
    for i in range(int((maxMult - minMult) / step) + 1):
        factor_value = minMult + i * step
        factors.append(factor_value)
        holder.append(Supertrend(0, 0, 0, 0, factor_value, 0))
