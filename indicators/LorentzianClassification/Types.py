import pandas as pd
from enum import IntEnum

# ======================
# ==== Custom Types ====
# ======================

# This section uses PineScript's new Type syntax to define important data structures
# used throughout the script.


class __Config__:
    def __init__(self, **kwargs):
        while kwargs:
            k, v = kwargs.popitem()
            setattr(self, k, v)


class Settings(__Config__):
    source: pd.Series  # Source of the input data
    neighborsCount = 16  # Number of neighbors to consider
    maxBarsBack = 2000  # Maximum number of bars to look back for calculations
    useDynamicExits = True  # Dynamic exits attempt to let profits ride by dynamically adjusting the exit threshold based on kernel regression logic

    # EMA Settings
    useEmaFilter = False
    emaPeriod = 200

    # SMA Settings
    useSmaFilter = False
    smaPeriod = 200


class Feature:
    type: str
    param1: int
    param2: int

    def __init__(self, type, param1, param2):
        self.type = type
        self.param1 = param1
        self.param2 = param2


# Nadaraya-Watson Kernel Regression Settings
class KernelFilter(__Config__):
    useKernelSmoothing = False  # Enhance Kernel Smoothing: Uses a crossover based mechanism to smoothen kernel color changes. This often results in less color transitions overall and may result in more ML entry signals being generated.
    lookbackWindow = 5  # Lookback Window: The number of bars used for the estimation. This is a sliding value that represents the most recent historical bars. Recommended range: 3-50
    relativeWeight = 8.0  # Relative Weighting: Relative weighting of time frames. As this value approaches zero, the longer time frames will exert more influence on the estimation. As this value approaches infinity, the behavior of the Rational Quadratic Kernel will become identical to the Gaussian kernel. Recommended range: 0.25-25
    regressionLevel = 25  # Regression Level: Bar index on which to start regression. Controls how tightly fit the kernel estimate is to the data. Smaller values are a tighter fit. Larger values are a looser fit. Recommended range: 2-25
    crossoverLag = 2  # Lag: Lag for crossover detection. Lower values result in earlier crossovers. Recommended range: 1-2


class FilterSettings(__Config__):
    useVolatilityFilter = (True,)  # Whether to use the volatility filter
    useRegimeFilter = (False,)  # Whether to use the trend detection filter
    useAdxFilter = (False,)  # Whether to use the ADX filter
    regimeThreshold = (0.0,)  # Threshold for detecting Trending/Ranging markets
    adxThreshold = 0  # Threshold for detecting Trending/Ranging markets

    kernelFilter: KernelFilter


class Filter(__Config__):
    volatility = True
    regime = False
    adx = False


# Label Object: Used for classifying historical data as training data for the ML Model
class Direction(IntEnum):
    LONG = 1
    SHORT = -1
    NEUTRAL = 0
