import pandas as pd
import talib
import pandas_ta as ta
import numpy as np
from script.utils import functions

np.seterr(divide="ignore", invalid="ignore")
pd.set_option("display.max_rows", None)


class Supertrend:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __str__(self):
        return f"Supertrend:\n{self.dataframe}"


# Supertrend instances stored in a list
holder = []

# List of factors
factors = []


def supertrend_ai(args):
    prices = args["prices"]
    length = int(args["length"])
    minMult = int(args["minMult"])
    maxMult = int(args["maxMult"])
    step = float(args["step"])
    perfAlpha = int(args["perfAlpha"])
    fromCluster = args["fromCluster"]
    maxIter = int(args["maxIter"])
    maxData = int(args["maxData"])
    if maxMult < minMult:
        raise ValueError("Minimum factor is greater than maximum factor in the range")

    atr = functions.atr(prices=prices, length=length)
    for i in range(int((maxMult - minMult) / step) + 1):
        factor_value = minMult + i * step
        factors.append(factor_value)

    for factor in factors:
        spt = functions.supertrend(prices=prices, length=length, factor=factor)

        # Create a DataFrame for each Supertrend instance
        supertrend_df = pd.DataFrame(
            {
                "upper": spt["upper"],
                "lower": spt["lower"],
                "output": spt["supertrend"],
                "perf": [0] * len(spt),
                "factor": [factor] * len(spt),
                "trend": spt["direction"],
            }
        )

        # Calculate diff and update perf for the Supertrend instance
        diff = pd.DataFrame(
            {
                "diff": (prices["close"] - supertrend_df["output"]).apply(
                    lambda x: 1 if x > 0 else 0
                )
            }
        )
        supertrend_df["perf"] += (2 / (perfAlpha + 1)) * (
            (diff["diff"] * (prices["close"] - prices["close"].shift(1))).fillna(0)
            - supertrend_df["perf"]
        )

        # Create a Supertrend instance with the DataFrame
        holder.append(Supertrend(supertrend_df))
    # Initialize empty lists for data and factor_array
    data = []
    factor_array = []
    # Get the index of the last row in the DataFrame
    last_index = prices.index[-1]

    # Iterate through elements in the holder
    for element in holder:
        # Append element.perf to the data list
        data.extend(element.dataframe["perf"].values)
        # extend element.factor to the factor_array list
        factor_array.extend(element.dataframe["factor"].values)
    # data = np.array(data)

    # Intialize centroids using quartiles
    centroids = [
        np.percentile(data, 25, method="linear"),
        np.percentile(data, 50, method="linear"),
        np.percentile(data, 75, method="linear"),
    ]
    # Initialize clusters
    # factors_clusters = [np.array([]), np.array([]), np.array([])]
    # perfclusters = [np.array([]), np.array([]), np.array([])]

    for _ in range(maxIter):
        factors_clusters = [np.array([]), np.array([]), np.array([])]
        perfclusters = [np.array([]), np.array([]), np.array([])]

        for i, value in enumerate(data):
            dist = [np.abs(value - centroid) for centroid in centroids]
            idx = dist.index(min(dist))
            perfclusters[idx] = np.append(perfclusters[idx], value)
            factors_clusters[idx] = np.append(factors_clusters[idx], factor_array[i])

        # Update centroids
        new_centroids = [np.mean(cluster) for cluster in perfclusters]
        # Test if centroids changed
        if np.allclose(new_centroids, centroids):
            break

        centroids = new_centroids

    # Get associated supertrend
    target_factor = np.nan
    perf_idx = np.nan
    perf_ama = np.nan
    from_cluster_options = {"Best": 2, "Average": 1, "Worst": 0}
    from_cluster = from_cluster_options[fromCluster]

    # Performance index denominator
    den = ta.ema(np.abs(prices["close"] - prices["close"].shift(1)), int(perfAlpha))
    # Check if any of the arrays within perfclusters is empty
    if any(not np.isnan(cluster).all() for cluster in perfclusters):
        # Get average factors within the target cluster
        target_factor = np.where(
            np.isnan(factors_clusters[from_cluster].mean()),
            target_factor,
            factors_clusters[from_cluster].mean(),
        )
        # Get performance index of the target cluster
        perf_idx = max(perfclusters[from_cluster].mean(), 0) / den
    new_supertrend = functions.supertrend(
        prices=prices,
        length=length,
        factor=target_factor,
    )
    new_supertrend_df = pd.DataFrame(
        {
            "upper": new_supertrend["upper"],
            "lower": new_supertrend["lower"],
            "output": new_supertrend["supertrend"],
            "perf": [0] * len(new_supertrend),
            "factor": [factor] * len(new_supertrend),
            "trend": new_supertrend["direction"],
        }
    )
    hl2 = (prices["high"] + prices["low"]) / 2
    os = pd.DataFrame(data={"os": [0] * len(prices)})
    # up = hl2 +
    # os["os"] = 1 * (prices["close"] > new_supertrend_df["upper"]) + 0 * (
    #     prices["close"] < new_supertrend_df["lower"]
    # )
    os["os"] = np.where(
        prices["close"] > new_supertrend_df["upper"],
        1,
        np.where(prices["close"] < new_supertrend_df["lower"], 0, os["os"]),
    )
    os["timestamp"] = prices["timestamp"].values
    os["symbol"] = prices["symbol"].values
    ts = pd.DataFrame(data={"os": [0] * len(prices)})
    ts["os"] = np.where(
        os["os"] == 1, new_supertrend_df["upper"], new_supertrend_df["lower"]
    )
    return os
