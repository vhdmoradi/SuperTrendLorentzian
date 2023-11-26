import pandas as pd
import talib
import pandas_ta as ta
import numpy as np

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


def supertrend_ai(
    prices,
    length=111,
    minMult=1,
    maxMult=5,
    step=0.5,
    perfAlpha=10,
    fromCluster="Best",
    maxIter=1000,
    maxData=10000,
):
    length = int(length)
    minMult = int(minMult)
    maxMult = int(maxMult)
    step = float(step)
    perfAlpha = int(perfAlpha)
    maxIter = int(maxIter)
    maxData = int(maxData)
    if maxMult < minMult:
        raise ValueError("Minimum factor is greater than maximum factor in the range")
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
    # Populate Supertrend type array
    symbol = prices["symbol"].iloc[-1]
    for i in range(int((maxMult - minMult) / step) + 1):
        factor_value = minMult + i * step
        factors.append(factor_value)

    for factor in factors:
        _props = f"_{length}_{factor}"

        # Calculate Supertrend using ta.supertrend
        supertrend_values = ta.supertrend(
            prices["high"],
            prices["low"],
            prices["close"],
            length=length,
            multiplier=factor,
        )

        # Create a DataFrame for each Supertrend instance
        supertrend_df = pd.DataFrame(
            {
                "upper": supertrend_values[f"SUPERTl{_props}"],
                "lower": supertrend_values[f"SUPERTs{_props}"],
                "output": supertrend_values[f"SUPERT{_props}"],
                "perf": [0] * len(supertrend_values),
                "factor": [factor] * len(supertrend_values),
                "trend": supertrend_values[f"SUPERT{_props}"],
            }
        )

        # Create a Supertrend instance with the DataFrame
        holder.append(Supertrend(supertrend_df))
        # Calculate diff and update perf for the Supertrend instance
        diff = np.sign(prices["close"].shift(1) - supertrend_df["output"])
        supertrend_df["perf"] += (
            2
            / (perfAlpha + 1)
            * (
                diff * (prices["close"] - prices["close"].shift(1))
                - supertrend_df["perf"]
            )
        )
    # Initialize empty lists for data and factor_array
    data = []
    factor_array = []
    # Get the index of the last row in the DataFrame
    last_index = prices.index[-1]

    # Iterate through elements in the holder
    for element in holder:
        # Check if the current index is within the specified maxData range
        if last_index - prices.index.get_loc(element.dataframe.index[-1]) <= maxData:
            # Append element.perf to the data list
            data.append(element.dataframe["perf"].values[-1])
            # Append element.factor to the factor_array list
            factor_array.append(element.dataframe["factor"].values[-1])

    data = np.array(data)
    # Intialize centroids using quartiles
    centroids = [
        np.percentile(data, 25, method="linear"),
        np.percentile(data, 50, method="linear"),
        np.percentile(data, 75, method="linear"),
    ]
    # Initialize clusters
    factors_clusters = [np.array([]), np.array([]), np.array([])]
    perfclusters = [np.array([]), np.array([]), np.array([])]
    if last_index - prices.index.get_loc(element.dataframe.index[-1]) <= maxData:
        for _ in range(maxIter):
            # Assign values to clusters
            for i, value in enumerate(data):
                dist = [np.abs(value - centroid) for centroid in centroids]
                idx = np.argmin(dist)
                perfclusters[idx] = np.append(perfclusters[idx], value)
                factors_clusters[idx] = np.append(
                    factors_clusters[idx], factor_array[i]
                )

            # Update centroids
            new_centroids = [np.mean(cluster) for cluster in perfclusters]

            # Test if centroids changed
            if np.array_equal(new_centroids, centroids):
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
        target_factor = (
            factors_clusters[from_cluster].mean()
            if from_cluster < len(factors_clusters)
            and factors_clusters[from_cluster].size > 0
            else np.nan
        )

        # Get performance index of the target cluster
        perf_idx = max(perfclusters[from_cluster].mean(), 0) / den
    new_supertrend = ta.supertrend(
        prices["high"],
        prices["low"],
        prices["close"],
        length=length,
        multiplier=target_factor,
    )
    _props = f"_{length}_{target_factor}"
    new_supertrend_df = pd.DataFrame(
        {
            "upper": new_supertrend[f"SUPERTl{_props}"],
            "lower": new_supertrend[f"SUPERTs{_props}"],
            "output": new_supertrend[f"SUPERT{_props}"],
            "perf": [0] * len(new_supertrend),
            "factor": [factor] * len(new_supertrend),
            "trend": new_supertrend[f"SUPERT{_props}"],
        }
    )
    hl2 = (prices["high"] + prices["low"]) / 2
    os = pd.DataFrame(data={"os": [0] * len(prices)})
    os["os"] = 1 * (prices["close"] > new_supertrend_df["upper"]) + 0 * (
        prices["close"] < new_supertrend_df["lower"]
    )
    os["timestamp"] = prices["timestamp"].values
    ts = pd.DataFrame(data={"os": [0] * len(prices)})
    ts["os"] = np.where(
        os["os"] == 1, new_supertrend_df["upper"], new_supertrend_df["lower"]
    )
    return os, symbol
