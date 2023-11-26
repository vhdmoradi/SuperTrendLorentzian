import math
import pandas as pd
import numpy as np


def rationalQuadratic(src: pd.Series, lookback: int, relativeWeight: float, startAtBar: int):
    """
    vectorized calculate for rational quadratic curve
    :param src:
    :param lookback:
    :param relativeWeight:
    :param startAtBar:
    :return:
    """

    size = startAtBar + 2
    windows = [src[i:i + size].values for i in range(0, len(src) - size + 1, 1)]
    weight = [math.pow(1 + (math.pow(i, 2) / (math.pow(lookback, 2) * 2 * relativeWeight)), -relativeWeight) for i in
              range(size)]
    current_weight = [np.sum(windows[i][::-1] * weight) for i in range(0, len(src) - size + 1, 1)]
    cumulative_weight = [np.sum(weight) for _ in range(0, len(src) - size + 1, 1)]
    kernel_line = np.array(current_weight) / np.array(cumulative_weight)
    kernel_line = np.concatenate((np.array([0.0] * (size - 1)), kernel_line))
    kernel_line = pd.Series(kernel_line.flatten())
    return kernel_line


def gaussian(src, lookback, startAtBar):
    """
    vectorized calculate for gaussian curve
    :param src:
    :param lookback:
    :param startAtBar:
    :return:
    """
    size = startAtBar + 2
    windows = [src[i:i + size].values for i in range(0, len(src) - size + 1, 1)]
    weight = [math.exp(-(i ** 2) / (2 * lookback ** 2)) for i in
              range(size)]
    current_weight = [np.sum(windows[i][::-1] * weight) for i in range(0, len(src) - size + 1, 1)]
    cumulative_weight = [np.sum(weight) for _ in range(0, len(src) - size + 1, 1)]
    gaussian_line = np.array(current_weight) / np.array(cumulative_weight)
    gaussian_line = np.concatenate((np.array([0.0] * (size - 1)), gaussian_line))
    gaussian_line = pd.Series(gaussian_line.flatten())
    return gaussian_line
