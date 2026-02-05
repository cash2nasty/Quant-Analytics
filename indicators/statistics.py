import pandas as pd
import numpy as np


def zscore(series: pd.Series, length: int = 20) -> pd.Series:
    mean = series.rolling(length).mean()
    std = series.rolling(length).std()
    return (series - mean) / std


def rolling_variance(series: pd.Series, length: int = 20) -> pd.Series:
    return series.rolling(length).var()


def estimate_drift(series: pd.Series, length: int = 20) -> float:
    """
    Drift = average return over window.
    """
    returns = series.pct_change()
    return returns.tail(length).mean()