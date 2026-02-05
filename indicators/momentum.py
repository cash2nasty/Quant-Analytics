import pandas as pd
import numpy as np


def roc(series: pd.Series, length: int = 10) -> pd.Series:
    return series.pct_change(length)


def momentum(series: pd.Series, length: int = 10) -> pd.Series:
    return series - series.shift(length)


def trend_strength(series: pd.Series, length: int = 20) -> float:
    """
    Trend strength = slope of linear regression line.
    """
    if len(series) < length:
        return 0.0

    y = series.tail(length).values
    x = np.arange(len(y))
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)