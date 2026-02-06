import pandas as pd
import numpy as np


def rolling_volatility(series: pd.Series, length: int = 20) -> pd.Series:
    return series.rolling(length).std()


def atr_like(df: pd.DataFrame, length: int = 14) -> pd.Series:
    ranges = df["high"] - df["low"]
    return ranges.rolling(length).mean()


def classify_volatility(vol_series: pd.Series, p_low: float = 0.30, p_high: float = 0.70) -> str:
    """
    Simple volatility regime classifier:
    - High volatility: above p_high percentile
    - Low volatility: below p_low percentile
    - Normal otherwise
    """
    if len(vol_series) < 20:
        return "normal"

    recent = vol_series.iloc[-1]
    p30 = vol_series.quantile(p_low)
    p70 = vol_series.quantile(p_high)

    if recent > p70:
        return "expanded"
    elif recent < p30:
        return "compressed"
    return "normal"