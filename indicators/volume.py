import pandas as pd
import numpy as np


def rvol(df: pd.DataFrame, length: int = 20) -> pd.Series:
    avg_vol = df["volume"].rolling(length).mean()
    return df["volume"] / avg_vol


def volume_trend(df: pd.DataFrame, length: int = 20) -> float:
    """
    Linear regression slope of volume.
    """
    if len(df) < length:
        return 0.0

    y = df["volume"].tail(length).values
    x = np.arange(len(y))
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)