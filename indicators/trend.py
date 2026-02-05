import pandas as pd


def classify_trend(df: pd.DataFrame, length: int = 20) -> str:
    """
    Simple trend classifier:
    - Uptrend: higher highs + higher lows
    - Downtrend: lower highs + lower lows
    - Neutral otherwise
    """
    if len(df) < length:
        return "neutral"

    highs = df["high"].tail(length)
    lows = df["low"].tail(length)

    hh = highs.is_monotonic_increasing
    ll = lows.is_monotonic_increasing

    hd = highs.is_monotonic_decreasing
    ld = lows.is_monotonic_decreasing

    if hh and ll:
        return "uptrend"
    if hd and ld:
        return "downtrend"
    return "neutral"