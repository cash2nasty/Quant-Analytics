import pandas as pd
from typing import Optional


def compute_daily_vwap(df: pd.DataFrame) -> pd.Series:
    """Compute intraday (per-date) VWAP and return a Series aligned with df.

    The function supports dataframes that include a `timestamp` column
    (datetime64) or use a DatetimeIndex. VWAP is computed as cumulative
    sum(typical_price * volume) / cumulative sum(volume) per-calendar-day.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)

    df = df.copy()

    # Ensure timestamp is available as a datetime series
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        dt_index = df["timestamp"]
    elif isinstance(df.index, pd.DatetimeIndex):
        dt_index = df.index.to_series()
        df = df.reset_index()
        df["timestamp"] = dt_index.values
    else:
        raise ValueError("DataFrame must have a 'timestamp' column or a DatetimeIndex")

    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    tpv = tp * df["volume"]

    # Group by calendar date and compute cumulative sums within each day
    day = df["timestamp"].dt.date
    cum_tpv = tpv.groupby(day).cumsum()
    cum_vol = df.groupby(day)["volume"].cumsum()

    vwap = cum_tpv / cum_vol

    # Align the result index with original df index
    vwap.index = df.index
    return vwap


def compute_weekly_vwap(df: pd.DataFrame) -> pd.Series:
    """Compute VWAP per-week and return a Series aligned with df.

    VWAP is computed as cumulative sum(typical_price * volume) / cumulative sum(volume)
    within each ISO week (Monday-Sunday grouping). If only a single day is
    provided the weekly VWAP will equal the daily VWAP for that day.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)

    df = df.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    elif isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
    else:
        raise ValueError("DataFrame must have a 'timestamp' column or a DatetimeIndex")

    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    tpv = tp * df["volume"]

    # Group by ISO week period
    week = df["timestamp"].dt.to_period("W")
    cum_tpv = tpv.groupby(week).cumsum()
    cum_vol = df["volume"].groupby(week).cumsum()

    wvwap = cum_tpv / cum_vol
    wvwap.index = df.index
    return wvwap


def render_settings():
    """Compatibility placeholder for the settings UI (kept for backward compatibility)."""
    try:
        import streamlit as st

        st.header("Settings")
        st.write("Indicator settings will be added here in future.")
    except Exception:
        # silent fallback when running outside Streamlit
        pass