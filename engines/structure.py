from dataclasses import dataclass
import datetime as dt
from typing import Dict, Optional, Tuple

import pandas as pd

from engines.zones import resample_ohlcv
from indicators.momentum import trend_strength


@dataclass
class StructureSummary:
    trend_15m: str
    bos_1m: str
    bos_time: Optional[pd.Timestamp]
    bos_price: Optional[float]
    bos_level: Optional[float]
    alignment: str


def _filter_trading_day(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "timestamp" not in df.columns:
        return pd.DataFrame()
    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values("timestamp")
    last_ts = data["timestamp"].iloc[-1]
    trading_date = last_ts.date()
    start = dt.datetime.combine(trading_date - dt.timedelta(days=1), dt.time(18, 0))
    end = dt.datetime.combine(trading_date, dt.time(17, 0))
    mask = (data["timestamp"] >= start) & (data["timestamp"] <= end)
    return data.loc[mask].reset_index(drop=True)


def _find_swing_points(df: pd.DataFrame, window: int) -> Tuple[Dict[int, float], Dict[int, float]]:
    swing_highs: Dict[int, float] = {}
    swing_lows: Dict[int, float] = {}
    highs = df["high"].values
    lows = df["low"].values
    for i in range(window, len(df) - window):
        if highs[i] == max(highs[i - window : i + window + 1]):
            swing_highs[i] = float(highs[i])
        if lows[i] == min(lows[i - window : i + window + 1]):
            swing_lows[i] = float(lows[i])
    return swing_highs, swing_lows


def _detect_last_bos(
    df: pd.DataFrame,
    window: int = 3,
) -> Tuple[str, Optional[pd.Timestamp], Optional[float], Optional[float]]:
    if df is None or df.empty:
        return "None", None, None, None
    required = {"timestamp", "high", "low", "close"}
    if not required.issubset(set(df.columns)):
        return "None", None, None, None
    if len(df) < (window * 2 + 2):
        return "None", None, None, None

    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values("timestamp").reset_index(drop=True)

    swing_highs, swing_lows = _find_swing_points(data, window)

    last_swing_high = None
    last_swing_low = None
    last_bos = ("None", None, None, None)

    for i in range(1, len(data)):
        if i in swing_highs:
            last_swing_high = (i, swing_highs[i])
        if i in swing_lows:
            last_swing_low = (i, swing_lows[i])

        close = float(data["close"].iloc[i])
        prev_close = float(data["close"].iloc[i - 1])
        ts = data["timestamp"].iloc[i]

        if last_swing_high is not None:
            level = last_swing_high[1]
            if prev_close <= level and close > level:
                last_bos = ("Bullish", ts, close, level)
        if last_swing_low is not None:
            level = last_swing_low[1]
            if prev_close >= level and close < level:
                last_bos = ("Bearish", ts, close, level)

    return last_bos


def _trend_label_15m(df: pd.DataFrame, length: int = 10) -> str:
    if df is None or df.empty:
        return "n/a"
    rs = resample_ohlcv(df, "15T")
    if len(rs) < length:
        return "n/a"
    slope = float(trend_strength(rs["close"], length=length))
    if slope > 0:
        return "Bullish"
    if slope < 0:
        return "Bearish"
    return "Neutral"


def detect_market_structure(df_1m: pd.DataFrame) -> StructureSummary:
    df_day = _filter_trading_day(df_1m)
    trend_15m = _trend_label_15m(df_day)
    bos_dir, bos_time, bos_price, bos_level = _detect_last_bos(df_day, window=3)

    if bos_dir in ("Bullish", "Bearish") and trend_15m in ("Bullish", "Bearish"):
        alignment = "Aligned" if bos_dir == trend_15m else "Opposite"
    else:
        alignment = "n/a"

    return StructureSummary(
        trend_15m=trend_15m,
        bos_1m=bos_dir,
        bos_time=bos_time,
        bos_price=bos_price,
        bos_level=bos_level,
        alignment=alignment,
    )
