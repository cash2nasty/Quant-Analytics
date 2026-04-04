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
    choch_1m: str
    choch_time: Optional[pd.Timestamp]
    choch_price: Optional[float]
    choch_level: Optional[float]
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


def _detect_last_bos_by_direction(
    df: pd.DataFrame,
    window: int = 3,
) -> Tuple[
    Tuple[str, Optional[pd.Timestamp], Optional[float], Optional[float]],
    Tuple[str, Optional[pd.Timestamp], Optional[float], Optional[float]],
]:
    if df is None or df.empty:
        empty = ("None", None, None, None)
        return empty, empty
    required = {"timestamp", "high", "low", "close"}
    if not required.issubset(set(df.columns)):
        empty = ("None", None, None, None)
        return empty, empty
    if len(df) < (window * 2 + 2):
        empty = ("None", None, None, None)
        return empty, empty

    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values("timestamp").reset_index(drop=True)

    swing_highs, swing_lows = _find_swing_points(data, window)

    last_swing_high = None
    last_swing_low = None
    last_bull = ("None", None, None, None)
    last_bear = ("None", None, None, None)

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
                last_bull = ("Bullish", ts, close, level)
        if last_swing_low is not None:
            level = last_swing_low[1]
            if prev_close >= level and close < level:
                last_bear = ("Bearish", ts, close, level)

    return last_bull, last_bear


def _pick_latest_event(
    event_a: Tuple[str, Optional[pd.Timestamp], Optional[float], Optional[float]],
    event_b: Tuple[str, Optional[pd.Timestamp], Optional[float], Optional[float]],
) -> Tuple[str, Optional[pd.Timestamp], Optional[float], Optional[float]]:
    time_a = event_a[1]
    time_b = event_b[1]
    if time_a is None and time_b is None:
        return ("None", None, None, None)
    if time_a is None:
        return event_b
    if time_b is None:
        return event_a
    return event_a if time_a >= time_b else event_b


def _detect_last_bos(
    df: pd.DataFrame,
    window: int = 3,
) -> Tuple[str, Optional[pd.Timestamp], Optional[float], Optional[float]]:
    last_bull, last_bear = _detect_last_bos_by_direction(df, window=window)
    return _pick_latest_event(last_bull, last_bear)


def _trend_label_15m(df: pd.DataFrame, length: int = 10) -> str:
    if df is None or df.empty:
        return "n/a"
    rs = resample_ohlcv(df, "15min")
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
    last_bull, last_bear = _detect_last_bos_by_direction(df_day, window=3)
    bos_dir, bos_time, bos_price, bos_level = _pick_latest_event(last_bull, last_bear)

    if trend_15m == "Bullish":
        choch_dir, choch_time, choch_price, choch_level = last_bear
    elif trend_15m == "Bearish":
        choch_dir, choch_time, choch_price, choch_level = last_bull
    else:
        choch_dir, choch_time, choch_price, choch_level = ("None", None, None, None)

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
        choch_1m=choch_dir,
        choch_time=choch_time,
        choch_price=choch_price,
        choch_level=choch_level,
        alignment=alignment,
    )
