from storage.history_manager import SessionStats, PatternSummary
from typing import Dict, Optional, Tuple

import pandas as pd

from indicators.moving_averages import compute_daily_vwap


def _opening_range(df: pd.DataFrame, minutes: int) -> Optional[Tuple[float, float]]:
    if df is None or df.empty or "timestamp" not in df.columns:
        return None
    date = df["timestamp"].iloc[0].date()
    start = pd.Timestamp.combine(date, pd.Timestamp("09:30").time())
    end = start + pd.Timedelta(minutes=minutes)
    window = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
    if window.empty:
        return None
    return float(window["high"].max()), float(window["low"].min())


def _orb_signal(df: pd.DataFrame, minutes: int) -> Tuple[bool, str]:
    or_range = _opening_range(df, minutes)
    if or_range is None:
        return False, "Neutral"
    or_high, or_low = or_range
    or_size = max(or_high - or_low, 1e-6)
    start = pd.Timestamp.combine(df["timestamp"].iloc[0].date(), pd.Timestamp("09:30").time())
    end = start + pd.Timedelta(minutes=minutes)
    after = df[df["timestamp"] > end]
    if after.empty:
        return False, "Neutral"
    bullish = after[after["close"] > or_high + 0.1 * or_size]
    bearish = after[after["close"] < or_low - 0.1 * or_size]
    if not bullish.empty and (bearish.empty or bullish.index[0] < bearish.index[0]):
        return True, "Bullish"
    if not bearish.empty:
        return True, "Bearish"
    return False, "Neutral"


def _failed_orb_signal(df: pd.DataFrame, minutes: int) -> Tuple[bool, str]:
    or_range = _opening_range(df, minutes)
    if or_range is None:
        return False, "Neutral"
    or_high, or_low = or_range
    start = pd.Timestamp.combine(df["timestamp"].iloc[0].date(), pd.Timestamp("09:30").time())
    end = start + pd.Timedelta(minutes=minutes)
    after = df[df["timestamp"] > end]
    if after.empty:
        return False, "Neutral"
    break_up = after[after["high"] > or_high]
    break_down = after[after["low"] < or_low]
    if break_up.empty and break_down.empty:
        return False, "Neutral"
    if not break_up.empty:
        t0 = break_up.iloc[0]["timestamp"]
        window = after[(after["timestamp"] > t0) & (after["timestamp"] <= t0 + pd.Timedelta(minutes=30))]
        if not window.empty and (window["close"] < or_high).any():
            return True, "Bearish"
    if not break_down.empty:
        t0 = break_down.iloc[0]["timestamp"]
        window = after[(after["timestamp"] > t0) & (after["timestamp"] <= t0 + pd.Timedelta(minutes=30))]
        if not window.empty and (window["close"] > or_low).any():
            return True, "Bullish"
    return False, "Neutral"


def _power_hour_trend(df: pd.DataFrame) -> Tuple[bool, str]:
    if df is None or df.empty or "timestamp" not in df.columns:
        return False, "Neutral"
    date = df["timestamp"].iloc[0].date()
    start = pd.Timestamp.combine(date, pd.Timestamp("14:00").time())
    end = pd.Timestamp.combine(date, pd.Timestamp("16:00").time())
    window = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
    if window.empty:
        return False, "Neutral"
    day_range = float(df["high"].max() - df["low"].min())
    move = float(window["close"].iloc[-1] - window["open"].iloc[0])
    if day_range <= 0:
        return False, "Neutral"
    if abs(move) < 0.3 * day_range:
        return False, "Neutral"
    return True, "Bullish" if move > 0 else "Bearish"


def _vwap_reclaim_reject(df: pd.DataFrame) -> Tuple[bool, str]:
    if df is None or df.empty or "timestamp" not in df.columns:
        return False, "Neutral"
    vwap = compute_daily_vwap(df)
    if vwap.empty:
        return False, "Neutral"
    closes = df["close"].values
    vwap_vals = vwap.values
    if len(closes) < 5:
        return False, "Neutral"
    for i in range(1, len(closes) - 4):
        prev = closes[i - 1] - vwap_vals[i - 1]
        curr = closes[i] - vwap_vals[i]
        if prev <= 0 and curr > 0:
            if all(closes[i:i + 4] > vwap_vals[i:i + 4]):
                return True, "Bullish"
        if prev >= 0 and curr < 0:
            if all(closes[i:i + 4] < vwap_vals[i:i + 4]):
                return True, "Bearish"
    return False, "Neutral"


def detect_patterns(
    sessions: Dict[str, SessionStats],
    df_today: Optional[pd.DataFrame] = None,
    df_prev: Optional[pd.DataFrame] = None,
) -> PatternSummary:
    """
    Simple structural pattern detection using session stats.
    """
    asia = sessions.get("Asia")
    london = sessions.get("London")
    us = sessions.get("US")

    london_breakout = False
    whipsaw = False
    trend_day = False
    vol_expansion = False
    asia_range_hold = False
    asia_range_sweep = False
    asia_range_sweep_bias = "Neutral"
    london_continuation = False
    london_continuation_bias = "Neutral"
    us_open_gap_fill = False
    us_open_gap_fill_bias = "Neutral"
    orb_30 = False
    orb_30_bias = "Neutral"
    orb_60 = False
    orb_60_bias = "Neutral"
    failed_orb_30 = False
    failed_orb_30_bias = "Neutral"
    failed_orb_60 = False
    failed_orb_60_bias = "Neutral"
    power_hour_trend = False
    power_hour_bias = "Neutral"
    vwap_reclaim_reject = False
    vwap_reclaim_reject_bias = "Neutral"
    notes_parts = []

    if asia and london:
        # London Breakout: London range extends beyond Asia high/low
        if london.high > asia.high and london.low < asia.low:
            london_breakout = True
            notes_parts.append("London extended beyond Asia range (London Breakout).")

        # Asia Range Hold: London stayed within Asia range
        if london.high <= asia.high and london.low >= asia.low:
            asia_range_hold = True
            notes_parts.append("London stayed inside Asia range (Asia Range Hold).")

        # Asia Range Sweep: London sweeps Asia high/low and closes inside
        if london.high > asia.high and asia.low < london.close < asia.high:
            asia_range_sweep = True
            asia_range_sweep_bias = "Bearish"
            notes_parts.append("London swept Asia high and closed inside (Asia Range Sweep).")
        elif london.low < asia.low and asia.low < london.close < asia.high:
            asia_range_sweep = True
            asia_range_sweep_bias = "Bullish"
            notes_parts.append("London swept Asia low and closed inside (Asia Range Sweep).")

    if london and us:
        # Whipsaw: London up, US down (or vice versa)
        london_dir = london.close - london.open
        us_dir = us.close - us.open
        if london_dir * us_dir < 0:
            whipsaw = True
            notes_parts.append("London and US moved in opposite directions (Whipsaw).")

        # Trend Day: same direction across London and US with strong ranges
        if london_dir * us_dir > 0 and abs(london_dir) > london.range * 0.3 and abs(us_dir) > us.range * 0.3:
            trend_day = True
            notes_parts.append("London and US aligned directionally with strong ranges (Trend Day).")

        # Volatility Expansion: US range > 1.5x London range
        if us.range > 1.5 * london.range:
            vol_expansion = True
            notes_parts.append("US range expanded significantly vs London (Volatility Expansion).")

        # London Continuation: US continues London break of Asia range
        if asia and london.high > asia.high and us.high > london.high and us.close > us.open:
            london_continuation = True
            london_continuation_bias = "Bullish"
            notes_parts.append("London broke Asia high and US continued higher (London Continuation).")
        elif asia and london.low < asia.low and us.low < london.low and us.close < us.open:
            london_continuation = True
            london_continuation_bias = "Bearish"
            notes_parts.append("London broke Asia low and US continued lower (London Continuation).")

    if df_today is not None and not df_today.empty:
        # US Open Gap Fill
        if df_prev is not None and not df_prev.empty:
            prev_close = float(df_prev["close"].iloc[-1])
            prev_range = float(df_prev["high"].max() - df_prev["low"].min())
            today_open = float(df_today["open"].iloc[0])
            gap = today_open - prev_close
            if prev_range > 0 and abs(gap) >= 0.2 * prev_range:
                date = df_today["timestamp"].iloc[0].date()
                start = pd.Timestamp.combine(date, pd.Timestamp("09:30").time())
                end = pd.Timestamp.combine(date, pd.Timestamp("10:30").time())
                window = df_today[(df_today["timestamp"] >= start) & (df_today["timestamp"] <= end)]
                if not window.empty:
                    if gap > 0 and (window["low"] <= prev_close).any():
                        us_open_gap_fill = True
                        us_open_gap_fill_bias = "Bearish"
                        notes_parts.append("US open gap up filled by 10:30 (Gap Fill).")
                    elif gap < 0 and (window["high"] >= prev_close).any():
                        us_open_gap_fill = True
                        us_open_gap_fill_bias = "Bullish"
                        notes_parts.append("US open gap down filled by 10:30 (Gap Fill).")

        # ORB and Failed ORB
        orb_30, orb_30_bias = _orb_signal(df_today, 30)
        orb_60, orb_60_bias = _orb_signal(df_today, 60)
        failed_orb_30, failed_orb_30_bias = _failed_orb_signal(df_today, 30)
        failed_orb_60, failed_orb_60_bias = _failed_orb_signal(df_today, 60)
        if orb_30:
            notes_parts.append(f"ORB 30m confirmed {orb_30_bias.lower()}.")
        if orb_60:
            notes_parts.append(f"ORB 60m confirmed {orb_60_bias.lower()}.")
        if failed_orb_30:
            notes_parts.append(f"Failed ORB 30m favors {failed_orb_30_bias.lower()}.")
        if failed_orb_60:
            notes_parts.append(f"Failed ORB 60m favors {failed_orb_60_bias.lower()}.")

        # Power Hour Trend
        power_hour_trend, power_hour_bias = _power_hour_trend(df_today)
        if power_hour_trend:
            notes_parts.append(f"Power hour trend aligned {power_hour_bias.lower()}.")

        # VWAP Reclaim/Reject
        vwap_reclaim_reject, vwap_reclaim_reject_bias = _vwap_reclaim_reject(df_today)
        if vwap_reclaim_reject:
            notes_parts.append(f"VWAP {vwap_reclaim_reject_bias.lower()} reclaim/reject.")

    if not notes_parts:
        notes_parts.append("No major structural patterns detected.")

    return PatternSummary(
        london_breakout=london_breakout,
        whipsaw=whipsaw,
        trend_day=trend_day,
        volatility_expansion=vol_expansion,
        notes=" ".join(notes_parts),
        asia_range_hold=asia_range_hold,
        asia_range_sweep=asia_range_sweep,
        asia_range_sweep_bias=asia_range_sweep_bias,
        london_continuation=london_continuation,
        london_continuation_bias=london_continuation_bias,
        us_open_gap_fill=us_open_gap_fill,
        us_open_gap_fill_bias=us_open_gap_fill_bias,
        orb_30=orb_30,
        orb_30_bias=orb_30_bias,
        orb_60=orb_60,
        orb_60_bias=orb_60_bias,
        failed_orb_30=failed_orb_30,
        failed_orb_30_bias=failed_orb_30_bias,
        failed_orb_60=failed_orb_60,
        failed_orb_60_bias=failed_orb_60_bias,
        power_hour_trend=power_hour_trend,
        power_hour_bias=power_hour_bias,
        vwap_reclaim_reject=vwap_reclaim_reject,
        vwap_reclaim_reject_bias=vwap_reclaim_reject_bias,
    )