import datetime as dt
from typing import Dict, List, Optional, Tuple

import pandas as pd

from data.session_reference import get_session_windows_for_date
from engines.patterns import PatternSummary
from engines.sessions import SessionStats
from engines.zones import (
    Zone,
    find_rejection_candles,
    is_fvg_inversed,
    is_zone_failed,
    is_zone_touched,
    zone_formed_timestamp,
    zone_liquidity_scores,
)
from indicators.moving_averages import compute_anchored_vwap, compute_daily_vwap, compute_weekly_vwap


OPEN_PATTERN_CANDLE3_STRICTNESS = "balanced"
OPEN_PATTERN_RECLAIM_SPEED_WINDOW_MINUTES = 30.0


def _fmt_ts(value: Optional[pd.Timestamp]) -> Optional[str]:
    if value is None:
        return None
    ts = pd.to_datetime(value)
    if pd.isna(ts):
        return None
    return ts.strftime("%Y-%m-%d %H:%M")


def _first_after(df: pd.DataFrame, mask: pd.Series) -> Optional[Tuple[pd.Timestamp, float]]:
    filtered = df[mask]
    if filtered.empty:
        return None
    row = filtered.iloc[0]
    return pd.to_datetime(row["timestamp"]), float(row["close"])


def _detect_orb_trigger(df: pd.DataFrame, minutes: int) -> Optional[Dict[str, object]]:
    if df is None or df.empty:
        return None
    date = pd.to_datetime(df["timestamp"].iloc[0]).date()
    start = pd.Timestamp.combine(date, pd.Timestamp("09:30").time())
    end = start + pd.Timedelta(minutes=minutes)
    opening = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
    if opening.empty:
        return None
    or_high = float(opening["high"].max())
    or_low = float(opening["low"].min())
    or_size = max(or_high - or_low, 1e-6)

    after = df[df["timestamp"] > end]
    if after.empty:
        return None

    bullish = _first_after(after, after["close"] > (or_high + 0.1 * or_size))
    bearish = _first_after(after, after["close"] < (or_low - 0.1 * or_size))

    if bullish and (not bearish or bullish[0] <= bearish[0]):
        return {
            "name": f"ORB {minutes}m Breakout",
            "direction": "Bullish",
            "time": _fmt_ts(bullish[0]),
            "price": bullish[1],
            "timeframe": "5m",
            "details": f"Close broke above OR high + 10% OR range ({or_high:.2f}).",
        }
    if bearish:
        return {
            "name": f"ORB {minutes}m Breakout",
            "direction": "Bearish",
            "time": _fmt_ts(bearish[0]),
            "price": bearish[1],
            "timeframe": "5m",
            "details": f"Close broke below OR low - 10% OR range ({or_low:.2f}).",
        }
    return None


def _detect_failed_orb_trigger(df: pd.DataFrame, minutes: int) -> Optional[Dict[str, object]]:
    if df is None or df.empty:
        return None
    date = pd.to_datetime(df["timestamp"].iloc[0]).date()
    start = pd.Timestamp.combine(date, pd.Timestamp("09:30").time())
    end = start + pd.Timedelta(minutes=minutes)
    opening = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
    if opening.empty:
        return None

    or_high = float(opening["high"].max())
    or_low = float(opening["low"].min())
    after = df[df["timestamp"] > end]
    if after.empty:
        return None

    break_up = after[after["high"] > or_high]
    if not break_up.empty:
        t0 = pd.to_datetime(break_up.iloc[0]["timestamp"])
        window = after[(after["timestamp"] > t0) & (after["timestamp"] <= t0 + pd.Timedelta(minutes=30))]
        recover = _first_after(window, window["close"] < or_high)
        if recover:
            return {
                "name": f"Failed ORB {minutes}m",
                "direction": "Bearish",
                "time": _fmt_ts(recover[0]),
                "price": recover[1],
                "timeframe": "5m",
                "details": "Break above OR failed and closed back inside within 30 minutes.",
            }

    break_down = after[after["low"] < or_low]
    if not break_down.empty:
        t0 = pd.to_datetime(break_down.iloc[0]["timestamp"])
        window = after[(after["timestamp"] > t0) & (after["timestamp"] <= t0 + pd.Timedelta(minutes=30))]
        recover = _first_after(window, window["close"] > or_low)
        if recover:
            return {
                "name": f"Failed ORB {minutes}m",
                "direction": "Bullish",
                "time": _fmt_ts(recover[0]),
                "price": recover[1],
                "timeframe": "5m",
                "details": "Break below OR failed and closed back inside within 30 minutes.",
            }

    return None


def _detect_vwap_trigger(df: pd.DataFrame) -> Optional[Dict[str, object]]:
    if df is None or df.empty or len(df) < 6:
        return None

    vwap = compute_daily_vwap(df)
    if vwap.empty:
        return None

    closes = df["close"].values
    vwap_vals = vwap.values
    ts_vals = pd.to_datetime(df["timestamp"]).values

    for i in range(1, len(closes) - 4):
        prev = closes[i - 1] - vwap_vals[i - 1]
        curr = closes[i] - vwap_vals[i]

        if prev <= 0 and curr > 0:
            if all(closes[i : i + 4] > vwap_vals[i : i + 4]):
                return {
                    "name": "VWAP Reclaim Hold",
                    "direction": "Bullish",
                    "time": _fmt_ts(pd.Timestamp(ts_vals[i + 3])),
                    "price": float(closes[i + 3]),
                    "timeframe": "1m-5m",
                    "details": "Crossed above VWAP and held 4 consecutive closes above.",
                }

        if prev >= 0 and curr < 0:
            if all(closes[i : i + 4] < vwap_vals[i : i + 4]):
                return {
                    "name": "VWAP Reject Hold",
                    "direction": "Bearish",
                    "time": _fmt_ts(pd.Timestamp(ts_vals[i + 3])),
                    "price": float(closes[i + 3]),
                    "timeframe": "1m-5m",
                    "details": "Crossed below VWAP and held 4 consecutive closes below.",
                }

    return None


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _infer_bar_minutes(df: pd.DataFrame, default_minutes: int = 5) -> int:
    if df is None or df.empty or "timestamp" not in df.columns or len(df) < 3:
        return default_minutes
    ts = pd.to_datetime(df["timestamp"]).sort_values()
    deltas = ts.diff().dropna().dt.total_seconds() / 60.0
    deltas = deltas[(deltas >= 1) & (deltas <= 60)]
    if deltas.empty:
        return default_minutes
    mode = deltas.mode()
    minutes = int(round(float(mode.iloc[0]))) if not mode.empty else default_minutes
    return max(1, minutes)


def _us_open_reclaim_watch(
    df: pd.DataFrame,
    trade_date: dt.date,
    now: dt.datetime,
    confluences: Optional[List[Dict[str, object]]] = None,
) -> Dict[str, object]:
    start_0930 = pd.Timestamp.combine(trade_date, dt.time(9, 30))
    end_1030 = pd.Timestamp.combine(trade_date, dt.time(10, 30))
    confluences = confluences or []

    base = {
        "name": "US Open Reclaim Pattern",
        "status": "Not Active",
        "direction": "Bullish",
        "confidence": 0.0,
        "reason": "Waiting for US open structure.",
        "checklist": [
            "9:30 candle shows two-sided rejection (long upper and lower wick).",
            "9:31 candle closes green.",
            "Price reclaims daily VWAP by 10:30 ET.",
        ],
        "entry": None,
        "stop": None,
        "targets": [],
        "invalidation": "No VWAP reclaim by 10:30 ET or a break below setup low after trigger.",
        "anticipation_factors": [],
        "provisional_plan": {
            "entry_rule": "After 9:31, first close back above Daily VWAP.",
            "stop_rule": "Lowest low from 9:30 through trigger candle.",
            "target_rule": "Target 1 = pre-open high (09:00-09:29), Target 2 = 2R extension.",
        },
        "score_breakdown": [],
        "penalty_points": 0.0,
        "model_settings": {
            "candle3_strictness": OPEN_PATTERN_CANDLE3_STRICTNESS,
            "reclaim_speed_window_minutes": OPEN_PATTERN_RECLAIM_SPEED_WINDOW_MINUTES,
        },
    }

    if df is None or df.empty:
        base["reason"] = "No intraday bars available."
        return base

    work = df.copy().sort_values("timestamp").reset_index(drop=True)
    work["timestamp"] = pd.to_datetime(work["timestamp"])
    bar_minutes = _infer_bar_minutes(work, default_minutes=5)

    second_candle_ts = start_0930 + pd.Timedelta(minutes=bar_minutes)
    third_candle_ts = second_candle_ts + pd.Timedelta(minutes=bar_minutes)
    if bar_minutes <= 1:
        base["checklist"] = [
            "9:30 candle shows two-sided rejection (long upper and lower wick).",
            "9:31 candle closes green.",
            "Price reclaims daily VWAP by 10:30 ET.",
        ]
    else:
        base["checklist"] = [
            "9:30 candle shows sweep/rejection behavior.",
            "Next 5m candle is bullish or followed by a strong candle-3 close above candle-2 body/high.",
            "Price reclaims/accepts above daily VWAP during the open window.",
        ]

    pre_open_window = work[
        (work["timestamp"] >= pd.Timestamp.combine(trade_date, dt.time(8, 0)))
        & (work["timestamp"] < start_0930)
    ]

    vwap = compute_daily_vwap(work)
    if not vwap.empty:
        work["daily_vwap"] = vwap.values
    else:
        work["daily_vwap"] = pd.NA

    component_max = {
        "Sweep Quality": 20.0,
        "VWAP Reclaim Quality": 20.0,
        "Confirmation Candle": 15.0,
        "Retest Behavior": 15.0,
        "Structure Alignment": 15.0,
        "Liquidity Path Clarity": 10.0,
    }

    def _build_score(components: Dict[str, float], notes: Dict[str, str], penalty_points: float) -> Tuple[float, List[Dict[str, object]]]:
        rows: List[Dict[str, object]] = []
        total = 0.0
        for name, max_score in component_max.items():
            val = max(0.0, min(float(components.get(name, 0.0)), max_score))
            total += val
            rows.append(
                {
                    "component": name,
                    "score": round(val, 1),
                    "max": max_score,
                    "note": notes.get(name, ""),
                }
            )
        total -= max(0.0, min(float(penalty_points), 15.0))
        return max(0.0, min(total, 100.0)), rows

    if now < start_0930:
        factors: List[str] = []
        notes: Dict[str, str] = {}
        comp = {name: 0.0 for name in component_max.keys()}
        penalty_points = 0.0

        if not pre_open_window.empty:
            pre_high = _safe_float(pre_open_window["high"].max())
            pre_low = _safe_float(pre_open_window["low"].min())
            pre_range = max(pre_high - pre_low, 1e-6)
            last_close = _safe_float(pre_open_window["close"].iloc[-1])
            last_open = _safe_float(pre_open_window["open"].iloc[-1])

            early = pre_open_window[pre_open_window["timestamp"] < pd.Timestamp.combine(trade_date, dt.time(9, 0))]
            late = pre_open_window[pre_open_window["timestamp"] >= pd.Timestamp.combine(trade_date, dt.time(9, 0))]

            if not early.empty and not late.empty:
                early_low = _safe_float(early["low"].min())
                late_low = _safe_float(late["low"].min())
                recovery_ratio = (last_close - late_low) / pre_range
                if late_low < early_low:
                    comp["Sweep Quality"] = min(12.0 + max(recovery_ratio, 0.0) * 16.0, 20.0)
                    notes["Sweep Quality"] = "Pre-open low sweep and recovery detected."
                    factors.append("Pre-open swept earlier lows and recovered (bullish sweep behavior).")
                else:
                    comp["Sweep Quality"] = 6.0
                    notes["Sweep Quality"] = "No clear pre-open sweep; partial credit only."
            else:
                comp["Sweep Quality"] = 5.0
                notes["Sweep Quality"] = "Limited early/late pre-open segmentation available."

            if pd.notna(work["daily_vwap"].iloc[-1]):
                vwap_last = _safe_float(work["daily_vwap"].iloc[-1])
                dist = abs(last_close - vwap_last)
                if last_close <= vwap_last:
                    comp["VWAP Reclaim Quality"] = min(16.0 + max(0.0, 1.0 - dist / max(pre_range, 1e-6)) * 4.0, 20.0)
                    notes["VWAP Reclaim Quality"] = "Below/near VWAP pre-open with reclaim potential."
                    factors.append("Price is at/below Daily VWAP pre-open (reclaim potential).")
                else:
                    comp["VWAP Reclaim Quality"] = max(6.0, 12.0 - min(dist / max(pre_range, 1e-6), 1.0) * 6.0)
                    notes["VWAP Reclaim Quality"] = "Already above VWAP; less reclaim edge but still actionable."
                    factors.append("Price already above Daily VWAP pre-open (less reclaim distance).")
            else:
                comp["VWAP Reclaim Quality"] = 8.0
                notes["VWAP Reclaim Quality"] = "VWAP unavailable; using reduced default score."

            body_ratio = abs(last_close - last_open) / pre_range
            if body_ratio > 0.18:
                comp["Confirmation Candle"] = min(4.0 + body_ratio * 30.0, 9.0)
                notes["Confirmation Candle"] = "Premarket impulse candle present; open confirmation pending."
            else:
                comp["Confirmation Candle"] = 3.0
                notes["Confirmation Candle"] = "No strong pre-open impulse candle yet."

            late_window = pre_open_window[pre_open_window["timestamp"] >= pd.Timestamp.combine(trade_date, dt.time(9, 15))]
            if not late_window.empty:
                lows = late_window["low"].astype(float).values
                higher_low = lows[-1] >= min(lows)
                comp["Retest Behavior"] = 8.0 if higher_low else 4.0
                notes["Retest Behavior"] = "Pre-open pullbacks are stabilizing." if higher_low else "Pre-open retests still unstable."
            else:
                comp["Retest Behavior"] = 4.0
                notes["Retest Behavior"] = "Insufficient late pre-open bars for retest assessment."

            if not pre_open_window.empty and len(pre_open_window) >= 8:
                recent = pre_open_window.tail(8)
                first_mid = (_safe_float(recent.iloc[0]["open"]) + _safe_float(recent.iloc[0]["close"])) / 2.0
                last_mid = (_safe_float(recent.iloc[-1]["open"]) + _safe_float(recent.iloc[-1]["close"])) / 2.0
                slope_up = last_mid > first_mid
                comp["Structure Alignment"] = 10.0 if slope_up else 5.0
                notes["Structure Alignment"] = "Short-term pre-open slope is rising." if slope_up else "Pre-open slope not yet aligned bullish."
            else:
                comp["Structure Alignment"] = 5.0
                notes["Structure Alignment"] = "Not enough bars for structure slope estimate."

            bullish_confluences = [
                c
                for c in confluences
                if str(c.get("Status", "")).lower() != "invalidated" and str(c.get("Side", "")).lower() == "bullish"
            ]
            near = False
            if bullish_confluences:
                for c in bullish_confluences[:7]:
                    low_px = _safe_float(c.get("Price Low"))
                    high_px = _safe_float(c.get("Price High"))
                    if low_px - 10.0 <= last_close <= high_px + 10.0:
                        near = True
                        break
            if near:
                comp["Liquidity Path Clarity"] = 8.0
                notes["Liquidity Path Clarity"] = "Price is near bullish confluence with upside path to pre-open high."
                factors.append("Nearby bullish liquidity confluence detected.")
            else:
                comp["Liquidity Path Clarity"] = 4.0
                notes["Liquidity Path Clarity"] = "Liquidity path less clear pre-open."

            if abs(last_close - (pre_low + pre_range * 0.5)) < pre_range * 0.10:
                penalty_points += 4.0
                factors.append("Penalty: pre-open close near range midpoint (less directional edge).")

            if pd.notna(work["daily_vwap"].iloc[-1]) and len(pre_open_window) >= 20:
                recent = pre_open_window.tail(20).copy()
                rv = work[(work["timestamp"].isin(recent["timestamp"]))].copy()
                if not rv.empty and "daily_vwap" in rv.columns:
                    side = (rv["close"].astype(float) > rv["daily_vwap"].astype(float)).astype(int)
                    flips = int((side.diff().abs().fillna(0) > 0).sum())
                    if flips >= 6:
                        penalty_points += 5.0
                        factors.append("Penalty: frequent VWAP side flips pre-open (chop risk).")

            base["provisional_plan"] = {
                "entry_rule": "If 9:30/9:31 structure confirms, enter on first close above Daily VWAP.",
                "stop_rule": f"Use setup low; pre-open reference low is {pre_low:.2f}.",
                "target_rule": f"Target 1 near pre-open high {pre_high:.2f}; Target 2 at 2R extension.",
            }
        else:
            factors.append("Limited pre-open bars; use reduced confidence.")
            comp["Sweep Quality"] = 4.0
            comp["VWAP Reclaim Quality"] = 6.0
            comp["Confirmation Candle"] = 2.0
            comp["Retest Behavior"] = 2.0
            comp["Structure Alignment"] = 3.0
            comp["Liquidity Path Clarity"] = 2.0
            notes["Sweep Quality"] = "No pre-open structure data."
            notes["VWAP Reclaim Quality"] = "No pre-open structure data."
            notes["Confirmation Candle"] = "No pre-open structure data."
            notes["Retest Behavior"] = "No pre-open structure data."
            notes["Structure Alignment"] = "No pre-open structure data."
            notes["Liquidity Path Clarity"] = "No pre-open structure data."

        total_score, breakdown = _build_score(comp, notes, penalty_points)

        base["status"] = "Watch"
        base["confidence"] = round(min(total_score, 70.0), 1)
        base["reason"] = "Pre-open anticipation mode: monitor open for sweep/reclaim confirmation."
        base["anticipation_factors"] = factors
        base["score_breakdown"] = breakdown
        base["penalty_points"] = round(min(max(penalty_points, 0.0), 15.0), 1)
        return base

    c1_df = work[work["timestamp"] == start_0930]
    c2_df = work[work["timestamp"] == second_candle_ts]
    c3_df = work[work["timestamp"] == third_candle_ts]

    if c1_df.empty:
        base["status"] = "Invalidated"
        base["reason"] = "Missing 9:30 candle data for this date."
        return base

    c1 = c1_df.iloc[0]
    c1_high = _safe_float(c1["high"])
    c1_low = _safe_float(c1["low"])
    c1_open = _safe_float(c1["open"])
    c1_close = _safe_float(c1["close"])
    c1_range = max(c1_high - c1_low, 1e-6)
    c1_upper = c1_high - max(c1_open, c1_close)
    c1_lower = min(c1_open, c1_close) - c1_low
    c1_body = abs(c1_close - c1_open)
    wick_ok = c1_upper >= 0.20 * c1_range and c1_lower >= 0.20 * c1_range and c1_body <= 0.60 * c1_range

    notes: Dict[str, str] = {}
    comp = {name: 0.0 for name in component_max.keys()}
    penalty_points = 0.0

    pre_open_low = _safe_float(pre_open_window["low"].min()) if not pre_open_window.empty else c1_low
    pre_open_high = _safe_float(pre_open_window["high"].max()) if not pre_open_window.empty else c1_high

    lower_ratio = c1_lower / c1_range
    upper_ratio = c1_upper / c1_range
    sweep_bonus = 5.0 if c1_low < pre_open_low else 0.0
    comp["Sweep Quality"] = min(max(lower_ratio * 10.0 + upper_ratio * 5.0 + sweep_bonus, 0.0), 20.0)
    notes["Sweep Quality"] = "9:30 two-sided rejection quality plus low sweep context."

    if c2_df.empty:
        base["status"] = "Watch"
        comp["Confirmation Candle"] = 4.0 if wick_ok else 1.5
        notes["Confirmation Candle"] = "Waiting for second open-window candle confirmation."
        comp["Retest Behavior"] = 3.0
        notes["Retest Behavior"] = "Retest sequence not formed yet."
        comp["Structure Alignment"] = 6.0 if wick_ok else 3.0
        notes["Structure Alignment"] = "Provisional structure read after 9:30 only."
        comp["Liquidity Path Clarity"] = 5.0 if pre_open_high > c1_close else 3.0
        notes["Liquidity Path Clarity"] = "Path estimated from pre-open highs."
        comp["VWAP Reclaim Quality"] = 5.0
        notes["VWAP Reclaim Quality"] = "VWAP reclaim not evaluated yet."
        total_score, breakdown = _build_score(comp, notes, penalty_points)
        base["confidence"] = round(min(total_score, 74.0), 1)
        base["reason"] = "Open candle printed; waiting for second-candle confirmation."
        base["score_breakdown"] = breakdown
        base["penalty_points"] = 0.0
        return base

    c2 = c2_df.iloc[0]
    c2_open = _safe_float(c2["open"])
    c2_close = _safe_float(c2["close"])
    c2_high = _safe_float(c2["high"])
    c2_green = c2_close > c2_open

    c3_close = _safe_float(c3_df.iloc[0]["close"]) if not c3_df.empty else c2_close
    c2_body_top = max(c2_open, c2_close)
    c2_body_mid = (c2_open + c2_close) / 2.0
    c3_break_mid = c3_close > c2_body_mid
    c3_break_body = c3_close > c2_body_top
    c3_break_high = c3_close > c2_high

    strictness = str(OPEN_PATTERN_CANDLE3_STRICTNESS).strip().lower()
    if strictness not in {"conservative", "balanced", "aggressive"}:
        strictness = "balanced"

    if strictness == "conservative":
        c3_confirm = c3_break_high
    elif strictness == "aggressive":
        c3_confirm = c3_break_mid
    else:
        c3_confirm = c3_break_body

    if bar_minutes <= 1:
        setup_confirmed = c2_green
    else:
        setup_confirmed = c2_green or c3_confirm

    if not wick_ok or not setup_confirmed:
        base["status"] = "Invalidated"
        why = []
        if not wick_ok:
            why.append("9:30 candle did not meet two-sided wick quality")
        if not setup_confirmed:
            if bar_minutes <= 1:
                why.append("9:31 did not close green")
            else:
                why.append("second/third open candles did not confirm bullish continuation")
        base["reason"] = "; ".join(why) + "."
        base["confidence"] = 15.0
        comp["Confirmation Candle"] = 0.0
        notes["Confirmation Candle"] = "Open-window candle confirmation failed."
        comp["VWAP Reclaim Quality"] = 2.0
        notes["VWAP Reclaim Quality"] = "Setup invalid before reclaim."
        comp["Retest Behavior"] = 2.0
        notes["Retest Behavior"] = "No valid continuation retest sequence."
        comp["Structure Alignment"] = 2.0
        notes["Structure Alignment"] = "Structure did not confirm."
        comp["Liquidity Path Clarity"] = 2.0
        notes["Liquidity Path Clarity"] = "Path is unclear after invalidation."
        _, breakdown = _build_score(comp, notes, 8.0)
        base["score_breakdown"] = breakdown
        base["penalty_points"] = 8.0
        return base

    body_strength = (c2_close - c2_open) / c1_range
    engulfing = c2_close >= c1_open
    c3_bonus = 0.0
    if bar_minutes > 1:
        if c3_break_body:
            c3_bonus += 2.0
        if c3_break_high:
            c3_bonus += 2.0
    comp["Confirmation Candle"] = min(8.0 + max(body_strength, 0.0) * 10.0 + (4.0 if engulfing else 0.0) + c3_bonus, 15.0)
    notes["Confirmation Candle"] = f"Open-window confirmation strength (candle-3 strictness: {strictness})."

    vwap = compute_daily_vwap(work)
    if vwap.empty:
        base["status"] = "Armed"
        comp["VWAP Reclaim Quality"] = 7.0
        notes["VWAP Reclaim Quality"] = "VWAP unavailable; reduced confidence until reclaim proxy."
        comp["Retest Behavior"] = 5.0
        notes["Retest Behavior"] = "Retest pending."
        comp["Structure Alignment"] = 8.0
        notes["Structure Alignment"] = "Early structure favorable; needs reclaim confirmation."
        comp["Liquidity Path Clarity"] = 6.0
        notes["Liquidity Path Clarity"] = "Upside path based on pre-open high reference."
        total_score, breakdown = _build_score(comp, notes, penalty_points)
        base["confidence"] = round(min(total_score, 78.0), 1)
        base["reason"] = "Setup formed; VWAP unavailable, waiting for reclaim confirmation."
        base["score_breakdown"] = breakdown
        base["penalty_points"] = round(penalty_points, 1)
        return base

    work["daily_vwap"] = vwap.values
    reclaim_window = work[
        (work["timestamp"] >= second_candle_ts)
        & (work["timestamp"] <= end_1030)
        & (work["close"] > work["daily_vwap"])
    ]

    if reclaim_window.empty:
        latest_ts = pd.to_datetime(work["timestamp"].iloc[-1])
        comp["VWAP Reclaim Quality"] = 6.0
        notes["VWAP Reclaim Quality"] = "Reclaim not printed yet; still waiting."
        comp["Retest Behavior"] = 6.0
        notes["Retest Behavior"] = "Post-open retest developing; no reclaim confirmation yet."
        comp["Structure Alignment"] = 8.0
        notes["Structure Alignment"] = "Structure favorable but unconfirmed by VWAP acceptance."
        comp["Liquidity Path Clarity"] = 6.0
        notes["Liquidity Path Clarity"] = "Path remains provisional until reclaim."
        total_score, breakdown = _build_score(comp, notes, penalty_points)
        if latest_ts <= end_1030:
            base["status"] = "Armed"
            base["confidence"] = round(min(total_score, 82.0), 1)
            base["reason"] = "Setup confirmed; waiting for first close above VWAP."
        else:
            base["status"] = "Invalidated"
            base["confidence"] = 22.0
            base["reason"] = "No VWAP reclaim by 10:30 ET."
            penalty_points += 9.0
        base["score_breakdown"] = breakdown
        base["penalty_points"] = round(min(max(penalty_points, 0.0), 15.0), 1)
        return base

    trigger = reclaim_window.iloc[0]
    trigger_ts = pd.to_datetime(trigger["timestamp"])
    entry = _safe_float(trigger["close"])

    stop_window = work[(work["timestamp"] >= start_0930) & (work["timestamp"] <= trigger_ts)]
    stop_price = _safe_float(stop_window["low"].min()) if not stop_window.empty else _safe_float(c1_low)
    risk = max(entry - stop_price, 0.25)

    pre_open_high = _safe_float(pre_open_window["high"].max()) if not pre_open_window.empty else entry + risk
    if pre_open_high <= entry:
        pre_open_high = entry + risk

    target_1 = pre_open_high
    target_2 = entry + 2.0 * risk
    rr_1 = (target_1 - entry) / risk

    reclaim_delay = max((trigger_ts - second_candle_ts).total_seconds() / 60.0, 0.0)
    configured_speed = max(float(OPEN_PATTERN_RECLAIM_SPEED_WINDOW_MINUTES), 10.0)
    speed_norm = max(12.0, configured_speed * 0.6) if bar_minutes <= 1 else configured_speed
    reclaim_speed_score = max(0.0, 1.0 - min(reclaim_delay / speed_norm, 1.0))
    reclaim_distance = max(entry - _safe_float(trigger["daily_vwap"]), 0.0)
    comp["VWAP Reclaim Quality"] = min(10.0 + reclaim_speed_score * 6.0 + min(reclaim_distance / max(risk, 1e-6), 1.0) * 4.0, 20.0)
    notes["VWAP Reclaim Quality"] = f"Scored from reclaim speed (window {speed_norm:.0f}m) and close strength above VWAP."

    post_trigger_minutes = 8 if bar_minutes <= 1 else 20
    post_trigger = work[(work["timestamp"] > trigger_ts) & (work["timestamp"] <= trigger_ts + pd.Timedelta(minutes=post_trigger_minutes))].copy()
    retest_score = 6.0
    if not post_trigger.empty:
        touched_vwap = (post_trigger["low"] <= post_trigger["daily_vwap"]).any()
        held_after_touch = False
        if touched_vwap:
            touch_idx = post_trigger[post_trigger["low"] <= post_trigger["daily_vwap"]].index[0]
            after_touch = post_trigger.loc[touch_idx:]
            held_after_touch = bool((after_touch["close"] >= after_touch["daily_vwap"]).all())
        if touched_vwap and held_after_touch:
            retest_score = 14.0
        elif touched_vwap:
            retest_score = 8.0
            penalty_points += 3.0
        else:
            retest_score = 9.0
    comp["Retest Behavior"] = min(retest_score, 15.0)
    notes["Retest Behavior"] = "Scored from post-reclaim VWAP retest and hold behavior."

    pre_trigger = work[(work["timestamp"] >= start_0930) & (work["timestamp"] <= trigger_ts)]
    recent_high = _safe_float(pre_trigger["high"].tail(6).max()) if not pre_trigger.empty else entry
    structure_break = entry >= recent_high
    comp["Structure Alignment"] = 12.0 if structure_break else 8.0
    notes["Structure Alignment"] = "Scored from local structure break strength into trigger."

    bearish_blocks_between = 0
    for c in confluences[:10]:
        side = str(c.get("Side", "")).lower()
        status = str(c.get("Status", "")).lower()
        if side != "bearish" or status == "invalidated":
            continue
        low_px = _safe_float(c.get("Price Low"))
        if entry < low_px < target_1:
            bearish_blocks_between += 1
    path_score = 10.0 - min(bearish_blocks_between * 2.5, 7.5)
    path_score += min(max(rr_1 - 1.0, 0.0) * 1.5, 2.5)
    comp["Liquidity Path Clarity"] = max(2.0, min(path_score, 10.0))
    notes["Liquidity Path Clarity"] = "Scored by target path friction and R multiple quality."

    if rr_1 < 1.0:
        penalty_points += 5.0
    if not post_trigger.empty:
        adverse = float((entry - post_trigger["low"]).max()) if len(post_trigger) else 0.0
        if adverse > 0.8 * risk:
            penalty_points += 4.0

    confidence, breakdown = _build_score(comp, notes, penalty_points)

    base["status"] = "Triggered"
    base["confidence"] = round(confidence, 1)
    if bar_minutes <= 1:
        base["reason"] = "9:30/9:31 setup formed and first close above daily VWAP confirmed the trigger."
    else:
        base["reason"] = "Open-window setup formed and first close above daily VWAP confirmed the trigger."
    base["entry"] = {
        "time": _fmt_ts(trigger_ts),
        "price": round(entry, 2),
    }
    base["stop"] = {
        "price": round(stop_price, 2),
        "rule": "Lowest low from 9:30 through trigger candle.",
    }
    base["targets"] = [
        {
            "name": "Target 1 (Pre-open High)",
            "price": round(target_1, 2),
            "rr": round(rr_1, 2),
        },
        {
            "name": "Target 2 (2R Extension)",
            "price": round(target_2, 2),
            "rr": 2.0,
        },
    ]
    base["anticipation_factors"] = [
        "Post-open confirmation complete: 9:30/9:31 structure + VWAP reclaim.",
        f"Pre-open high reference used for Target 1 ({target_1:.2f}).",
    ]
    base["score_breakdown"] = breakdown
    base["penalty_points"] = round(min(max(penalty_points, 0.0), 15.0), 1)
    return base


def _zone_touch_times(df: pd.DataFrame, zone: Zone) -> Tuple[Optional[str], Optional[str], int]:
    if df is None or df.empty:
        return None, None, 0
    formed_ts = zone_formed_timestamp(zone)
    after = df[df["timestamp"] > formed_ts]
    if after.empty:
        return None, None, 0
    hits = after[(after["low"] <= zone.high) & (after["high"] >= zone.low)]
    if hits.empty:
        return None, None, 0
    first = _fmt_ts(pd.to_datetime(hits.iloc[0]["timestamp"]))
    retest = _fmt_ts(pd.to_datetime(hits.iloc[1]["timestamp"])) if len(hits) > 1 else None
    return first, retest, max(len(hits) - 1, 0)


def _stdv_levels(df: pd.DataFrame, center: float, length: int = 20) -> List[Dict[str, object]]:
    if df is None or df.empty:
        return []
    std = df["close"].rolling(length).std()
    if std.empty or pd.isna(std.iloc[-1]):
        return []
    sigma = float(std.iloc[-1])
    if sigma <= 0:
        return []

    day_high = float(df["high"].max())
    day_low = float(df["low"].min())

    levels: List[Dict[str, object]] = []
    for n in [1, 2, 3]:
        up = center + n * sigma
        down = center - n * sigma
        levels.append(
            {
                "Level": f"+{n}σ",
                "Price": float(up),
                "From Center": float(n * sigma),
                "Status": "Reached" if day_high >= up else "Untouched",
            }
        )
        levels.append(
            {
                "Level": f"-{n}σ",
                "Price": float(down),
                "From Center": float(n * sigma),
                "Status": "Reached" if day_low <= down else "Untouched",
            }
        )
    return levels


def _liquidity_level(score: float) -> str:
    if score < 0.25:
        return "Very Low"
    if score < 0.60:
        return "Low"
    if score < 1.00:
        return "Medium"
    if score < 1.50:
        return "High"
    if score < 2.20:
        return "Very High"
    return "Major"


def _enrich_trigger(trigger: Dict[str, object]) -> Dict[str, object]:
    name = str(trigger.get("name", ""))
    if "VWAP Reject Hold" in name:
        trigger["what_happened"] = "Price crossed below VWAP and held below it for 4 consecutive closes."
        trigger["must_happen"] = "On a pullback, price should fail to reclaim VWAP."
        trigger["execution_look_for"] = "Bearish retest rejection candle at/under VWAP plus momentum expansion lower."
        trigger["invalidation"] = "A sustained close back above VWAP."
        return trigger
    if "VWAP Reclaim Hold" in name:
        trigger["what_happened"] = "Price crossed above VWAP and held above it for 4 consecutive closes."
        trigger["must_happen"] = "On a pullback, price should hold above VWAP."
        trigger["execution_look_for"] = "Bullish hold/rejection candle above VWAP plus momentum expansion higher."
        trigger["invalidation"] = "A sustained close back below VWAP."
        return trigger
    if "Failed ORB" in name:
        trigger["what_happened"] = "An OR breakout failed and price closed back inside the OR window."
        trigger["must_happen"] = "Price should continue away from the failed breakout side."
        trigger["execution_look_for"] = "Retest failure of OR edge and continuation in failure direction."
        trigger["invalidation"] = "Re-break and hold beyond original OR breakout side."
        return trigger
    if "ORB" in name:
        trigger["what_happened"] = "Price closed decisively beyond the opening range threshold."
        trigger["must_happen"] = "Retest should hold on the breakout side."
        trigger["execution_look_for"] = "Break-and-retest continuation with no immediate rejection back into OR."
        trigger["invalidation"] = "Close back inside OR with failed follow-through."
        return trigger

    trigger["what_happened"] = trigger.get("details", "Trigger condition was detected.")
    trigger["must_happen"] = "Price must continue to respect the trigger side."
    trigger["execution_look_for"] = "A confirmed retest and continuation candle sequence."
    trigger["invalidation"] = "Clear rejection of trigger direction."
    return trigger


def _pick_primary_trigger(triggers: List[Dict[str, object]], direction: str) -> Optional[Dict[str, object]]:
    if not triggers:
        return None
    if direction in {"Bullish", "Bearish"}:
        aligned = [t for t in triggers if t.get("direction") == direction]
        if aligned:
            return aligned[0]
    return triggers[0]


def _select_reference_confluences(
    confluences: List[Dict[str, object]],
    direction: str,
    max_items: int = 2,
) -> List[str]:
    if not confluences:
        return []

    filtered = [c for c in confluences if c.get("Status") != "Invalidated"]
    if direction == "Bullish":
        directional = [c for c in filtered if str(c.get("Side", "")).lower() == "bullish"]
    elif direction == "Bearish":
        directional = [c for c in filtered if str(c.get("Side", "")).lower() == "bearish"]
    else:
        directional = filtered

    candidates = directional if directional else filtered
    candidates = sorted(
        candidates,
        key=lambda x: (not bool(x.get("Top Liquidity", False)), -float(x.get("Liquidity Score", 0.0))),
    )

    refs: List[str] = []
    for row in candidates[:max_items]:
        refs.append(
            f"{row.get('Confluence')} ({float(row.get('Price Low', 0.0)):.2f}-{float(row.get('Price High', 0.0)):.2f})"
        )
    return refs


def _select_reference_confluence_rows(
    confluences: List[Dict[str, object]],
    direction: str,
    max_items: int = 2,
) -> List[Dict[str, object]]:
    if not confluences:
        return []

    filtered = [c for c in confluences if c.get("Status") != "Invalidated"]
    if direction == "Bullish":
        directional = [c for c in filtered if str(c.get("Side", "")).lower() == "bullish"]
    elif direction == "Bearish":
        directional = [c for c in filtered if str(c.get("Side", "")).lower() == "bearish"]
    else:
        directional = filtered

    candidates = directional if directional else filtered
    return sorted(
        candidates,
        key=lambda x: (not bool(x.get("Top Liquidity", False)), -float(x.get("Liquidity Score", 0.0))),
    )[:max_items]


def _first_zone_tap_time(
    df: pd.DataFrame,
    zone_low: float,
    zone_high: float,
    formed_time: Optional[str] = None,
) -> Optional[str]:
    if df is None or df.empty:
        return None
    work = df
    if formed_time and formed_time != "n/a":
        formed_ts = pd.to_datetime(formed_time, errors="coerce")
        if pd.notna(formed_ts):
            work = work[pd.to_datetime(work["timestamp"]) > formed_ts]
    hits = work[(work["low"] <= zone_high) & (work["high"] >= zone_low)]
    if hits.empty:
        return None
    return _fmt_ts(pd.to_datetime(hits.iloc[0]["timestamp"]))


def _first_midline_hit_time(
    df: pd.DataFrame,
    midline: float,
    formed_time: Optional[str] = None,
) -> Optional[str]:
    if df is None or df.empty:
        return None
    work = df
    if formed_time and formed_time != "n/a":
        formed_ts = pd.to_datetime(formed_time, errors="coerce")
        if pd.notna(formed_ts):
            work = work[pd.to_datetime(work["timestamp"]) > formed_ts]
    hits = work[(work["low"] <= midline) & (work["high"] >= midline)]
    if hits.empty:
        return None
    return _fmt_ts(pd.to_datetime(hits.iloc[0]["timestamp"]))


def _entry_style_target_prices(
    target_ladder: List[Dict[str, object]],
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    prices: List[float] = []
    for t in target_ladder[:8]:
        try:
            prices.append(float(t.get("Price")))
        except Exception:
            continue

    if not prices:
        return None, None, None

    preferred = prices[0]
    minimum = min(prices)
    maximum = max(prices)
    return preferred, minimum, maximum


def _analyze_zone_reaction_candles(
    df: pd.DataFrame,
    zone_low: float,
    zone_high: float,
    side: str,
    lookahead_bars: int = 3,
) -> Dict[str, object]:
    if df is None or df.empty:
        return {
            "signal": "None",
            "score": 0.0,
            "reason": "No intraday bars available for reaction analysis.",
            "event_time": "n/a",
        }

    work = df.copy().sort_values("timestamp").reset_index(drop=True)
    touches = work[(work["low"] <= zone_high) & (work["high"] >= zone_low)]
    if touches.empty:
        return {
            "signal": "None",
            "score": 0.0,
            "reason": "Zone has not been touched yet.",
            "event_time": "n/a",
        }

    bullish = str(side).lower() == "bullish"
    best_score = -1.0
    best = {
        "signal": "None",
        "score": 0.0,
        "reason": "No qualifying reaction candle found yet.",
        "event_time": "n/a",
    }

    for idx in touches.index[:5]:
        row = work.iloc[idx]
        o = _safe_float(row.get("open"))
        h = _safe_float(row.get("high"))
        l = _safe_float(row.get("low"))
        c = _safe_float(row.get("close"))
        rng = max(h - l, 1e-6)
        body = abs(c - o)
        lower_wick = min(o, c) - l
        upper_wick = h - max(o, c)

        engulfing = False
        rejection = False
        rejection_score = 0.0
        engulfing_score = 0.0

        if idx > 0:
            prev = work.iloc[idx - 1]
            po = _safe_float(prev.get("open"))
            pc = _safe_float(prev.get("close"))
            if bullish:
                engulfing = (c > o) and (pc < po) and (o <= pc) and (c >= po)
            else:
                engulfing = (c < o) and (pc > po) and (o >= pc) and (c <= po)
            if engulfing:
                engulfing_score = 35.0

        if bullish:
            wick_ok = lower_wick >= max(0.8 * body, 0.20 * rng)
            close_away = c >= (zone_low + zone_high) / 2.0
            color_ok = c >= o
            rejection = wick_ok and close_away and color_ok
            if rejection:
                rejection_score = 30.0
        else:
            wick_ok = upper_wick >= max(0.8 * body, 0.20 * rng)
            close_away = c <= (zone_low + zone_high) / 2.0
            color_ok = c <= o
            rejection = wick_ok and close_away and color_ok
            if rejection:
                rejection_score = 30.0

        end_idx = min(idx + max(1, lookahead_bars), len(work) - 1)
        forward = work.iloc[idx : end_idx + 1]
        if bullish:
            departure_points = max(_safe_float(forward["high"].max()) - c, 0.0)
        else:
            departure_points = max(c - _safe_float(forward["low"].min()), 0.0)
        departure_score = min(35.0, departure_points * 4.0)

        score = min(100.0, engulfing_score + rejection_score + departure_score)
        if score > best_score:
            best_score = score
            signal = "None"
            if engulfing and rejection:
                signal = "Engulfing + Rejection"
            elif engulfing:
                signal = "Engulfing"
            elif rejection:
                signal = "Rejection"

            best = {
                "signal": signal,
                "score": round(score, 1),
                "reason": (
                    f"{signal} reaction with departure strength {departure_points:.2f} points "
                    f"over next {max(1, lookahead_bars)} bars."
                    if signal != "None"
                    else f"Touch occurred but no clear engulfing/rejection; departure {departure_points:.2f} points."
                ),
                "event_time": _fmt_ts(pd.to_datetime(row["timestamp"])) or "n/a",
            }

    return best


def _suggest_confluence_entry_styles(
    reference_rows: List[Dict[str, object]],
    df: pd.DataFrame,
    target_ladder: List[Dict[str, object]],
    mode: str,
    whipsaw_risk: bool,
) -> List[Dict[str, object]]:
    suggestions: List[Dict[str, object]] = []
    preferred_tgt, min_tgt, max_tgt = _entry_style_target_prices(target_ladder)
    for row in reference_rows:
        name = str(row.get("Confluence", "n/a"))
        name_l = name.lower()
        low = float(row.get("Price Low", 0.0))
        high = float(row.get("Price High", 0.0))
        mid = (low + high) / 2.0
        width = abs(high - low)
        status = str(row.get("Status", "Fresh"))
        formed_time = str(row.get("Formed Time", "n/a"))

        tap_time = _first_zone_tap_time(df, low, high, formed_time=formed_time)
        midline_time = _first_midline_hit_time(df, mid, formed_time=formed_time)
        tap_hit = tap_time is not None
        midline_hit = midline_time is not None
        invalidated = status == "Invalidated"

        zone_kind = "Other"
        if "fvg" in name_l:
            zone_kind = "FVG"
        elif "ob" in name_l or "order block" in name_l:
            zone_kind = "OB"
        elif "bb" in name_l or "breaker" in name_l:
            zone_kind = "BB"

        style = "First Tap"
        reason = "Momentum context favors immediate reaction entries."

        reaction = _analyze_zone_reaction_candles(
            df=df,
            zone_low=low,
            zone_high=high,
            side=str(row.get("Side", "")),
            lookahead_bars=3,
        )
        reaction_signal = str(reaction.get("signal", "None"))
        reaction_score = float(reaction.get("score", 0.0) or 0.0)
        reaction_reason = str(reaction.get("reason", "n/a"))
        reaction_time = str(reaction.get("event_time", "n/a"))

        if zone_kind == "FVG":
            style = "Midline"
            reason = "FVG setups typically improve fill quality near the 50% balance point."
        elif zone_kind in {"OB", "BB"}:
            style = "First Tap"
            reason = "OB/BB zones often react on first contact during directional conditions."

        if status in {"Retested", "Tested"}:
            style = "Midline"
            reason = "Zone has already been interacted with; waiting for midline can reduce weak edge taps."

        if whipsaw_risk or mode == "Whipsaw Risk":
            style = "Midline"
            reason = "Whipsaw conditions favor more selective pricing over immediate touch entries."

        if width < 1.5 and style == "Midline":
            style = "First Tap"
            reason = "Zone is narrow enough that midline and tap fills are effectively equivalent."

        # Adaptive FVG mode: strong reaction at touch prioritizes participation over perfect fill.
        if zone_kind == "FVG":
            if reaction_score >= 70.0 and reaction_signal in {"Engulfing", "Rejection", "Engulfing + Rejection"}:
                style = "First Tap"
                reason = f"Strong {reaction_signal.lower()} reaction detected at the zone ({reaction_score:.1f} score)."
            elif 45.0 <= reaction_score < 70.0 and reaction_signal in {"Engulfing", "Rejection", "Engulfing + Rejection"}:
                style = "Split (Tap + Midline)"
                reason = f"Moderate {reaction_signal.lower()} reaction suggests partial fill at tap and add near midline."

        if preferred_tgt is not None:
            exit_plan = f"Scale at preferred target {preferred_tgt:.2f}; extend toward {max_tgt:.2f}."
        else:
            exit_plan = "Use nearest qualified ladder target; exit early on invalidation."

        suggestions.append(
            {
                "Confluence": name,
                "Kind": zone_kind,
                "Status": status,
                "Formed Time": formed_time,
                "Suggested Entry": style,
                "First Tap Price": round(low if "bullish" in name_l else high, 2),
                "Midline Price": round(mid, 2),
                "Tap Hit": "Yes" if tap_hit else "No",
                "Tap Time": tap_time or "n/a",
                "Midline Hit": "Yes" if midline_hit else "No",
                "Midline Time": midline_time or "n/a",
                "Zone Invalidated": "Yes" if invalidated else "No",
                "Exit": exit_plan,
                "Reaction Signal": reaction_signal,
                "Reaction Score": round(reaction_score, 1),
                "Reaction Time": reaction_time,
                "Reaction Why": reaction_reason,
                "Preferred Target Price": round(preferred_tgt, 2) if preferred_tgt is not None else None,
                "Minimum Target Price": round(min_tgt, 2) if min_tgt is not None else None,
                "Maximum Target Price": round(max_tgt, 2) if max_tgt is not None else None,
                "Entry Style Reason": reason,
                "Reason": reason,
            }
        )
    return suggestions


def _estimate_daily_entry_capacity(
    mode: str,
    direction: str,
    trade_today: str,
    confidence: float,
    reference_rows: List[Dict[str, object]],
    primary_trigger: Optional[Dict[str, object]],
    now: dt.datetime,
    windows: Dict[str, Dict[str, dt.datetime]],
) -> Dict[str, object]:
    us_start = windows.get("US", {}).get("start")
    us_end = windows.get("US", {}).get("end")

    if trade_today == "No":
        return {
            "expected_min": 0,
            "expected_max": 0,
            "likely": 0,
            "remaining_estimate": 0,
            "planning_note": "Trade-today decision is No, so no entries are planned.",
            "eod_execution_status": "Not Executed",
            "eod_execution_note": "Trade-today decision prevented execution.",
        }

    min_e, max_e = 0, 2
    if mode in {"AMD", "NY Continuation"} and direction in {"Bullish", "Bearish"}:
        min_e, max_e = 1, 3
    elif mode == "Whipsaw Risk":
        min_e, max_e = 0, 1

    if direction not in {"Bullish", "Bearish"}:
        min_e = 0
        max_e = min(max_e, 2)

    top_liq_count = sum(1 for r in reference_rows if bool(r.get("Top Liquidity", False)))
    if top_liq_count >= 2 and mode != "Whipsaw Risk":
        max_e = min(max_e + 1, 4)

    if primary_trigger is not None and trade_today in {"Yes", "Wait"}:
        min_e = max(min_e, 1)

    if us_start and now < us_start:
        note = "Pre-US open estimate based on mode, directional alignment, and confluence quality."
        eod_status = "Pending"
        eod_note = "Session is not complete yet."
    elif us_end and now > us_end:
        if primary_trigger is None:
            min_e, max_e = 0, 0
            note = "US session has ended without a trigger; no additional entries expected."
            eod_status = "Not Executed"
            eod_note = "No qualifying trigger was confirmed by session close."
        else:
            min_e, max_e = min(min_e, 1), min(max_e, 1)
            note = "US session is largely complete; at most one late opportunity remains."
            eod_status = "Executed"
            eod_note = "A qualifying trigger fired during session hours."
    else:
        note = "In-session estimate adjusts as trigger state and confluence validity evolve."
        eod_status = "Pending"
        eod_note = "Waiting for end-of-day session close to finalize execution status."

    if confidence >= 0.70:
        likely = min(max(min_e + 1, min_e), max_e)
    elif confidence <= 0.55:
        likely = min_e
    else:
        likely = min(min_e + 1, max_e)

    return {
        "expected_min": int(min_e),
        "expected_max": int(max_e),
        "likely": int(likely),
        "remaining_estimate": int(max_e),
        "planning_note": note,
        "eod_execution_status": eod_status,
        "eod_execution_note": eod_note,
    }


def _entry_blueprints(
    mode: str,
    direction: str,
    windows: Dict[str, Dict[str, dt.datetime]],
    reference_confluences: List[str],
    vwap_levels: List[Dict[str, object]],
    entry_style_suggestions: Optional[List[Dict[str, object]]] = None,
) -> List[Dict[str, str]]:
    us_start = windows.get("US", {}).get("start")
    us_end = windows.get("US", {}).get("end")
    us_window = f"{us_start:%H:%M}-{us_end:%H:%M} ET" if us_start and us_end else "US session"

    primary_confluence = reference_confluences[0] if reference_confluences else "nearest active confluence"
    secondary_confluence = reference_confluences[1] if len(reference_confluences) > 1 else primary_confluence
    daily_vwap_label = "Daily VWAP"
    for v in vwap_levels:
        if str(v.get("Name", "")).lower().startswith("daily vwap"):
            daily_vwap_label = str(v.get("Name"))
            break

    primary_style = "First Tap"
    primary_style_reason = "Use default confluence tap execution."
    if entry_style_suggestions:
        primary_style = str(entry_style_suggestions[0].get("Suggested Entry", primary_style))
        primary_style_reason = str(entry_style_suggestions[0].get("Reason", primary_style_reason))

    if mode == "Whipsaw Risk":
        return [
            {
                "Setup": "Risk-Off / Two-Sided",
                "IfThen": f"If price keeps crossing around {daily_vwap_label} and cannot hold either side of OR, then avoid directional entries.",
                "Trigger": "Use only exceptional reclaim/reject + structure alignment at high-liquidity confluence.",
                "Confluence Entry": primary_style,
                "Entry Reason": primary_style_reason,
                "Execution TF": "5m",
                "Session": us_window,
            }
        ]

    direction_word = "long" if direction == "Bullish" else "short" if direction == "Bearish" else "directional"
    opposite = "below" if direction == "Bullish" else "above"

    base = [
        {
            "Setup": "VWAP Acceptance",
            "IfThen": f"If price retests {daily_vwap_label} and holds the {direction.lower()} side near {primary_confluence}, then look for {direction_word} continuation.",
            "Trigger": f"VWAP reclaim/reject hold + 1m structure confirmation while staying {opposite} invalidation side.",
            "Confluence Entry": primary_style,
            "Entry Reason": primary_style_reason,
            "Execution TF": "1m",
            "Session": us_window,
        },
        {
            "Setup": "OR Retest",
            "IfThen": f"If OR break aligns {direction.lower()} and retest holds into {secondary_confluence}, then execute continuation.",
            "Trigger": "ORB or Failed ORB confirmation.",
            "Confluence Entry": primary_style,
            "Entry Reason": primary_style_reason,
            "Execution TF": "1m-5m",
            "Session": "09:30-11:30 ET",
        },
    ]

    if mode == "AMD":
        base.insert(
            0,
            {
                "Setup": "AMD Reversal",
                "IfThen": f"If London sweep is rejected and US reclaims through {primary_confluence}, then look for {direction_word} toward opposite liquidity.",
                "Trigger": "Sweep rejection + reclaim + micro BOS/CHOCH.",
                "Confluence Entry": primary_style,
                "Entry Reason": primary_style_reason,
                "Execution TF": "1m",
                "Session": "09:30-12:00 ET",
            },
        )

    if mode == "NY Continuation":
        base.insert(
            0,
            {
                "Setup": "NY Continuation Pullback",
                "IfThen": f"If price pulls back to {primary_confluence} and holds with {daily_vwap_label} support/resistance, then execute {direction_word}.",
                "Trigger": "Retest hold + momentum re-expansion.",
                "Confluence Entry": primary_style,
                "Entry Reason": primary_style_reason,
                "Execution TF": "1m-5m",
                "Session": us_window,
            },
        )

    return base


def _timeframe_playbook(mode: str) -> List[Dict[str, str]]:
    if mode == "Whipsaw Risk":
        return [
            {
                "Context TF": "15m",
                "Setup TF": "5m",
                "Trigger TF": "1m",
                "Execution TF": "5m",
                "Management TF": "1m-5m",
                "Use Case": "Noise control / defensive mode",
            }
        ]

    return [
        {
            "Context TF": "15m-1H",
            "Setup TF": "5m-15m",
            "Trigger TF": "1m-5m",
            "Execution TF": "1m",
            "Management TF": "1m-5m",
            "Use Case": "Primary intraday execution",
        },
        {
            "Context TF": "15m",
            "Setup TF": "5m",
            "Trigger TF": "1m",
            "Execution TF": "5m",
            "Management TF": "5m",
            "Use Case": "High-volatility fallback execution",
        },
    ]


def build_strategy_playbook(
    df_today: pd.DataFrame,
    df_prev: Optional[pd.DataFrame],
    sessions: Dict[str, SessionStats],
    patterns: PatternSummary,
    zones: List[Zone],
    now_et: Optional[dt.datetime] = None,
    whipsaw_threshold: float = 3.0,
) -> Dict[str, object]:
    if df_today is None or df_today.empty:
        return {
            "decision": {
                "trade_today": "Wait",
                "ny_mode": "n/a",
                "ny_direction": "Neutral",
                "confidence": 0.0,
                "primary_reason": "No intraday data available.",
                "supporting_factors": [],
                "blocking_factors": ["No today data"],
            },
            "triggers": [],
            "confluences": [],
            "targets": [],
            "vwap_levels": [],
            "stdv_levels": [],
            "power_hour": {
                "focus": "No",
                "bias": "Neutral",
                "reason": "No data.",
                "entries": [],
            },
            "entry_blueprints": [],
            "timeframe_playbook": [],
            "open_pattern_watch": {
                "name": "US Open Reclaim Pattern",
                "status": "Not Active",
                "direction": "Bullish",
                "confidence": 0.0,
                "reason": "No intraday data available.",
                "checklist": [],
                "entry": None,
                "stop": None,
                "targets": [],
                "invalidation": "n/a",
            },
        }

    df = df_today.copy().sort_values("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    trade_date = pd.to_datetime(df["timestamp"].iloc[-1]).date()
    windows = get_session_windows_for_date(trade_date)
    now = now_et or pd.to_datetime(df["timestamp"].iloc[-1]).to_pydatetime()

    asia = sessions.get("Asia")
    london = sessions.get("London")

    asia_range = (asia.high - asia.low) if asia else None
    london_range = (london.high - london.low) if london else None
    whipsaw_ratio = None
    if asia_range and london_range and asia_range > 0:
        whipsaw_ratio = london_range / asia_range
    whipsaw_risk = bool(whipsaw_ratio is not None and whipsaw_ratio > whipsaw_threshold)

    ny_mode = "Neutral"
    if whipsaw_risk or bool(getattr(patterns, "whipsaw", False)):
        ny_mode = "Whipsaw Risk"
    elif bool(getattr(patterns, "asia_range_sweep", False)):
        ny_mode = "AMD"
    elif bool(getattr(patterns, "london_continuation", False)):
        ny_mode = "NY Continuation"
    elif asia and london:
        if london.close > asia.high or london.close < asia.low:
            ny_mode = "NY Continuation"

    ny_direction = "Neutral"
    if getattr(patterns, "asia_range_sweep_bias", "Neutral") in {"Bullish", "Bearish"}:
        ny_direction = getattr(patterns, "asia_range_sweep_bias")
    elif getattr(patterns, "london_continuation_bias", "Neutral") in {"Bullish", "Bearish"}:
        ny_direction = getattr(patterns, "london_continuation_bias")
    elif getattr(patterns, "power_hour_bias", "Neutral") in {"Bullish", "Bearish"}:
        ny_direction = getattr(patterns, "power_hour_bias")
    elif london:
        move = london.close - london.open
        ny_direction = "Bullish" if move > 0 else "Bearish" if move < 0 else "Neutral"

    triggers: List[Dict[str, object]] = []
    for detected in [
        _detect_orb_trigger(df, 30),
        _detect_orb_trigger(df, 60),
        _detect_failed_orb_trigger(df, 30),
        _detect_failed_orb_trigger(df, 60),
        _detect_vwap_trigger(df),
    ]:
        if detected:
            triggers.append(detected)

    triggers = sorted(triggers, key=lambda x: x.get("time") or "")
    triggers = [_enrich_trigger(t) for t in triggers]
    primary_trigger = _pick_primary_trigger(triggers, ny_direction)

    supporting_factors: List[str] = []
    blocking_factors: List[str] = []

    if ny_mode != "Neutral":
        supporting_factors.append(f"Mode classified as {ny_mode}")
    if ny_direction in {"Bullish", "Bearish"}:
        supporting_factors.append(f"Directional bias: {ny_direction}")
    if primary_trigger:
        supporting_factors.append(f"Trigger fired: {primary_trigger['name']}")

    if whipsaw_risk:
        blocking_factors.append(f"Whipsaw ratio elevated ({whipsaw_ratio:.2f})")
    if ny_direction == "Neutral":
        blocking_factors.append("No clear directional bias")

    us_start = windows.get("US", {}).get("start")
    us_confirm = us_start + dt.timedelta(minutes=60) if us_start else None
    us_end = windows.get("US", {}).get("end")

    trade_today = "Wait"
    primary_reason = "Waiting for session confirmation."
    confidence = 0.45

    if ny_mode == "Whipsaw Risk":
        trade_today = "No"
        primary_reason = "Whipsaw risk is elevated; avoid directional setup unless exceptional confirmation appears."
        confidence = 0.75
    elif us_start and now < us_start:
        trade_today = "Wait"
        primary_reason = "US session not started yet."
        confidence = 0.55
    elif primary_trigger is not None:
        trade_today = "Yes"
        primary_reason = f"Qualified trigger detected: {primary_trigger['name']}."
        confidence = 0.70
    elif us_confirm and now < us_confirm:
        trade_today = "Wait"
        primary_reason = "Waiting for opening-hour confirmation window."
        confidence = 0.60
    elif us_end and now > us_end and primary_trigger is None:
        trade_today = "No"
        primary_reason = "No valid trigger occurred during the US session."
        confidence = 0.72
    else:
        trade_today = "Wait"
        primary_reason = "Bias exists but no execution trigger has fired yet."
        confidence = 0.58

    last_price = float(df["close"].iloc[-1])
    day_high = float(df["high"].max())
    day_low = float(df["low"].min())

    dvwap = compute_daily_vwap(df)
    wvwap = compute_weekly_vwap(df)
    daily_vwap = float(dvwap.iloc[-1]) if not dvwap.empty else None
    weekly_vwap = float(wvwap.iloc[-1]) if not wvwap.empty else None

    vwap_levels: List[Dict[str, object]] = []
    if daily_vwap is not None:
        vwap_levels.append(
            {
                "Name": "Daily VWAP",
                "Price": daily_vwap,
                "Distance": last_price - daily_vwap,
                "Status": "Above" if last_price > daily_vwap else "Below" if last_price < daily_vwap else "At",
                "Time": _fmt_ts(pd.to_datetime(df["timestamp"].iloc[-1])),
            }
        )
    if weekly_vwap is not None:
        vwap_levels.append(
            {
                "Name": "Weekly VWAP",
                "Price": weekly_vwap,
                "Distance": last_price - weekly_vwap,
                "Status": "Above" if last_price > weekly_vwap else "Below" if last_price < weekly_vwap else "At",
                "Time": _fmt_ts(pd.to_datetime(df["timestamp"].iloc[-1])),
            }
        )

    if df_prev is not None and not df_prev.empty:
        prev_high_idx = df_prev["high"].idxmax()
        prev_low_idx = df_prev["low"].idxmin()
        anchors = [
            ("Anchored VWAP (PDH)", pd.to_datetime(df_prev.loc[prev_high_idx, "timestamp"])),
            ("Anchored VWAP (PDL)", pd.to_datetime(df_prev.loc[prev_low_idx, "timestamp"])),
        ]
        for name, anchor_ts in anchors:
            av = compute_anchored_vwap(pd.concat([df_prev, df], ignore_index=True), anchor_ts)
            if av is None or av.empty:
                continue
            px = float(av.iloc[-1])
            vwap_levels.append(
                {
                    "Name": name,
                    "Price": px,
                    "Distance": last_price - px,
                    "Status": "Above" if last_price > px else "Below" if last_price < px else "At",
                    "Time": _fmt_ts(anchor_ts),
                }
            )

    stdv_center = daily_vwap if daily_vwap is not None else last_price
    stdv_levels = _stdv_levels(df, center=stdv_center, length=20)

    confluences: List[Dict[str, object]] = []
    for z in zones:
        first_test, retest_time, retest_count = _zone_touch_times(df, z)
        touched = is_zone_touched(df, z)
        inversed = is_fvg_inversed(df, z) if z.kind == "fvg" else False
        failed = is_zone_failed(df, z) or inversed
        density, volume_score = zone_liquidity_scores(df, z)
        liquidity_score = float(density * 0.2 + volume_score)
        liquidity_level = _liquidity_level(liquidity_score)
        rejection_times = [t.strftime("%Y-%m-%d %H:%M") for t in find_rejection_candles(df, z)]

        if failed:
            status = "Invalidated"
        elif retest_count > 0:
            status = "Retested"
        elif touched:
            status = "Tested"
        else:
            status = "Fresh"

        confluences.append(
            {
                "Confluence": f"{z.timeframe} {z.kind} {z.side}",
                "Price Low": float(z.low),
                "Price High": float(z.high),
                "Formed Time": _fmt_ts(zone_formed_timestamp(z)),
                "First Test": first_test,
                "Retest Time": retest_time,
                "Retest Count": int(retest_count),
                "Rejection Times": ", ".join(rejection_times[:3]) if rejection_times else "",
                "Status": status,
                "Side": z.side,
                "Liquidity Score": liquidity_score,
                "Liquidity Level": liquidity_level,
                "Top Liquidity": False,
            }
        )

    if confluences:
        ranked = sorted(range(len(confluences)), key=lambda i: confluences[i]["Liquidity Score"], reverse=True)
        for idx in ranked[:7]:
            confluences[idx]["Top Liquidity"] = True
        confluences = sorted(
            confluences,
            key=lambda row: (not bool(row.get("Top Liquidity", False)), -float(row.get("Liquidity Score", 0.0))),
        )

    direction_sign = 1 if ny_direction == "Bullish" else -1 if ny_direction == "Bearish" else 0
    targets: List[Dict[str, object]] = []

    def _add_target(name: str, price: float, source: str):
        distance = price - last_price
        if direction_sign == 1 and distance <= 0:
            return
        if direction_sign == -1 and distance >= 0:
            return
        targets.append(
            {
                "Target": name,
                "Price": float(price),
                "Distance": float(distance),
                "Source": source,
            }
        )

    for z in zones:
        candidate = z.high if ny_direction == "Bullish" else z.low if ny_direction == "Bearish" else (z.high + z.low) / 2
        _add_target(f"{z.timeframe} {z.kind} {z.side}", float(candidate), "Zone")

    for level in vwap_levels:
        _add_target(level["Name"], float(level["Price"]), "VWAP")

    for level in stdv_levels:
        _add_target(f"STDV {level['Level']}", float(level["Price"]), "STDV")

    targets_sorted = sorted(targets, key=lambda x: abs(float(x["Distance"])))
    target_ladder = targets_sorted[:12]

    power_hour_focus = "Conditional"
    power_hour_bias = getattr(patterns, "power_hour_bias", "Neutral") or "Neutral"
    power_hour_reason = "Awaiting 14:00-16:00 development."
    power_hour_entries: List[str] = []

    if getattr(patterns, "power_hour_trend", False):
        power_hour_focus = "Yes"
        power_hour_reason = "Power-hour trend condition is active."
    elif ny_mode in {"AMD", "NY Continuation"} and ny_direction in {"Bullish", "Bearish"}:
        power_hour_focus = "Conditional"
        power_hour_bias = ny_direction
        power_hour_reason = "Directional context supports a potential late-session continuation if momentum persists."
    else:
        power_hour_focus = "No"
        power_hour_reason = "No directional alignment for late-session focus."

    if power_hour_focus in {"Yes", "Conditional"}:
        direction_word = "long" if power_hour_bias == "Bullish" else "short" if power_hour_bias == "Bearish" else "directional"
        power_hour_entries = [
            f"Look for {direction_word} pullback entries at VWAP or nearest confluence after 14:00 ET hold.",
            "Look for break-retest continuation on 1m-5m structure before entering.",
            "Avoid first impulsive candle chase; prefer retest confirmation.",
        ]

    reference_rows = _select_reference_confluence_rows(confluences, ny_direction, max_items=2)
    reference_confluences = _select_reference_confluences(confluences, ny_direction, max_items=2)
    entry_style_suggestions = _suggest_confluence_entry_styles(
        reference_rows,
        df=df,
        target_ladder=target_ladder,
        mode=ny_mode,
        whipsaw_risk=whipsaw_risk,
    )
    entry_blueprints = _entry_blueprints(
        ny_mode,
        ny_direction,
        windows,
        reference_confluences=reference_confluences,
        vwap_levels=vwap_levels,
        entry_style_suggestions=entry_style_suggestions,
    )
    daily_entry_capacity = _estimate_daily_entry_capacity(
        mode=ny_mode,
        direction=ny_direction,
        trade_today=trade_today,
        confidence=float(confidence),
        reference_rows=reference_rows,
        primary_trigger=primary_trigger,
        now=now,
        windows=windows,
    )
    open_pattern_watch = _us_open_reclaim_watch(df=df, trade_date=trade_date, now=now, confluences=confluences)

    return {
        "decision": {
            "trade_today": trade_today,
            "ny_mode": ny_mode,
            "ny_direction": ny_direction,
            "confidence": float(confidence),
            "primary_reason": primary_reason,
            "supporting_factors": supporting_factors,
            "blocking_factors": blocking_factors,
            "whipsaw_ratio": float(whipsaw_ratio) if whipsaw_ratio is not None else None,
        },
        "triggers": triggers,
        "primary_trigger": primary_trigger,
        "confluences": confluences,
        "targets": target_ladder,
        "vwap_levels": vwap_levels,
        "stdv_levels": stdv_levels,
        "power_hour": {
            "focus": power_hour_focus,
            "bias": power_hour_bias,
            "reason": power_hour_reason,
            "entries": power_hour_entries,
        },
        "entry_blueprints": entry_blueprints,
        "confluence_entry_styles": entry_style_suggestions,
        "daily_entry_capacity": daily_entry_capacity,
        "timeframe_playbook": _timeframe_playbook(ny_mode),
        "open_pattern_watch": open_pattern_watch,
        "market_snapshot": {
            "last_price": last_price,
            "day_high": day_high,
            "day_low": day_low,
            "trade_date": str(trade_date),
        },
    }
