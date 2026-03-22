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
    zone_liquidity_scores,
)
from indicators.moving_averages import compute_anchored_vwap, compute_daily_vwap, compute_weekly_vwap


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


def _zone_touch_times(df: pd.DataFrame, zone: Zone) -> Tuple[Optional[str], Optional[str], int]:
    if df is None or df.empty:
        return None, None, 0
    after = df[df["timestamp"] > zone.start]
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


def _entry_blueprints(
    mode: str,
    direction: str,
    windows: Dict[str, Dict[str, dt.datetime]],
    reference_confluences: List[str],
    vwap_levels: List[Dict[str, object]],
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

    if mode == "Whipsaw Risk":
        return [
            {
                "Setup": "Risk-Off / Two-Sided",
                "IfThen": f"If price keeps crossing around {daily_vwap_label} and cannot hold either side of OR, then avoid directional entries.",
                "Trigger": "Use only exceptional reclaim/reject + structure alignment at high-liquidity confluence.",
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
            "Execution TF": "1m",
            "Session": us_window,
        },
        {
            "Setup": "OR Retest",
            "IfThen": f"If OR break aligns {direction.lower()} and retest holds into {secondary_confluence}, then execute continuation.",
            "Trigger": "ORB or Failed ORB confirmation.",
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
                "Formed Time": _fmt_ts(z.start),
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

    reference_confluences = _select_reference_confluences(confluences, ny_direction, max_items=2)
    entry_blueprints = _entry_blueprints(
        ny_mode,
        ny_direction,
        windows,
        reference_confluences=reference_confluences,
        vwap_levels=vwap_levels,
    )

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
        "timeframe_playbook": _timeframe_playbook(ny_mode),
        "market_snapshot": {
            "last_price": last_price,
            "day_high": day_high,
            "day_low": day_low,
            "trade_date": str(trade_date),
        },
    }
