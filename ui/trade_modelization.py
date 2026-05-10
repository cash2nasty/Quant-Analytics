import datetime as dt
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from data.data_fetcher import fetch_intraday_ohlcv
from engines.patterns import detect_patterns
from engines.sessions import compute_session_stats
from engines.strategy_playbook import build_strategy_playbook
from engines.volume_profile import build_session_profiles
from engines.zones import build_htf_zones
from ui.live_analysis import get_prev_trading_day


try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


def _now_et() -> dt.datetime:
    if ZoneInfo is None:
        return dt.datetime.now()
    return dt.datetime.now(ZoneInfo("America/New_York")).replace(tzinfo=None)


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    out = df.copy()
    if "timestamp" not in out.columns:
        out = out.reset_index().rename(columns={out.index.name or "index": "timestamp"})
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    keep = [c for c in ["timestamp", "open", "high", "low", "close", "volume"] if c in out.columns]
    return out[keep].sort_values("timestamp").reset_index(drop=True)


def _trading_day_bounds(trading_day: dt.date) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp.combine(trading_day - dt.timedelta(days=1), dt.time(18, 0))
    end = pd.Timestamp.combine(trading_day, dt.time(16, 59, 59))
    return start, end


def _trade_suggestion_bounds(trading_day: dt.date) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp.combine(trading_day, dt.time(9, 30))
    end = pd.Timestamp.combine(trading_day, dt.time(14, 30))
    return start, end


def _slice_trading_day(df: pd.DataFrame, trading_day: dt.date) -> pd.DataFrame:
    if df is None or df.empty or "timestamp" not in df.columns:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    start, end = _trading_day_bounds(trading_day)
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    mask = (out["timestamp"] >= start) & (out["timestamp"] <= end)
    return out.loc[mask].sort_values("timestamp").reset_index(drop=True)


def _slice_window(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if df is None or df.empty or "timestamp" not in df.columns:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    mask = (out["timestamp"] >= start) & (out["timestamp"] <= end)
    return out.loc[mask].sort_values("timestamp").reset_index(drop=True)


def _to_ts(value: Any) -> Optional[pd.Timestamp]:
    if value in (None, "", "n/a"):
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts)


def _to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value in (None, "", "n/a"):
            return default
        return float(value)
    except Exception:
        return default


def _extract_point_value(value: Any) -> Optional[float]:
    text = str(value or "").strip()
    if not text or text == "n/a":
        return None
    left = text.split(" ", 1)[0]
    try:
        return float(left)
    except Exception:
        return _to_float(text, None)


def _build_trade_key(row: Dict[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("Confluence", "n/a")),
            str(row.get("Suggested Time", "n/a")),
            str(row.get("Action", "Wait")),
            str(row.get("Entry Price", "n/a")),
        ]
    )


def _status_from_row(row: Dict[str, Any]) -> str:
    executed = str(row.get("Executed", "No"))
    outcome = str(row.get("Outcome", "")).lower()
    if executed == "Yes" and outcome == "successful":
        return "target_hit"
    if executed == "Yes" and outcome == "failed":
        return "invalidated"
    if executed == "Yes" and outcome == "open":
        return "triggered"
    if outcome in {"pending", "unfilled"}:
        return "forming"
    if outcome == "skipped":
        return "skipped"
    return "forming"


def _status_label(status: str) -> str:
    if status == "triggered":
        return "🟢 Trigger Hit (Active)"
    if status == "target_hit":
        return "✅ Target Hit"
    if status == "invalidated":
        return "❌ Invalidated / Stop Hit"
    if status == "expired":
        return "⌛ Expired (Window End)"
    if status == "time_limited":
        return "⏱ Time-Limited (14:30 Cutoff)"
    if status == "skipped":
        return "⏸ Skipped"
    return "🟡 Setup Forming"


def _target_stop_levels(row: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    action = str(row.get("Action", "Wait"))
    entry = _to_float(row.get("Entry Price"), None)
    if action not in {"Long", "Short"} or entry is None:
        return None, None

    risk_points = _extract_point_value(row.get("Risk Pts(Ticks)"))
    target_points = _extract_point_value(row.get("Target Pts(Ticks)"))

    if risk_points is None:
        risk_points = 2.0
    if target_points is None:
        target_points = max(3.0, risk_points * 1.4)

    if action == "Long":
        stop = entry - risk_points
        target = entry + target_points
    else:
        stop = entry + risk_points
        target = entry - target_points
    return target, stop


def _calc_probabilities(
    row: Dict[str, Any],
    liquidity_norm: float,
    momentum_prediction: str,
) -> Tuple[float, float, float]:
    conf = max(0.0, min(100.0, _to_float(row.get("Entry Confidence"), 0.0) or 0.0))
    rr = max(0.3, _to_float(row.get("RR"), 1.0) or 1.0)
    base_fill = 28.0 + conf * 0.45 + (rr - 1.0) * 14.0 + liquidity_norm * 10.0

    action = str(row.get("Action", "Wait"))
    momentum_boost = 0.0
    if momentum_prediction == "Bullish" and action == "Long":
        momentum_boost = 4.0
    elif momentum_prediction == "Bearish" and action == "Short":
        momentum_boost = 4.0
    elif momentum_prediction in {"Bullish", "Bearish"}:
        momentum_boost = -4.0

    fill_prob = max(5.0, min(95.0, base_fill + momentum_boost))

    target_given_fill = 24.0 + conf * 0.50 + (rr - 1.0) * 36.0 + liquidity_norm * 8.0 + momentum_boost
    target_given_fill = max(5.0, min(95.0, target_given_fill))

    target_prob = max(1.0, min(95.0, fill_prob * (target_given_fill / 100.0)))
    invalidation_prob = max(
        1.0,
        min(95.0, fill_prob * ((100.0 - target_given_fill) / 100.0) + (100.0 - fill_prob) * 0.30),
    )

    return round(fill_prob, 1), round(target_prob, 1), round(invalidation_prob, 1)


def _grade_trade(fill_prob: float, target_prob: float, invalidation_prob: float, conf: float) -> str:
    score = 0.30 * fill_prob + 0.40 * target_prob + 0.20 * (100.0 - invalidation_prob) + 0.10 * conf
    if score >= 85:
        return "A+"
    if score >= 80:
        return "A"
    if score >= 75:
        return "A-"
    if score >= 70:
        return "B+"
    if score >= 65:
        return "B"
    if score >= 60:
        return "B-"
    if score >= 55:
        return "C+"
    if score >= 50:
        return "C"
    return "C-"


def _build_model_rows(playbook: Dict[str, Any], start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> List[Dict[str, Any]]:
    execution_rows = playbook.get("entry_execution_tracker", []) or []
    decision = playbook.get("decision", {}) or {}
    risk_engine = playbook.get("risk_engine", {}) or {}
    volatility = playbook.get("volatility_metrics", {}) or {}
    volume = playbook.get("volume_detector", {}) or {}
    momentum = playbook.get("momentum_prediction", {}) or {}
    primary_trigger = playbook.get("primary_trigger") or {}

    confluence_liquidity: Dict[str, float] = {}
    for row in playbook.get("confluences", []) or []:
        name = str(row.get("Confluence", "n/a"))
        score = _to_float(row.get("Liquidity Score"), 0.0) or 0.0
        confluence_liquidity[name] = max(confluence_liquidity.get(name, 0.0), score)

    models: List[Dict[str, Any]] = []
    for row in execution_rows:
        action = str(row.get("Action", "Wait"))
        if action not in {"Long", "Short"}:
            continue

        suggested_ts = _to_ts(row.get("Suggested Time"))
        if suggested_ts is None or suggested_ts < start_ts or suggested_ts > end_ts:
            continue

        status = _status_from_row(row)
        target_level, stop_level = _target_stop_levels(row)
        confluence_name = str(row.get("Confluence", "n/a"))
        liquidity_norm = max(0.0, min(1.0, confluence_liquidity.get(confluence_name, 0.0)))

        fill_prob, target_prob, invalidation_prob = _calc_probabilities(
            row=row,
            liquidity_norm=liquidity_norm,
            momentum_prediction=str(momentum.get("predicted", "Neutral")),
        )

        conf = max(0.0, min(100.0, _to_float(row.get("Entry Confidence"), 0.0) or 0.0))
        grade = _grade_trade(fill_prob, target_prob, invalidation_prob, conf)

        boosts: List[str] = []
        invalidators: List[str] = []

        for item in (decision.get("supporting_factors", []) or [])[:3]:
            boosts.append(str(item))
        for item in (decision.get("blocking_factors", []) or [])[:3]:
            invalidators.append(str(item))

        if str(volume.get("state", "normal")) == "high":
            boosts.append("Above-normal participation (RVOL high).")
        if str(volatility.get("regime", "normal")) == "expanded":
            invalidators.append("Expanded volatility can invalidate setups faster.")

        momentum_pred = str(momentum.get("predicted", "Neutral"))
        if (momentum_pred == "Bullish" and action == "Long") or (momentum_pred == "Bearish" and action == "Short"):
            boosts.append("Momentum model aligns with setup direction.")
        elif momentum_pred in {"Bullish", "Bearish"}:
            invalidators.append("Momentum model conflicts with setup direction.")

        if status == "invalidated":
            invalidators.append("Stop/invalidation has already been hit.")
        elif status == "target_hit":
            boosts.append("Target already completed.")

        data_used = [
            f"Trade Today decision: {decision.get('trade_today', 'Wait')}",
            f"NY mode/direction: {decision.get('ny_mode', 'n/a')} / {decision.get('ny_direction', 'Neutral')}",
            f"Primary trigger: {primary_trigger.get('name', 'No trigger yet')}",
            f"Confluence: {confluence_name}",
            f"Entry confidence: {row.get('Entry Confidence', 'n/a')}",
            f"R:R: {row.get('RR', 'n/a')}",
            f"Vol regime: {volatility.get('regime', 'normal')}",
            f"RVOL state: {volume.get('state', 'normal')}",
            f"Momentum: {momentum_pred}",
        ]

        models.append(
            {
                "trade_key": _build_trade_key(row),
                "suggested_time": suggested_ts,
                "suggested_time_text": suggested_ts.strftime("%Y-%m-%d %H:%M"),
                "action": action,
                "confluence": confluence_name,
                "entry": _to_float(row.get("Entry Price"), None),
                "target": target_level,
                "stop": stop_level,
                "status": status,
                "status_label": _status_label(status),
                "grade": grade,
                "fill_prob": fill_prob,
                "target_prob": target_prob,
                "invalidation_prob": invalidation_prob,
                "executed": str(row.get("Executed", "No")),
                "execution_time": str(row.get("Execution Time", "n/a")),
                "outcome": str(row.get("Outcome", "n/a")),
                "exit_time": str(row.get("Exit Time", "n/a")),
                "entry_hit": "Yes" if str(row.get("Executed", "No")) == "Yes" else "No",
                "boosts": boosts[:5],
                "invalidators": invalidators[:5],
                "data_used": data_used,
                "confidence_reasoning": str(row.get("Confidence Reasoning", row.get("Why", "n/a"))),
            }
        )

    return sorted(models, key=lambda r: r.get("suggested_time") or pd.Timestamp.min, reverse=True)


def _level_base_confidence(source: str) -> float:
    s = str(source or "")
    if "HVN" in s:
        return 66.0
    if "LVN" in s:
        return 62.0
    if "VWAP" in s:
        return 64.0
    if "POC" in s or "VAH" in s or "VAL" in s:
        return 63.0
    if "Session" in s:
        return 60.0
    return 58.0


def _build_level_models(
    playbook: Dict[str, Any],
    df_trading_day: pd.DataFrame,
    sessions: Dict[str, Any],
    selected_date: dt.date,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> List[Dict[str, Any]]:
    if df_trading_day is None or df_trading_day.empty:
        return []

    decision = playbook.get("decision", {}) or {}
    momentum = playbook.get("momentum_prediction", {}) or {}
    volatility = playbook.get("volatility_metrics", {}) or {}
    volume = playbook.get("volume_detector", {}) or {}

    last_ts = pd.to_datetime(df_trading_day["timestamp"].iloc[-1])
    if last_ts < start_ts or last_ts > end_ts:
        return []

    last_price = float(df_trading_day["close"].iloc[-1])
    day_high = float(df_trading_day["high"].max())
    day_low = float(df_trading_day["low"].min())
    atr_like = _to_float(volatility.get("atr_like"), 2.0) or 2.0
    risk_points = max(1.25, atr_like * 0.65)
    target_points = max(2.0, risk_points * 1.6)

    ny_direction = str(decision.get("ny_direction", "Neutral"))
    momentum_pred = str(momentum.get("predicted", "Neutral"))

    levels: List[Dict[str, Any]] = []

    for session_name, sess in sessions.items():
        try:
            levels.append(
                {
                    "name": f"{session_name} Session High",
                    "source": "Session High/Low",
                    "price": float(sess.high),
                }
            )
            levels.append(
                {
                    "name": f"{session_name} Session Low",
                    "source": "Session High/Low",
                    "price": float(sess.low),
                }
            )
        except Exception:
            continue

    profiles = build_session_profiles(df_trading_day, selected_date)
    full_day = profiles.get("Full Day")
    if full_day is not None:
        for label, value in (
            ("Full Day POC", full_day.poc),
            ("Full Day VAH", full_day.vah),
            ("Full Day VAL", full_day.val),
        ):
            if value is not None:
                levels.append({"name": label, "source": "Profile Value", "price": float(value)})

        for node in (full_day.hvns or [])[:3]:
            levels.append(
                {
                    "name": f"HVN {node.low:.2f}-{node.high:.2f}",
                    "source": "HVN",
                    "price": float(node.center),
                }
            )
        for node in (full_day.lvns or [])[:3]:
            levels.append(
                {
                    "name": f"LVN {node.low:.2f}-{node.high:.2f}",
                    "source": "LVN",
                    "price": float(node.center),
                }
            )

    for row in playbook.get("vwap_levels", []) or []:
        price = _to_float(row.get("Price"), None)
        if price is None:
            continue
        levels.append(
            {
                "name": str(row.get("Name", "VWAP Level")),
                "source": "VWAP",
                "price": float(price),
            }
        )

    dedup: Dict[Tuple[str, float], Dict[str, Any]] = {}
    for lv in levels:
        key = (str(lv.get("source", "Level")), round(float(lv.get("price", 0.0)), 2))
        if key not in dedup:
            dedup[key] = lv

    out: List[Dict[str, Any]] = []
    for lv in dedup.values():
        entry = float(lv["price"])
        source = str(lv.get("source", "Level"))
        level_name = str(lv.get("name", source))

        if ny_direction == "Bullish":
            action = "Long"
        elif ny_direction == "Bearish":
            action = "Short"
        else:
            action = "Long" if entry <= last_price else "Short"

        if action == "Long":
            target = entry + target_points
            stop = entry - risk_points
        else:
            target = entry - target_points
            stop = entry + risk_points

        proximity = abs(last_price - entry)
        status = "triggered" if proximity <= max(0.5, risk_points * 0.25) else "forming"
        outcome = "Open" if status == "triggered" else "Pending"
        if status == "triggered":
            if action == "Long" and day_high >= target:
                status = "target_hit"
                outcome = "Successful"
            elif action == "Long" and day_low <= stop:
                status = "invalidated"
                outcome = "Failed"
            elif action == "Short" and day_low <= target:
                status = "target_hit"
                outcome = "Successful"
            elif action == "Short" and day_high >= stop:
                status = "invalidated"
                outcome = "Failed"

        confidence = _level_base_confidence(source)
        if momentum_pred == "Bullish" and action == "Long":
            confidence += 4.0
        elif momentum_pred == "Bearish" and action == "Short":
            confidence += 4.0
        elif momentum_pred in {"Bullish", "Bearish"}:
            confidence -= 4.0
        confidence = max(40.0, min(92.0, confidence))
        rr = abs(target - entry) / max(abs(entry - stop), 0.25)

        fill_prob, target_prob, invalidation_prob = _calc_probabilities(
            row={"Action": action, "Entry Confidence": confidence, "RR": rr},
            liquidity_norm=0.55 if source in {"HVN", "Profile Value", "VWAP"} else 0.45,
            momentum_prediction=momentum_pred,
        )
        grade = _grade_trade(fill_prob, target_prob, invalidation_prob, confidence)

        boosts = [
            f"{source} level used: {level_name}.",
            f"NY directional context: {ny_direction}.",
            f"Momentum state: {momentum_pred}.",
        ]
        if str(volume.get("state", "normal")) == "high":
            boosts.append("RVOL high supports follow-through.")

        invalidators = [
            f"Break and acceptance through stop level {_format_level(stop)}.",
            "Loss of directional alignment in NY mode.",
        ]
        if str(volatility.get("regime", "normal")) == "expanded":
            invalidators.append("Expanded volatility can break levels faster.")

        out.append(
            {
                "trade_key": f"LEVEL|{source}|{level_name}|{entry:.2f}|{action}",
                "suggested_time": last_ts,
                "suggested_time_text": last_ts.strftime("%Y-%m-%d %H:%M"),
                "action": action,
                "confluence": f"{source}: {level_name}",
                "entry": entry,
                "target": target,
                "stop": stop,
                "status": status,
                "status_label": _status_label(status),
                "grade": grade,
                "fill_prob": fill_prob,
                "target_prob": target_prob,
                "invalidation_prob": invalidation_prob,
                "executed": "Yes" if status in {"triggered", "target_hit", "invalidated"} else "No",
                "execution_time": last_ts.strftime("%Y-%m-%d %H:%M") if status in {"triggered", "target_hit", "invalidated"} else "n/a",
                "outcome": outcome,
                "exit_time": "n/a",
                "entry_hit": "Yes" if status in {"triggered", "target_hit", "invalidated"} else "No",
                "boosts": boosts,
                "invalidators": invalidators,
                "data_used": [
                    f"Model source: {source}",
                    f"Level: {level_name} @ {entry:.2f}",
                    f"NY mode/direction: {decision.get('ny_mode', 'n/a')} / {ny_direction}",
                    f"Momentum: {momentum_pred}",
                    f"Vol regime: {volatility.get('regime', 'normal')}",
                    f"RVOL state: {volume.get('state', 'normal')}",
                ],
                "confidence_reasoning": (
                    f"Level-driven setup from {source} with direction filter and volatility-scaled risk. "
                    f"Distance to level is {proximity:.2f} points."
                ),
            }
        )

    return sorted(out, key=lambda r: r.get("suggested_time") or pd.Timestamp.min, reverse=True)


def _merge_daily_registry(
    symbol: str,
    selected_date: dt.date,
    current_models: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    registry_key = f"trade_model_registry::{symbol}::{selected_date.isoformat()}"
    if registry_key not in st.session_state:
        st.session_state[registry_key] = {}

    registry: Dict[str, Dict[str, Any]] = st.session_state[registry_key]

    for model in current_models:
        key = str(model.get("trade_key", ""))
        if not key:
            continue
        prev = registry.get(key)
        if prev is None:
            registry[key] = model
            continue

        merged = dict(prev)
        merged.update(model)

        if prev.get("suggested_time") is not None:
            merged["suggested_time"] = prev.get("suggested_time")
            merged["suggested_time_text"] = prev.get("suggested_time_text", merged.get("suggested_time_text"))

        prev_status = str(prev.get("status", "forming"))
        curr_status = str(model.get("status", "forming"))
        if prev_status in {"target_hit", "invalidated"} and curr_status in {"forming", "triggered"}:
            merged["status"] = prev_status
            merged["status_label"] = _status_label(prev_status)

        registry[key] = merged

    st.session_state[registry_key] = registry

    rows = list(registry.values())
    rows = sorted(rows, key=lambda r: r.get("suggested_time") or pd.Timestamp.min, reverse=True)
    return rows


def _format_level(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _risk_ratio_text(model: Dict[str, Any]) -> str:
    entry = _to_float(model.get("entry"), None)
    target = _to_float(model.get("target"), None)
    stop = _to_float(model.get("stop"), None)
    if entry is None or target is None or stop is None:
        return "n/a"
    risk = abs(entry - stop)
    reward = abs(target - entry)
    if risk <= 0:
        return "n/a"
    return f"{(reward / risk):.2f}"


def _selector_result_text(model: Dict[str, Any]) -> str:
    status = str(model.get("status", "forming"))
    outcome = str(model.get("outcome", "")).lower()
    entry_hit = str(model.get("entry_hit", "No"))
    cutoff_price = model.get("cutoff_price")

    if status == "time_limited":
        if cutoff_price is None:
            return "14:30 Cutoff"
        return f"14:30 Price {_format_level(cutoff_price)}"

    if status == "target_hit" or outcome == "successful":
        return "Target Hit"
    if status == "invalidated" or outcome == "failed":
        return "Stop Loss Hit"
    if outcome in {"unfilled", "pending"} and entry_hit != "Yes":
        return "Not Filled"
    if status == "expired" and entry_hit != "Yes":
        return "Not Filled"
    if entry_hit == "Yes":
        return "Filled (Open)"
    return "Not Filled"


def _enforce_trade_window(
    models: List[Dict[str, Any]],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    now_ts: pd.Timestamp,
    reference_df: Optional[pd.DataFrame] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    is_after_window = bool(now_ts > end_ts)

    cutoff_price: Optional[float] = None
    if reference_df is not None and not reference_df.empty:
        ref = reference_df.copy()
        ref["timestamp"] = pd.to_datetime(ref["timestamp"])
        ref = ref[ref["timestamp"] <= end_ts]
        if not ref.empty:
            cutoff_price = float(ref.iloc[-1]["close"])

    for model in models:
        suggested_ts = model.get("suggested_time")
        if suggested_ts is None:
            continue
        ts = pd.Timestamp(suggested_ts)
        if ts < start_ts or ts > end_ts:
            continue

        row = dict(model)
        status = str(row.get("status", "forming"))
        if is_after_window and status in {"forming", "triggered"}:
            row["status"] = "time_limited"
            row["status_label"] = _status_label("time_limited")
            row["cutoff_price"] = cutoff_price
            row["outcome"] = (
                f"14:30 Price {_format_level(cutoff_price)}"
                if cutoff_price is not None
                else "14:30 Cutoff"
            )
            row["exit_time"] = end_ts.strftime("%Y-%m-%d %H:%M")
        out.append(row)
    return out


def _build_repeated_setup_rows(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for row in models:
        key = f"{row.get('action', 'Wait')} | {row.get('confluence', 'n/a')}"
        if key not in grouped:
            grouped[key] = {
                "setup": key,
                "count": 0,
                "target_hit": 0,
                "stop_hit": 0,
                "not_filled": 0,
                "time_limited": 0,
                "latest_suggested": "n/a",
                "avg_grade_score": 0.0,
            }
        g = grouped[key]
        g["count"] += 1
        status = str(row.get("status", "forming"))
        if status == "target_hit":
            g["target_hit"] += 1
        elif status == "invalidated":
            g["stop_hit"] += 1
        elif status == "time_limited":
            g["time_limited"] += 1
        elif str(row.get("entry_hit", "No")) != "Yes":
            g["not_filled"] += 1

        suggested_text = str(row.get("suggested_time_text", "n/a"))
        if suggested_text > str(g.get("latest_suggested", "n/a")):
            g["latest_suggested"] = suggested_text

        grade = str(row.get("grade", "C"))
        grade_map = {
            "A+": 4.3,
            "A": 4.0,
            "A-": 3.7,
            "B+": 3.3,
            "B": 3.0,
            "B-": 2.7,
            "C+": 2.3,
            "C": 2.0,
            "C-": 1.7,
        }
        g["avg_grade_score"] += grade_map.get(grade, 2.0)

    rows: List[Dict[str, Any]] = []
    for g in grouped.values():
        if g["count"] < 2:
            continue
        avg_score = g["avg_grade_score"] / float(g["count"])
        rows.append(
            {
                "Setup": g["setup"],
                "Times Identified": g["count"],
                "Target Hit": g["target_hit"],
                "Stop Loss Hit": g["stop_hit"],
                "Not Filled": g["not_filled"],
                "14:30 Time-Limited": g["time_limited"],
                "Latest Suggested": g["latest_suggested"],
                "Avg Grade Score": round(avg_score, 2),
            }
        )

    return sorted(rows, key=lambda r: (int(r.get("Times Identified", 0)), float(r.get("Avg Grade Score", 0.0))), reverse=True)


def _run_auto_update(auto_update: bool, selected_date: dt.date, today: dt.date, refresh_seconds: int, symbol: str) -> None:
    if not auto_update or selected_date != today:
        return

    st.caption(f"Auto update is enabled ({refresh_seconds} sec cadence for today).")

    if hasattr(st, "fragment"):
        interval = f"{int(refresh_seconds)}s"

        @st.fragment(run_every=interval)
        def _refresh_tick() -> None:
            marker_key = f"trade_model_last_auto_rerun::{symbol}::{selected_date.isoformat()}"
            now_ts = time.time()
            last_ts = float(st.session_state.get(marker_key, 0.0))
            if now_ts - last_ts >= max(int(refresh_seconds) - 1, 1):
                st.session_state[marker_key] = now_ts
                st.rerun()

        _refresh_tick()
    else:
        st.caption("Auto update compatibility mode: use Refresh now.")


def render_trade_modelization_tab() -> None:
    st.header("Trade Modelization")
    st.caption(
        "Models setup-forming trades with concrete entry/target/invalidation levels, probability stack, confidence grading, and lifecycle tracking."
    )

    today = _now_et().date()
    refresh_options = {
        "60secs": 60,
        "2min": 120,
        "3min": 180,
        "5min": 300,
    }

    with st.sidebar:
        st.subheader("Trade Model Inputs")
        symbol = st.text_input("Symbol", value="NQH26", key="trade_model_symbol")
        selected_date = st.date_input("Analysis date", value=today, key="trade_model_date")
        auto_update = st.checkbox("Auto update", value=True, key="trade_model_auto_update")
        refresh_label = st.selectbox(
            "Auto update every",
            options=list(refresh_options.keys()),
            index=0,
            disabled=not auto_update,
            key="trade_model_refresh_label",
        )
        refresh_now = st.button("Refresh now", key="trade_model_refresh_now", use_container_width=True)

    if refresh_now:
        st.rerun()

    refresh_seconds = refresh_options.get(refresh_label, 60)

    res_today = fetch_intraday_ohlcv(symbol, selected_date)
    if isinstance(res_today, tuple):
        df_today, used_ticker = res_today
    else:
        df_today = res_today
        used_ticker = symbol

    prev_date = get_prev_trading_day(selected_date)
    res_prev = fetch_intraday_ohlcv(symbol, prev_date)
    if isinstance(res_prev, tuple):
        df_prev, _ = res_prev
    else:
        df_prev = res_prev

    df_today = _prepare_df(df_today)
    df_prev = _prepare_df(df_prev)

    combined_source = pd.concat([df_prev, df_today], ignore_index=True).sort_values("timestamp")
    td_analysis_start, td_analysis_end = _trading_day_bounds(selected_date)
    suggest_start, suggest_end = _trade_suggestion_bounds(selected_date)

    df_trading_day = _slice_window(combined_source, td_analysis_start, td_analysis_end)
    prev_analysis_start, prev_analysis_end = _trading_day_bounds(prev_date)
    df_prev_trading_day = _slice_window(df_prev, prev_analysis_start, prev_analysis_end)

    st.caption(f"Data source ticker: {used_ticker or symbol}")
    st.caption(
        f"Trade suggestion window: {suggest_start.strftime('%Y-%m-%d %H:%M')} to {suggest_end.strftime('%Y-%m-%d %H:%M')} ET"
    )

    if df_trading_day.empty:
        st.warning("No intraday data available for this trading day context.")
        _run_auto_update(auto_update=auto_update, selected_date=selected_date, today=today, refresh_seconds=refresh_seconds, symbol=symbol)
        return

    session_source = pd.concat([df_prev_trading_day, df_trading_day], ignore_index=True).sort_values("timestamp")
    sessions = compute_session_stats(session_source, selected_date)
    patterns = detect_patterns(sessions, df_trading_day, df_prev_trading_day)
    zones = build_htf_zones(session_source) if not session_source.empty else []

    playbook = build_strategy_playbook(
        df_today=df_trading_day,
        df_prev=df_prev_trading_day if not df_prev_trading_day.empty else None,
        sessions=sessions,
        patterns=patterns,
        zones=zones,
        now_et=_now_et(),
        whipsaw_threshold=3.0,
        trading_day=selected_date,
    )

    confluence_models = _build_model_rows(playbook, start_ts=suggest_start, end_ts=suggest_end)
    level_models = _build_level_models(
        playbook=playbook,
        df_trading_day=df_trading_day,
        sessions=sessions,
        selected_date=selected_date,
        start_ts=suggest_start,
        end_ts=suggest_end,
    )
    current_models = confluence_models + level_models
    all_models = _merge_daily_registry(symbol=symbol, selected_date=selected_date, current_models=current_models)
    all_models = _enforce_trade_window(
        models=all_models,
        start_ts=suggest_start,
        end_ts=suggest_end,
        now_ts=pd.Timestamp(_now_et()),
        reference_df=df_trading_day,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Setups Tracked", f"{len(all_models)}")
    c2.metric("Setup Forming", f"{sum(1 for r in all_models if r.get('status') == 'forming')}")
    c3.metric("Trigger Hit", f"{sum(1 for r in all_models if r.get('status') == 'triggered')}")
    c4.metric("Resolved", f"{sum(1 for r in all_models if r.get('status') in {'target_hit', 'invalidated'})}")

    st.markdown("### Trade Setup Models")
    if not all_models:
        st.info("No directional trade setups are available yet for this trading day.")
    else:
        latest_model = all_models[0]
        with st.expander("Select trade model to view", expanded=False):
            options: List[str] = []
            for idx, model in enumerate(all_models, start=1):
                result_text = _selector_result_text(model)
                options.append(
                    f"{idx}. {model.get('suggested_time_text', 'n/a')} | {model.get('action', 'Wait')} | "
                    f"{model.get('confluence', 'n/a')} | {result_text} | {model.get('grade', 'C')}"
                )
            selected_label = st.selectbox(
                "Trade model",
                options=options,
                index=0,
                key=f"trade_model_selected::{symbol}::{selected_date.isoformat()}",
            )
            selected_idx = options.index(selected_label) if selected_label in options else 0
            selected_model = all_models[selected_idx]

            st.markdown(
                f"#### Selected: {selected_model.get('action', 'Wait')} | {selected_model.get('confluence', 'n/a')} | Grade {selected_model.get('grade', 'C')}"
            )
            h1, h2, h3, h4 = st.columns(4)
            h1.metric("Status", selected_model.get("status_label", "🟡 Setup Forming"))
            h2.metric("Entry", _format_level(selected_model.get("entry")))
            h3.metric("Target", _format_level(selected_model.get("target")))
            h4.metric("Stop/Invalidation", _format_level(selected_model.get("stop")))

            p1, p2, p3 = st.columns(3)
            p1.metric("P(Fill)", f"{float(selected_model.get('fill_prob', 0.0)):.1f}%")
            p2.metric("P(Target)", f"{float(selected_model.get('target_prob', 0.0)):.1f}%")
            p3.metric("P(Invalidation)", f"{float(selected_model.get('invalidation_prob', 0.0)):.1f}%")

            trigger_hit = selected_model.get("status") in {"triggered", "target_hit", "invalidated", "time_limited"}
            st.caption(f"Trigger hit: {'🟢 Yes' if trigger_hit else 'No'}")
            st.caption(
                f"Suggested at {selected_model.get('suggested_time_text', 'n/a')} | Entry hit: {selected_model.get('entry_hit', 'No')} | "
                f"Entry time: {selected_model.get('execution_time', 'n/a')} | Outcome: {selected_model.get('outcome', 'n/a')}"
            )
            st.caption(str(selected_model.get("confidence_reasoning", "n/a")))

            boosts = selected_model.get("boosts", []) or []
            invalidators = selected_model.get("invalidators", []) or []
            if boosts:
                st.markdown("Boost confidence factors:")
                for item in boosts:
                    st.write(f"- {item}")
            if invalidators:
                st.markdown("Invalidation/exit risk factors:")
                for item in invalidators:
                    st.write(f"- {item}")

            with st.expander("Data used for this model"):
                for item in selected_model.get("data_used", []) or []:
                    st.write(f"- {item}")

        st.markdown("#### Latest Suggested Trade")
        st.caption(
            f"{latest_model.get('suggested_time_text', 'n/a')} | {latest_model.get('action', 'Wait')} | "
            f"{latest_model.get('confluence', 'n/a')} | {latest_model.get('status_label', 'n/a')} | Grade {latest_model.get('grade', 'C')}"
        )

    st.markdown("### Repeated Setups")
    with st.expander("Show/Hide repeated setups identified", expanded=False):
        repeated_rows = _build_repeated_setup_rows(all_models)
        if repeated_rows:
            st.dataframe(pd.DataFrame(repeated_rows), use_container_width=True)
        else:
            st.write("No repeated setups identified yet (minimum 2 occurrences required).")

    st.markdown("### Daily Trade Outcome Log")
    st.caption(
        "Retractable daily log includes suggestion time, entry-hit status and time, and whether target or stop/invalidation was hit with exit time."
    )
    with st.expander("Show/Hide all trades for this day", expanded=False):
        log_rows: List[Dict[str, Any]] = []
        for model in all_models:
            log_rows.append(
                {
                    "Suggested Time": model.get("suggested_time_text", "n/a"),
                    "Action": model.get("action", "Wait"),
                    "Confluence": model.get("confluence", "n/a"),
                    "Entry": _format_level(model.get("entry")),
                    "Target": _format_level(model.get("target")),
                    "Stop/Invalidation": _format_level(model.get("stop")),
                    "Risk Ratio": _risk_ratio_text(model),
                    "Trigger Hit": "Yes" if model.get("status") in {"triggered", "target_hit", "invalidated", "time_limited"} else "No",
                    "Entry Hit": model.get("entry_hit", "No"),
                    "Entry Hit Time": model.get("execution_time", "n/a"),
                    "Result": model.get("outcome", "n/a"),
                    "Target or Stop": (
                        "Target"
                        if model.get("status") == "target_hit"
                        else "Stop/Invalidation"
                        if model.get("status") == "invalidated"
                        else "14:30 Cutoff"
                        if model.get("status") == "time_limited"
                        else "Expired"
                        if model.get("status") == "expired"
                        else "Open/Pending"
                    ),
                    "Target/Stop Time": model.get("exit_time", "n/a"),
                    "Grade": model.get("grade", "C"),
                }
            )

        if log_rows:
            st.dataframe(pd.DataFrame(log_rows), use_container_width=True)
        else:
            st.write("No trade models logged yet for this day.")

    _run_auto_update(
        auto_update=bool(auto_update),
        selected_date=selected_date,
        today=today,
        refresh_seconds=int(refresh_seconds),
        symbol=symbol,
    )
