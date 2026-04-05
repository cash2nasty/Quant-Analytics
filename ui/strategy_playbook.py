import datetime as dt
import hashlib
import json
import time
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from data.data_fetcher import fetch_intraday_ohlcv
from engines.patterns import detect_patterns
from engines.sessions import compute_session_stats
from engines.strategy_playbook import build_strategy_playbook
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


def _trading_day_bounds(trading_day: dt.date) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp.combine(trading_day - dt.timedelta(days=1), dt.time(18, 0))
    end = pd.Timestamp.combine(trading_day, dt.time(16, 59, 59))
    return start, end


def _slice_trading_day(df: pd.DataFrame, trading_day: dt.date) -> pd.DataFrame:
    if df is None or df.empty or "timestamp" not in df.columns:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    start, end = _trading_day_bounds(trading_day)
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    mask = (out["timestamp"] >= start) & (out["timestamp"] <= end)
    return out.loc[mask].sort_values("timestamp").reset_index(drop=True)


def _parse_hhmm(value: str) -> tuple[dt.time | None, str | None]:
    text = str(value or "").strip()
    if len(text) != 5 or text[2] != ":":
        return None, "Use HH:MM format (for example 09:35 or 18:10)."
    hh, mm = text[:2], text[3:5]
    if not (hh.isdigit() and mm.isdigit()):
        return None, "Use numeric HH:MM format only."
    h = int(hh)
    m = int(mm)
    if h < 0 or h > 23 or m < 0 or m > 59:
        return None, "Hour must be 00-23 and minute must be 00-59."
    return dt.time(h, m), None


def _resolve_trading_day_asof(selected_date: dt.date, hhmm: str) -> tuple[pd.Timestamp | None, str | None]:
    t, err = _parse_hhmm(hhmm)
    if err:
        return None, err
    if t is None:
        return None, "Invalid time."

    # Trading-day window is 18:00 (prior calendar day) through 16:59 (selected date).
    if dt.time(17, 0) <= t <= dt.time(17, 59, 59):
        return None, "Time must be inside trading-day window 18:00-16:59 ET."

    if t >= dt.time(18, 0):
        ts_date = selected_date - dt.timedelta(days=1)
    else:
        ts_date = selected_date

    return pd.Timestamp.combine(ts_date, t), None


def _status_color(value: str) -> str:
    v = (value or "").lower()
    if v in {"yes", "bullish", "fresh", "tested", "retested", "tradeable whipsaw"}:
        return "🟢"
    if v in {"no", "bearish", "invalidated", "untradeable whipsaw"}:
        return "🔴"
    return "🟡"


def _fmt_price(value: object) -> str:
    try:
        return f"{float(value):.2f}"
    except Exception:
        return "n/a"


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _trade_row_key(row: Dict[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("Confluence", "n/a")),
            str(row.get("Suggested Time", "n/a")),
            str(row.get("Action", "Wait")),
            str(row.get("Suggested Entry", "n/a")),
        ]
    )


def _build_top_unfilled_trade_picks(
    playbook: Dict[str, Any],
    asof_ts: pd.Timestamp | None = None,
    suggested_at_app_time: str | None = None,
) -> Dict[str, Any]:
    execution_rows = playbook.get("entry_execution_tracker", []) or []
    confluences = playbook.get("confluences", []) or []

    liquidity_by_name: Dict[str, float] = {}
    for row in confluences:
        name = str(row.get("Confluence", "n/a"))
        liquidity_by_name[name] = max(
            liquidity_by_name.get(name, 0.0),
            _to_float(row.get("Liquidity Score", 0.0), 0.0),
        )

    candidates: List[Dict[str, Any]] = []
    for row in execution_rows:
        action = str(row.get("Action", "Wait"))
        executed = str(row.get("Executed", "No"))
        outcome = str(row.get("Outcome", "")).lower()
        if action not in {"Long", "Short"}:
            continue
        if executed == "Yes":
            continue
        if outcome in {"skipped", "failed", "successful", "open"}:
            continue

        if asof_ts is not None:
            suggested_time = pd.to_datetime(row.get("Suggested Time", ""), errors="coerce")
            if pd.notna(suggested_time) and suggested_time > asof_ts:
                continue

        rr = _to_float(row.get("RR", 0.0), 0.0)
        conf = _to_float(row.get("Entry Confidence", 0.0), 0.0)
        liq = liquidity_by_name.get(str(row.get("Confluence", "n/a")), 0.0)
        liq_score = max(0.0, min(100.0, liq * 35.0))

        fill_conf = max(
            5.0,
            min(
                95.0,
                30.0 + conf * 0.50 + (rr - 0.90) * 18.0 + liq_score * 0.12,
            ),
        )
        success_given_fill = max(
            5.0,
            min(
                95.0,
                24.0 + conf * 0.55 + (rr - 0.90) * 42.0 + liq_score * 0.10,
            ),
        )
        success_conf = max(1.0, min(95.0, fill_conf * (success_given_fill / 100.0)))
        fail_conf = max(1.0, min(95.0, fill_conf * ((100.0 - success_given_fill) / 100.0)))

        rank_score = (
            fill_conf * 0.42
            + success_conf * 0.43
            - fail_conf * 0.35
            + conf * 0.20
            + liq_score * 0.08
        )

        row_copy = dict(row)
        row_copy["Success confidence"] = round(success_conf, 1)
        row_copy["Fill confidence"] = round(fill_conf, 1)
        row_copy["Fail confidence"] = round(fail_conf, 1)
        row_copy["_liq_score"] = round(liq, 2)
        row_copy["_rank_score"] = round(rank_score, 2)
        row_copy["_trade_key"] = _trade_row_key(row_copy)
        candidates.append(row_copy)

    ranked = sorted(candidates, key=lambda r: (_to_float(r.get("_rank_score"), 0.0)), reverse=True)
    top_two = ranked[:2]

    best_key = ""
    if top_two:
        best = sorted(
            top_two,
            key=lambda r: (
                -_to_float(r.get("Fill confidence"), 0.0),
                _to_float(r.get("Fail confidence"), 100.0),
                -_to_float(r.get("Success confidence"), 0.0),
                -_to_float(r.get("_rank_score"), 0.0),
            ),
        )[0]
        best_key = str(best.get("_trade_key", ""))

    table_rows = []
    for row in top_two:
        table_rows.append(
            {
                "Confluence": row.get("Confluence", "n/a"),
                "Confluence Range": row.get("Confluence Range", "n/a"),
                "Suggested time": row.get("Suggested Time", "n/a"),
                "Suggested at (App Time)": row.get("Suggested App Time", suggested_at_app_time or "n/a"),
                "Action": row.get("Action", "Wait"),
                "Suggested Entry": row.get("Suggested Entry", "n/a"),
                "Entry Price": row.get("Entry Price", "n/a"),
                "Risk Price(Ticks)": row.get("Risk Pts(Ticks)", "n/a"),
                "Target Price(Ticks)": row.get("Target Pts(Ticks)", "n/a"),
                "RR": row.get("RR", "n/a"),
                "Success confidence": f"{_to_float(row.get('Success confidence', 0.0), 0.0):.1f}%",
                "Fill confidence": f"{_to_float(row.get('Fill confidence', 0.0), 0.0):.1f}%",
                "Fail confidence": f"{_to_float(row.get('Fail confidence', 0.0), 0.0):.1f}%",
            }
        )

    return {
        "rows": table_rows,
        "detailed_rows": top_two,
        "best_trade_key": best_key,
        "active": True,
        "status": "Top picks active.",
    }


def _build_daily_summary(playbook: dict) -> tuple[str, list[str]]:
    decision = playbook.get("decision", {})
    primary_trigger = playbook.get("primary_trigger") or {}
    entry_blueprints = playbook.get("entry_blueprints", []) or []
    confluence_entry_styles = playbook.get("confluence_entry_styles", []) or []
    daily_entry_capacity = playbook.get("daily_entry_capacity", {}) or {}
    confluences = playbook.get("confluences", []) or []
    targets = playbook.get("targets", []) or []
    vwap_levels = playbook.get("vwap_levels", []) or []
    power_hour = playbook.get("power_hour", {}) or {}
    open_watch = playbook.get("open_pattern_watch", {}) or {}
    risk_engine = playbook.get("risk_engine", {}) or {}
    vwap_probs = playbook.get("vwap_probabilities", {}) or {}
    momentum = playbook.get("momentum_prediction", {}) or {}

    trade_today = str(decision.get("trade_today", "Wait"))
    ny_mode = str(decision.get("ny_mode", "n/a"))
    ny_direction = str(decision.get("ny_direction", "Neutral"))
    confidence = float(decision.get("confidence", 0.0))

    trigger_name = str(primary_trigger.get("name", "No trigger yet"))
    trigger_time = str(primary_trigger.get("time", "n/a"))

    liquidity_top = [c for c in confluences if bool(c.get("Top Liquidity", False))][:3]
    if not liquidity_top:
        liquidity_top = sorted(confluences, key=lambda row: -float(row.get("Liquidity Score", 0.0)))[:3]

    top_target = targets[0] if targets else None
    top_setup = entry_blueprints[0] if entry_blueprints else None

    daily_vwap = next((row for row in vwap_levels if str(row.get("Name", "")).lower().startswith("daily vwap")), None)

    paragraph = (
        f"Today is classified as {ny_mode} with {ny_direction} bias. The current call is {trade_today} "
        f"at {confidence * 100:.0f}% confidence. "
        f"Primary trigger status: {trigger_name} ({trigger_time}). "
        f"Use this as the anchor for risk-on execution only if follow-through and retest behavior remain valid."
    )

    bullets: list[str] = []
    if top_setup:
        bullets.append(
            f"Entry focus: {top_setup.get('Setup', 'n/a')} ({top_setup.get('Action', 'Wait')}) — {top_setup.get('IfThen', 'n/a')}"
        )
        bullets.append(
            f"Execution trigger: {top_setup.get('Trigger', 'n/a')} | TF: {top_setup.get('Execution TF', 'n/a')} | Session: {top_setup.get('Session', 'n/a')}"
        )
        if top_setup.get("Confluence Entry"):
            bullets.append(
                f"Confluence execution: {top_setup.get('Confluence Entry')} ({top_setup.get('Entry Reason', 'n/a')})"
            )

    if confluence_entry_styles:
        top_style = confluence_entry_styles[0]
        bullets.append(
            f"Top zone style: {top_style.get('Confluence', 'n/a')} -> {top_style.get('Suggested Entry', 'n/a')} "
            f"(tap {top_style.get('First Tap Price', 'n/a')} | midline {top_style.get('Midline Price', 'n/a')})"
        )
        bullets.append(
            f"Entry style timing/exit: formed {top_style.get('Formed Time', 'n/a')} | exit {top_style.get('Exit', 'n/a')}"
        )

    if daily_entry_capacity:
        bullets.append(
            "Daily entry estimate: "
            f"{daily_entry_capacity.get('expected_min', 0)}-{daily_entry_capacity.get('expected_max', 0)} "
            f"(likely {daily_entry_capacity.get('likely', 0)})."
        )
        bullets.append(
            f"End-of-day execution status: {daily_entry_capacity.get('eod_execution_status', 'Pending')}"
        )

    if daily_vwap:
        bullets.append(
            f"Key VWAP level: Daily VWAP at {_fmt_price(daily_vwap.get('Price'))} ({daily_vwap.get('Status', 'n/a')})"
        )

    if liquidity_top:
        levels = ", ".join(
            [
                f"{row.get('Confluence', 'n/a')} [{_fmt_price(row.get('Price Low'))}-{_fmt_price(row.get('Price High'))}] ({row.get('Liquidity Level', 'n/a')})"
                for row in liquidity_top
            ]
        )
        bullets.append(f"Important liquidity zones: {levels}")

    if top_target:
        bullets.append(
            f"Nearest target: {top_target.get('Target', 'n/a')} at {_fmt_price(top_target.get('Price'))} (distance {_fmt_price(top_target.get('Distance'))})"
        )

    if power_hour:
        bullets.append(
            f"Power hour plan: focus {power_hour.get('focus', 'No')}, bias {power_hour.get('bias', 'Neutral')} — {power_hour.get('reason', 'n/a')}"
        )

    if open_watch:
        bullets.append(
            f"US Open Pattern Watch: {open_watch.get('status', 'Not Active')} ({float(open_watch.get('confidence', 0.0)):.0f}% confidence) — {open_watch.get('reason', 'n/a')}"
        )

    if risk_engine:
        bullets.append(
            f"Risk engine: {risk_engine.get('status', 'Caution')} | side {risk_engine.get('side', 'Wait')} | R:R {risk_engine.get('rr', 'n/a')}"
        )
        if risk_engine.get("trade_summary"):
            bullets.append(f"Risk engine summary: {risk_engine.get('trade_summary')}")
        if risk_engine.get("expected_behavior"):
            bullets.append(f"Expected behavior: {risk_engine.get('expected_behavior')}")

    if vwap_probs:
        bullets.append(
            "NY VWAP probabilities: "
            f"reversion {float(vwap_probs.get('mean_reversion_prob', 50.0)):.1f}% | "
            f"expansion {float(vwap_probs.get('expansion_prob', 50.0)):.1f}%"
        )

    if momentum:
        bullets.append(
            f"Momentum model: {momentum.get('predicted', 'Neutral')} "
            f"(up {float(momentum.get('prob_up', 50.0)):.1f}% / down {float(momentum.get('prob_down', 50.0)):.1f}%)"
        )

    support = decision.get("supporting_factors", []) or []
    block = decision.get("blocking_factors", []) or []
    if support:
        bullets.append(f"Supporting context: {', '.join([str(s) for s in support[:3]])}")
    if block:
        bullets.append(f"Risk flags: {', '.join([str(b) for b in block[:3]])}")

    return paragraph, bullets


MARKET_UPDATE_FILTERS = [
    "FVGs",
    "OB's",
    "BB's",
    "Trade Today Decision",
    "Trigger Detection",
    "Entry Playbook Updates",
    "Top Trade Picks",
    "Power Hour Focus",
    "Important Updates",
]


def _confluence_filter_label(confluence_name: str) -> str:
    name = str(confluence_name or "").upper()
    if "FVG" in name:
        return "FVGs"
    if "OB" in name or "ORDER BLOCK" in name:
        return "OB's"
    if "BB" in name or "BREAKER" in name:
        return "BB's"
    return ""


def _confluence_id(row: Dict[str, Any]) -> str:
    name = str(row.get("Confluence", "n/a"))
    low = _fmt_price(row.get("Price Low"))
    high = _fmt_price(row.get("Price High"))
    formed = str(row.get("Formed Time", "n/a"))
    return f"{name}|{low}|{high}|{formed}"


def _snapshot_playbook(playbook: Dict[str, Any]) -> Dict[str, Any]:
    decision = playbook.get("decision", {}) or {}
    trigger = playbook.get("primary_trigger") or {}
    triggers = playbook.get("triggers", []) or []
    confirmations = playbook.get("confirmation_triggers", []) or []
    entry_rows = playbook.get("entry_blueprints", []) or []
    entry_style_rows = playbook.get("confluence_entry_styles", []) or []
    power_hour = playbook.get("power_hour", {}) or {}
    confluence_rows = playbook.get("confluences", []) or []
    top_trade_picks = playbook.get("top_trade_picks", {}) or {}

    confluence_map: Dict[str, Dict[str, Any]] = {}
    for row in confluence_rows:
        cid = _confluence_id(row)
        confluence_map[cid] = {
            "id": cid,
            "name": str(row.get("Confluence", "n/a")),
            "type": _confluence_filter_label(str(row.get("Confluence", ""))),
            "price_low": _fmt_price(row.get("Price Low")),
            "price_high": _fmt_price(row.get("Price High")),
            "formed_time": str(row.get("Formed Time", "n/a")),
            "retest_count": int(float(row.get("Retest Count", 0) or 0)),
            "status": str(row.get("Status", "n/a")),
            "liquidity_level": str(row.get("Liquidity Level", "n/a")),
            "top_liquidity": bool(row.get("Top Liquidity", False)),
        }

    entry_signature = [
        {
            "Setup": str(r.get("Setup", "n/a")),
            "Action": str(r.get("Action", "Wait")),
            "IfThen": str(r.get("IfThen", "n/a")),
            "Trigger": str(r.get("Trigger", "n/a")),
            "Confluence Entry": str(r.get("Confluence Entry", "n/a")),
            "Entry Reason": str(r.get("Entry Reason", "n/a")),
            "Execution TF": str(r.get("Execution TF", "n/a")),
            "Session": str(r.get("Session", "n/a")),
        }
        for r in entry_rows
    ]

    entry_style_map: Dict[str, Dict[str, Any]] = {}
    for row in entry_style_rows:
        sid = _confluence_id(
            {
                "Confluence": row.get("Confluence", "n/a"),
                "Price Low": row.get("First Tap Price", "n/a"),
                "Price High": row.get("Midline Price", "n/a"),
                "Formed Time": row.get("Formed Time", "n/a"),
            }
        )
        entry_style_map[sid] = {
            "id": sid,
            "confluence": str(row.get("Confluence", "n/a")),
            "formed_time": str(row.get("Formed Time", "n/a")),
            "suggested_entry": str(row.get("Suggested Entry", "n/a")),
            "action": str(row.get("Action", "Wait")),
            "side": str(row.get("Side", "Neutral")),
            "exit": str(row.get("Exit", "n/a")),
            "tap_hit": str(row.get("Tap Hit", "No")),
            "tap_time": str(row.get("Tap Time", "n/a")),
            "midline_hit": str(row.get("Midline Hit", "No")),
            "midline_time": str(row.get("Midline Time", "n/a")),
            "zone_invalidated": str(row.get("Zone Invalidated", "No")),
            "preferred_target": row.get("Preferred Target Price"),
            "min_target": row.get("Minimum Target Price"),
            "max_target": row.get("Maximum Target Price"),
            "reaction_signal": str(row.get("Reaction Signal", "None")),
            "reaction_score": row.get("Reaction Score"),
            "reaction_time": str(row.get("Reaction Time", "n/a")),
            "reaction_why": str(row.get("Reaction Why", "n/a")),
            "entry_style_reason": str(row.get("Entry Style Reason", row.get("Reason", "n/a"))),
            "expected_retests": str(row.get("Expected Retests", "n/a")),
            "entry_timing": str(row.get("Entry Timing", "n/a")),
            "reason": str(row.get("Reason", "n/a")),
        }

    return {
        "decision": {
            "trade_today": str(decision.get("trade_today", "Wait")),
            "ny_mode": str(decision.get("ny_mode", "n/a")),
            "ny_direction": str(decision.get("ny_direction", "Neutral")),
            "primary_reason": str(decision.get("primary_reason", "n/a")),
            "confidence": float(decision.get("confidence", 0.0) or 0.0),
        },
        "trigger": {
            "name": str(trigger.get("name", "")),
            "direction": str(trigger.get("direction", "")),
            "time": str(trigger.get("time", "")),
            "price": _fmt_price(trigger.get("price")),
            "details": str(trigger.get("details", "")),
            "must_happen": str(trigger.get("must_happen", "")),
        },
        "triggers": [
            {
                "name": str(t.get("name", "")),
                "direction": str(t.get("direction", "")),
                "time": str(t.get("time", "")),
                "details": str(t.get("details", "")),
            }
            for t in triggers
        ],
        "confirmations": [
            {
                "name": str(t.get("name", "")),
                "direction": str(t.get("direction", "")),
                "time": str(t.get("time", "")),
                "details": str(t.get("details", "")),
            }
            for t in confirmations
        ],
        "entry": entry_signature,
        "entry_styles": entry_style_map,
        "power_hour": {
            "focus": str(power_hour.get("focus", "No")),
            "bias": str(power_hour.get("bias", "Neutral")),
            "reason": str(power_hour.get("reason", "n/a")),
        },
        "top_trade_picks": {
            "active": bool(top_trade_picks.get("active", False)),
            "status": str(top_trade_picks.get("status", "")),
            "best_trade_key": str(top_trade_picks.get("best_trade_key", "")),
            "rows": [
                {
                    "Confluence": str(r.get("Confluence", "n/a")),
                    "Suggested time": str(r.get("Suggested time", "n/a")),
                    "Action": str(r.get("Action", "Wait")),
                    "RR": str(r.get("RR", "n/a")),
                    "Success confidence": str(r.get("Success confidence", "n/a")),
                    "Fill confidence": str(r.get("Fill confidence", "n/a")),
                    "Fail confidence": str(r.get("Fail confidence", "n/a")),
                }
                for r in (top_trade_picks.get("rows", []) or [])
            ],
        },
        "confluences": confluence_map,
    }


def _event_signature(event: Dict[str, Any]) -> str:
    payload = {
        "kind": event.get("kind"),
        "summary": event.get("summary"),
        "where": event.get("where"),
        "before": event.get("before"),
        "after": event.get("after"),
        "event_time": event.get("event_time"),
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _event_base(reported_at: str, kind: str, tags: List[str], where: str, summary: str) -> Dict[str, Any]:
    return {
        "reported_at": reported_at,
        "event_time": reported_at,
        "kind": kind,
        "tags": tags,
        "where": where,
        "summary": summary,
        "before": "n/a",
        "after": "n/a",
        "why": "n/a",
        "expect": "n/a",
        "important": "Important Updates" in tags,
        "confluence_type": "",
        "formed_time": "n/a",
        "retest_count": 0,
        "liquidity_level": "n/a",
        "invalidated": "No",
    }


def _build_update_events(prev_snapshot: Dict[str, Any], curr_snapshot: Dict[str, Any], now_et: dt.datetime) -> List[Dict[str, Any]]:
    reported_at = now_et.strftime("%Y-%m-%d %H:%M:%S")
    events: List[Dict[str, Any]] = []

    prev_decision = prev_snapshot.get("decision", {})
    curr_decision = curr_snapshot.get("decision", {})
    if prev_decision != curr_decision:
        before_status = (
            f"{_status_color(prev_decision.get('trade_today', 'Wait'))} {prev_decision.get('trade_today', 'Wait')}"
        )
        after_status = (
            f"{_status_color(curr_decision.get('trade_today', 'Wait'))} {curr_decision.get('trade_today', 'Wait')}"
        )
        event = _event_base(
            reported_at,
            "decision_change",
            ["Trade Today Decision", "Important Updates"],
            "Trade Decision",
            f"Trade decision updated to {after_status}.",
        )
        event["before"] = (
            f"{before_status} | Mode: {prev_decision.get('ny_mode', 'n/a')} | Direction: {prev_decision.get('ny_direction', 'n/a')}"
        )
        event["after"] = (
            f"{after_status} | Mode: {curr_decision.get('ny_mode', 'n/a')} | Direction: {curr_decision.get('ny_direction', 'n/a')}"
        )
        event["why"] = curr_decision.get("primary_reason", "n/a")
        event["expect"] = "Follow current decision and wait for confirming trigger/structure follow-through."
        events.append(event)

    prev_trigger = prev_snapshot.get("trigger", {})
    curr_trigger = curr_snapshot.get("trigger", {})
    if prev_trigger != curr_trigger:
        prev_name = prev_trigger.get("name", "")
        curr_name = curr_trigger.get("name", "")
        if not prev_name and curr_name:
            summary = f"Primary trigger detected: {curr_name}."
        elif prev_name and not curr_name:
            summary = "Primary trigger is no longer active."
        else:
            summary = f"Primary trigger changed from {prev_name or 'none'} to {curr_name or 'none'}."

        event = _event_base(
            reported_at,
            "trigger_change",
            ["Trigger Detection", "Important Updates"],
            "Trigger Report",
            summary,
        )
        event["before"] = (
            f"{prev_name or 'No trigger'} | {prev_trigger.get('direction', 'n/a')} | {prev_trigger.get('time', 'n/a')}"
        )
        event["after"] = (
            f"{curr_name or 'No trigger'} | {curr_trigger.get('direction', 'n/a')} | {curr_trigger.get('time', 'n/a')}"
        )
        event["event_time"] = curr_trigger.get("time", reported_at)
        event["why"] = curr_trigger.get("details", "Trigger conditions were re-evaluated.")
        event["expect"] = curr_trigger.get("must_happen", "Watch for retest/continuation confirmation.")
        events.append(event)

    prev_triggers = prev_snapshot.get("triggers", []) or []
    curr_triggers = curr_snapshot.get("triggers", []) or []
    if prev_triggers != curr_triggers:
        prev_names = [str(t.get("name", "")) for t in prev_triggers if t.get("name")]
        curr_names = [str(t.get("name", "")) for t in curr_triggers if t.get("name")]
        event = _event_base(
            reported_at,
            "trigger_monitoring_change",
            ["Trigger Detection", "Important Updates"],
            "Trigger Monitor",
            "Trigger monitoring list changed.",
        )
        event["before"] = ", ".join(prev_names) if prev_names else "No triggers"
        event["after"] = ", ".join(curr_names) if curr_names else "No triggers"
        latest_time = curr_triggers[-1].get("time", reported_at) if curr_triggers else reported_at
        event["event_time"] = latest_time
        event["why"] = "The model keeps scanning for additional confirmations even after first trigger." 
        event["expect"] = "Continue monitoring OR and bias-confirmation triggers for upgrades/downgrades."
        events.append(event)

    prev_entry = prev_snapshot.get("entry", [])
    curr_entry = curr_snapshot.get("entry", [])
    if prev_entry != curr_entry:
        prev_setup = prev_entry[0]["Setup"] if prev_entry else "No setup"
        curr_setup = curr_entry[0]["Setup"] if curr_entry else "No setup"
        event = _event_base(
            reported_at,
            "entry_playbook_change",
            ["Entry Playbook Updates", "Important Updates"],
            "Entry Playbooks",
            f"Entry playbook updated to {curr_setup}.",
        )
        event["before"] = prev_setup
        event["after"] = curr_setup
        event["why"] = "Playbook alignment changed based on current trigger, mode, and confluence state."
        event["expect"] = "Use updated if/then conditions for next executable setup."
        events.append(event)

    prev_entry_styles = prev_snapshot.get("entry_styles", {})
    curr_entry_styles = curr_snapshot.get("entry_styles", {})
    for sid, curr_style in curr_entry_styles.items():
        prev_style = prev_entry_styles.get(sid, {})

        if not prev_style:
            event = _event_base(
                reported_at,
                "entry_style_baseline",
                ["Entry Playbook Updates", "Important Updates"],
                "Entry Playbook",
                f"Confirmed entry style loaded for {curr_style.get('confluence', 'n/a')}: {curr_style.get('suggested_entry', 'n/a')}",
            )
            event["before"] = "No prior confirmed style"
            event["after"] = (
                f"Entry: {curr_style.get('suggested_entry', 'n/a')} | Exit: {curr_style.get('exit', 'n/a')}"
            )
            event["why"] = curr_style.get("reason", "n/a")
            event["entry_style_reason"] = curr_style.get("entry_style_reason", curr_style.get("reason", "n/a"))
            event["expect"] = (
                "Preferred target: "
                f"{curr_style.get('preferred_target', 'n/a')} | "
                f"Min target: {curr_style.get('min_target', 'n/a')} | "
                f"Max target: {curr_style.get('max_target', 'n/a')}"
            )
            events.append(event)

        if prev_style.get("tap_hit", "No") != "Yes" and curr_style.get("tap_hit", "No") == "Yes":
            event = _event_base(
                reported_at,
                "entry_style_tap_hit",
                ["Entry Playbook Updates", "Important Updates"],
                "Entry Playbook",
                f"Price tapped confluence for {curr_style.get('confluence', 'n/a')}.",
            )
            event["event_time"] = curr_style.get("tap_time", reported_at)
            event["before"] = "Tap status: No"
            event["after"] = f"Tap status: Yes @ {curr_style.get('tap_time', 'n/a')}"
            event["why"] = "Price entered the confluence boundaries and activated touch condition."
            event["entry_style_reason"] = curr_style.get("entry_style_reason", curr_style.get("reason", "n/a"))
            event["expect"] = (
                "Watch follow-through toward targets. "
                f"Preferred target: {curr_style.get('preferred_target', 'n/a')} | "
                f"Min: {curr_style.get('min_target', 'n/a')} | "
                f"Max: {curr_style.get('max_target', 'n/a')}"
            )
            events.append(event)

        if prev_style.get("midline_hit", "No") != "Yes" and curr_style.get("midline_hit", "No") == "Yes":
            event = _event_base(
                reported_at,
                "entry_style_midline_hit",
                ["Entry Playbook Updates", "Important Updates"],
                "Entry Playbook",
                f"Price reached confluence midline for {curr_style.get('confluence', 'n/a')}.",
            )
            event["event_time"] = curr_style.get("midline_time", reported_at)
            event["before"] = "Midline status: No"
            event["after"] = f"Midline status: Yes @ {curr_style.get('midline_time', 'n/a')}"
            event["why"] = "Midline threshold was touched inside the active confluence zone."
            event["entry_style_reason"] = curr_style.get("entry_style_reason", curr_style.get("reason", "n/a"))
            event["expect"] = (
                "Monitor reaction quality and continuation probability. "
                f"Preferred target: {curr_style.get('preferred_target', 'n/a')} | "
                f"Min: {curr_style.get('min_target', 'n/a')} | "
                f"Max: {curr_style.get('max_target', 'n/a')}"
            )
            events.append(event)

        if prev_style.get("zone_invalidated", "No") != "Yes" and curr_style.get("zone_invalidated", "No") == "Yes":
            event = _event_base(
                reported_at,
                "entry_style_invalidated",
                ["Entry Playbook Updates", "Important Updates"],
                "Entry Playbook",
                f"Confluence invalidated for {curr_style.get('confluence', 'n/a')}.",
            )
            event["before"] = "Zone invalidated: No"
            event["after"] = "Zone invalidated: Yes"
            event["why"] = "Price behavior breached confluence validity rules for this setup."
            event["entry_style_reason"] = curr_style.get("entry_style_reason", curr_style.get("reason", "n/a"))
            event["expect"] = (
                "Stand down on this confluence and re-anchor to next valid setup. "
                f"Preferred target was {curr_style.get('preferred_target', 'n/a')} | "
                f"Min: {curr_style.get('min_target', 'n/a')} | "
                f"Max: {curr_style.get('max_target', 'n/a')}"
            )
            events.append(event)

        if (
            prev_style
            and (
                prev_style.get("suggested_entry", "n/a") != curr_style.get("suggested_entry", "n/a")
                or prev_style.get("reaction_signal", "None") != curr_style.get("reaction_signal", "None")
            )
        ):
            event = _event_base(
                reported_at,
                "entry_style_changed",
                ["Entry Playbook Updates", "Important Updates"],
                "Entry Playbook",
                f"Confirmed entry style changed for {curr_style.get('confluence', 'n/a')}.",
            )
            event["before"] = (
                f"Entry: {prev_style.get('suggested_entry', 'n/a')} | "
                f"Reaction: {prev_style.get('reaction_signal', 'None')}"
            )
            event["after"] = (
                f"Entry: {curr_style.get('suggested_entry', 'n/a')} | "
                f"Reaction: {curr_style.get('reaction_signal', 'None')}"
            )
            event["event_time"] = curr_style.get("reaction_time", reported_at)
            event["why"] = curr_style.get("reaction_why", curr_style.get("reason", "n/a"))
            event["entry_style_reason"] = curr_style.get("entry_style_reason", curr_style.get("reason", "n/a"))
            event["expect"] = (
                "Adjusted execution plan. "
                f"Preferred target: {curr_style.get('preferred_target', 'n/a')} | "
                f"Min: {curr_style.get('min_target', 'n/a')} | "
                f"Max: {curr_style.get('max_target', 'n/a')}"
            )
            events.append(event)

    prev_power = prev_snapshot.get("power_hour", {})
    curr_power = curr_snapshot.get("power_hour", {})
    if prev_power != curr_power:
        event = _event_base(
            reported_at,
            "power_hour_change",
            ["Power Hour Focus", "Important Updates"],
            "Power Hour Focus",
            f"Power-hour focus updated: {curr_power.get('focus', 'No')} / {curr_power.get('bias', 'Neutral')}",
        )
        event["before"] = (
            f"Focus: {prev_power.get('focus', 'No')} | Bias: {prev_power.get('bias', 'Neutral')}"
        )
        event["after"] = (
            f"Focus: {curr_power.get('focus', 'No')} | Bias: {curr_power.get('bias', 'Neutral')}"
        )
        event["why"] = curr_power.get("reason", "n/a")
        event["expect"] = "Use power-hour bias only if liquidity/trigger alignment confirms."
        events.append(event)

    prev_top = prev_snapshot.get("top_trade_picks", {}) or {}
    curr_top = curr_snapshot.get("top_trade_picks", {}) or {}
    if bool(curr_top.get("active", False)) and prev_top != curr_top:
        prev_rows = prev_top.get("rows", []) or []
        curr_rows = curr_top.get("rows", []) or []
        prev_best = str(prev_top.get("best_trade_key", ""))
        curr_best = str(curr_top.get("best_trade_key", ""))

        event = _event_base(
            reported_at,
            "top_trade_picks_change",
            ["Top Trade Picks", "Important Updates"],
            "Top Trade Picks",
            "Top unfilled trade picks were refreshed.",
        )
        event["before"] = f"Count {len(prev_rows)} | best key {prev_best or 'n/a'}"
        event["after"] = f"Count {len(curr_rows)} | best key {curr_best or 'n/a'}"
        if curr_rows:
            top = curr_rows[0]
            event["why"] = (
                f"Top pick now {top.get('Confluence', 'n/a')} ({top.get('Action', 'Wait')}) with "
                f"fill {top.get('Fill confidence', 'n/a')}, success {top.get('Success confidence', 'n/a')}, "
                f"fail {top.get('Fail confidence', 'n/a')}."
            )
            event["expect"] = "Risk Engine default trade focus will shift to the highest-ranked unfilled trade." 
        else:
            event["why"] = "No eligible unfilled directional trades are currently available."
            event["expect"] = "Wait for a new unfilled directional setup to appear."
        events.append(event)

    prev_confs = prev_snapshot.get("confluences", {})
    curr_confs = curr_snapshot.get("confluences", {})

    for cid, conf in curr_confs.items():
        if cid not in prev_confs:
            tags: List[str] = []
            if conf.get("type"):
                tags.append(conf["type"])
            if conf.get("top_liquidity"):
                tags.append("Important Updates")

            event = _event_base(
                reported_at,
                "confluence_formed",
                tags,
                f"{conf.get('name', 'Confluence')} [{conf.get('price_low', 'n/a')}-{conf.get('price_high', 'n/a')} ]",
                f"{conf.get('name', 'Confluence')} formed.",
            )
            event["event_time"] = conf.get("formed_time", reported_at)
            event["formed_time"] = conf.get("formed_time", "n/a")
            event["confluence_type"] = conf.get("type", "")
            event["retest_count"] = int(conf.get("retest_count", 0))
            event["liquidity_level"] = conf.get("liquidity_level", "n/a")
            event["invalidated"] = "Yes" if str(conf.get("status", "")).lower() == "invalidated" else "No"
            event["why"] = "New confluence was detected by current price/structure logic."
            event["expect"] = "Watch first test/retest behavior for confirmation or invalidation."
            events.append(event)

        else:
            prev_conf = prev_confs[cid]
            curr_status = str(conf.get("status", "n/a"))
            prev_status = str(prev_conf.get("status", "n/a"))
            if prev_status != curr_status:
                tags = [conf.get("type", "")] if conf.get("type") else []
                if conf.get("top_liquidity") or prev_conf.get("top_liquidity"):
                    tags.append("Important Updates")

                event = _event_base(
                    reported_at,
                    "confluence_status_change",
                    tags,
                    f"{conf.get('name', 'Confluence')} [{conf.get('price_low', 'n/a')}-{conf.get('price_high', 'n/a')} ]",
                    f"Confluence status changed: {prev_status} -> {curr_status}.",
                )
                event["before"] = prev_status
                event["after"] = curr_status
                event["event_time"] = reported_at
                event["formed_time"] = conf.get("formed_time", "n/a")
                event["confluence_type"] = conf.get("type", "")
                event["retest_count"] = int(conf.get("retest_count", 0))
                event["liquidity_level"] = conf.get("liquidity_level", "n/a")
                event["invalidated"] = "Yes" if curr_status.lower() == "invalidated" else "No"
                event["why"] = "Confluence validity changed after latest price action." 
                event["expect"] = "Re-anchor entries to still-valid confluence zones."
                events.append(event)

            prev_retests = int(prev_conf.get("retest_count", 0))
            curr_retests = int(conf.get("retest_count", 0))
            if curr_retests > prev_retests:
                tags = [conf.get("type", "")] if conf.get("type") else []
                if conf.get("top_liquidity"):
                    tags.append("Important Updates")

                event = _event_base(
                    reported_at,
                    "confluence_retest",
                    tags,
                    f"{conf.get('name', 'Confluence')} [{conf.get('price_low', 'n/a')}-{conf.get('price_high', 'n/a')} ]",
                    f"Confluence retest count increased to {curr_retests}.",
                )
                event["event_time"] = reported_at
                event["formed_time"] = conf.get("formed_time", "n/a")
                event["confluence_type"] = conf.get("type", "")
                event["retest_count"] = curr_retests
                event["liquidity_level"] = conf.get("liquidity_level", "n/a")
                event["invalidated"] = "Yes" if curr_status.lower() == "invalidated" else "No"
                event["why"] = "Price revisited the zone and registered another test." 
                event["expect"] = "Repeated tests can weaken zone reliability; watch rejection quality."
                events.append(event)

            if not bool(prev_conf.get("top_liquidity", False)) and bool(conf.get("top_liquidity", False)):
                tags = [conf.get("type", "")] if conf.get("type") else []
                tags.append("Important Updates")
                event = _event_base(
                    reported_at,
                    "high_liquidity_upgrade",
                    tags,
                    f"{conf.get('name', 'Confluence')} [{conf.get('price_low', 'n/a')}-{conf.get('price_high', 'n/a')} ]",
                    "Confluence upgraded to high-liquidity tier.",
                )
                event["event_time"] = reported_at
                event["formed_time"] = conf.get("formed_time", "n/a")
                event["confluence_type"] = conf.get("type", "")
                event["retest_count"] = int(conf.get("retest_count", 0))
                event["liquidity_level"] = conf.get("liquidity_level", "n/a")
                event["invalidated"] = "Yes" if curr_status.lower() == "invalidated" else "No"
                event["why"] = "Liquidity score moved into top-liquidity group."
                event["expect"] = "Prioritize this zone in execution mapping while it remains valid."
                events.append(event)

    return events


def _build_initial_events(curr_snapshot: Dict[str, Any], now_et: dt.datetime) -> List[Dict[str, Any]]:
    reported_at = now_et.strftime("%Y-%m-%d %H:%M:%S")
    events: List[Dict[str, Any]] = []

    decision = curr_snapshot.get("decision", {})
    decision_event = _event_base(
        reported_at,
        "decision_baseline",
        ["Trade Today Decision", "Important Updates"],
        "Trade Decision",
        f"Baseline decision loaded: {_status_color(decision.get('trade_today', 'Wait'))} {decision.get('trade_today', 'Wait')}.",
    )
    decision_event["after"] = (
        f"{_status_color(decision.get('trade_today', 'Wait'))} {decision.get('trade_today', 'Wait')} | "
        f"Mode: {decision.get('ny_mode', 'n/a')} | Direction: {decision.get('ny_direction', 'n/a')}"
    )
    decision_event["why"] = decision.get("primary_reason", "n/a")
    decision_event["expect"] = "Use this as the current anchor until a new update event is reported."
    events.append(decision_event)

    trigger = curr_snapshot.get("trigger", {})
    trigger_name = str(trigger.get("name", ""))
    trigger_event = _event_base(
        reported_at,
        "trigger_baseline",
        ["Trigger Detection", "Important Updates"],
        "Trigger Report",
        f"Baseline trigger state: {trigger_name or 'No trigger'}.",
    )
    trigger_event["after"] = f"{trigger_name or 'No trigger'} | {trigger.get('direction', 'n/a')} | {trigger.get('time', 'n/a')}"
    trigger_event["event_time"] = trigger.get("time", reported_at)
    trigger_event["why"] = trigger.get("details", "n/a")
    trigger_event["expect"] = trigger.get("must_happen", "Wait for valid trigger conditions.")
    events.append(trigger_event)

    entry = curr_snapshot.get("entry", [])
    if entry:
        event = _event_base(
            reported_at,
            "entry_baseline",
            ["Entry Playbook Updates", "Important Updates"],
            "Entry Playbooks",
            f"Baseline entry playbook loaded: {entry[0].get('Setup', 'No setup')}.",
        )
        event["after"] = entry[0].get("Setup", "No setup")
        event["why"] = "Current confluence/trigger context generated this active playbook view."
        event["expect"] = "Track playbook updates as market structure evolves."
        events.append(event)

    entry_styles = curr_snapshot.get("entry_styles", {})
    for _, style in entry_styles.items():
        event = _event_base(
            reported_at,
            "entry_style_baseline",
            ["Entry Playbook Updates", "Important Updates"],
            "Entry Playbook",
            f"Baseline confirmed entry style: {style.get('confluence', 'n/a')} -> {style.get('suggested_entry', 'n/a')}",
        )
        event["before"] = "n/a"
        event["after"] = (
            f"Entry: {style.get('suggested_entry', 'n/a')} | "
            f"Tap: {style.get('tap_hit', 'No')} | "
            f"Midline: {style.get('midline_hit', 'No')} | "
            f"Invalidated: {style.get('zone_invalidated', 'No')}"
        )
        event["why"] = style.get("reason", "n/a")
        event["entry_style_reason"] = style.get("entry_style_reason", style.get("reason", "n/a"))
        event["expect"] = (
            "Preferred target: "
            f"{style.get('preferred_target', 'n/a')} | "
            f"Min target: {style.get('min_target', 'n/a')} | "
            f"Max target: {style.get('max_target', 'n/a')}"
        )
        events.append(event)

    power = curr_snapshot.get("power_hour", {})
    event = _event_base(
        reported_at,
        "power_hour_baseline",
        ["Power Hour Focus", "Important Updates"],
        "Power Hour Focus",
        f"Baseline power-hour focus: {power.get('focus', 'No')} / {power.get('bias', 'Neutral')}.",
    )
    event["after"] = f"Focus: {power.get('focus', 'No')} | Bias: {power.get('bias', 'Neutral')}"
    event["why"] = power.get("reason", "n/a")
    event["expect"] = "Recheck focus as the session advances toward close."
    events.append(event)

    top_trade_picks = curr_snapshot.get("top_trade_picks", {}) or {}
    top_rows = top_trade_picks.get("rows", []) or []
    if bool(top_trade_picks.get("active", False)) and top_rows:
        first = top_rows[0]
        event = _event_base(
            reported_at,
            "top_trade_picks_baseline",
            ["Top Trade Picks", "Important Updates"],
            "Top Trade Picks",
            "Baseline top unfilled trade picks loaded.",
        )
        event["after"] = (
            f"Top 1: {first.get('Confluence', 'n/a')} | {first.get('Action', 'Wait')} | "
            f"fill {first.get('Fill confidence', 'n/a')}"
        )
        event["why"] = "Trades were ranked by fill/success/fail confidence, liquidity alignment, and confidence strength."
        event["expect"] = "Risk Engine defaults to the highest-ranked unfilled trade while selection remains user-editable."
        events.append(event)

    confluences = curr_snapshot.get("confluences", {})
    for _, conf in confluences.items():
        tags: List[str] = []
        if conf.get("type"):
            tags.append(conf["type"])
        if conf.get("top_liquidity"):
            tags.append("Important Updates")
        event = _event_base(
            reported_at,
            "confluence_baseline",
            tags,
            f"{conf.get('name', 'Confluence')} [{conf.get('price_low', 'n/a')}-{conf.get('price_high', 'n/a')} ]",
            f"Baseline confluence loaded: {conf.get('name', 'Confluence')} ({conf.get('status', 'n/a')}).",
        )
        event["event_time"] = conf.get("formed_time", reported_at)
        event["formed_time"] = conf.get("formed_time", "n/a")
        event["confluence_type"] = conf.get("type", "")
        event["retest_count"] = int(conf.get("retest_count", 0))
        event["liquidity_level"] = conf.get("liquidity_level", "n/a")
        event["invalidated"] = "Yes" if str(conf.get("status", "")).lower() == "invalidated" else "No"
        event["why"] = "Detected in current confluence map at load time."
        event["expect"] = "Watch for retests, status changes, or invalidation."
        events.append(event)

    return events


def _filter_market_events(events: List[Dict[str, Any]], selected_filters: List[str]) -> List[Dict[str, Any]]:
    if not selected_filters:
        return events
    selected = set(selected_filters)
    filtered: List[Dict[str, Any]] = []
    for event in events:
        tags = set([t for t in event.get("tags", []) if t])
        if "Important Updates" in selected and event.get("important", False):
            filtered.append(event)
            continue
        if tags.intersection(selected):
            filtered.append(event)
    return filtered


def _render_market_updates_panel(events: List[Dict[str, Any]], panel_key: str = "default") -> None:
    selected_filters = st.multiselect(
        "Filter updates",
        options=MARKET_UPDATE_FILTERS,
        default=[],
        help="Leave empty to show all updates in compact mode.",
        key=f"market_updates_filters::{panel_key}",
    )

    filtered = _filter_market_events(events, selected_filters)
    filtered = sorted(filtered, key=lambda e: str(e.get("reported_at", "")), reverse=True)

    st.markdown(
        """
        <style>
        .market-updates-box {
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-radius: 12px;
            padding: 10px 12px;
            height: 520px;
            overflow-y: auto;
        }
        .market-update-item {
            border-bottom: 1px dashed rgba(49, 51, 63, 0.15);
            padding: 8px 0;
            font-size: 0.92rem;
        }
        .market-update-item:last-child {
            border-bottom: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if not filtered:
        st.info("No market updates to show for current filters yet.")
        return

    single_mode = len(selected_filters) == 1
    selected_one = selected_filters[0] if single_mode else ""

    detail_section_filters = {
        "Trade Today Decision",
        "Trigger Detection",
        "Entry Playbook Updates",
        "Power Hour Focus",
        "Top Trade Picks",
    }
    confluence_detail_filters = {"FVGs", "OB's", "BB's"}

    st.markdown('<div class="market-updates-box">', unsafe_allow_html=True)

    for event in filtered:
        if single_mode and selected_one in detail_section_filters:
            st.markdown('<div class="market-update-item">', unsafe_allow_html=True)
            st.write(f"**Reported at:** {event.get('reported_at', 'n/a')}")
            st.write(f"**Event time:** {event.get('event_time', 'n/a')}")
            st.write(f"**What happened:** {event.get('summary', 'n/a')}")
            st.write(f"**Before:** {event.get('before', 'n/a')}")
            st.write(f"**After:** {event.get('after', 'n/a')}")
            st.write(f"**Why:** {event.get('why', 'n/a')}")
            if selected_one == "Entry Playbook Updates" and event.get("entry_style_reason"):
                st.write(f"**Entry style reasoning:** {event.get('entry_style_reason', 'n/a')}")
            st.write(f"**What to expect:** {event.get('expect', 'n/a')}")
            st.markdown('</div>', unsafe_allow_html=True)
            continue

        if single_mode and selected_one in confluence_detail_filters:
            st.markdown('<div class="market-update-item">', unsafe_allow_html=True)
            st.write(f"**Reported at:** {event.get('reported_at', 'n/a')}")
            st.write(f"**Event time:** {event.get('event_time', 'n/a')}")
            st.write(f"**What happened:** {event.get('summary', 'n/a')}")
            st.write(f"**Formed time:** {event.get('formed_time', event.get('event_time', 'n/a'))}")
            st.write(f"**Where:** {event.get('where', 'n/a')}")
            st.write(f"**Times tested:** {event.get('retest_count', 0)}")
            st.write(f"**Liquidity level:** {event.get('liquidity_level', 'n/a')}")
            st.write(f"**Invalidated:** {event.get('invalidated', 'No')}")
            st.markdown('</div>', unsafe_allow_html=True)
            continue

        st.markdown('<div class="market-update-item">', unsafe_allow_html=True)
        st.write(
            f"{event.get('reported_at', 'n/a')} | event: {event.get('event_time', 'n/a')} | {event.get('summary', 'n/a')} | {event.get('where', 'n/a')}"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def _update_market_event_state(
    symbol: str,
    selected_date: dt.date,
    playbook: Dict[str, Any],
    now_et: dt.datetime,
    max_events: int = 0,
) -> List[Dict[str, Any]]:
    state_key = f"market_updates_state::{symbol}::{selected_date.isoformat()}"
    if state_key not in st.session_state:
        st.session_state[state_key] = {
            "snapshot": None,
            "events": [],
            "seen_signatures": set(),
        }

    state = st.session_state[state_key]
    curr_snapshot = _snapshot_playbook(playbook)
    prev_snapshot = state.get("snapshot")

    if prev_snapshot is not None:
        new_events = _build_update_events(prev_snapshot, curr_snapshot, now_et)
        for event in new_events:
            signature = _event_signature(event)
            if signature in state["seen_signatures"]:
                continue
            state["seen_signatures"].add(signature)
            state["events"].append(event)

        if max_events > 0 and len(state["events"]) > max_events:
            state["events"] = state["events"][-max_events:]

    else:
        baseline_events = _build_initial_events(curr_snapshot, now_et)
        for event in baseline_events:
            signature = _event_signature(event)
            if signature in state["seen_signatures"]:
                continue
            state["seen_signatures"].add(signature)
            state["events"].append(event)

    state["snapshot"] = curr_snapshot
    st.session_state[state_key] = state
    return state.get("events", [])


def _run_auto_refresh(auto_refresh: bool, selected_date: dt.date, today: dt.date, refresh_seconds: int, symbol: str) -> None:
    should_refresh = bool(auto_refresh) and selected_date == today
    if not should_refresh:
        return

    st.caption(f"Auto-refresh is on ({int(refresh_seconds)}s cadence for today).")

    if hasattr(st, "fragment"):
        interval = f"{int(refresh_seconds)}s"

        @st.fragment(run_every=interval)
        def _refresh_tick() -> None:
            marker_key = f"playbook_last_auto_rerun::{symbol}::{selected_date.isoformat()}"
            now_ts = time.time()
            last_ts = float(st.session_state.get(marker_key, 0.0))
            if now_ts - last_ts >= max(int(refresh_seconds) - 1, 1):
                st.session_state[marker_key] = now_ts
                st.rerun()

        _refresh_tick()
    else:
        c1, c2 = st.columns([3, 1])
        with c1:
            st.caption("Auto-refresh compatibility mode: press Refresh now to pull latest state.")
        with c2:
            if st.button("Refresh now", key="playbook_refresh_now"):
                st.rerun()


def render_strategy_playbook() -> None:
    st.header("Strategy Playbook")
    st.caption(
        "Decision engine for Trade Today, NY mode, triggers, confluence timing, VWAP/STDV prices, targets, and timeframe execution guidance."
    )
    with st.expander("Key Definitions", expanded=True):
        st.markdown(
            "- **VWAP Reject**: price crosses from above to below VWAP (initial loss of VWAP).\n"
            "- **VWAP Reject Hold**: after VWAP reject, price remains below VWAP for the hold condition (4 closes in current logic).\n"
            "- **Invalidated (Confluence)**: price action has broken a zone/confluence in a way that removes its current setup validity.\n"
            "- **Regular Manipulation (AMD)**: one-sided trap/sweep then directional move.\n"
            "- **Whipsaw**: unstable two-sided movement with conflicting direction and frequent flips around key levels."
        )

    today = _now_et().date()

    with st.sidebar:
        st.subheader("Playbook Inputs")
        symbol = st.text_input("Symbol", value="NQH26")
        selected_date = st.date_input("Analysis date", value=today)
        whipsaw_threshold = st.slider("Whipsaw Risk Ratio", min_value=1.5, max_value=6.0, value=3.0, step=0.5)
        auto_refresh = st.checkbox("Auto-refresh", value=True)
        refresh_seconds = st.selectbox("Refresh every (sec)", options=[30, 60], index=1, disabled=not auto_refresh)

    res_today = fetch_intraday_ohlcv(symbol, selected_date)
    if isinstance(res_today, tuple):
        df_today, used_ticker = res_today
    else:
        df_today = res_today
        used_ticker = ""

    prev_date = get_prev_trading_day(selected_date)
    prev_prev_date = get_prev_trading_day(prev_date)

    res_prev = fetch_intraday_ohlcv(symbol, prev_date)
    if isinstance(res_prev, tuple):
        df_prev, _ = res_prev
    else:
        df_prev = res_prev

    res_prev_prev = fetch_intraday_ohlcv(symbol, prev_prev_date)
    if isinstance(res_prev_prev, tuple):
        df_prev_prev, _ = res_prev_prev
    else:
        df_prev_prev = res_prev_prev

    df_today = _prepare_df(df_today)
    df_prev = _prepare_df(df_prev)
    df_prev_prev = _prepare_df(df_prev_prev)

    current_trading_source = pd.concat([df_prev, df_today], ignore_index=True).sort_values("timestamp")
    previous_trading_source = pd.concat([df_prev_prev, df_prev], ignore_index=True).sort_values("timestamp")

    df_trading_day = _slice_trading_day(current_trading_source, selected_date)
    df_prev_trading_day = _slice_trading_day(previous_trading_source, prev_date)

    has_today = df_trading_day is not None and not df_trading_day.empty
    has_prev = df_prev_trading_day is not None and not df_prev_trading_day.empty

    td_start, td_end = _trading_day_bounds(selected_date)

    st.caption(f"Data source ticker: {used_ticker or symbol}")
    st.caption(
        f"Trading day window: {td_start.strftime('%Y-%m-%d %H:%M')} to {td_end.strftime('%Y-%m-%d %H:%M')} ET"
    )

    mode_col, time_col, live_col = st.columns([1.2, 2.4, 1.2])
    with mode_col:
        view_mode = st.radio(
            "View mode",
            options=["Live", "Time Travel"],
            horizontal=True,
            key=f"playbook_view_mode::{symbol}::{selected_date.isoformat()}",
        )

    asof_ts: pd.Timestamp | None = None
    asof_error: str | None = None
    with time_col:
        asof_input = st.text_input(
            "As-of time (HH:MM)",
            value="09:30",
            key=f"playbook_asof_time::{symbol}::{selected_date.isoformat()}",
            disabled=view_mode != "Time Travel",
            help="Trading-day intervals are 18:00-16:59 ET.",
        )
    with live_col:
        st.write("")
        st.write("")
        if st.button(
            "Return To Live",
            key=f"playbook_return_live::{symbol}::{selected_date.isoformat()}",
            use_container_width=True,
            disabled=view_mode == "Live",
        ):
            st.session_state[f"playbook_view_mode::{symbol}::{selected_date.isoformat()}"] = "Live"
            st.rerun()

    st.caption("Time entry uses 24-hour HH:MM. Valid trading-day intervals: 18:00-23:59 and 00:00-16:59 ET.")

    if view_mode == "Time Travel":
        asof_ts, asof_error = _resolve_trading_day_asof(selected_date, asof_input)
        if asof_error:
            st.warning(asof_error)
        elif asof_ts is not None:
            st.info(f"Time-travel active: showing page state as of {asof_ts.strftime('%Y-%m-%d %H:%M')} ET.")

    now_live = _now_et()
    top_picks_active = False
    top_picks_status = ""
    if view_mode == "Time Travel":
        if asof_ts is not None and asof_error is None:
            top_picks_active = True
            top_picks_status = f"Top picks active for Time Travel as-of {asof_ts.strftime('%Y-%m-%d %H:%M')} ET."
        else:
            top_picks_status = "Top picks are inactive until a valid Time Travel as-of time is set."
    else:
        if selected_date != today:
            top_picks_status = "Top picks are inactive for ended historical trading days in Live mode."
        elif td_start <= pd.Timestamp(now_live) <= td_end:
            top_picks_active = True
            top_picks_status = "Top picks active for the current live trading day."
        else:
            top_picks_status = "Top picks are inactive because the current trading-day window is not live."

    if not has_today and not has_prev:
        st.warning("No intraday data available for selected trading-day window (18:00 to 16:59 ET).")
        return

    df_trading_day_view = df_trading_day
    if view_mode == "Time Travel" and asof_ts is not None:
        df_trading_day_view = df_trading_day[df_trading_day["timestamp"] <= asof_ts].reset_index(drop=True)

    has_today_view = df_trading_day_view is not None and not df_trading_day_view.empty
    if view_mode == "Time Travel" and not has_today_view:
        st.warning("No bars are available yet by the selected time in this trading-day window.")
        return

    if has_today_view and has_prev:
        session_source = pd.concat([df_prev_trading_day, df_trading_day_view], ignore_index=True).sort_values("timestamp")
    else:
        session_source = df_trading_day_view if has_today_view else df_prev_trading_day

    sessions = compute_session_stats(session_source, selected_date) if session_source is not None else {}
    patterns = detect_patterns(sessions, df_trading_day_view if has_today_view else None, df_prev_trading_day if has_prev else None)
    zones = build_htf_zones(session_source) if session_source is not None and not session_source.empty else []

    run_now = asof_ts.to_pydatetime() if (view_mode == "Time Travel" and asof_ts is not None) else now_live

    playbook = build_strategy_playbook(
        df_today=df_trading_day_view if has_today_view else session_source,
        df_prev=df_prev_trading_day if has_prev else None,
        sessions=sessions,
        patterns=patterns,
        zones=zones,
        now_et=run_now,
        whipsaw_threshold=float(whipsaw_threshold),
        trading_day=selected_date,
    )
    app_suggested_at = pd.Timestamp(run_now).strftime("%Y-%m-%d %H:%M")
    if top_picks_active:
        playbook["top_trade_picks"] = _build_top_unfilled_trade_picks(
            playbook,
            asof_ts=asof_ts,
            suggested_at_app_time=app_suggested_at,
        )
        playbook["top_trade_picks"]["active"] = True
        playbook["top_trade_picks"]["status"] = top_picks_status
    else:
        playbook["top_trade_picks"] = {
            "rows": [],
            "detailed_rows": [],
            "best_trade_key": "",
            "active": False,
            "status": top_picks_status,
        }

    market_events = _update_market_event_state(
        symbol=symbol,
        selected_date=selected_date,
        playbook=playbook,
        now_et=run_now,
    )

    header_left, header_right = st.columns([5, 1])
    with header_left:
        st.caption("Use Market Updates button to open/retract the floating updates drawer.")
    with header_right:
        if hasattr(st, "popover"):
            with st.popover("Market Updates", use_container_width=True):
                st.markdown("**Market Updates**")
                _render_market_updates_panel(
                    market_events,
                    panel_key=f"{symbol}::{selected_date.isoformat()}",
                )
        else:
            panel_state_key = f"market_updates_open::{symbol}::{selected_date.isoformat()}"
            if panel_state_key not in st.session_state:
                st.session_state[panel_state_key] = False
            toggle_label = "Hide Updates" if bool(st.session_state[panel_state_key]) else "Show Updates"
            if st.button(toggle_label, key=f"market_updates_toggle_btn::{symbol}::{selected_date.isoformat()}", use_container_width=True):
                st.session_state[panel_state_key] = not bool(st.session_state[panel_state_key])

    if not hasattr(st, "popover") and bool(st.session_state.get(f"market_updates_open::{symbol}::{selected_date.isoformat()}", False)):
        _, updates_col = st.columns([3, 2], gap="large")
        with updates_col:
            st.markdown("**Market Updates**")
            _render_market_updates_panel(
                market_events,
                panel_key=f"{symbol}::{selected_date.isoformat()}",
            )

    decision = playbook.get("decision", {})
    expectation_summaries = playbook.get("expectation_summaries", {}) or {}

    st.markdown("### Trade Decision")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trade Today", f"{_status_color(decision.get('trade_today', 'Wait'))} {decision.get('trade_today', 'Wait')}")
    c2.metric("NY Mode", f"{decision.get('ny_mode', 'n/a')}")
    c3.metric("NY Direction", f"{decision.get('ny_direction', 'Neutral')}")
    c4.metric("Confidence", f"{100.0 * float(decision.get('confidence', 0.0)):.0f}%")

    st.info(decision.get("primary_reason", "No reason available."))
    wait_for = decision.get("wait_for_confirmations", []) or []
    if wait_for:
        st.markdown("**Wait For These Confirmations**")
        for item in wait_for:
            st.write(f"- {item}")

    reasons_col1, reasons_col2 = st.columns(2)
    with reasons_col1:
        st.markdown("**Supporting factors**")
        support = decision.get("supporting_factors", [])
        if support:
            for item in support:
                st.write(f"- {item}")
        else:
            st.write("- None")

    with reasons_col2:
        st.markdown("**Blocking factors**")
        block = decision.get("blocking_factors", [])
        if block:
            for item in block:
                st.write(f"- {item}")
        else:
            st.write("- None")

    if decision.get("whipsaw_ratio") is not None:
        st.caption(f"Whipsaw ratio (London/Asia): {float(decision['whipsaw_ratio']):.2f}")

    st.markdown("### Trigger Report")
    primary_trigger = playbook.get("primary_trigger")
    if primary_trigger:
        st.success(
            f"Primary Trigger: {primary_trigger.get('name')} | {primary_trigger.get('direction')} | "
            f"{primary_trigger.get('time')} @ {primary_trigger.get('price'):.2f} | TF {primary_trigger.get('timeframe')}"
        )
        st.caption("Monitoring remains active for OR and bias-confirmation triggers even after this trigger fired.")
        st.caption("ORB triggers are evaluated from current-day OR windows only (post 10:00 / 10:30 ET).")
        st.caption(primary_trigger.get("details", ""))
        st.markdown("**What happened**")
        st.write(f"- {primary_trigger.get('what_happened', 'n/a')}")
        st.markdown("**What must happen next for execution**")
        st.write(f"- {primary_trigger.get('must_happen', 'n/a')}")
        st.markdown("**Execution look-for**")
        st.write(f"- {primary_trigger.get('execution_look_for', 'n/a')}")
        st.markdown("**Invalidation**")
        st.write(f"- {primary_trigger.get('invalidation', 'n/a')}")
    else:
        st.warning("No qualified trigger has fired yet.")

    triggers = playbook.get("triggers", [])
    if triggers:
        st.dataframe(pd.DataFrame(triggers), use_container_width=True)

    confirmations = playbook.get("confirmation_triggers", []) or []
    if confirmations:
        st.markdown("**Bias Confirmation Triggers**")
        st.dataframe(pd.DataFrame(confirmations), use_container_width=True)

    watchlist = playbook.get("trigger_watchlist", []) or []
    if watchlist:
        st.markdown("**Trigger Watchlist (Continuous Monitoring)**")
        st.dataframe(pd.DataFrame(watchlist), use_container_width=True)

    st.markdown("### US Open Pattern Watch")
    open_watch = playbook.get("open_pattern_watch", {}) or {}
    status = str(open_watch.get("status", "Not Active"))
    confidence = float(open_watch.get("confidence", 0.0))
    st.write(f"- Status: {_status_color(status)} {status}")
    st.write(f"- Confidence: {confidence:.0f}%")
    st.write(f"- Reason: {open_watch.get('reason', 'n/a')}")

    checklist = open_watch.get("checklist", []) or []
    if checklist:
        st.markdown("**Checklist**")
        for item in checklist:
            st.write(f"- {item}")

    anticipation = open_watch.get("anticipation_factors", []) or []
    if anticipation:
        st.markdown("**Anticipation factors**")
        for item in anticipation:
            st.write(f"- {item}")

    provisional_plan = open_watch.get("provisional_plan", {}) or {}
    if provisional_plan:
        st.markdown("**Provisional plan**")
        st.write(f"- Entry rule: {provisional_plan.get('entry_rule', 'n/a')}")
        st.write(f"- Stop rule: {provisional_plan.get('stop_rule', 'n/a')}")
        st.write(f"- Target rule: {provisional_plan.get('target_rule', 'n/a')}")

    score_breakdown = open_watch.get("score_breakdown", []) or []
    if score_breakdown:
        st.markdown("**Confidence model**")
        settings = open_watch.get("model_settings", {}) or {}
        if settings:
            st.caption(
                f"Settings: strictness={settings.get('candle3_strictness', 'balanced')} | "
                f"reclaim_window={float(settings.get('reclaim_speed_window_minutes', 30.0)):.0f}m"
            )
        score_df = pd.DataFrame(score_breakdown)
        preferred = ["component", "score", "max", "note"]
        cols = [c for c in preferred if c in score_df.columns] + [c for c in score_df.columns if c not in preferred]
        st.dataframe(score_df[cols], use_container_width=True)
        penalty = float(open_watch.get("penalty_points", 0.0))
        if penalty > 0:
            st.caption(f"Penalty applied: -{penalty:.1f} points")

    if status in {"Armed", "Triggered"}:
        entry = open_watch.get("entry")
        stop = open_watch.get("stop")
        targets = open_watch.get("targets", []) or []
        if entry:
            st.write(f"- Entry: {entry.get('time', 'n/a')} @ {_fmt_price(entry.get('price'))}")
        if stop:
            st.write(f"- Stop: {_fmt_price(stop.get('price'))} ({stop.get('rule', 'n/a')})")
        if targets:
            st.markdown("**Targets**")
            for t in targets[:2]:
                st.write(
                    f"- {t.get('name', 'Target')}: {_fmt_price(t.get('price'))} "
                    f"(R:R {float(t.get('rr', 0.0)):.2f})"
                )
        st.caption(f"Invalidation: {open_watch.get('invalidation', 'n/a')}")

    st.markdown("### Entry Playbooks (If/Then)")
    blueprints = playbook.get("entry_blueprints", [])
    if blueprints:
        st.dataframe(pd.DataFrame(blueprints), use_container_width=True)
    else:
        st.write("No entry playbooks available for current state.")

    st.markdown("### Confluence Entry Style Suggestions")
    style_rows = playbook.get("confluence_entry_styles", []) or []
    if style_rows:
        style_df = pd.DataFrame(style_rows)
        if "Suggested App Time" not in style_df.columns and "Suggested Time" in style_df.columns:
            style_df["Suggested App Time"] = style_df["Suggested Time"]
        elif "Suggested App Time" in style_df.columns and "Suggested Time" in style_df.columns:
            style_df["Suggested App Time"] = style_df["Suggested App Time"].fillna(style_df["Suggested Time"])
        if "Suggested Time" in style_df.columns:
            style_df["_suggested_time_sort"] = pd.to_datetime(style_df["Suggested Time"], errors="coerce")
            style_df = style_df.sort_values("_suggested_time_sort", ascending=False, na_position="last")
            style_df = style_df.drop(columns=["_suggested_time_sort"])
        style_rows_sorted = style_df.to_dict("records")
        style_df = style_df.rename(
            columns={
                "Session Marker": "Session marker",
                "Suggested Time": "Suggested time",
                "Suggested App Time": "Suggested app time",
                "Entry Confidence": "Entry confidence",
                "Confidence Tier": "Confidence tier",
                "Confidence Reasoning": "Confidence reasoning",
                "Execution Time": "Execution time",
                "Exit Time": "Exit time",
                "Min Target": "Min Target",
                "Max Target": "Max Target",
                "Risk Pts(Ticks)": "Risk Price(Ticks)",
                "Target Pts(Ticks)": "Target Price(Ticks)",
            }
        )
        ordered = [
            "Session marker",
            "Suggested time",
            "Suggested app time",
            "Entry ETA (HH:MM)",
            "Entry Window",
            "Action",
            "Confluence",
            "Suggested Entry",
            "Entry Price",
            "Risk Price(Ticks)",
            "Target Price(Ticks)",
            "RR",
            "Entry confidence",
            "Confidence tier",
            "Confidence reasoning",
            "Min Target",
            "Max Target",
        ]
        for c in ordered:
            if c not in style_df.columns:
                style_df[c] = "n/a"
        st.dataframe(style_df[ordered], use_container_width=True)
        timing_key = f"show_confluence_timing::{symbol}::{selected_date.isoformat()}"
        if timing_key not in st.session_state:
            st.session_state[timing_key] = False
        timing_btn_label = (
            "Hide Confluence Entry Timing Summary"
            if bool(st.session_state[timing_key])
            else "Show Confluence Entry Timing Summary"
        )
        if st.button(timing_btn_label, key=f"{timing_key}::btn"):
            st.session_state[timing_key] = not bool(st.session_state[timing_key])

        if bool(st.session_state[timing_key]):
            st.markdown("**Confluence Entry Timing Summary**")
            for row in style_rows_sorted[:3]:
                st.write(
                    "- "
                    f"{row.get('Confluence', 'n/a')} | {row.get('Action', 'Wait')} | "
                    f"confidence {row.get('Entry Confidence', 'n/a')} ({row.get('Confidence Tier', 'n/a')}) | "
                    f"Suggested: {row.get('Suggested Time', 'n/a')} | "
                    f"Expect entry at: {row.get('Entry ETA (HH:MM)', 'n/a')} ({row.get('Entry Window', 'n/a')}) | "
                    f"Confluence formed: {row.get('Formed Time', 'n/a')} | "
                    f"Confluence range: {row.get('Confluence Range', 'n/a')} | "
                    f"Expected retests: {row.get('Expected Retests', 'n/a')} | "
                    f"Entry timing: {row.get('Entry Timing', 'n/a')}"
                )

        conf_reason_key = f"show_confluence_reasoning::{symbol}::{selected_date.isoformat()}"
        if conf_reason_key not in st.session_state:
            st.session_state[conf_reason_key] = False
        conf_btn_label = (
            "Hide Confluence Entry Reasoning"
            if bool(st.session_state[conf_reason_key])
            else "Show Confluence Entry Reasoning"
        )
        if st.button(conf_btn_label, key=f"{conf_reason_key}::btn"):
            st.session_state[conf_reason_key] = not bool(st.session_state[conf_reason_key])

        if bool(st.session_state[conf_reason_key]):
            st.markdown("**Entry Confidence And Reasoning**")
            for row in style_rows_sorted[:5]:
                st.write(
                    "- "
                    f"{row.get('Confluence', 'n/a')} -> "
                    f"{row.get('Entry Confidence', 'n/a')}/100 ({row.get('Confidence Tier', 'n/a')}) | "
                    f"Suggested: {row.get('Suggested Time', 'n/a')} | "
                    f"ETA: {row.get('Entry ETA (HH:MM)', 'n/a')} ({row.get('Entry Window', 'n/a')}) | "
                    f"Formed: {row.get('Formed Time', 'n/a')} | "
                    f"Range: {row.get('Confluence Range', 'n/a')}"
                )
                st.caption(str(row.get("Confidence Reasoning", "n/a")))
                st.caption(str(row.get("Entry ETA Detail", "n/a")))
    else:
        st.write("No confluence entry style suggestions available.")

    st.markdown("### Entry Execution Tracker")
    execution_rows = playbook.get("entry_execution_tracker", []) or []
    if execution_rows:
        exec_df = pd.DataFrame(execution_rows)
        if "Suggested App Time" not in exec_df.columns and "Suggested Time" in exec_df.columns:
            exec_df["Suggested App Time"] = exec_df["Suggested Time"]
        elif "Suggested App Time" in exec_df.columns and "Suggested Time" in exec_df.columns:
            exec_df["Suggested App Time"] = exec_df["Suggested App Time"].fillna(exec_df["Suggested Time"])
        if "Suggested Time" in exec_df.columns:
            exec_df["_suggested_time_sort"] = pd.to_datetime(exec_df["Suggested Time"], errors="coerce")
            exec_df = exec_df.sort_values("_suggested_time_sort", ascending=False, na_position="last")
            exec_df = exec_df.drop(columns=["_suggested_time_sort"])
        execution_rows_sorted = exec_df.to_dict("records")
        exec_df = exec_df.rename(
            columns={
                "Session Marker": "Session marker",
                "Suggested Time": "Suggested time",
                "Suggested App Time": "Suggested app time",
                "Entry Confidence": "Entry confidence",
                "Confidence Tier": "Confidence tier",
                "Confidence Reasoning": "Confidence reasoning",
                "Execution Time": "Execution time",
                "Exit Time": "Exit time",
                "Risk Pts(Ticks)": "Risk Price(Ticks)",
                "Target Pts(Ticks)": "Target Price(Ticks)",
            }
        )
        ordered = [
            "Session marker",
            "Suggested time",
            "Suggested app time",
            "Entry ETA (HH:MM)",
            "Entry Window",
            "Action",
            "Confluence",
            "Suggested Entry",
            "Entry Price",
            "Risk Price(Ticks)",
            "Target Price(Ticks)",
            "RR",
            "Entry confidence",
            "Confidence tier",
            "Confidence reasoning",
            "Executed",
            "Execution time",
            "Outcome",
            "Exit time",
            "Exit Price",
            "Why",
            "Min Target",
            "Max Target",
        ]
        for c in ordered:
            if c not in exec_df.columns:
                exec_df[c] = "n/a"
        st.dataframe(exec_df[ordered], use_container_width=True)
        successful = sum(1 for r in execution_rows if str(r.get("Outcome", "")).lower() == "successful")
        failed = sum(1 for r in execution_rows if str(r.get("Outcome", "")).lower() == "failed")
        open_count = sum(1 for r in execution_rows if str(r.get("Outcome", "")).lower() == "open")
        pending = sum(1 for r in execution_rows if str(r.get("Outcome", "")).lower() == "pending")
        unfilled = sum(1 for r in execution_rows if str(r.get("Outcome", "")).lower() == "unfilled")
        skipped = sum(1 for r in execution_rows if str(r.get("Outcome", "")).lower() == "skipped")
        st.caption(
            "Execution summary across all suggested entries for this trading day: "
            f"successful={successful}, failed={failed}, open={open_count}, pending={pending}, unfilled={unfilled}, skipped={skipped}."
        )

        tracker_reason_key = f"show_tracker_reasoning::{symbol}::{selected_date.isoformat()}"
        if tracker_reason_key not in st.session_state:
            st.session_state[tracker_reason_key] = False
        tracker_btn_label = (
            "Hide Entry Tracker Reasoning"
            if bool(st.session_state[tracker_reason_key])
            else "Show Entry Tracker Reasoning"
        )
        if st.button(tracker_btn_label, key=f"{tracker_reason_key}::btn"):
            st.session_state[tracker_reason_key] = not bool(st.session_state[tracker_reason_key])

        if bool(st.session_state[tracker_reason_key]):
            st.markdown("**Entry Tracker Confidence And Reasoning**")
            for row in execution_rows_sorted[:5]:
                st.write(
                    "- "
                    f"{row.get('Confluence', 'n/a')} | Suggested: {row.get('Suggested Time', 'n/a')} | "
                    f"ETA: {row.get('Entry ETA (HH:MM)', 'n/a')} ({row.get('Entry Window', 'n/a')}) | "
                    f"Formed: {row.get('Confluence Formed Time', 'n/a')} | "
                    f"Range: {row.get('Confluence Range', 'n/a')} | "
                    f"Confidence {row.get('Entry Confidence', 'n/a')} ({row.get('Confidence Tier', 'n/a')})"
                )
                st.caption(str(row.get("Confidence Reasoning", row.get("Why", "n/a"))))
                st.caption(str(row.get("Entry ETA Detail", "n/a")))
    else:
        st.write("No execution records available yet.")

    st.markdown("### Top 2 Unfilled Trade Picks")
    top_picks = playbook.get("top_trade_picks", {}) or {}
    top_picks_active = bool(top_picks.get("active", False))
    top_picks_status = str(top_picks.get("status", ""))
    if top_picks_status:
        st.caption(top_picks_status)
    top_rows = top_picks.get("rows", []) or []
    top_detailed = top_picks.get("detailed_rows", []) or []
    if top_picks_active and top_rows:
        st.dataframe(pd.DataFrame(top_rows), use_container_width=True)
        st.markdown("**Why These Trades Were Chosen**")
        for row in top_detailed:
            st.write(
                "- "
                f"{row.get('Confluence', 'n/a')} ({row.get('Action', 'Wait')}) was selected because it combines "
                f"fill {float(row.get('Fill confidence', 0.0)):.1f}%, success {float(row.get('Success confidence', 0.0)):.1f}%, "
                f"and low fail {float(row.get('Fail confidence', 0.0)):.1f}% with RR {row.get('RR', 'n/a')} and "
                f"entry confidence {row.get('Entry Confidence', 'n/a')}.")
            st.caption(
                f"Liquidity score proxy: {row.get('_liq_score', 'n/a')} | Ranking score: {row.get('_rank_score', 'n/a')}"
            )
    elif top_picks_active:
        st.write("No eligible unfilled directional trades are available yet.")
    else:
        st.write("Top picks are currently disabled for this view.")

    st.markdown("### Risk Engine")
    risk_engine = playbook.get("risk_engine", {}) or {}
    trade_rows = playbook.get("entry_execution_tracker", []) or []
    trade_options: list[tuple[str, dict]] = []
    for row in trade_rows:
        label = (
            f"{row.get('Suggested Time', 'n/a')} | {row.get('Action', 'Wait')} | "
            f"{row.get('Confluence', 'n/a')} | ETA {row.get('Entry ETA (HH:MM)', 'n/a')} | RR {row.get('RR', 'n/a')}"
        )
        trade_options.append((label, row))

    selected_trade = None
    if trade_options:
        best_trade_key = str(top_picks.get("best_trade_key", "")) if top_picks_active else ""
        default_index = 0
        if best_trade_key:
            for i, (_, row) in enumerate(trade_options):
                if _trade_row_key(row) == best_trade_key:
                    default_index = i
                    break

        left_col, right_col = st.columns([2.5, 2.0])
        with left_col:
            st.markdown("**Risk Engine View**")
        with right_col:
            selected_label = st.selectbox(
                "Suggested trade focus",
                options=[lbl for lbl, _ in trade_options],
                index=default_index,
                key="risk_engine_trade_focus",
            )
        selected_trade = next((row for lbl, row in trade_options if lbl == selected_label), None)

    if risk_engine:
        display_side = str(risk_engine.get("side", "Wait"))
        display_rr = risk_engine.get("rr")
        display_quality = float(risk_engine.get("quality_score", 0.0) or 0.0)
        display_status = str(risk_engine.get("status", "Caution"))
        display_entry = risk_engine.get("entry", "n/a")
        display_stop = risk_engine.get("stop", "n/a")
        display_target = risk_engine.get("target", "n/a")
        display_size = risk_engine.get("size_contracts_10k", 0)
        display_suggested_target = risk_engine.get("suggested_target", "n/a")
        display_min_target = risk_engine.get("minimum_target", "n/a")
        display_max_target = risk_engine.get("maximum_target", "n/a")
        display_notes = str(risk_engine.get("notes", "n/a"))
        alignment_factors = risk_engine.get("alignment_factors", []) or []

        if selected_trade is not None:
            display_side = str(selected_trade.get("Action", "Wait"))
            try:
                display_rr = float(selected_trade.get("RR", 0.0))
            except Exception:
                display_rr = None
            try:
                display_quality = float(selected_trade.get("Entry Confidence", 0.0) or 0.0)
            except Exception:
                display_quality = 0.0
            display_status = str(selected_trade.get("Confidence Tier", "n/a"))
            display_entry = selected_trade.get("Entry Price", "n/a")
            display_stop = "n/a"
            display_target = selected_trade.get("Min Target", selected_trade.get("Max Target", "n/a"))
            display_size = "n/a"
            display_suggested_target = selected_trade.get("Preferred Target Price", selected_trade.get("Min Target", "n/a"))
            display_min_target = selected_trade.get("Min Target", "n/a")
            display_max_target = selected_trade.get("Max Target", "n/a")
            display_notes = (
                f"Trade-focused view for {selected_trade.get('Confluence', 'selected setup')}. "
                f"Suggested at {selected_trade.get('Suggested Time', 'n/a')} with RR {selected_trade.get('RR', 'n/a')}."
            )
            reasoning = str(selected_trade.get("Confidence Reasoning", selected_trade.get("Reason", "n/a")))
            alignment_factors = [f"Confluence range: {selected_trade.get('Confluence Range', 'n/a')}"]
            alignment_factors.extend([part.strip() for part in reasoning.split(";") if part.strip()][:6])

        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Side", display_side)
        rc2.metric("R:R", f"{float(display_rr):.2f}" if display_rr is not None else "n/a")
        rc3.metric("Quality", f"{display_quality:.1f}" if display_quality else "n/a")
        rc4.metric("Status", display_status)
        st.write(
            "- "
            f"Entry {display_entry} | Stop {display_stop} | "
            f"Target {display_target} | Size/10k {display_size}"
        )
        st.write(
            "- "
            f"Suggested target {display_suggested_target} | "
            f"Minimum target {display_min_target} | "
            f"Maximum target {display_max_target}"
        )
        st.caption(display_notes)

        p1, p2, p3 = st.columns(3)
        p1.metric("Fill Confidence", f"{float(risk_engine.get('fill_confidence_pct', 0.0)):.1f}%")
        p2.metric("Stop-Hit Confidence", f"{float(risk_engine.get('stop_hit_confidence_pct', 0.0)):.1f}%")
        p3.metric("Target-Hit Confidence", f"{float(risk_engine.get('target_hit_confidence_pct', 0.0)):.1f}%")
        st.caption(str(risk_engine.get("outcome_probability_note", "")))

        if alignment_factors:
            st.markdown("**What Aligned For This Risk Read**")
            for item in alignment_factors:
                st.write(f"- {item}")

        if selected_trade is not None:
            st.caption("Showing selected-trade-driven risk view. Global risk model context is listed below.")
            st.write(
                "- "
                f"Global model snapshot: side {risk_engine.get('side', 'Wait')} | "
                f"RR {risk_engine.get('rr', 'n/a')} | quality {risk_engine.get('quality_score', 'n/a')}"
            )

        if risk_engine.get("trade_summary"):
            st.markdown("**Trade Summary**")
            st.write(f"- {risk_engine.get('trade_summary')}")
        if risk_engine.get("expected_behavior"):
            st.markdown("**What We Expect If Entered**")
            st.write(f"- {risk_engine.get('expected_behavior')}")
        if risk_engine.get("post_trade_assumption"):
            st.markdown("**What Must Stay True After Entry**")
            st.write(f"- {risk_engine.get('post_trade_assumption')}")

        strengthening = risk_engine.get("strengthening_alignments", []) or []
        if strengthening:
            st.markdown("**What Would Strengthen This Setup (And Chance It Happens)**")
            for item in strengthening:
                st.write(f"- {item}")

        if selected_trade is not None:
            st.markdown("**Selected Suggested Trade Details**")
            st.write(
                "- "
                f"Suggested: {selected_trade.get('Suggested Time', 'n/a')} | "
                f"Expected entry: {selected_trade.get('Entry ETA (HH:MM)', 'n/a')} ({selected_trade.get('Entry Window', 'n/a')}) | "
                f"Action: {selected_trade.get('Action', 'Wait')} | "
                f"Confluence: {selected_trade.get('Confluence', 'n/a')} | "
                f"Range: {selected_trade.get('Confluence Range', 'n/a')}"
            )
            st.write(
                "- "
                f"Entry: {selected_trade.get('Entry Price', 'n/a')} | "
                f"Risk: {selected_trade.get('Risk Pts(Ticks)', 'n/a')} | "
                f"Target: {selected_trade.get('Target Pts(Ticks)', 'n/a')} | "
                f"RR: {selected_trade.get('RR', 'n/a')}"
            )
            st.write(
                "- "
                f"Confidence: {selected_trade.get('Entry Confidence', 'n/a')} "
                f"({selected_trade.get('Confidence Tier', 'n/a')})"
            )
            st.caption(str(selected_trade.get("Confidence Reasoning", selected_trade.get("Reason", "n/a"))))
            st.caption(str(selected_trade.get("Entry ETA Detail", "n/a")))

        waiting_items = risk_engine.get("waiting_for_confirmations", []) or []
        if waiting_items and (risk_engine.get("stop") is None or risk_engine.get("target") is None):
            st.markdown("**Risk Engine Waiting For Confirmations**")
            for item in waiting_items:
                st.write(f"- {item}")

    st.markdown("### VWAP Reversion vs Expansion (NY)")
    vwap_probs = playbook.get("vwap_probabilities", {}) or {}
    if vwap_probs:
        vc1, vc2, vc3 = st.columns(3)
        vc1.metric("Price vs VWAP", str(vwap_probs.get("price_vs_vwap", "n/a")))
        vc2.metric("Mean Reversion %", f"{float(vwap_probs.get('mean_reversion_prob', 50.0)):.1f}%")
        vc3.metric("Expansion %", f"{float(vwap_probs.get('expansion_prob', 50.0)):.1f}%")
        st.caption(str(vwap_probs.get("note", "n/a")))

    st.markdown("### VWAP Strength After Dip")
    dip = playbook.get("vwap_strength_after_dip", {}) or {}
    if dip:
        dc1, dc2, dc3 = st.columns(3)
        dc1.metric("Score", str(dip.get("score_out_of", f"{float(dip.get('score', 0.0)):.1f}/100")))
        dc2.metric("Label", str(dip.get("label", "Not Active")))
        dc3.metric("Reclaim Time", str(dip.get("reclaim_time", "n/a")))
        st.write(f"- Entry timing: {dip.get('entry_timing', 'n/a')}")
        st.write(f"- Good vs bad: {dip.get('good_bad_guide', 'n/a')}")
        st.write(f"- What to expect: {expectation_summaries.get('vwap_dip', dip.get('rest_of_day_assumption', 'n/a'))}")
        st.caption(str(dip.get("note", "n/a")))

    st.markdown("### Volatility, Volume, Momentum")
    volm = playbook.get("volatility_metrics", {}) or {}
    volume_det = playbook.get("volume_detector", {}) or {}
    momentum = playbook.get("momentum_prediction", {}) or {}
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Vol Regime", str(volm.get("regime", "normal")))
    mc2.metric("STDV", f"{float(volm.get('std', 0.0) or 0.0):.5f}" if volm.get("std") is not None else "n/a")
    mc3.metric("RVOL", f"{float(volume_det.get('rvol', 0.0) or 0.0):.2f}" if volume_det.get("rvol") is not None else "n/a")
    mc4.metric("Momentum Bias", str(momentum.get("predicted", "Neutral")))
    st.write(
        "- "
        f"Momentum probs: up {float(momentum.get('prob_up', 50.0)):.1f}% / "
        f"down {float(momentum.get('prob_down', 50.0)):.1f}%"
    )
    outcomes = volm.get("regime_outcomes", {}) or {}
    if outcomes:
        st.write("- Vol regime meanings:")
        st.write(f"  - Compressed: {outcomes.get('compressed', 'n/a')}")
        st.write(f"  - Normal: {outcomes.get('normal', 'n/a')}")
        st.write(f"  - Expanded: {outcomes.get('expanded', 'n/a')}")
    st.write(f"- Vol good vs bad: {volm.get('good_bad_guide', 'n/a')}")
    st.write(f"- Volume good vs bad: {volume_det.get('good_bad_guide', 'n/a')}")
    st.write(f"- Momentum good vs bad: {momentum.get('good_bad_guide', 'n/a')}")
    st.write(f"- What to expect (volatility): {expectation_summaries.get('volatility', volm.get('rest_of_day_assumption', 'n/a'))}")
    st.write(f"- What to expect (volume): {expectation_summaries.get('volume', volume_det.get('rest_of_day_assumption', 'n/a'))}")
    st.write(f"- What to expect (momentum): {expectation_summaries.get('momentum', momentum.get('rest_of_day_assumption', 'n/a'))}")
    st.caption(str(volm.get("regime_note", "")))
    st.caption(str(volume_det.get("note", "")))
    st.caption(str(momentum.get("note", "")))

    st.markdown("### Drawdown, EV, Sharpe")
    perf = playbook.get("performance_metrics", {}) or {}
    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.metric("Max DD %", f"{float(perf.get('max_drawdown_pct', 0.0)):.2f}%")
    pc2.metric("Current DD %", f"{float(perf.get('current_drawdown_pct', 0.0)):.2f}%")
    pc3.metric("EV (bps)", f"{float(perf.get('expected_value_bps', 0.0)):.2f}")
    pc4.metric("Raw Sharpe", f"{float(perf.get('raw_sharpe', 0.0)):.3f}")
    st.write(
        "- "
        f"Annualized Sharpe: {float(perf.get('annualized_sharpe', 0.0)):.3f} | "
        f"DD duration bars: {int(perf.get('drawdown_duration_bars', 0))}"
    )
    guides = perf.get("good_bad_guide", {}) or {}
    if guides:
        st.write("- Good vs bad interpretation:")
        st.write(f"  - Drawdown: {guides.get('max_drawdown_pct', 'n/a')}")
        st.write(f"  - EV: {guides.get('expected_value_bps', 'n/a')}")
        st.write(f"  - Sharpe: {guides.get('raw_sharpe', 'n/a')}")
    st.write(f"- What to expect: {expectation_summaries.get('performance', perf.get('rest_of_day_assumption', 'n/a'))}")
    st.caption(str(perf.get("notes", "n/a")))

    st.markdown("### Daily Entry Capacity")
    capacity = playbook.get("daily_entry_capacity", {}) or {}
    if capacity:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Expected Min", f"{int(capacity.get('expected_min', 0))}")
        c2.metric("Expected Max", f"{int(capacity.get('expected_max', 0))}")
        c3.metric("Likely", f"{int(capacity.get('likely', 0))}")
        c4.metric("Remaining Estimate", f"{int(capacity.get('remaining_estimate', 0))}")
        st.caption(str(capacity.get("planning_note", "")))
        st.write(
            "- End-of-day execution: "
            f"{capacity.get('eod_execution_status', 'Pending')}"
        )
        st.caption(str(capacity.get("eod_execution_note", "")))
    else:
        st.write("No daily entry capacity estimate available.")

    st.markdown("### Timeframe Playbook")
    tf_rows = playbook.get("timeframe_playbook", [])
    if tf_rows:
        st.dataframe(pd.DataFrame(tf_rows), use_container_width=True)

    st.markdown("### Confluence Map (Price + Time)")
    conf_rows = playbook.get("confluences", [])
    if conf_rows:
        conf_df = pd.DataFrame(conf_rows)
        if "Top Liquidity" in conf_df.columns:
            conf_df["Top 7"] = conf_df["Top Liquidity"].map(lambda v: "⭐" if bool(v) else "")

        preferred_order = [
            "Top 7",
            "Confluence",
            "Side",
            "Price Low",
            "Price High",
            "Formed Time",
            "First Test",
            "Retest Time",
            "Retest Count",
            "Status",
            "Liquidity Level",
            "Liquidity Score",
            "Rejection Times",
        ]
        cols = [c for c in preferred_order if c in conf_df.columns] + [
            c for c in conf_df.columns if c not in preferred_order
        ]
        conf_df = conf_df[cols]

        if "Top Liquidity" in conf_df.columns:
            styled = conf_df.style.apply(
                lambda row: ["font-weight: bold" if bool(row.get("Top Liquidity", False)) else "" for _ in row],
                axis=1,
            )
            st.dataframe(styled, use_container_width=True)
        else:
            st.dataframe(conf_df, use_container_width=True)
    else:
        st.write("No confluences detected yet.")

    st.markdown("### VWAP Price Levels")
    vwap_rows = playbook.get("vwap_levels", [])
    if vwap_rows:
        st.dataframe(pd.DataFrame(vwap_rows), use_container_width=True)
    else:
        st.write("No VWAP levels available.")

    st.markdown("### STDV Price Targets")
    stdv_rows = playbook.get("stdv_levels", [])
    if stdv_rows:
        st.dataframe(pd.DataFrame(stdv_rows), use_container_width=True)
    else:
        st.write("No STDV levels available (insufficient bars).")

    st.markdown("### Target Ladder")
    targets = playbook.get("targets", [])
    if targets:
        st.dataframe(pd.DataFrame(targets), use_container_width=True)
    else:
        st.write("No directional targets available for the current state.")

    st.markdown("### Power Hour Focus")
    power_hour = playbook.get("power_hour", {})
    st.write(f"- Focus: {power_hour.get('focus', 'No')}")
    st.write(f"- Bias: {power_hour.get('bias', 'Neutral')}")
    st.write(f"- Reason: {power_hour.get('reason', 'n/a')}")
    entries = power_hour.get("entries", [])
    if entries:
        st.markdown("**Power-hour entry focus**")
        for item in entries:
            st.write(f"- {item}")

    snapshot = playbook.get("market_snapshot", {})
    if snapshot:
        st.caption(
            f"Snapshot | Date: {snapshot.get('trade_date')} | Last: {snapshot.get('last_price', 0.0):.2f} | "
            f"Day High: {snapshot.get('day_high', 0.0):.2f} | Day Low: {snapshot.get('day_low', 0.0):.2f}"
        )

    st.markdown("### Daily Summary")
    summary_paragraph, summary_bullets = _build_daily_summary(playbook)
    st.write(summary_paragraph)
    for line in summary_bullets:
        st.write(f"- {line}")

    _run_auto_refresh(
        auto_refresh=bool(auto_refresh) and view_mode == "Live",
        selected_date=selected_date,
        today=today,
        refresh_seconds=int(refresh_seconds),
        symbol=symbol,
    )
