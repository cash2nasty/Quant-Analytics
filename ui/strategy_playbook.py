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


def _status_color(value: str) -> str:
    v = (value or "").lower()
    if v in {"yes", "bullish", "fresh", "tested", "retested"}:
        return "🟢"
    if v in {"no", "bearish", "invalidated"}:
        return "🔴"
    return "🟡"


def _fmt_price(value: object) -> str:
    try:
        return f"{float(value):.2f}"
    except Exception:
        return "n/a"


def _build_daily_summary(playbook: dict) -> tuple[str, list[str]]:
    decision = playbook.get("decision", {})
    primary_trigger = playbook.get("primary_trigger") or {}
    entry_blueprints = playbook.get("entry_blueprints", []) or []
    confluences = playbook.get("confluences", []) or []
    targets = playbook.get("targets", []) or []
    vwap_levels = playbook.get("vwap_levels", []) or []
    power_hour = playbook.get("power_hour", {}) or {}
    open_watch = playbook.get("open_pattern_watch", {}) or {}

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
            f"Entry focus: {top_setup.get('Setup', 'n/a')} — {top_setup.get('IfThen', 'n/a')}"
        )
        bullets.append(
            f"Execution trigger: {top_setup.get('Trigger', 'n/a')} | TF: {top_setup.get('Execution TF', 'n/a')} | Session: {top_setup.get('Session', 'n/a')}"
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
    entry_rows = playbook.get("entry_blueprints", []) or []
    power_hour = playbook.get("power_hour", {}) or {}
    confluence_rows = playbook.get("confluences", []) or []

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
            "IfThen": str(r.get("IfThen", "n/a")),
            "Trigger": str(r.get("Trigger", "n/a")),
            "Execution TF": str(r.get("Execution TF", "n/a")),
            "Session": str(r.get("Session", "n/a")),
        }
        for r in entry_rows
    ]

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
        "entry": entry_signature,
        "power_hour": {
            "focus": str(power_hour.get("focus", "No")),
            "bias": str(power_hour.get("bias", "Neutral")),
            "reason": str(power_hour.get("reason", "n/a")),
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
    }
    confluence_detail_filters = {"FVGs", "OB's", "BB's"}

    st.markdown('<div class="market-updates-box">', unsafe_allow_html=True)

    for event in filtered:
        if single_mode and selected_one in detail_section_filters:
            st.markdown('<div class="market-update-item">', unsafe_allow_html=True)
            st.write(f"**Time:** {event.get('reported_at', 'n/a')}")
            st.write(f"**What happened:** {event.get('summary', 'n/a')}")
            st.write(f"**Before:** {event.get('before', 'n/a')}")
            st.write(f"**After:** {event.get('after', 'n/a')}")
            st.write(f"**Why:** {event.get('why', 'n/a')}")
            st.write(f"**What to expect:** {event.get('expect', 'n/a')}")
            st.markdown('</div>', unsafe_allow_html=True)
            continue

        if single_mode and selected_one in confluence_detail_filters:
            st.markdown('<div class="market-update-item">', unsafe_allow_html=True)
            st.write(f"**Time:** {event.get('reported_at', 'n/a')}")
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
            f"{event.get('reported_at', 'n/a')} | {event.get('summary', 'n/a')} | {event.get('where', 'n/a')}"
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
    res_prev = fetch_intraday_ohlcv(symbol, prev_date)
    if isinstance(res_prev, tuple):
        df_prev, _ = res_prev
    else:
        df_prev = res_prev

    df_today = _prepare_df(df_today)
    df_prev = _prepare_df(df_prev)

    has_today = df_today is not None and not df_today.empty
    has_prev = df_prev is not None and not df_prev.empty

    st.caption(f"Data source ticker: {used_ticker or symbol}")

    if not has_today and not has_prev:
        st.warning("No intraday data available for selected symbol/date.")
        return

    if has_today and has_prev:
        session_source = pd.concat([df_prev, df_today], ignore_index=True).sort_values("timestamp")
    else:
        session_source = df_today if has_today else df_prev

    sessions = compute_session_stats(session_source, selected_date) if session_source is not None else {}
    patterns = detect_patterns(sessions, df_today if has_today else None, df_prev if has_prev else None)
    zones = build_htf_zones(session_source) if session_source is not None and not session_source.empty else []

    playbook = build_strategy_playbook(
        df_today=df_today if has_today else session_source,
        df_prev=df_prev if has_prev else None,
        sessions=sessions,
        patterns=patterns,
        zones=zones,
        now_et=_now_et(),
        whipsaw_threshold=float(whipsaw_threshold),
    )

    market_events = _update_market_event_state(
        symbol=symbol,
        selected_date=selected_date,
        playbook=playbook,
        now_et=_now_et(),
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

    st.markdown("### Trade Decision")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trade Today", f"{_status_color(decision.get('trade_today', 'Wait'))} {decision.get('trade_today', 'Wait')}")
    c2.metric("NY Mode", f"{decision.get('ny_mode', 'n/a')}")
    c3.metric("NY Direction", f"{decision.get('ny_direction', 'Neutral')}")
    c4.metric("Confidence", f"{100.0 * float(decision.get('confidence', 0.0)):.0f}%")

    st.info(decision.get("primary_reason", "No reason available."))

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
        auto_refresh=bool(auto_refresh),
        selected_date=selected_date,
        today=today,
        refresh_seconds=int(refresh_seconds),
        symbol=symbol,
    )
