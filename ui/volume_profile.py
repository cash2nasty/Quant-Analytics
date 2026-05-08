import datetime as dt
import hashlib
import json
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from data.data_fetcher import fetch_intraday_ohlcv
from data.session_reference import get_session_windows_for_date
from engines.volume_profile import (
    VolumeProfileResult,
    anticipate_ny_from_sessions,
    build_session_profiles,
    summarize_previous_day,
    trading_day_bounds,
)
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
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"])
    keep = [c for c in ["timestamp", "open", "high", "low", "close", "volume"] if c in out.columns]
    return out[keep].sort_values("timestamp").reset_index(drop=True)


def _slice_window(df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    mask = (out["timestamp"] >= start_ts) & (out["timestamp"] <= end_ts)
    return out.loc[mask].sort_values("timestamp").reset_index(drop=True)


def _fmt_price(v: Optional[float]) -> str:
    try:
        return f"{float(v):.2f}"
    except Exception:
        return "n/a"


def _profile_card(name: str, p: VolumeProfileResult, finished: bool, last_price: Optional[float]) -> None:
    st.markdown(f"**{name}**")
    st.write(f"POC: {_fmt_price(p.poc)}")
    st.write(f"VAH/VAL: {_fmt_price(p.vah)} / {_fmt_price(p.val)}")
    st.write(f"Close vs Value: {p.close_location}")
    st.write(f"Shape: {p.shape if finished else 'Pending (session not finished)'}")
    st.caption(f"Value width: {_fmt_price(p.value_area_width)} | Volume: {p.total_volume:,.0f}")
    if last_price is not None and p.poc is not None:
        st.caption(f"Last vs POC: {_fmt_price(last_price - p.poc)} pts")


def _node_hit_likelihood(node: Any, last_price: Optional[float], profile_name: str) -> float:
    # Heuristic probability-like score, bounded [0, 100], for practical ranking.
    if last_price is None:
        return 50.0

    width = max(float(node.high - node.low), 0.25)
    distance = abs(float(last_price) - float(node.center))
    distance_score = 1.0 / (1.0 + (distance / width))

    prom = max(float(node.prominence), 0.0)
    prom_score = prom / (prom + 1.0)

    profile_weight = {
        "NY": 1.00,
        "Full Day": 0.95,
        "London": 0.90,
        "Asia": 0.85,
    }.get(profile_name, 0.88)

    inside_bonus = 0.10 if (float(node.low) <= float(last_price) <= float(node.high)) else 0.0

    score = 100.0 * (
        0.52 * distance_score
        + 0.33 * prom_score
        + 0.15 * profile_weight
        + inside_bonus
    )
    return float(max(1.0, min(99.0, score)))


def _node_interpretation(profile_name: str, node_type: str, node: Any, last_price: Optional[float]) -> Dict[str, str]:
    zone = f"{_fmt_price(node.low)} - {_fmt_price(node.high)}"
    relation = "n/a"
    if last_price is not None:
        if last_price > node.high:
            relation = "price is above this zone"
        elif last_price < node.low:
            relation = "price is below this zone"
        else:
            relation = "price is currently inside this zone"

    if node_type == "HVN":
        meaning = f"{profile_name} HVN zone {zone} is an acceptance area where auction tends to slow and rotate ({relation})."
        expect = "Expect pause/rotation around this zone; sustained acceptance can anchor directional continuation from this area."
        look_for = "Look for repeated closes around the zone center and failed breaks at zone edges before directional move resumes."
    else:
        meaning = f"{profile_name} LVN zone {zone} is a low-acceptance area where price often moves quickly ({relation})."
        expect = "Expect fast travel through this zone or sharp rejection at its boundary depending on acceptance quality."
        look_for = "Look for impulse through the zone with follow-through or immediate rejection back out of the zone boundary."

    return {
        "Node": f"{profile_name} {node_type}",
        "Zone": zone,
        "Meaning": meaning,
        "Expect": expect,
        "Look For": look_for,
        "Prominence": f"{float(node.prominence):.3f}",
    }


def _build_bias_blocks(
    selected_date: dt.date,
    now_ref: dt.datetime,
    day_df: pd.DataFrame,
    prev_full: VolumeProfileResult,
    prev_ny: VolumeProfileResult,
    curr_full: VolumeProfileResult,
) -> Dict[str, Dict[str, str]]:
    open_finalize = dt.datetime.combine(selected_date, dt.time(9, 10))
    or30_finalize = dt.datetime.combine(selected_date, dt.time(10, 0))
    or60_finalize = dt.datetime.combine(selected_date, dt.time(10, 30))
    eod_finalize = dt.datetime.combine(selected_date, dt.time(17, 0))

    out: Dict[str, Dict[str, str]] = {}

    # Market open bias (finalized 09:10)
    if now_ref >= open_finalize or selected_date < _now_et().date():
        cut = day_df[day_df["timestamp"] <= pd.Timestamp(open_finalize)]
        if cut.empty:
            open_bias = "Neutral"
            open_reason = "Insufficient bars into 09:10 ET."
        else:
            px = float(cut["close"].iloc[-1])
            threshold = prev_full.poc if prev_full.poc is not None else px
            if px > threshold and prev_full.close_location != "Below VAL":
                open_bias = "Bullish"
            elif px < threshold and prev_full.close_location != "Above VAH":
                open_bias = "Bearish"
            else:
                open_bias = "Neutral"
            open_reason = (
                f"09:10 close {_fmt_price(px)} vs prior full-day POC {_fmt_price(prev_full.poc)} "
                f"and prior NY close location {prev_ny.close_location}."
            )
        out["open"] = {"status": "Final", "bias": open_bias, "reason": open_reason}
    else:
        out["open"] = {
            "status": "Pending",
            "bias": "Pending",
            "reason": "Finalizes at 09:10 ET once opening flow is established.",
        }

    # OR bias 30m/60m
    us_start = dt.datetime.combine(selected_date, dt.time(9, 30))
    us = day_df[day_df["timestamp"] >= pd.Timestamp(us_start)]
    if (now_ref >= or30_finalize or selected_date < _now_et().date()) and not us.empty:
        first30 = us[us["timestamp"] <= pd.Timestamp(or30_finalize)]
        if not first30.empty:
            open_px = float(first30["open"].iloc[0])
            close30 = float(first30["close"].iloc[-1])
            b30 = "Bullish" if close30 > open_px else "Bearish" if close30 < open_px else "Neutral"
            r30 = f"30m OR close {_fmt_price(close30)} vs OR open {_fmt_price(open_px)}."
        else:
            b30, r30 = "Neutral", "Insufficient 30m OR bars."
    else:
        b30, r30 = "Pending", "30m OR bias finalizes at 10:00 ET."

    if (now_ref >= or60_finalize or selected_date < _now_et().date()) and not us.empty:
        first60 = us[us["timestamp"] <= pd.Timestamp(or60_finalize)]
        if not first60.empty:
            open_px = float(first60["open"].iloc[0])
            close60 = float(first60["close"].iloc[-1])
            b60 = "Bullish" if close60 > open_px else "Bearish" if close60 < open_px else "Neutral"
            r60 = f"60m OR close {_fmt_price(close60)} vs OR open {_fmt_price(open_px)}."
        else:
            b60, r60 = "Neutral", "Insufficient 60m OR bars."
    else:
        b60, r60 = "Pending", "60m OR bias finalizes at 10:30 ET."

    out["or"] = {
        "status": "Final" if (b30 != "Pending" and b60 != "Pending") else "Pending",
        "bias": f"30m: {b30} | 60m: {b60}",
        "reason": f"{r30} {r60}",
    }

    # EOD summary bias (finalized 17:00)
    if now_ref >= eod_finalize or selected_date < _now_et().date():
        if curr_full.close_location == "Above VAH":
            eod_bias = "Bullish acceptance"
        elif curr_full.close_location == "Below VAL":
            eod_bias = "Bearish acceptance"
        else:
            eod_bias = "Balanced"
        out["eod"] = {
            "status": "Final",
            "bias": eod_bias,
            "reason": f"EOD close location {curr_full.close_location} with {curr_full.shape} shape.",
        }
    else:
        out["eod"] = {
            "status": "Pending",
            "bias": "Pending",
            "reason": "Finalizes at 17:00 ET.",
        }

    return out


def _profile_snapshot(profiles: Dict[str, VolumeProfileResult], last_price: Optional[float]) -> Dict[str, Any]:
    full = profiles.get("Full Day")
    if full is None:
        return {}

    def _top_node(nodes: List[Any]) -> Optional[Dict[str, float]]:
        if not nodes:
            return None
        n = nodes[0]
        return {
            "low": float(n.low),
            "high": float(n.high),
            "center": float(n.center),
            "prom": float(n.prominence),
            "volume": float(n.volume),
        }

    close_state = "inside"
    if last_price is not None and full.vah is not None and full.val is not None:
        if last_price > full.vah:
            close_state = "above"
        elif last_price < full.val:
            close_state = "below"

    return {
        "poc": full.poc,
        "close_state": close_state,
        "hvn": _top_node(full.hvns),
        "lvn": _top_node(full.lvns),
        "vah": full.vah,
        "val": full.val,
    }


def _event_signature(event: Dict[str, Any]) -> str:
    s = json.dumps(event, sort_keys=True, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _build_profile_events(prev: Dict[str, Any], curr: Dict[str, Any], now_ref: dt.datetime) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    ts = now_ref.strftime("%Y-%m-%d %H:%M")

    prev_poc = prev.get("poc")
    curr_poc = curr.get("poc")
    if prev_poc is not None and curr_poc is not None and abs(float(curr_poc) - float(prev_poc)) >= 0.25:
        direction = "up" if curr_poc > prev_poc else "down"
        events.append(
            {
                "time": ts,
                "event": f"POC shifted {direction}",
                "reason": f"Full-day POC moved from {_fmt_price(prev_poc)} to {_fmt_price(curr_poc)}.",
                "horizon": "30-90m",
                "look_for": "Acceptance around new POC and value migration follow-through.",
                "invalidation": "Immediate rejection and fast return to prior POC region.",
            }
        )

    prev_hvn = prev.get("hvn") or {}
    curr_hvn = curr.get("hvn") or {}
    if prev_hvn and curr_hvn:
        if curr_hvn.get("prom", 0.0) > prev_hvn.get("prom", 0.0) * 1.15:
            events.append(
                {
                    "time": ts,
                    "event": "Stronger HVN formed",
                    "reason": f"HVN prominence improved from {prev_hvn.get('prom', 0.0):.2f} to {curr_hvn.get('prom', 0.0):.2f}.",
                    "horizon": "30-90m",
                    "look_for": "Price acceptance/rotation around the new high-volume node.",
                    "invalidation": "Price fails to hold around HVN and quickly leaves value.",
                }
            )

    prev_lvn = prev.get("lvn") or {}
    curr_lvn = curr.get("lvn") or {}
    if prev_lvn and curr_lvn:
        if curr_lvn.get("prom", 0.0) > prev_lvn.get("prom", 0.0) * 1.15:
            events.append(
                {
                    "time": ts,
                    "event": "Stronger LVN formed",
                    "reason": f"LVN prominence improved from {prev_lvn.get('prom', 0.0):.2f} to {curr_lvn.get('prom', 0.0):.2f}.",
                    "horizon": "15-45m",
                    "look_for": "Fast travel through LVN on rejection or acceptance tests at LVN edge.",
                    "invalidation": "Price lingers and builds volume at LVN, reducing imbalance.",
                }
            )

    prev_state = prev.get("close_state")
    curr_state = curr.get("close_state")
    if prev_state in {"above", "below"} and curr_state == "inside":
        side = "VAH" if prev_state == "above" else "VAL"
        events.append(
            {
                "time": ts,
                "event": "Returned inside value",
                "reason": f"Price moved back inside value after trading beyond {side}.",
                "horizon": "15-60m",
                "look_for": "Rotation toward POC if acceptance persists inside value.",
                "invalidation": "Re-break and acceptance back outside value.",
            }
        )
    elif prev_state == "inside" and curr_state in {"above", "below"}:
        side = "VAH" if curr_state == "above" else "VAL"
        events.append(
            {
                "time": ts,
                "event": f"Closed beyond {side}",
                "reason": f"Price transitioned from inside value to outside at {side}.",
                "horizon": "15-45m",
                "look_for": "Continuation if subsequent closes maintain acceptance outside value.",
                "invalidation": "Fast failure back inside value.",
            }
        )

    return events


def _update_profile_market_updates(
    key: str,
    profiles: Dict[str, VolumeProfileResult],
    last_price: Optional[float],
    now_ref: dt.datetime,
) -> List[Dict[str, Any]]:
    if key not in st.session_state:
        st.session_state[key] = {"snapshot": None, "events": [], "seen": set()}

    state = st.session_state[key]
    curr = _profile_snapshot(profiles, last_price)
    prev = state.get("snapshot")

    if prev is not None and curr:
        for ev in _build_profile_events(prev, curr, now_ref):
            sig = _event_signature(ev)
            if sig in state["seen"]:
                continue
            state["seen"].add(sig)
            state["events"].append(ev)

    state["snapshot"] = curr
    state["events"] = state["events"][-60:]
    st.session_state[key] = state
    return state["events"]


def _run_auto_refresh(auto_refresh: bool, refresh_seconds: int, selected_date: dt.date, today: dt.date, symbol: str) -> None:
    if not (auto_refresh and selected_date == today):
        return
    st.caption(f"Auto-refresh is on ({int(refresh_seconds)}s cadence for today).")

    if hasattr(st, "fragment"):
        interval = f"{int(refresh_seconds)}s"

        @st.fragment(run_every=interval)
        def _refresh_tick() -> None:
            marker = f"vp_last_auto::{symbol}::{selected_date.isoformat()}"
            now_ts = time.time()
            last_ts = float(st.session_state.get(marker, 0.0))
            if now_ts - last_ts >= max(int(refresh_seconds) - 1, 1):
                st.session_state[marker] = now_ts
                st.rerun()

        _refresh_tick()
    else:
        if st.button("Refresh now", key=f"vp_refresh::{symbol}::{selected_date.isoformat()}"):
            st.rerun()


def _candidate_trade(
    selected_date: dt.date,
    now_ref: dt.datetime,
    day_df: pd.DataFrame,
    profiles: Dict[str, VolumeProfileResult],
) -> Optional[Dict[str, Any]]:
    if selected_date != _now_et().date():
        return None
    if not (dt.time(8, 0) <= now_ref.time() <= dt.time(16, 0)):
        return None
    if day_df is None or len(day_df) < 2:
        return None

    full = profiles.get("Full Day")
    if full is None or full.val is None or full.vah is None or full.poc is None:
        return None

    last = day_df.iloc[-1]
    prev = day_df.iloc[-2]
    last_close = float(last["close"])
    prev_close = float(prev["close"])
    width = max(float(full.value_area_width), 1.0)

    # Reclaim from below -> long
    if prev_close < full.val <= last_close:
        entry = last_close
        stop = min(float(last["low"]), float(full.val - 0.25 * width))
        target = max(float(full.poc), float(full.vah))
        risk = max(entry - stop, 0.25)
        rr = max((target - entry) / risk, 0.1)
        return {
            "id": f"long-reclaim-{now_ref.strftime('%Y%m%d%H%M%S')}",
            "time": now_ref.strftime("%Y-%m-%d %H:%M"),
            "action": "Long",
            "entry": round(entry, 2),
            "stop": round(stop, 2),
            "target": round(target, 2),
            "rr": round(rr, 2),
            "status": "Active",
            "reason": "Price reclaimed value from below (VAL reclaim) with improving acceptance.",
            "invalidated_by": "Close back below VAL with follow-through.",
            "score": 65 + min(rr * 10.0, 20.0),
        }

    # Reclaim from above -> short
    if prev_close > full.vah >= last_close:
        entry = last_close
        stop = max(float(last["high"]), float(full.vah + 0.25 * width))
        target = min(float(full.poc), float(full.val))
        risk = max(stop - entry, 0.25)
        rr = max((entry - target) / risk, 0.1)
        return {
            "id": f"short-reclaim-{now_ref.strftime('%Y%m%d%H%M%S')}",
            "time": now_ref.strftime("%Y-%m-%d %H:%M"),
            "action": "Short",
            "entry": round(entry, 2),
            "stop": round(stop, 2),
            "target": round(target, 2),
            "rr": round(rr, 2),
            "status": "Active",
            "reason": "Price failed above value and re-entered from above (VAH reject).",
            "invalidated_by": "Close back above VAH with acceptance.",
            "score": 65 + min(rr * 10.0, 20.0),
        }

    # Continuation outside value
    if prev_close > full.vah and last_close > full.vah:
        entry = last_close
        stop = float(full.vah - 0.20 * width)
        target = float(entry + (entry - stop) * 1.7)
        rr = max((target - entry) / max(entry - stop, 0.25), 0.1)
        return {
            "id": f"long-cont-{now_ref.strftime('%Y%m%d%H%M%S')}",
            "time": now_ref.strftime("%Y-%m-%d %H:%M"),
            "action": "Long",
            "entry": round(entry, 2),
            "stop": round(stop, 2),
            "target": round(target, 2),
            "rr": round(rr, 2),
            "status": "Active",
            "reason": "Price accepted above VAH; continuation setup through low-volume area.",
            "invalidated_by": "Return and acceptance back inside value.",
            "score": 58 + min(rr * 10.0, 20.0),
        }

    if prev_close < full.val and last_close < full.val:
        entry = last_close
        stop = float(full.val + 0.20 * width)
        target = float(entry - (stop - entry) * 1.7)
        rr = max((entry - target) / max(stop - entry, 0.25), 0.1)
        return {
            "id": f"short-cont-{now_ref.strftime('%Y%m%d%H%M%S')}",
            "time": now_ref.strftime("%Y-%m-%d %H:%M"),
            "action": "Short",
            "entry": round(entry, 2),
            "stop": round(stop, 2),
            "target": round(target, 2),
            "rr": round(rr, 2),
            "status": "Active",
            "reason": "Price accepted below VAL; continuation setup through low-volume area.",
            "invalidated_by": "Return and acceptance back inside value.",
            "score": 58 + min(rr * 10.0, 20.0),
        }

    return None


def _update_trade_state(
    key: str,
    candidate: Optional[Dict[str, Any]],
    day_df: pd.DataFrame,
    now_ref: dt.datetime,
) -> Dict[str, Any]:
    if key not in st.session_state:
        st.session_state[key] = {"active": None, "history": []}
    state = st.session_state[key]
    active = state.get("active")

    # Update existing active status from observed bars.
    if active is not None and day_df is not None and not day_df.empty:
        sug_ts = pd.to_datetime(active.get("time"), errors="coerce")
        if pd.notna(sug_ts):
            after = day_df[day_df["timestamp"] >= sug_ts]
        else:
            after = day_df

        side = str(active.get("action", ""))
        entry = float(active.get("entry", 0.0))
        stop = float(active.get("stop", 0.0))

        status = str(active.get("status", "Active"))
        if status in {"Active", "Filled"} and not after.empty:
            filled = status == "Filled"
            for _, r in after.iterrows():
                hi = float(r.get("high", 0.0))
                lo = float(r.get("low", 0.0))
                if not filled and lo <= entry <= hi:
                    filled = True
                    status = "Filled"
                if side == "Long" and lo <= stop:
                    status = "Invalidated"
                    break
                if side == "Short" and hi >= stop:
                    status = "Invalidated"
                    break
            active["status"] = status

        if now_ref.time() > dt.time(16, 0) and active.get("status") == "Active":
            active["status"] = "Expired"

        if active.get("status") in {"Invalidated", "Expired"}:
            state["history"].insert(0, active)
            state["active"] = None
            active = None

    # Suggest/replace by higher-score setup
    if candidate is not None:
        if active is None:
            state["active"] = candidate
        else:
            if float(candidate.get("score", 0.0)) > float(active.get("score", 0.0)) + 6.0:
                state["history"].insert(0, active)
                state["active"] = candidate

    state["history"] = state["history"][:25]
    st.session_state[key] = state
    return state


def render_volume_profile_tab() -> None:
    st.header("Volume Profile")
    st.caption(
        "Session and full-day volume profiles with HVN/LVN structure, shape labeling, profile-driven biases, market updates, and one-trade-at-a-time execution ideas."
    )

    today = _now_et().date()
    with st.sidebar:
        st.subheader("Volume Profile Inputs")
        symbol = st.text_input("Symbol", value="NQH26", key="vp_symbol")
        selected_date = st.date_input("Analysis date", value=today, key="vp_date")
        tick_size = st.selectbox("Price bin (tick size)", options=[0.25, 0.5, 1.0], index=0)
        value_area = st.slider("Value area %", min_value=0.60, max_value=0.90, value=0.70, step=0.01)
        auto_refresh = st.checkbox("Auto-refresh", value=True, key="vp_auto_refresh")
        refresh_seconds = st.selectbox("Refresh every (sec)", options=[30, 60], index=1, disabled=not auto_refresh)

    prev_date = get_prev_trading_day(selected_date)
    fetch_start = get_prev_trading_day(prev_date) - dt.timedelta(days=1)
    fetch_end = selected_date

    raw_res = fetch_intraday_ohlcv(symbol, (fetch_start, fetch_end))
    raw_df = raw_res[0] if isinstance(raw_res, tuple) else raw_res
    used_ticker = raw_res[1] if isinstance(raw_res, tuple) else ""
    all_df = _prepare_df(raw_df)

    if all_df.empty:
        st.warning("No intraday data available for selected window.")
        return

    td_start, td_end = trading_day_bounds(selected_date)
    prev_start, prev_end = trading_day_bounds(prev_date)
    day_df = _slice_window(all_df, td_start, td_end)
    prev_day_df = _slice_window(all_df, prev_start, prev_end)

    if day_df.empty:
        st.warning("No bars available for the selected trading day window.")
        return

    curr_profiles = build_session_profiles(day_df, selected_date, tick_size=float(tick_size), value_area_pct=float(value_area))
    prev_profiles = build_session_profiles(prev_day_df, prev_date, tick_size=float(tick_size), value_area_pct=float(value_area))

    now_ref = _now_et() if selected_date == today else dt.datetime.combine(selected_date, dt.time(17, 1))
    last_price = float(day_df["close"].iloc[-1]) if not day_df.empty else None

    st.caption(f"Data source ticker: {used_ticker or symbol}")
    st.caption(f"Trading day window: {td_start.strftime('%Y-%m-%d %H:%M')} to {td_end.strftime('%Y-%m-%d %H:%M')} ET")

    _run_auto_refresh(auto_refresh=bool(auto_refresh), refresh_seconds=int(refresh_seconds), selected_date=selected_date, today=today, symbol=symbol)

    update_key = f"vp_updates::{symbol}::{selected_date.isoformat()}"
    updates = _update_profile_market_updates(update_key, curr_profiles, last_price, now_ref)

    h_left, h_right = st.columns([5, 1])
    with h_left:
        st.caption("Use Market Updates to view profile-state changes.")
    with h_right:
        if hasattr(st, "popover"):
            with st.popover("Market Updates", use_container_width=True):
                st.markdown("**Volume Profile Updates**")
                if updates:
                    for ev in reversed(updates[-25:]):
                        st.markdown(f"- **{ev.get('time', 'n/a')}** {ev.get('event', 'Update')}")
                        st.caption(f"Why: {ev.get('reason', '')}")
                        st.caption(f"Horizon: {ev.get('horizon', '')}")
                        st.caption(f"Look for: {ev.get('look_for', '')}")
                        st.caption(f"Invalidation: {ev.get('invalidation', '')}")
                else:
                    st.caption("No profile updates yet.")

    st.markdown("### Session Volume Profiles")
    windows = get_session_windows_for_date(selected_date)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _profile_card("Asia", curr_profiles["Asia"], now_ref >= windows["Asia"]["end"], last_price)
    with c2:
        _profile_card("London", curr_profiles["London"], now_ref >= windows["London"]["end"], last_price)
    with c3:
        _profile_card("NY", curr_profiles["NY"], now_ref >= windows["US"]["end"], last_price)
    with c4:
        _profile_card("Full Day", curr_profiles["Full Day"], now_ref >= dt.datetime.combine(selected_date, dt.time(17, 0)), last_price)

    st.markdown("### Previous Day Profile Summary")
    prev_full = prev_profiles["Full Day"]
    prev_ny = prev_profiles["NY"]
    prev_summary, prev_expect = summarize_previous_day(prev_full, prev_ny)
    st.write(
        f"Previous day full profile close location: {prev_full.close_location} | Shape: {prev_full.shape}"
    )
    st.write(
        f"Previous day NY profile close location: {prev_ny.close_location} | Shape: {prev_ny.shape}"
    )
    st.info(prev_summary)
    st.caption(f"Expectation for next open/day: {prev_expect}")

    st.markdown("### NY Session Anticipation")
    st.write(anticipate_ny_from_sessions(curr_profiles["Asia"], curr_profiles["London"]))

    st.markdown("### Profile-Driven Biases")
    biases = _build_bias_blocks(
        selected_date=selected_date,
        now_ref=now_ref,
        day_df=day_df,
        prev_full=prev_full,
        prev_ny=prev_ny,
        curr_full=curr_profiles["Full Day"],
    )
    b1, b2, b3 = st.columns(3)
    with b1:
        st.markdown("**Market Open Bias (09:10)**")
        st.write(f"Status: {biases['open']['status']}")
        st.write(f"Bias: {biases['open']['bias']}")
        st.caption(biases["open"]["reason"])
    with b2:
        st.markdown("**OR Bias (30m/60m)**")
        st.write(f"Status: {biases['or']['status']}")
        st.write(f"Bias: {biases['or']['bias']}")
        st.caption(biases["or"]["reason"])
    with b3:
        st.markdown("**End Of Day Summary (17:00)**")
        st.write(f"Status: {biases['eod']['status']}")
        st.write(f"Bias: {biases['eod']['bias']}")
        st.caption(biases["eod"]["reason"])

    st.markdown("### HVN / LVN Highlights")
    hvn_rows = []
    lvn_rows = []
    for name, p in curr_profiles.items():
        for n in p.hvns[:3]:
            hvn_rows.append(
                {
                    "Profile": name,
                    "Box Low": round(n.low, 2),
                    "Box High": round(n.high, 2),
                    "Center": round(n.center, 2),
                    "Volume": round(n.volume, 1),
                    "Prominence": round(n.prominence, 3),
                    "Method": getattr(n, "method", "Primary"),
                    "Confidence": getattr(n, "confidence", "High"),
                    "Likelihood Hit %": round(_node_hit_likelihood(n, last_price, name), 1),
                }
            )
        for n in p.lvns[:3]:
            lvn_rows.append(
                {
                    "Profile": name,
                    "Box Low": round(n.low, 2),
                    "Box High": round(n.high, 2),
                    "Center": round(n.center, 2),
                    "Volume": round(n.volume, 1),
                    "Prominence": round(n.prominence, 3),
                    "Method": getattr(n, "method", "Primary"),
                    "Confidence": getattr(n, "confidence", "High"),
                    "Likelihood Hit %": round(_node_hit_likelihood(n, last_price, name), 1),
                }
            )
    hcol, lcol = st.columns(2)
    with hcol:
        st.markdown("**Top HVNs**")
        st.dataframe(pd.DataFrame(hvn_rows), use_container_width=True)
    with lcol:
        st.markdown("**Top LVNs**")
        st.dataframe(pd.DataFrame(lvn_rows), use_container_width=True)

    st.markdown("### Major HVN/LVN Interpretation")
    interp_rows: List[Dict[str, str]] = []
    for profile_name, p in curr_profiles.items():
        if p.hvns:
            interp_rows.append(_node_interpretation(profile_name, "HVN", p.hvns[0], last_price))
        if p.lvns:
            interp_rows.append(_node_interpretation(profile_name, "LVN", p.lvns[0], last_price))
    if interp_rows:
        st.dataframe(pd.DataFrame(interp_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No major HVN/LVN zones available yet for interpretation.")

    st.markdown("### Trade Suggestion")
    candidate = _candidate_trade(selected_date, now_ref, day_df, curr_profiles)
    trade_state_key = f"vp_trades::{symbol}::{selected_date.isoformat()}"
    trade_state = _update_trade_state(trade_state_key, candidate, day_df, now_ref)
    active = trade_state.get("active")

    if active is None:
        st.info("No trades available at the moment.")
    else:
        st.success(
            f"{active.get('action', 'Trade')} now | Entry {_fmt_price(active.get('entry'))} | "
            f"Stop {_fmt_price(active.get('stop'))} | Target {_fmt_price(active.get('target'))} | RR {active.get('rr', 'n/a')}"
        )
        st.caption(f"Status: {active.get('status', 'Active')}")
        st.caption(f"Why: {active.get('reason', 'n/a')}")
        st.caption(f"Invalidation: {active.get('invalidated_by', 'n/a')}")

    bottom_left, bottom_right = st.columns([3, 2])
    with bottom_right:
        with st.expander("Previous Trades", expanded=False):
            history = trade_state.get("history", [])
            if not history:
                st.caption("No previous trades yet.")
            else:
                rows = []
                for t in history:
                    rows.append(
                        {
                            "Time": t.get("time", "n/a"),
                            "Action": t.get("action", "n/a"),
                            "Entry": t.get("entry", "n/a"),
                            "Stop": t.get("stop", "n/a"),
                            "Target": t.get("target", "n/a"),
                            "RR": t.get("rr", "n/a"),
                            "Status": t.get("status", "n/a"),
                        }
                    )
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
