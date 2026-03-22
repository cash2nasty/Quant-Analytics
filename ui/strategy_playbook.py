import datetime as dt

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

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

    support = decision.get("supporting_factors", []) or []
    block = decision.get("blocking_factors", []) or []
    if support:
        bullets.append(f"Supporting context: {', '.join([str(s) for s in support[:3]])}")
    if block:
        bullets.append(f"Risk flags: {', '.join([str(b) for b in block[:3]])}")

    return paragraph, bullets


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

    should_refresh = bool(auto_refresh) and selected_date == today
    if should_refresh:
        refresh_ms = int(refresh_seconds) * 1000
        st.caption(f"Auto-refresh is on ({int(refresh_seconds)}s cadence for today).")
        components.html(
            f"""
            <script>
                setTimeout(function() {{
                    window.parent.location.reload();
                }}, {refresh_ms});
            </script>
            """,
            height=0,
            width=0,
        )
