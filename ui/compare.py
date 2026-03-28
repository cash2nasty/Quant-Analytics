import datetime as dt
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from data.data_fetcher import fetch_intraday_ohlcv
from engines.patterns import detect_patterns
from engines.sessions import compute_session_stats
from engines.strategy_playbook import build_strategy_playbook
from engines.zones import build_htf_zones
from storage.history_manager import load_all_summaries
from ui.live_analysis import get_prev_trading_day


NQ_TICK_SIZE = 0.25
NQ_TICK_VALUE_1_CONTRACT = 5.0
NQ_TICK_VALUE_2_CONTRACTS = 10.0


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    out = df.copy()
    if "timestamp" not in out.columns:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    keep = [c for c in ["timestamp", "open", "high", "low", "close", "volume"] if c in out.columns]
    return out[keep].sort_values("timestamp").reset_index(drop=True)


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _entry_price_from_style(row: Dict[str, Any]) -> Optional[float]:
    tap = _safe_float(row.get("First Tap Price"))
    mid = _safe_float(row.get("Midline Price"))
    style = str(row.get("Suggested Entry", "First Tap"))

    if style == "Midline":
        return mid if mid is not None else tap
    if style.startswith("Split"):
        if tap is not None and mid is not None:
            return (tap + mid) / 2.0
        return tap if tap is not None else mid
    return tap if tap is not None else mid


def _target_pnl(entry: Optional[float], target: Optional[float]) -> Dict[str, Optional[float]]:
    if entry is None or target is None:
        return {
            "ticks": None,
            "usd_1": None,
            "usd_2": None,
        }
    ticks = abs(target - entry) / NQ_TICK_SIZE
    return {
        "ticks": round(ticks, 1),
        "usd_1": round(ticks * NQ_TICK_VALUE_1_CONTRACT, 2),
        "usd_2": round(ticks * NQ_TICK_VALUE_2_CONTRACTS, 2),
    }


def _compute_day_opportunity(symbol: str, date_str: str) -> Dict[str, Any]:
    try:
        trade_date = dt.date.fromisoformat(str(date_str))
    except Exception:
        return {"error": "Invalid date format in saved summary."}

    today_raw, _ = fetch_intraday_ohlcv(symbol, trade_date)
    prev_raw, _ = fetch_intraday_ohlcv(symbol, get_prev_trading_day(trade_date))
    df_today = _prepare_df(today_raw)
    df_prev = _prepare_df(prev_raw)

    if df_today.empty:
        return {"error": "No intraday data found for selected day."}

    session_source = df_today
    if not df_prev.empty:
        session_source = pd.concat([df_prev, df_today], ignore_index=True).sort_values("timestamp").reset_index(drop=True)

    sessions = compute_session_stats(session_source, trade_date)
    patterns = detect_patterns(sessions, df_today, df_prev if not df_prev.empty else None)
    zones = build_htf_zones(session_source)

    playbook = build_strategy_playbook(
        df_today=df_today,
        df_prev=df_prev if not df_prev.empty else None,
        sessions=sessions,
        patterns=patterns,
        zones=zones,
        now_et=dt.datetime.combine(trade_date, dt.time(17, 30)),
        whipsaw_threshold=3.0,
    )

    styles = playbook.get("confluence_entry_styles", []) or []
    if not styles:
        return {"error": "No confluence entry styles available for this day."}

    top = styles[0]
    entry_price = _entry_price_from_style(top)
    preferred_target = _safe_float(top.get("Preferred Target Price"))
    min_target = _safe_float(top.get("Minimum Target Price"))
    max_target = _safe_float(top.get("Maximum Target Price"))

    return {
        "symbol": symbol,
        "date": str(trade_date),
        "confluence": top.get("Confluence", "n/a"),
        "formed_time": top.get("Formed Time", "n/a"),
        "suggested_entry": top.get("Suggested Entry", "n/a"),
        "entry_price": entry_price,
        "preferred_target": preferred_target,
        "min_target": min_target,
        "max_target": max_target,
        "preferred": _target_pnl(entry_price, preferred_target),
        "minimum": _target_pnl(entry_price, min_target),
        "maximum": _target_pnl(entry_price, max_target),
    }


def _fmt_money(value: Optional[float]) -> str:
    return "n/a" if value is None else f"${value:,.2f}"


def _fmt_ticks(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def _render_opportunity_block(title: str, opp: Dict[str, Any]) -> None:
    st.subheader(title)
    if opp.get("error"):
        st.warning(str(opp["error"]))
        return

    st.write(f"Confluence: {opp.get('confluence', 'n/a')}")
    st.write(f"Formed: {opp.get('formed_time', 'n/a')}")
    st.write(f"Suggested Entry: {opp.get('suggested_entry', 'n/a')}")
    st.write(f"Entry Price: {opp.get('entry_price', 'n/a')}")

    rows = []
    for label, key in [("Preferred", "preferred"), ("Minimum", "minimum"), ("Maximum", "maximum")]:
        item = opp.get(key, {}) or {}
        rows.append(
            {
                "Target": label,
                "Ticks": _fmt_ticks(item.get("ticks")),
                "$ (1 Contract)": _fmt_money(item.get("usd_1")),
                "$ (2 Contracts)": _fmt_money(item.get("usd_2")),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def render_compare():
    st.header("Compare Days")

    summaries = load_all_summaries()
    has_saved = len(summaries) >= 2
    source_mode = st.radio(
        "Comparison source",
        ["Saved history days", "Any dates (recompute)"],
        horizontal=True,
    )

    left_symbol = "NQH26"
    right_symbol = "NQH26"
    left_date = None
    right_date = None

    if source_mode == "Saved history days":
        if not has_saved:
            st.info("Need at least two saved days for this mode. Switch to 'Any dates (recompute)'.")
            return

        labels = [f"{s.date} | {s.symbol}" for s in summaries]
        col1, col2 = st.columns(2)
        with col1:
            left_label = st.selectbox("Left Day", labels, index=0)
        with col2:
            right_label = st.selectbox("Right Day", labels, index=1)

        if left_label == right_label:
            st.warning("Select two different days.")
            return

        left = summaries[labels.index(left_label)]
        right = summaries[labels.index(right_label)]
        left_symbol, right_symbol = left.symbol, right.symbol
        left_date, right_date = left.date, right.date

        c1, c2 = st.columns(2)

        with c1:
            st.subheader(f"{left.date} ({left.symbol})")
            st.markdown("**Bias**")
            st.write(f"Daily: {left.bias.daily_bias} ({left.bias.daily_confidence:.0%})")
            us30_left = getattr(left.bias, "us_open_bias_30", None) or left.bias.us_open_bias
            us60_left = getattr(left.bias, "us_open_bias_60", None) or left.bias.us_open_bias
            us30_left_conf = getattr(left.bias, "us_open_confidence_30", None)
            us60_left_conf = getattr(left.bias, "us_open_confidence_60", None)
            if us30_left_conf is None:
                us30_left_conf = left.bias.us_open_confidence
            if us60_left_conf is None:
                us60_left_conf = left.bias.us_open_confidence
            st.write(f"US 30m: {us30_left} ({us30_left_conf:.0%})")
            st.write(f"US 60m: {us60_left} ({us60_left_conf:.0%})")
            st.markdown("**Patterns**")
            st.write(f"London Breakout: {left.patterns.london_breakout}")
            st.write(f"Whipsaw: {left.patterns.whipsaw}")
            st.write(f"Trend Day: {left.patterns.trend_day}")
            st.write(f"Vol Expansion: {left.patterns.volatility_expansion}")
            st.markdown("**Accuracy**")
            st.write(f"Actual: {left.accuracy.actual_direction}")
            st.write(f"Correct: {left.accuracy.bias_correct}")
            st.write(f"Used Bias: {getattr(left.accuracy,'used_bias', 'n/a')}")
            st.write(f"US Open Correct 30m: {getattr(left.accuracy,'us_open_bias_correct_30', 'n/a')}")
            st.write(f"US Open Correct 60m: {getattr(left.accuracy,'us_open_bias_correct_60', 'n/a')}")

        with c2:
            st.subheader(f"{right.date} ({right.symbol})")
            st.markdown("**Bias**")
            st.write(f"Daily: {right.bias.daily_bias} ({right.bias.daily_confidence:.0%})")
            us30_right = getattr(right.bias, "us_open_bias_30", None) or right.bias.us_open_bias
            us60_right = getattr(right.bias, "us_open_bias_60", None) or right.bias.us_open_bias
            us30_right_conf = getattr(right.bias, "us_open_confidence_30", None)
            us60_right_conf = getattr(right.bias, "us_open_confidence_60", None)
            if us30_right_conf is None:
                us30_right_conf = right.bias.us_open_confidence
            if us60_right_conf is None:
                us60_right_conf = right.bias.us_open_confidence
            st.write(f"US 30m: {us30_right} ({us30_right_conf:.0%})")
            st.write(f"US 60m: {us60_right} ({us60_right_conf:.0%})")
            st.markdown("**Patterns**")
            st.write(f"London Breakout: {right.patterns.london_breakout}")
            st.write(f"Whipsaw: {right.patterns.whipsaw}")
            st.write(f"Trend Day: {right.patterns.trend_day}")
            st.write(f"Vol Expansion: {right.patterns.volatility_expansion}")
            st.markdown("**Accuracy**")
            st.write(f"Actual: {right.accuracy.actual_direction}")
            st.write(f"Correct: {right.accuracy.bias_correct}")
            st.write(f"Used Bias: {getattr(right.accuracy,'used_bias', 'n/a')}")
            st.write(f"US Open Correct 30m: {getattr(right.accuracy,'us_open_bias_correct_30', 'n/a')}")
            st.write(f"US Open Correct 60m: {getattr(right.accuracy,'us_open_bias_correct_60', 'n/a')}")
    else:
        default_symbol = summaries[-1].symbol if summaries else "NQH26"
        today = dt.date.today()
        c0, c1, c2 = st.columns([2, 1, 1])
        with c0:
            symbol = st.text_input("Symbol", value=default_symbol)
        with c1:
            left_date_input = st.date_input("Left date", value=today - dt.timedelta(days=1))
        with c2:
            right_date_input = st.date_input("Right date", value=today - dt.timedelta(days=2))

        if left_date_input == right_date_input:
            st.warning("Select two different dates.")
            return

        left_symbol = right_symbol = symbol
        left_date = left_date_input.isoformat()
        right_date = right_date_input.isoformat()
        st.caption("Dates are recomputed from raw intraday data and do not need to be saved in history first.")

    st.markdown("### End-of-Day NQ Opportunity (Entry -> Targets)")
    st.caption(
        "Tick math uses NQ tick size 0.25. Dollar conversion: 1 contract = $5/tick, 2 contracts = $10/tick."
    )

    left_opp = _compute_day_opportunity(left_symbol, left_date)
    right_opp = _compute_day_opportunity(right_symbol, right_date)

    o1, o2 = st.columns(2)
    with o1:
        _render_opportunity_block(f"{left_date} ({left_symbol})", left_opp)
    with o2:
        _render_opportunity_block(f"{right_date} ({right_symbol})", right_opp)

    if not left_opp.get("error") and not right_opp.get("error"):
        st.markdown("**Comparison Delta (Left - Right)**")
        left_pref_1 = (left_opp.get("preferred", {}) or {}).get("usd_1")
        right_pref_1 = (right_opp.get("preferred", {}) or {}).get("usd_1")
        left_pref_2 = (left_opp.get("preferred", {}) or {}).get("usd_2")
        right_pref_2 = (right_opp.get("preferred", {}) or {}).get("usd_2")

        d1 = left_pref_1 - right_pref_1 if left_pref_1 is not None and right_pref_1 is not None else None
        d2 = left_pref_2 - right_pref_2 if left_pref_2 is not None and right_pref_2 is not None else None

        dcol1, dcol2 = st.columns(2)
        dcol1.metric("Preferred $ Delta (1 Contract)", _fmt_money(d1))
        dcol2.metric("Preferred $ Delta (2 Contracts)", _fmt_money(d2))