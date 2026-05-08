import datetime as dt
import time
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from data.data_fetcher import fetch_intraday_ohlcv
from data.session_reference import get_session_windows_for_date
from engines.volume_profile import build_session_profiles, trading_day_bounds
from indicators.momentum import roc, trend_strength
from indicators.moving_averages import compute_daily_vwap
from indicators.statistics import zscore
from indicators.volatility import atr_like, classify_volatility, rolling_volatility
from indicators.volume import rvol
from engines.unified_bias import build_unified_bias
from ui.bias_composite import render_unified_bias_panel

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    out = df.copy()
    if "timestamp" not in out.columns:
        out = out.reset_index().rename(columns={out.index.name or "index": "timestamp"})
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"])
    keep = [c for c in ["timestamp", "open", "high", "low", "close", "volume"] if c in out.columns]
    out = out[keep].sort_values("timestamp").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


def _prev_trading_day(date: dt.date) -> dt.date:
    prev = date - dt.timedelta(days=1)
    while prev.weekday() >= 5:
        prev -= dt.timedelta(days=1)
    return prev


def _next_trading_day(date: dt.date) -> dt.date:
    nxt = date + dt.timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += dt.timedelta(days=1)
    return nxt


def _now_et() -> dt.datetime:
    if ZoneInfo is None:
        return dt.datetime.now()
    return dt.datetime.now(ZoneInfo("America/New_York")).replace(tzinfo=None)


def _twap(series: pd.Series) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(dtype=float)
    return series.expanding().mean()


def _label_from_value(value: float, threshold: float = 0.0) -> str:
    if value > threshold:
        return "Bullish"
    if value < -threshold:
        return "Bearish"
    return "Neutral"


def _vote_direction(votes: list) -> Tuple[str, float]:
    vals = [v for v in votes if v in ("Bullish", "Bearish")]
    if not vals:
        return "Neutral", 0.5
    bull = vals.count("Bullish")
    bear = vals.count("Bearish")
    if bull > bear:
        return "Bullish", bull / len(vals)
    if bear > bull:
        return "Bearish", bear / len(vals)
    return "Neutral", 0.5


def _sign_from_label(label: str) -> int:
    if label == "Bullish":
        return 1
    if label == "Bearish":
        return -1
    return 0


def _combine_directional_components(components: list) -> Tuple[str, float, float]:
    weighted = 0.0
    total_w = 0.0
    for label, conf in components:
        s = _sign_from_label(label)
        w = max(float(conf), 0.0)
        if s == 0 or w <= 0:
            continue
        weighted += s * w
        total_w += w

    if total_w <= 0:
        return "Neutral", 0.5, 0.0

    ratio = weighted / total_w
    if ratio > 0.10:
        direction = "Bullish"
    elif ratio < -0.10:
        direction = "Bearish"
    else:
        direction = "Neutral"
    confidence = abs(ratio)
    return direction, confidence, weighted


def _direction_from_frame(df: pd.DataFrame, threshold: float = 0.0) -> str:
    if df is None or df.empty:
        return "Neutral"
    move = float(df["close"].iloc[-1] - df["open"].iloc[0])
    return _label_from_value(move, threshold=threshold)


def _is_finalized(selected_date: dt.date, cutoff: dt.datetime, now_ref: dt.datetime) -> bool:
    if selected_date < now_ref.date():
        return True
    if selected_date > now_ref.date():
        return False
    return now_ref >= cutoff


def _phase_status_text(is_final: bool) -> str:
    return "Finalized" if is_final else "Not Finalized"


def _evaluate_alignment(predicted: str, actual: str) -> str:
    if predicted == "Neutral":
        return "The hypothesis was neutral, so directional alignment is inconclusive."
    if actual == "Neutral":
        return "The session finished neutral, so the directional hypothesis had partial alignment only."
    if predicted == actual:
        return "The hypothesis aligned with the realized direction."
    return "The hypothesis did not align with the realized direction."


def _session_path_sentence(name: str, df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return f"{name} session data is unavailable, so path analysis is limited."
    s_open = float(df["open"].iloc[0])
    s_close = float(df["close"].iloc[-1])
    s_high = float(df["high"].max())
    s_low = float(df["low"].min())
    direction = _direction_from_frame(df)
    return (
        f"{name} traded from {s_open:.2f} to {s_close:.2f} with a {direction.lower()} finish, "
        f"printing a range from {s_low:.2f} to {s_high:.2f}."
    )


def _direction_evidence_sentence(direction: str, evidence: list) -> str:
    if not evidence:
        return "Signals are mixed, so directional conviction is muted."
    lead = ", ".join(evidence[:3])
    if direction == "Bullish":
        return f"Bullish evidence is led by {lead}, which supports upside continuation if opening structure holds."
    if direction == "Bearish":
        return f"Bearish evidence is led by {lead}, which supports downside continuation if opening structure holds."
    return f"Evidence is mixed ({lead}), which favors rotational behavior until one side gains acceptance."


def _render_analysis_card(title: str, body: str) -> None:
    st.markdown(
        (
            "<div class='ms-analysis-card'>"
            f"<div class='ms-analysis-title'>{title}</div>"
            f"<div class='ms-analysis-body'>{body}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_analysis_grid(items: list) -> None:
    if not items:
        return
    rows = [items[i:i + 3] for i in range(0, len(items), 3)]
    for row in rows:
        cols = st.columns(3)
        for idx, col in enumerate(cols):
            if idx < len(row):
                with col:
                    _render_analysis_card(row[idx]["title"], row[idx]["body"])


def _unfinished_label(done: bool, finish_ts: dt.datetime) -> str:
    if done:
        return ""
    return f"Unfinished (expected by {finish_ts:%H:%M} ET)"


def _neutral_reason(
    not_enough_evidence: bool,
    mixed_signals: bool,
    weak_directional_move: bool,
    inside_range_behavior: bool,
) -> str:
    if not_enough_evidence:
        return "there's not enough evidence"
    if mixed_signals:
        return "there's mixed signals"
    if weak_directional_move:
        return "weak directional move"
    if inside_range_behavior:
        return "inside range behavior"
    return "there's mixed signals"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _partner_alignment(bias: str, partner_biases: Dict[str, str], required: list) -> bool:
    if bias not in ("Bullish", "Bearish"):
        return False
    vals = [partner_biases.get(k, "Neutral") for k in required]
    if not vals:
        return True
    return all(v == bias for v in vals)


def _classify_opening_drive(df_us: pd.DataFrame, atr_value: float) -> str:
    if df_us is None or df_us.empty or len(df_us) < 3:
        return "Neutral"
    first30 = df_us.head(6)
    if first30.empty:
        return "Neutral"
    open_px = float(first30["open"].iloc[0])
    close_px = float(first30["close"].iloc[-1])
    high_px = float(first30["high"].max())
    low_px = float(first30["low"].min())
    move = close_px - open_px
    range_30 = max(high_px - low_px, 1e-6)
    atr_safe = max(atr_value, 1e-6)

    if move > 0 and move >= 0.25 * atr_safe and close_px >= (high_px - 0.2 * range_30):
        return "Bullish"
    if move < 0 and abs(move) >= 0.25 * atr_safe and close_px <= (low_px + 0.2 * range_30):
        return "Bearish"
    return "Neutral"


def _classify_previous_day_type(prev_day_df: pd.DataFrame, prev_profile_location: str) -> str:
    if prev_day_df is None or prev_day_df.empty:
        return "n/a"
    day_open = float(prev_day_df["open"].iloc[0])
    day_close = float(prev_day_df["close"].iloc[-1])
    day_high = float(prev_day_df["high"].max())
    day_low = float(prev_day_df["low"].min())
    rng = max(day_high - day_low, 1e-6)
    close_pos = (day_close - day_low) / rng
    body = abs(day_close - day_open) / rng

    if body >= 0.55 and close_pos >= 0.75:
        return "Trend Up"
    if body >= 0.55 and close_pos <= 0.25:
        return "Trend Down"
    if prev_profile_location == "Within Value":
        return "Balanced"
    return "Rotational"


def _gap_fill_probability(gap_atr_ratio: float, vol_regime: str, ny_open_relation: str) -> float:
    prob = 0.55
    if gap_atr_ratio >= 1.0:
        prob -= 0.22
    elif gap_atr_ratio <= 0.35:
        prob += 0.12

    if vol_regime == "expanded":
        prob -= 0.10
    elif vol_regime == "compressed":
        prob += 0.08

    if ny_open_relation == "Inside overnight range":
        prob += 0.08
    else:
        prob -= 0.04

    return _clamp(prob, 0.05, 0.95)


def _format_distance(value_points: float, unit: str, tick_size: float) -> str:
    if not np.isfinite(value_points):
        return "n/a"
    if unit == "ticks":
        ticks = value_points / max(tick_size, 1e-6)
        return f"{ticks:+.1f} ticks"
    return f"{value_points:+.2f} pts"


def _distance_value_for_score(value_points: float, unit: str, tick_size: float) -> float:
    if not np.isfinite(value_points):
        return 0.0
    if unit == "ticks":
        return value_points / max(tick_size, 1e-6)
    return value_points


def _render_tab_reading_guide() -> None:
    with st.expander("How To Read This Tab", expanded=True):
        st.markdown(
            "- **VWAP / TWAP**: session fair-value benchmarks (volume-weighted and time-weighted). "
            "**Price impact**: sustained trade above favors bullish continuation; sustained trade below favors bearish continuation.\n"
            "- **Distance to VWAP/TWAP**: how stretched price is from fair value. "
            "**Price impact**: small distance supports balance; large distance raises either continuation potential (with momentum) or snap-back risk (without momentum).\n"
            "- **VWAP/TWAP Standard Deviation**: dispersion around fair value. "
            "**Price impact**: rising dispersion often accompanies trend expansion; contracting dispersion often precedes range behavior.\n"
            "- **Z-Score**: standardized distance from recent mean. "
            "**Price impact**: extreme values usually increase mean-reversion risk unless trend and volume are strongly aligned.\n"
            "- **Mean Reversion Distance (ATR)**: stretch from VWAP in ATR units. "
            "**Price impact**: high values suggest either strong trend acceleration or elevated pullback probability.\n"
            "- **ATR / ATR Ratio**: absolute volatility and normalized stretch. "
            "**Price impact**: larger ATR widens expected swings; high ratio means price is extended and needs stronger confirmation for continuation.\n"
            "- **Realized Volatility / Volatility Percentile / Volatility Regime**: current volatility state vs recent history. "
            "**Price impact**: compressed regimes often transition into expansion; expanded regimes support continuation but can reverse sharply at extremes.\n"
            "- **Overnight Range and Position**: overnight high/low and current location relative to it. "
            "**Price impact**: opening/holding outside overnight range often supports continuation; staying inside often supports rotation.\n"
            "- **Overnight VWAP and Overnight vs VWAP**: overnight fair value and posture. "
            "**Price impact**: acceptance above overnight VWAP supports upside auction; below supports downside auction.\n"
            "- **Asia High/Low and London High/Low**: session reference extremes. "
            "**Price impact**: session breaks and failures around these levels often define intraday directional bias.\n"
            "- **London Sweep Direction**: whether London ran Asia highs or lows first. "
            "**Price impact**: buy-side sweep can set bearish reversal context; sell-side sweep can set bullish reversal context.\n"
            "- **NY Open vs Overnight/London Range**: NY open location context. "
            "**Price impact**: opens outside ranges favor continuation; opens inside ranges favor mean-revert starts.\n"
            "- **Opening Drive Behavior**: first 30-minute directional quality. "
            "**Price impact**: strong drive near extremes often leads to follow-through; weak drive tends to rotate.\n"
            "- **Session Expansion Rate / Compression-Expansion Ratio**: how much range growth is occurring. "
            "**Price impact**: fast expansion supports trend participation; compression supports breakout-watch posture rather than chase entries.\n"
            "- **POC / VAH / VAL**: profile acceptance center and value boundaries. "
            "**Price impact**: acceptance above VAH is bullish, below VAL is bearish, inside value is rotational.\n"
            "- **Profile Shape (D/P/b/B)**: auction distribution type. "
            "**Price impact**: D favors two-way trade, P can support upside continuation, b can support downside continuation, B often signals two-sided distribution.\n"
            "- **Gap Size / Gap ATR Ratio / Gap Fill Probability**: opening dislocation and likelihood of reversion. "
            "**Price impact**: small gaps fill more often; large ATR-normalized gaps are less likely to fill immediately.\n"
            "- **Previous Day Close Location / Previous Day Type**: prior session structural context. "
            "**Price impact**: trend-day closes outside value can carry momentum; balanced closes increase rotational odds.\n"
            "- **Overnight Trend Strength**: directional slope during overnight trade. "
            "**Price impact**: stronger overnight trend can prime continuation if NY open confirms.\n"
            "- **Relative Volume**: current volume vs recent average. "
            "**Price impact**: high relative volume validates directional moves; low relative volume weakens breakout reliability.\n"
            "- **Variance / Covariance**: return variability and relation to volume changes. "
            "**Price impact**: higher variance means higher uncertainty/risk; positive return-volume covariance can support continuation quality.\n"
            "- **Trend/Balance/Compression/Expansion/Mean-Reversion/Momentum Regimes**: state classification layer. "
            "**Price impact**: regime determines whether continuation or fade setups should be weighted more heavily.\n"
            "- **Confluence Table**: weighted directional synthesis with partner alignment checks. "
            "**Price impact**: the overall score helps filter low-quality setups and prioritize aligned directional conditions."
        )

        st.markdown("### Quick Good / Normal / Bad + Direction Guide")
        guide_rows = [
            {
                "Statistic": "Z-Score",
                "Good": "0.5 to 1.5 with trend alignment",
                "Normal": "-0.5 to 0.5",
                "Bad": "Beyond +/-1.5 against setup",
                "Bullish Result": "Positive z-score with trend/momentum confirmation",
                "Bearish Result": "Negative z-score with trend/momentum confirmation",
            },
            {
                "Statistic": "|Px-VWAP| / ATR",
                "Good": "0.5 to 1.5 with momentum",
                "Normal": "0.0 to 0.5",
                "Bad": "> 1.5 without alignment",
                "Bullish Result": "Price above VWAP with rising momentum and aligned partners",
                "Bearish Result": "Price below VWAP with falling momentum and aligned partners",
            },
            {
                "Statistic": "Compression/Expansion Ratio",
                "Good": "> 1.2 for continuation",
                "Normal": "0.8 to 1.2",
                "Bad": "< 0.8 for breakout expectations",
                "Bullish Result": "Expansion with upside continuation structure",
                "Bearish Result": "Expansion with downside continuation structure",
            },
            {
                "Statistic": "Volatility Percentile",
                "Good": "Matches strategy context",
                "Normal": "30% to 70%",
                "Bad": "Extreme mismatch to setup",
                "Bullish Result": "Expanded volatility plus bullish directional alignment",
                "Bearish Result": "Expanded volatility plus bearish directional alignment",
            },
            {
                "Statistic": "Gap ATR Ratio",
                "Good": "Small-medium for fill setups",
                "Normal": "0.35 to 1.0",
                "Bad": "> 1.0 for immediate fade",
                "Bullish Result": "Up gap accepted above key levels and held",
                "Bearish Result": "Down gap accepted below key levels and held",
            },
            {
                "Statistic": "Relative Volume",
                "Good": "> 1.2 with directional thesis",
                "Normal": "0.8 to 1.2",
                "Bad": "< 0.8 for momentum continuation",
                "Bullish Result": "High relative volume while price holds above value/VWAP",
                "Bearish Result": "High relative volume while price holds below value/VWAP",
            },
        ]
        st.dataframe(pd.DataFrame(guide_rows), use_container_width=True)


def render_market_statistics_tab() -> None:
    st.header("Market Statistics")
    st.caption(
        "Statistical and session-structure dashboard using current data feed capabilities. "
        "Order-flow-only fields (true delta and resting liquidity) are intentionally excluded."
    )
    _render_tab_reading_guide()

    today = _now_et().date()
    with st.sidebar:
        symbol = st.text_input("Symbol", value="NQH26", key="ms_symbol")
        selected_date = st.date_input("Analysis date", value=today, key="ms_date")
        tick_size = st.number_input("Tick size", min_value=0.01, value=0.25, step=0.01, key="ms_tick_size")
        value_area = st.slider("Value area %", min_value=0.60, max_value=0.90, value=0.70, step=0.01, key="ms_value_area")
        distance_unit = st.selectbox("Distance Unit", options=["points", "ticks"], index=0, key="ms_distance_unit")
        auto_refresh = st.checkbox("Auto-refresh", value=True, key="ms_auto_refresh")
        refresh_seconds = st.selectbox("Refresh every", options=[30, 60, 300], index=1, key="ms_refresh_seconds")

    if auto_refresh:
        interval = f"{int(refresh_seconds)}s"

        @st.fragment(run_every=interval)
        def _refresh_tick() -> None:
            marker_key = f"market_stats_last_auto_rerun::{symbol}::{selected_date.isoformat()}"
            now_ts = time.time()
            last_ts = float(st.session_state.get(marker_key, 0.0))
            if now_ts - last_ts >= max(int(refresh_seconds) - 1, 1):
                st.session_state[marker_key] = now_ts
                st.rerun()

        _refresh_tick()
    else:
        c1, c2 = st.columns([3, 1])
        with c1:
            st.caption("Auto-refresh is off. Use Refresh now to pull latest values.")
        with c2:
            if st.button("Refresh now", key="ms_refresh_now"):
                st.rerun()

    prev_date = _prev_trading_day(selected_date)
    inherited_key = f"ms_inherited_outlook::{selected_date.isoformat()}"
    inherited_outlook = st.session_state.get(inherited_key)
    res_today = fetch_intraday_ohlcv(symbol, selected_date)
    res_prev = fetch_intraday_ohlcv(symbol, prev_date)

    df_today = _prepare_df(res_today[0] if isinstance(res_today, tuple) else res_today)
    df_prev = _prepare_df(res_prev[0] if isinstance(res_prev, tuple) else res_prev)

    if df_today.empty and df_prev.empty:
        st.warning("No intraday data available for selected inputs.")
        return

    td_start, td_end = trading_day_bounds(selected_date)
    combined = pd.concat([df_prev, df_today], ignore_index=True) if not df_prev.empty else df_today.copy()
    combined = combined.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    day_df = combined[(combined["timestamp"] >= td_start) & (combined["timestamp"] <= td_end)].copy()
    prev_td_start, prev_td_end = trading_day_bounds(prev_date)
    prev_day_df = df_prev[(df_prev["timestamp"] >= prev_td_start) & (df_prev["timestamp"] <= prev_td_end)].copy()

    if day_df.empty:
        st.warning("Selected trading-day window has no bars.")
        return

    windows = get_session_windows_for_date(selected_date)
    now_ref = _now_et()
    market_open_end = windows["Asia"]["start"] - dt.timedelta(minutes=15)
    ny_open_end = dt.datetime.combine(selected_date, dt.time(9, 45))
    opening_drive_end = dt.datetime.combine(selected_date, dt.time(10, 0))
    asia_done = _is_finalized(selected_date, windows["Asia"]["end"], now_ref)
    london_done = _is_finalized(selected_date, windows["London"]["end"], now_ref)
    ny_open_done = _is_finalized(selected_date, ny_open_end, now_ref)
    opening_drive_done = _is_finalized(selected_date, opening_drive_end, now_ref)
    ny_done = _is_finalized(selected_date, windows["US"]["end"], now_ref)
    market_open_done = _is_finalized(selected_date, market_open_end, now_ref)

    asia_df = day_df[(day_df["timestamp"] >= windows["Asia"]["start"]) & (day_df["timestamp"] <= windows["Asia"]["end"])].copy()
    london_df = day_df[(day_df["timestamp"] >= windows["London"]["start"]) & (day_df["timestamp"] <= windows["London"]["end"])].copy()
    us_df = day_df[(day_df["timestamp"] >= windows["US"]["start"]) & (day_df["timestamp"] <= windows["US"]["end"])].copy()
    overnight_df = day_df[(day_df["timestamp"] >= td_start) & (day_df["timestamp"] < windows["US"]["start"])].copy()
    last_data_ts = pd.to_datetime(day_df["timestamp"].max()) if not day_df.empty else None
    last_data_label = last_data_ts.strftime("%Y-%m-%d %H:%M ET") if last_data_ts is not None else "n/a"
    last_data_time_label = last_data_ts.strftime("%H:%M:%S ET") if last_data_ts is not None else "n/a"
    st.info(f"Last data update: {last_data_label}")

    close = day_df["close"]
    high = day_df["high"]
    low = day_df["low"]
    last_price = float(close.iloc[-1])

    vwap_series = compute_daily_vwap(day_df)
    twap_series = _twap(close)
    vwap_last = float(vwap_series.iloc[-1]) if not vwap_series.empty and pd.notna(vwap_series.iloc[-1]) else float("nan")
    twap_last = float(twap_series.iloc[-1]) if not twap_series.empty and pd.notna(twap_series.iloc[-1]) else float("nan")

    dist_vwap = last_price - vwap_last if np.isfinite(vwap_last) else float("nan")
    dist_twap = last_price - twap_last if np.isfinite(twap_last) else float("nan")
    dist_vwap_display = _format_distance(dist_vwap, distance_unit, float(tick_size))
    dist_twap_display = _format_distance(dist_twap, distance_unit, float(tick_size))

    spread_vwap = close - vwap_series if not vwap_series.empty else pd.Series(dtype=float)
    spread_twap = close - twap_series if not twap_series.empty else pd.Series(dtype=float)
    std_vwap = float(spread_vwap.std()) if len(spread_vwap.dropna()) > 5 else float("nan")
    std_twap = float(spread_twap.std()) if len(spread_twap.dropna()) > 5 else float("nan")

    z_series = zscore(close, length=20)
    z_last = float(z_series.iloc[-1]) if len(z_series) and pd.notna(z_series.iloc[-1]) else 0.0

    atr_series = atr_like(day_df, length=14)
    atr_last = float(atr_series.iloc[-1]) if len(atr_series) and pd.notna(atr_series.iloc[-1]) else 0.0
    atr_ratio = abs(dist_vwap) / max(atr_last, 1e-6) if np.isfinite(dist_vwap) else 0.0

    ret = close.pct_change().dropna()
    realized_vol = float(ret.rolling(20).std().iloc[-1]) if len(ret) >= 20 else 0.0
    vol_series = rolling_volatility(close, length=20)
    vol_regime = classify_volatility(vol_series)
    vol_pct = float((vol_series.rank(pct=True).iloc[-1])) if len(vol_series.dropna()) > 0 else 0.5

    overnight_high = float(overnight_df["high"].max()) if not overnight_df.empty else float("nan")
    overnight_low = float(overnight_df["low"].min()) if not overnight_df.empty else float("nan")
    overnight_range = (overnight_high - overnight_low) if np.isfinite(overnight_high) and np.isfinite(overnight_low) else 0.0
    overnight_vwap_series = compute_daily_vwap(overnight_df) if not overnight_df.empty else pd.Series(dtype=float)
    overnight_vwap = float(overnight_vwap_series.iloc[-1]) if len(overnight_vwap_series) else float("nan")
    overnight_position = (
        "Above overnight high"
        if np.isfinite(overnight_high) and last_price > overnight_high
        else "Below overnight low"
        if np.isfinite(overnight_low) and last_price < overnight_low
        else "Inside overnight range"
    )
    overnight_vs_vwap = (
        "Above overnight VWAP"
        if np.isfinite(overnight_vwap) and last_price > overnight_vwap
        else "Below overnight VWAP"
        if np.isfinite(overnight_vwap)
        else "n/a"
    )

    asia_high = float(asia_df["high"].max()) if not asia_df.empty else float("nan")
    asia_low = float(asia_df["low"].min()) if not asia_df.empty else float("nan")
    london_high = float(london_df["high"].max()) if not london_df.empty else float("nan")
    london_low = float(london_df["low"].min()) if not london_df.empty else float("nan")
    london_sweep = "None"
    if np.isfinite(asia_high) and np.isfinite(asia_low) and not london_df.empty:
        up_sweep = float(london_df["high"].max()) > asia_high
        down_sweep = float(london_df["low"].min()) < asia_low
        if up_sweep and not down_sweep:
            london_sweep = "Buy-side sweep"
        elif down_sweep and not up_sweep:
            london_sweep = "Sell-side sweep"
        elif up_sweep and down_sweep:
            london_sweep = "Two-sided sweep"

    ny_open = float(us_df["open"].iloc[0]) if not us_df.empty else last_price
    ny_open_overnight = (
        "Above overnight range"
        if np.isfinite(overnight_high) and ny_open > overnight_high
        else "Below overnight range"
        if np.isfinite(overnight_low) and ny_open < overnight_low
        else "Inside overnight range"
    )
    ny_open_london = (
        "Above London range"
        if np.isfinite(london_high) and ny_open > london_high
        else "Below London range"
        if np.isfinite(london_low) and ny_open < london_low
        else "Inside London range"
    )

    asia_range = float(asia_high - asia_low) if np.isfinite(asia_high) and np.isfinite(asia_low) else 0.0
    london_range = float(london_high - london_low) if np.isfinite(london_high) and np.isfinite(london_low) else 0.0
    us_range = float(us_df["high"].max() - us_df["low"].min()) if not us_df.empty else 0.0
    day_range = float(high.max() - low.min())
    prev_range = float(prev_day_df["high"].max() - prev_day_df["low"].min()) if not prev_day_df.empty else 0.0

    session_expansion_rate = (us_range - overnight_range) / max(overnight_range, 1e-6)
    compression_expansion_ratio = day_range / max(prev_range, 1e-6) if prev_range > 0 else 0.0

    opening_drive = _classify_opening_drive(us_df, atr_last)

    roc_series = roc(close, length=10)
    roc_last = float(roc_series.iloc[-1]) if len(roc_series) and pd.notna(roc_series.iloc[-1]) else 0.0
    trend_val = float(trend_strength(close, length=20)) if len(close) >= 20 else 0.0
    momentum_regime = _label_from_value(roc_last, threshold=0.0005)
    trend_regime = _label_from_value(trend_val, threshold=max(atr_last * 0.02, 1e-6))
    balance_regime = "Balanced" if abs(z_last) < 0.5 else "Not Balanced"
    compression_regime = "Compressed" if vol_regime == "compressed" else "Not Compressed"
    expansion_regime = "Expanded" if vol_regime == "expanded" else "Not Expanded"
    mean_reversion_regime = "Active" if abs(z_last) >= 1.5 and vol_regime != "expanded" else "Low"

    prev_close = float(prev_day_df["close"].iloc[-1]) if not prev_day_df.empty else float("nan")
    gap_size = ny_open - prev_close if np.isfinite(prev_close) else 0.0
    gap_atr_ratio = abs(gap_size) / max(atr_last, 1e-6) if atr_last > 0 else 0.0

    curr_profiles = build_session_profiles(day_df, selected_date, tick_size=float(tick_size), value_area_pct=float(value_area))
    prev_profiles = build_session_profiles(prev_day_df, prev_date, tick_size=float(tick_size), value_area_pct=float(value_area))
    full_profile = curr_profiles.get("Full Day")
    prev_full_profile = prev_profiles.get("Full Day")

    poc = float(full_profile.poc) if full_profile and full_profile.poc is not None else float("nan")
    vah = float(full_profile.vah) if full_profile and full_profile.vah is not None else float("nan")
    val = float(full_profile.val) if full_profile and full_profile.val is not None else float("nan")
    profile_shape = full_profile.shape if full_profile else "n/a"
    prev_close_location = prev_full_profile.close_location if prev_full_profile else "n/a"
    prev_day_type = _classify_previous_day_type(prev_day_df, prev_close_location)

    overnight_trend = float(trend_strength(overnight_df["close"], length=20)) if len(overnight_df) >= 20 else 0.0
    rvol_series = rvol(day_df, length=20)
    rel_volume = float(rvol_series.iloc[-1]) if len(rvol_series) and pd.notna(rvol_series.iloc[-1]) else 1.0

    ret_recent = ret.tail(50)
    vol_change_recent = day_df["volume"].pct_change().replace([np.inf, -np.inf], np.nan).dropna().tail(50)
    variance = float(ret_recent.var()) if len(ret_recent) > 1 else 0.0
    cov_df = pd.concat([ret_recent.rename("ret"), vol_change_recent.rename("vol_chg")], axis=1).dropna()
    covariance = float(cov_df["ret"].cov(cov_df["vol_chg"])) if len(cov_df) > 1 else 0.0

    gap_fill_prob = _gap_fill_probability(gap_atr_ratio, vol_regime, ny_open_overnight)
    mean_reversion_distance = abs(dist_vwap) / max(atr_last, 1e-6) if np.isfinite(dist_vwap) else 0.0

    vwap_bias = "Bullish" if np.isfinite(dist_vwap) and dist_vwap > 0 else "Bearish"
    twap_bias = "Bullish" if np.isfinite(dist_twap) and dist_twap > 0 else "Bearish"
    overnight_bias = "Bullish" if overnight_position.startswith("Above") else "Bearish" if overnight_position.startswith("Below") else "Neutral"
    london_bias = "Bullish" if london_sweep == "Sell-side sweep" else "Bearish" if london_sweep == "Buy-side sweep" else "Neutral"
    if not london_done:
        london_bias = "Neutral"
    ny_open_bias = "Bullish" if ny_open_overnight.startswith("Above") else "Bearish" if ny_open_overnight.startswith("Below") else "Neutral"
    if not ny_open_done:
        ny_open_bias = "Neutral"
    opening_drive_bias = opening_drive if opening_drive in ("Bullish", "Bearish") else "Neutral"
    if not opening_drive_done:
        opening_drive_bias = "Neutral"
    gap_bias = "Bullish" if gap_size > 0 else "Bearish" if gap_size < 0 else "Neutral"
    profile_bias = "Bullish" if np.isfinite(vah) and last_price > vah else "Bearish" if np.isfinite(val) and last_price < val else "Neutral"
    momentum_bias = momentum_regime if momentum_regime in ("Bullish", "Bearish") else "Neutral"

    st.subheader("Core Statistics")
    st.caption(f"Distance metrics are shown in {distance_unit} (tick size: {float(tick_size):.2f}).")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Last Price", f"{last_price:.2f}")
    col1.metric("VWAP", f"{vwap_last:.2f}" if np.isfinite(vwap_last) else "n/a")
    col1.metric("TWAP", f"{twap_last:.2f}" if np.isfinite(twap_last) else "n/a")

    col2.metric("Distance to VWAP", dist_vwap_display)
    col2.metric("Distance to TWAP", dist_twap_display)
    col2.metric("Z-Score", f"{z_last:.2f}")

    col3.metric("ATR (14)", f"{atr_last:.2f}")
    col3.metric("ATR Ratio (|Px-VWAP|/ATR)", f"{atr_ratio:.2f}")
    col3.metric("Realized Vol (20)", f"{realized_vol:.4f}")

    col4.metric("Volatility Regime", vol_regime)
    col4.metric("Volatility Percentile", f"{vol_pct:.2%}")
    col4.metric("Mean Reversion Distance", f"{mean_reversion_distance:.2f} ATR")

    asia_hilo_value = (
        f"{asia_high:.2f} / {asia_low:.2f}"
        if np.isfinite(asia_high) and np.isfinite(asia_low)
        else "n/a"
    )
    if not asia_done:
        asia_hilo_value = _unfinished_label(False, windows["Asia"]["end"])

    london_hilo_value = (
        f"{london_high:.2f} / {london_low:.2f}"
        if np.isfinite(london_high) and np.isfinite(london_low)
        else "n/a"
    )
    if not london_done:
        london_hilo_value = _unfinished_label(False, windows["London"]["end"])

    london_sweep_value = london_sweep if london_done else _unfinished_label(False, windows["London"]["end"])
    ny_open_overnight_value = ny_open_overnight if ny_open_done else _unfinished_label(False, ny_open_end)
    ny_open_london_value = ny_open_london if ny_open_done else _unfinished_label(False, ny_open_end)
    opening_drive_value = opening_drive if opening_drive_done else _unfinished_label(False, opening_drive_end)
    session_expansion_value = f"{session_expansion_rate:.2f}" if ny_done else _unfinished_label(False, windows["US"]["end"])

    st.subheader("Session Structure")
    st.caption(f"Last update: {last_data_time_label}")
    struct_rows = [
        {"Metric": "Overnight High/Low", "Value": f"{overnight_high:.2f} / {overnight_low:.2f}" if np.isfinite(overnight_high) and np.isfinite(overnight_low) else "n/a"},
        {"Metric": "Overnight Range", "Value": f"{overnight_range:.2f}"},
        {"Metric": "Overnight Position", "Value": overnight_position},
        {"Metric": "Overnight VWAP", "Value": f"{overnight_vwap:.2f}" if np.isfinite(overnight_vwap) else "n/a"},
        {"Metric": "Overnight vs VWAP", "Value": overnight_vs_vwap},
        {"Metric": "Asia High/Low", "Value": asia_hilo_value},
        {"Metric": "London High/Low", "Value": london_hilo_value},
        {"Metric": "London Sweep Direction", "Value": london_sweep_value},
        {"Metric": "NY Open vs Overnight", "Value": ny_open_overnight_value},
        {"Metric": "NY Open vs London", "Value": ny_open_london_value},
        {"Metric": "Opening Drive", "Value": opening_drive_value},
        {"Metric": "Session Expansion Rate", "Value": session_expansion_value},
        {"Metric": "Compression/Expansion Ratio", "Value": f"{compression_expansion_ratio:.2f}"},
    ]
    st.dataframe(pd.DataFrame(struct_rows), use_container_width=True)
    session_votes = [
        overnight_bias,
        london_bias,
        ny_open_bias,
        opening_drive_bias,
    ]
    session_result, session_conf_raw = _vote_direction(session_votes)
    session_conf = _clamp(0.45 + 0.45 * session_conf_raw, 0.35, 0.90)
    session_non_neutral = [v for v in session_votes if v in ("Bullish", "Bearish")]
    session_has_bull = any(v == "Bullish" for v in session_non_neutral)
    session_has_bear = any(v == "Bearish" for v in session_non_neutral)
    session_reason = _neutral_reason(
        not_enough_evidence=(not london_done) or (not ny_open_done) or (not opening_drive_done),
        mixed_signals=session_has_bull and session_has_bear,
        weak_directional_move=(opening_drive_bias == "Neutral" and abs(roc_last) < 0.001),
        inside_range_behavior=(
            overnight_position == "Inside overnight range"
            and ny_open_overnight == "Inside overnight range"
            and ny_open_london == "Inside London range"
        ),
    )
    s1, s2 = st.columns(2)
    s1.metric("Session Structure Result", session_result)
    s2.metric("Session Structure Confidence", f"{session_conf:.0%}")
    if session_result == "Neutral":
        st.caption(f"Session Structure is neutral because {session_reason}.")

    st.subheader("Profile and Context")
    st.caption(f"Last update: {last_data_time_label}")
    profile_rows = [
        {"Metric": "POC", "Value": f"{poc:.2f}" if np.isfinite(poc) else "n/a"},
        {"Metric": "VAH", "Value": f"{vah:.2f}" if np.isfinite(vah) else "n/a"},
        {"Metric": "VAL", "Value": f"{val:.2f}" if np.isfinite(val) else "n/a"},
        {"Metric": "Profile Shape", "Value": profile_shape},
        {"Metric": "Gap Size", "Value": f"{gap_size:+.2f}"},
        {"Metric": "Gap ATR Ratio", "Value": f"{gap_atr_ratio:.2f}"},
        {"Metric": "Gap Fill Probability", "Value": f"{gap_fill_prob:.0%}"},
        {"Metric": "Previous Day Close Location", "Value": prev_close_location},
        {"Metric": "Previous Day Type", "Value": prev_day_type},
        {"Metric": "Overnight Trend Strength", "Value": f"{overnight_trend:+.4f}"},
        {"Metric": "Relative Volume", "Value": f"{rel_volume:.2f}"},
        {"Metric": "Variance (returns)", "Value": f"{variance:.6f}"},
        {"Metric": "Covariance (returns, volume change)", "Value": f"{covariance:.6f}"},
    ]
    st.dataframe(pd.DataFrame(profile_rows), use_container_width=True)
    profile_context_votes = [
        gap_bias,
        profile_bias,
        "Bullish" if prev_close_location == "Above VAH" else "Bearish" if prev_close_location == "Below VAL" else "Neutral",
        "Bullish" if prev_day_type == "Trend Up" else "Bearish" if prev_day_type == "Trend Down" else "Neutral",
        _label_from_value(overnight_trend, threshold=max(atr_last * 0.01, 1e-6)),
    ]
    profile_context_result, profile_context_conf_raw = _vote_direction(profile_context_votes)
    profile_context_conf = _clamp(0.45 + 0.45 * profile_context_conf_raw, 0.35, 0.90)
    profile_non_neutral = [v for v in profile_context_votes if v in ("Bullish", "Bearish")]
    profile_has_bull = any(v == "Bullish" for v in profile_non_neutral)
    profile_has_bear = any(v == "Bearish" for v in profile_non_neutral)
    profile_reason = _neutral_reason(
        not_enough_evidence=(not np.isfinite(poc)) or (not np.isfinite(vah)) or (not np.isfinite(val)),
        mixed_signals=profile_has_bull and profile_has_bear,
        weak_directional_move=(abs(gap_size) < max(0.10 * atr_last, 1e-6) and abs(overnight_trend) < max(atr_last * 0.01, 1e-6)),
        inside_range_behavior=(profile_bias == "Neutral" and prev_close_location == "Within Value"),
    )
    p1, p2 = st.columns(2)
    p1.metric("Profile/Context Result", profile_context_result)
    p2.metric("Profile/Context Confidence", f"{profile_context_conf:.0%}")
    if profile_context_result == "Neutral":
        st.caption(f"Profile/Context is neutral because {profile_reason}.")

    st.subheader("Regime Dashboard")
    st.caption(f"Last update: {last_data_time_label}")
    regime_rows = [
        {"Regime": "Trend", "State": trend_regime},
        {"Regime": "Balance", "State": balance_regime},
        {"Regime": "Compression", "State": compression_regime},
        {"Regime": "Expansion", "State": expansion_regime},
        {"Regime": "Mean Reversion", "State": mean_reversion_regime},
        {"Regime": "Momentum", "State": momentum_regime},
    ]
    st.dataframe(pd.DataFrame(regime_rows), use_container_width=True)
    regime_votes = [trend_regime, momentum_regime]
    regime_result, regime_conf_raw = _vote_direction(regime_votes)
    regime_conf = _clamp(0.45 + 0.45 * regime_conf_raw, 0.35, 0.90)
    regime_non_neutral = [v for v in regime_votes if v in ("Bullish", "Bearish")]
    regime_has_bull = any(v == "Bullish" for v in regime_non_neutral)
    regime_has_bear = any(v == "Bearish" for v in regime_non_neutral)
    regime_reason = _neutral_reason(
        not_enough_evidence=len(close) < 20,
        mixed_signals=regime_has_bull and regime_has_bear,
        weak_directional_move=(abs(roc_last) < 0.0005 and abs(trend_val) < max(atr_last * 0.01, 1e-6)),
        inside_range_behavior=(balance_regime == "Balanced"),
    )
    r1, r2 = st.columns(2)
    r1.metric("Regime Result", regime_result)
    r2.metric("Regime Confidence", f"{regime_conf:.0%}")
    if regime_result == "Neutral":
        st.caption(f"Regime is neutral because {regime_reason}.")

    partner_map = {
        "vwap": vwap_bias,
        "twap": twap_bias,
        "overnight": overnight_bias,
        "london": london_bias,
        "ny_open": ny_open_bias,
        "opening_drive": opening_drive_bias,
        "gap": gap_bias,
        "profile": profile_bias,
        "momentum": momentum_bias,
    }

    score_rows = []

    def add_row(metric: str, value: str, bias: str, strength: float, partners: list, reason: str) -> None:
        aligned = _partner_alignment(bias, partner_map, partners)
        direction = 1 if bias == "Bullish" else -1 if bias == "Bearish" else 0
        score = direction * strength * (1.0 if aligned else 0.65)
        score_rows.append(
            {
                "Metric": metric,
                "Value": value,
                "Bias": bias,
                "Strength": round(strength, 2),
                "Partners": ", ".join(partners) if partners else "n/a",
                "Aligned": "Yes" if aligned else "No",
                "Score": round(score, 2),
                "Reason": reason,
            }
        )

    add_row(
        "VWAP posture",
        dist_vwap_display,
        vwap_bias,
        _clamp(abs(dist_vwap) / max(atr_last, 1e-6), 0.2, 1.0) if np.isfinite(dist_vwap) else 0.2,
        ["twap", "momentum"],
        f"Distance from VWAP normalized by ATR (displayed in {distance_unit}).",
    )
    add_row(
        "TWAP posture",
        dist_twap_display,
        twap_bias,
        _clamp(abs(dist_twap) / max(atr_last, 1e-6), 0.2, 1.0) if np.isfinite(dist_twap) else 0.2,
        ["vwap", "momentum"],
        f"Distance from TWAP normalized by ATR (displayed in {distance_unit}).",
    )
    add_row(
        "Overnight position",
        overnight_position,
        overnight_bias,
        0.75,
        ["ny_open", "gap"],
        "Location of price relative to overnight range.",
    )
    add_row(
        "London sweep",
        london_sweep_value,
        london_bias,
        0.65,
        ["ny_open", "opening_drive"],
        "Sweep direction often informs NY directional setup; if unfinished, this row becomes active after London close.",
    )
    add_row(
        "NY open location",
        ny_open_overnight_value,
        ny_open_bias,
        0.8,
        ["overnight", "opening_drive"],
        "NY open relative to overnight range; if unfinished, this finalizes by 09:45 ET.",
    )
    add_row(
        "Opening drive",
        opening_drive_value,
        opening_drive_bias,
        0.85,
        ["ny_open", "momentum"],
        "First 30m directional quality versus ATR; if unfinished, this finalizes by 10:00 ET.",
    )
    add_row(
        "Gap context",
        f"{gap_size:+.2f} ({gap_atr_ratio:.2f} ATR)",
        gap_bias,
        _clamp(gap_atr_ratio, 0.2, 1.0),
        ["overnight", "opening_drive"],
        "Opening gap and its ATR-normalized magnitude.",
    )
    add_row(
        "Profile location",
        f"POC {poc:.2f} | VAH/VAL {vah:.2f}/{val:.2f}" if np.isfinite(poc) and np.isfinite(vah) and np.isfinite(val) else "n/a",
        profile_bias,
        0.7,
        ["vwap", "momentum"],
        "Acceptance outside value can support continuation.",
    )
    add_row(
        "Momentum regime",
        momentum_regime,
        momentum_bias,
        _clamp(abs(roc_last) / 0.004, 0.2, 1.0),
        ["vwap", "opening_drive"],
        "Rate-of-change based directional momentum.",
    )

    score_df = pd.DataFrame(score_rows)
    net_score = float(score_df["Score"].sum()) if not score_df.empty else 0.0
    max_score = float(score_df["Strength"].sum()) if not score_df.empty else 1.0
    confidence = abs(net_score) / max(max_score, 1e-6)

    if net_score > 1.25:
        overall = "Bullish"
    elif net_score < -1.25:
        overall = "Bearish"
    else:
        overall = "Neutral"

    st.subheader("High-Impact Confluence Table")
    st.dataframe(score_df, use_container_width=True)

    confluence_updated_label = _now_et().strftime("%H:%M:%S ET")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall Result", overall)
    c2.metric("Net Score", f"{net_score:.2f}")
    c3.metric("Confidence", f"{confidence:.0%}")
    c4.metric("Last Update", confluence_updated_label)
    if overall == "Neutral":
        has_pos = bool((score_df["Score"] > 0).any()) if not score_df.empty else False
        has_neg = bool((score_df["Score"] < 0).any()) if not score_df.empty else False
        overall_reason = _neutral_reason(
            not_enough_evidence=(not london_done) or (not ny_open_done) or (not opening_drive_done),
            mixed_signals=has_pos and has_neg,
            weak_directional_move=abs(net_score) < 0.25,
            inside_range_behavior=(overnight_position == "Inside overnight range" and profile_bias == "Neutral"),
        )
        st.caption(f"Overall Result is neutral because {overall_reason}.")

    st.caption(
        "Confluence score combines directional metrics and down-weights signals when required partner metrics do not align."
    )

    daily_bias_cutoff = dt.datetime.combine(selected_date, dt.time(10, 45))
    daily_bias_finalized = _is_finalized(selected_date, daily_bias_cutoff, now_ref)
    stats_votes = [
        vwap_bias,
        twap_bias,
        overnight_bias,
        london_bias,
        ny_open_bias,
        opening_drive_bias,
        gap_bias,
        profile_bias,
        momentum_bias,
    ]
    stats_result, stats_conf_raw = _vote_direction(stats_votes)
    stats_conf = _clamp(0.40 + 0.50 * stats_conf_raw, 0.30, 0.92)

    finalized_votes = []
    if session_result in ("Bullish", "Bearish") and london_done and ny_open_done and opening_drive_done:
        finalized_votes.append(session_result)
    if profile_context_result in ("Bullish", "Bearish") and ny_done:
        finalized_votes.append(profile_context_result)
    if regime_result in ("Bullish", "Bearish") and ny_done:
        finalized_votes.append(regime_result)
    finalized_result, finalized_conf_raw = _vote_direction(finalized_votes)
    finalized_conf = _clamp(0.40 + 0.50 * finalized_conf_raw, 0.30, 0.92)

    daily_bias_components = [
        (overall, confidence),
        (session_result, session_conf),
        (profile_context_result, profile_context_conf),
        (regime_result, regime_conf),
        (stats_result, stats_conf),
        (finalized_result, finalized_conf),
    ]
    daily_bias, daily_bias_conf, daily_bias_weighted = _combine_directional_components(daily_bias_components)
    daily_bias_updated_label = _now_et().strftime("%H:%M:%S ET")
    st.subheader("Daily bias")
    st.caption(f"Last updated: {daily_bias_updated_label}")
    st.write(
        f"Daily bias is {daily_bias.lower()} with {daily_bias_conf:.0%} confidence, combined from High-Impact Confluence, Session Structure, "
        f"Profile/Context, Regime, broader statistics, and finalized summaries on this tab. "
        f"Status: {'Finalized' if daily_bias_finalized else 'Not Finalized'}"
    )
    st.caption(f"Daily bias finalization time: {daily_bias_cutoff:%H:%M} ET")

    st.markdown(
        """
        <style>
        .ms-analysis-card {
            border: 1px solid rgba(128,128,128,0.35);
            border-radius: 10px;
            padding: 12px 12px;
            min-height: 220px;
            background: rgba(20, 24, 30, 0.08);
            margin-bottom: 12px;
        }
        .ms-analysis-title {
            font-weight: 700;
            margin-bottom: 8px;
        }
        .ms-analysis-body {
            font-size: 0.92rem;
            line-height: 1.4;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Pre Session Analysis")
    if inherited_outlook:
        st.info(f"Inherited Context: {inherited_outlook}")

    atr_threshold = max(atr_last * 0.05, 0.0)

    asia_actual = _direction_from_frame(asia_df, threshold=atr_threshold)
    london_actual = _direction_from_frame(london_df, threshold=atr_threshold)
    ny_actual = _direction_from_frame(us_df, threshold=atr_threshold)
    day_actual = _direction_from_frame(day_df, threshold=atr_threshold)

    prev_day_move_dir = _direction_from_frame(prev_day_df, threshold=atr_threshold)
    prev_profile_dir = (
        "Bullish" if prev_close_location == "Above VAH" else "Bearish" if prev_close_location == "Below VAL" else "Neutral"
    )
    prev_day_type_dir = (
        "Bullish" if prev_day_type == "Trend Up" else "Bearish" if prev_day_type == "Trend Down" else "Neutral"
    )

    market_open_dir, market_open_conf_raw = _vote_direction([prev_day_move_dir, prev_profile_dir, prev_day_type_dir])
    market_open_conf = _clamp(0.45 + 0.45 * market_open_conf_raw, 0.35, 0.90)

    overnight_dir = _label_from_value(overnight_trend, threshold=max(atr_last * 0.01, 1e-6))
    asia_hyp_dir, asia_hyp_conf_raw = _vote_direction([market_open_dir, overnight_dir, gap_bias])
    asia_hyp_conf = _clamp(0.45 + 0.45 * asia_hyp_conf_raw, 0.35, 0.90)

    london_hyp_dir, london_hyp_conf_raw = _vote_direction([asia_actual, asia_hyp_dir, overnight_bias])
    london_hyp_conf = _clamp(0.45 + 0.45 * london_hyp_conf_raw, 0.35, 0.90)

    ny_open_hyp_dir, ny_open_hyp_conf_raw = _vote_direction([london_actual, ny_open_bias, gap_bias, overnight_bias])
    ny_open_hyp_conf = _clamp(0.45 + 0.45 * ny_open_hyp_conf_raw, 0.35, 0.90)

    ny_session_hyp_dir, ny_session_hyp_conf_raw = _vote_direction([opening_drive_bias, momentum_bias, profile_bias, ny_open_bias])
    ny_session_hyp_conf = _clamp(0.45 + 0.45 * ny_session_hyp_conf_raw, 0.35, 0.90)

    market_open_df = day_df[(day_df["timestamp"] >= td_start) & (day_df["timestamp"] <= market_open_end)].copy()
    market_open_actual = _direction_from_frame(market_open_df, threshold=atr_threshold)

    ny_open_df = us_df[us_df["timestamp"] <= ny_open_end].copy()
    ny_open_actual = _direction_from_frame(ny_open_df, threshold=atr_threshold)

    eod_cutoff = td_end
    mkt_open_plan_cutoff = dt.datetime.combine(selected_date, dt.time(17, 30))
    asia_finalize = windows["Asia"]["start"] + dt.timedelta(minutes=15)
    london_finalize = windows["London"]["start"] + dt.timedelta(minutes=15)
    ny_open_finalize = dt.datetime.combine(selected_date, dt.time(9, 15))
    ny_session_finalize = dt.datetime.combine(selected_date, dt.time(10, 35))

    eod_final = _is_finalized(selected_date, eod_cutoff, now_ref)
    mkt_open_plan_final = _is_finalized(selected_date, mkt_open_plan_cutoff, now_ref)
    asia_final = _is_finalized(selected_date, asia_finalize, now_ref)
    london_final = _is_finalized(selected_date, london_finalize, now_ref)
    ny_open_final = _is_finalized(selected_date, ny_open_finalize, now_ref)
    ny_session_final = _is_finalized(selected_date, ny_session_finalize, now_ref)

    eod_alignment = _evaluate_alignment(overall, day_actual)
    market_open_alignment = _evaluate_alignment(market_open_dir, market_open_actual)
    asia_alignment = _evaluate_alignment(asia_hyp_dir, asia_actual)
    london_alignment = _evaluate_alignment(london_hyp_dir, london_actual)
    ny_open_alignment = _evaluate_alignment(ny_open_hyp_dir, ny_open_actual)
    ny_session_alignment = _evaluate_alignment(ny_session_hyp_dir, ny_actual)

    market_open_evidence = [
        f"previous day type {prev_day_type}",
        f"previous close location {prev_close_location}",
        f"gap context {gap_bias.lower()}",
    ]
    asia_evidence = [
        f"overnight trend {overnight_trend:+.4f}",
        f"market-open hypothesis {market_open_dir.lower()}",
        f"overnight position {overnight_position.lower()}",
    ]
    london_evidence = [
        f"Asia realized direction {asia_actual.lower()}",
        f"London sweep {london_sweep.lower()}",
        f"session expansion rate {session_expansion_rate:.2f}",
    ]
    ny_open_evidence = [
        f"NY open location {ny_open_overnight.lower()}",
        f"opening drive {opening_drive_bias.lower()}",
        f"gap-fill probability {gap_fill_prob:.0%}",
    ]
    ny_session_evidence = [
        f"momentum regime {momentum_bias.lower()}",
        f"profile location bias {profile_bias.lower()}",
        f"relative volume {rel_volume:.2f}",
    ]

    market_open_sentence = (
        f"Market Open Plan ({_phase_status_text(mkt_open_plan_final)} by 17:30 ET): using previous-day statistics including day type ({prev_day_type}), "
        f"close location ({prev_close_location}), and prior directional move ({prev_day_move_dir.lower()}), the hypothesis projects a {market_open_dir.lower()} open with "
        f"{market_open_conf:.0%} confidence, a gap-fill probability of {gap_fill_prob:.0%}, and an expectation that price will "
        f"{'seek continuation away from value' if market_open_dir in ('Bullish', 'Bearish') else 'trade rotationally around value'}; {market_open_alignment} "
        f"{_direction_evidence_sentence(market_open_dir, market_open_evidence)}"
    )
    asia_sentence = (
        f"Asia Session Hypothesis ({_phase_status_text(asia_final)} 15 minutes after Asia open): using previous-day context, market-open plan, overnight trend "
        f"({overnight_trend:+.4f}), and gap posture ({gap_bias.lower()}), Asia was expected to be {asia_hyp_dir.lower()} with {asia_hyp_conf:.0%} confidence, "
        f"and the realized Asia outcome was {asia_actual.lower()}; {asia_alignment} "
        f"{_direction_evidence_sentence(asia_hyp_dir, asia_evidence)}"
    )
    london_sentence = (
        f"London Session Hypothesis ({_phase_status_text(london_final)} 15 minutes after London open): using previous-day context, market-open plan, and Asia outcome "
        f"({asia_actual.lower()}), London was expected to be {london_hyp_dir.lower()} with {london_hyp_conf:.0%} confidence, and the realized London outcome was "
        f"{london_actual.lower()}; {london_alignment} "
        f"{_direction_evidence_sentence(london_hyp_dir, london_evidence)}"
    )
    ny_open_sentence = (
        f"NY Open Hypothesis ({_phase_status_text(ny_open_final)} by 09:15 ET): using previous-day, market-open, Asia, and London statistics, NY open was expected "
        f"to be {ny_open_hyp_dir.lower()} with {ny_open_hyp_conf:.0%} confidence, with immediate behavior biased to "
        f"{'continuation' if ny_open_hyp_dir in ('Bullish', 'Bearish') else 'rotation'} at the open and then "
        f"{'follow-through if OR breaks and holds' if ny_open_hyp_dir in ('Bullish', 'Bearish') else 'reversion unless OR expands with volume'} after the 30-minute OR forms; "
        f"{ny_open_alignment} {_direction_evidence_sentence(ny_open_hyp_dir, ny_open_evidence)}"
    )
    ny_session_sentence = (
        f"NY Session Hypothesis ({_phase_status_text(ny_session_final)} by 10:35 ET): after 30-minute and 60-minute OR information, the combined statistics expected "
        f"a {ny_session_hyp_dir.lower()} NY session with {ny_session_hyp_conf:.0%} confidence, while realized NY session direction was {ny_actual.lower()}; "
        f"{ny_session_alignment} {_direction_evidence_sentence(ny_session_hyp_dir, ny_session_evidence)}"
    )

    unified_payload = build_unified_bias(
        df_today=day_df,
        df_prev=prev_day_df,
        trading_date=selected_date,
        now_et=now_ref,
    )
    render_unified_bias_panel(
        panel_title="Combined Daily + NY Session/Open Bias",
        panel_key=f"market_stats::{selected_date.isoformat()}",
        unified_payload=unified_payload,
    )

    pre_cards = [
        {"title": "Market Open", "body": market_open_sentence},
        {"title": "Asia", "body": asia_sentence},
        {"title": "London", "body": london_sentence},
        {"title": "NY Open", "body": ny_open_sentence},
        {"title": "NY Session", "body": ny_session_sentence},
    ]
    _render_analysis_grid(pre_cards)

    st.subheader("Post Session Analysis")

    market_open_review = (
        f"Market Open Post-Session Review ({_phase_status_text(market_open_done)}): {_session_path_sentence('Market Open (18:00 to Asia-15m)', market_open_df)} "
        f"The statistics had assumed a {market_open_dir.lower()} path at {market_open_conf:.0%} confidence, and {market_open_alignment.lower()}"
    )
    asia_review = (
        f"Asia Post-Session Review ({_phase_status_text(asia_done)}): {_session_path_sentence('Asia', asia_df)} "
        f"The statistics had assumed a {asia_hyp_dir.lower()} path at {asia_hyp_conf:.0%} confidence, and {asia_alignment.lower()}"
    )
    london_review = (
        f"London Post-Session Review ({_phase_status_text(london_done)}): {_session_path_sentence('London', london_df)} "
        f"The statistics had assumed a {london_hyp_dir.lower()} path at {london_hyp_conf:.0%} confidence, and {london_alignment.lower()}"
    )
    ny_open_review = (
        f"NY Open Post-Session Review ({_phase_status_text(ny_open_done)}): {_session_path_sentence('NY Open (09:30-09:45)', ny_open_df)} "
        f"The statistics had assumed a {ny_open_hyp_dir.lower()} path at {ny_open_hyp_conf:.0%} confidence, and {ny_open_alignment.lower()}"
    )
    ny_review = (
        f"NY Session Post-Session Review ({_phase_status_text(ny_done)}): {_session_path_sentence('NY Session', us_df)} "
        f"The statistics had assumed a {ny_session_hyp_dir.lower()} path at {ny_session_hyp_conf:.0%} confidence, and {ny_session_alignment.lower()} "
        f"Stat drivers were momentum ({momentum_regime}), profile location ({profile_bias}), and opening-drive carry ({opening_drive_bias})."
    )

    post_cards = [
        {"title": "Market Open", "body": market_open_review},
        {"title": "Asia", "body": asia_review},
        {"title": "London", "body": london_review},
        {"title": "NY Open", "body": ny_open_review},
        {"title": "NY Session", "body": ny_review},
    ]
    _render_analysis_grid(post_cards)

    st.subheader("End of day summary")
    if not eod_final:
        st.write("Unavailable at the moment.")
    else:
        if day_actual in ("Bullish", "Bearish"):
            bias_truth = "True" if daily_bias == day_actual else "False"
        else:
            bias_truth = "Inconclusive"
        close_pos_pct = ((float(day_df["close"].iloc[-1]) - float(day_df["low"].min())) / max(float(day_df["high"].max() - day_df["low"].min()), 1e-6)) * 100.0
        if close_pos_pct <= 30:
            close_quality = "weak close"
        elif close_pos_pct >= 70:
            close_quality = "strong close"
        else:
            close_quality = "normal close"
        eod_sentence = (
            f"End of Day Summary ({_phase_status_text(eod_final)} at selected trading-day close): Asia was {asia_actual.lower()}, London was {london_actual.lower()}, "
            f"and NY was {ny_actual.lower()}, while the full-day confluence model expected a {overall.lower()} outcome at {confidence:.0%} confidence; "
            f"{eod_alignment} The selected trading day closed with {day_actual.lower()} direction and profile shape {profile_shape}. "
            f"Daily bias validation was {bias_truth} (daily bias: {daily_bias}, realized day direction: {day_actual}). "
            f"Close quality was {close_quality} at {close_pos_pct:.1f}% of the session range."
        )
        st.write(eod_sentence)

        next_session_bias, _, _ = _combine_directional_components(
            [
                (day_actual, 0.55),
                (daily_bias, daily_bias_conf),
                (ny_actual, 0.50),
                ("Bullish" if close_pos_pct >= 70 else "Bearish" if close_pos_pct <= 30 else "Neutral", 0.45),
            ]
        )
        st.markdown("**Next Session Outlook**")
        st.write(
            f"Based on session statistics, close quality ({close_quality}), and profile posture ({profile_shape}), "
            f"the next session opens with a {next_session_bias.lower()} expectation. "
            "Treat this as inherited context and refresh it with live overnight structure."
        )
        next_day_key = f"ms_inherited_outlook::{_next_trading_day(selected_date).isoformat()}"
        st.session_state[next_day_key] = (
            f"Prior session ended {day_actual.lower()} with a {close_quality} ({close_pos_pct:.1f}% range close), "
            f"profile shape {profile_shape}, and next-session directional lean {next_session_bias.lower()}."
        )
