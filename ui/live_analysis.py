import datetime as dt
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

from data.data_fetcher import fetch_intraday_ohlcv
from data.session_reference import get_session_windows_for_date
from engines.sessions import compute_session_stats
from engines.patterns import detect_patterns
from engines.bias import build_bias, anchored_vwap_anchor_times, build_anchored_vwap_rows
from engines.structure import detect_market_structure
from engines.accuracy import evaluate_bias_accuracy
from engines.zones import (
    build_htf_zones,
    find_rejection_candles,
    is_zone_touched,
    is_fvg_inversed,
    score_zone_setup,
    summarize_zone_confluence,
    zone_liquidity_scores,
    zone_size_points,
)
from indicators.moving_averages import compute_daily_vwap, compute_weekly_vwap
from indicators.statistics import zscore
from indicators.volatility import rolling_volatility, atr_like
from indicators.momentum import roc, trend_strength
from indicators.volume import rvol
from storage.history_manager import (
    BiasSummary,
    DaySummary,
    save_day_summary,
    load_all_summaries,
)
from engines.manifold import (
    apply_standardization,
    build_feature_matrix,
    build_session_feature_vector,
    cluster_labels,
    describe_regimes,
    isomap_embedding,
    kmeans_fit,
    kmeans_predict,
    nearest_neighbors,
    pca_embedding,
    spd_covariance_from_intraday,
    spd_log_euclidean_distance,
    standardize_features,
)

try:
    from engines.trade_suggestions import build_trade_suggestion
except Exception:
    def build_trade_suggestion(bias):
        return type("Suggestion", (), {"action": "HOLD", "rationale": "Trade suggestion module not available"})


def classify_range_size(curr_range: float, prev_range: float) -> str:
    try:
        if prev_range is None or prev_range == 0:
            return "unknown"
        ratio = curr_range / prev_range
        if ratio >= 1.5:
            return "huge"
        if ratio >= 1.2:
            return "big"
        if ratio >= 0.8:
            return "medium"
        return "small"
    except Exception:
        return "unknown"


def session_trend_label(sess) -> str:
    if not sess:
        return "n/a"
    try:
        move = sess.close - sess.open
        threshold = 0.1 * sess.range if getattr(sess, "range", None) else 0
        if abs(move) < threshold:
            return "consolidation"
        return "uptrend" if move > 0 else "downtrend"
    except Exception:
        return "n/a"


def trend_day_direction(sessions) -> str:
    london = sessions.get("London") if sessions else None
    us = sessions.get("US") if sessions else None
    if not london or not us:
        return "n/a"
    direction = (london.close - london.open) + (us.close - us.open)
    if direction > 0:
        return "Bullish"
    if direction < 0:
        return "Bearish"
    return "Neutral"


def first_break(df: pd.DataFrame, level: float, above: bool = True):
    if level is None or df.empty:
        return None
    if above:
        m = df[df["close"] > level]
    else:
        m = df[df["close"] < level]
    if m.empty:
        return None
    r = m.iloc[0]
    return (r.get("timestamp", None), float(r["close"]))


def first_break_in_window(
    df: pd.DataFrame,
    level: float,
    start: dt.datetime,
    end: dt.datetime,
    above: bool = True,
):
    if level is None or df.empty:
        return None
    m = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
    return first_break(m, level, above=above)


def format_break(event) -> str:
    if not event:
        return "No break"
    ts, price = event
    if ts is None:
        return "Break detected"
    return f"{ts:%Y-%m-%d %H:%M} @ {price:.2f}"


def session_range_std(df: pd.DataFrame, start: dt.datetime, end: dt.datetime):
    if df.empty:
        return None
    sdf = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
    if len(sdf) < 2:
        return None
    return float((sdf["high"] - sdf["low"]).std())


def _level_rejection(last_row: pd.Series, level: float):
    if last_row is None or level is None:
        return None
    if last_row["low"] <= level <= last_row["high"]:
        if last_row["close"] > level and last_row["open"] > level:
            return "Bullish"
        if last_row["close"] < level and last_row["open"] < level:
            return "Bearish"
    return None


def _direction_label(value: float) -> str:
    if value > 0:
        return "Bullish"
    if value < 0:
        return "Bearish"
    return "Neutral"


def get_next_trading_day(date: dt.date) -> dt.date:
    next_date = date + dt.timedelta(days=1)
    while next_date.weekday() >= 5:
        next_date += dt.timedelta(days=1)
    return next_date


def get_prev_trading_day(date: dt.date) -> dt.date:
    prev_date = date - dt.timedelta(days=1)
    while prev_date.weekday() >= 5:
        prev_date -= dt.timedelta(days=1)
    return prev_date


def _now_et() -> dt.datetime:
    if ZoneInfo is None:
        return dt.datetime.now()
    return dt.datetime.now(ZoneInfo("America/New_York")).replace(tzinfo=None)


def trading_day_window(trading_date: dt.date) -> tuple:
    start = dt.datetime.combine(trading_date - dt.timedelta(days=1), dt.time(18, 0))
    end = dt.datetime.combine(trading_date, dt.time(17, 0))
    return start, end


def trading_date_for_timestamp(ts: dt.datetime) -> dt.date:
    date = ts.date()
    if ts >= dt.datetime.combine(date, dt.time(18, 0)):
        date = date + dt.timedelta(days=1)
    while date.weekday() >= 5:
        date += dt.timedelta(days=1)
    return date


def compute_trading_day_extremes(df: pd.DataFrame, trading_date: dt.date):
    if df is None or df.empty:
        return (None, None, None, None)
    start = dt.datetime.combine(trading_date - dt.timedelta(days=1), dt.time(18, 0))
    end = dt.datetime.combine(trading_date, dt.time(17, 0))
    mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
    sdf = df.loc[mask]
    if sdf.empty:
        return (None, None, start, end)
    return (float(sdf["high"].max()), float(sdf["low"].min()), start, end)


def build_preview_bias(
    df_prev: pd.DataFrame,
    sessions,
    prev_sessions=None,
    zone_confluence=None,
) -> BiasSummary:
    prev_label = "Neutral"
    if df_prev is not None and not df_prev.empty:
        prev_open = df_prev["open"].iloc[0]
        prev_close = df_prev["close"].iloc[-1]
        prev_label = _direction_label(prev_close - prev_open)

    asia = sessions.get("Asia")
    london = sessions.get("London")
    votes = [prev_label]
    if asia:
        votes.append(_direction_label(asia.close - asia.open))
    if london:
        votes.append(_direction_label(london.close - london.open))

    daily_bias = max(set(votes), key=votes.count)
    daily_conf = 0.35 + (0.1 if asia else 0.0) + (0.1 if london else 0.0)
    daily_conf = min(0.6, daily_conf)

    explanation = (
        "Pre-session bias based on the previous trading day. "
        "Asia and London will update the bias as their sessions complete."
    )

    if df_prev is not None and not df_prev.empty and prev_sessions:
        prev_patterns = detect_patterns(prev_sessions, df_prev, None)
        next_bias = getattr(prev_patterns, "power_hour_bias", None)
        if next_bias in ("Bullish", "Bearish"):
            if daily_bias == "Neutral":
                daily_bias = next_bias
                daily_conf += 0.05
                explanation += f" Late-day power hour suggests {next_bias} next-day bias."
            elif daily_bias == next_bias:
                daily_conf += 0.03
                explanation += f" Power hour aligned with next-day {next_bias} bias."
        vwap_bias = getattr(prev_patterns, "vwap_reclaim_reject_bias", None)
        if vwap_bias in ("Bullish", "Bearish"):
            if daily_bias == "Neutral":
                daily_bias = vwap_bias
                daily_conf += 0.05
                explanation += f" VWAP reclaim/reject suggests {vwap_bias} next-day bias."
            elif daily_bias == vwap_bias:
                daily_conf += 0.03
                explanation += f" VWAP reclaim/reject aligned with {vwap_bias} bias."

    if zone_confluence is not None and zone_confluence.bias != "Neutral":
        score = abs(zone_confluence.score)
        daily_conf += min(0.1, 0.03 * score)
        if score >= 2 and daily_bias != zone_confluence.bias:
            daily_bias = zone_confluence.bias
        explanation += (
            f" HTF zones suggest {zone_confluence.bias} bias "
            f"(score {zone_confluence.score:.2f}; {zone_confluence.notes})."
        )

    daily_conf = min(0.75, max(0.1, daily_conf))

    return BiasSummary(
        daily_bias=daily_bias,
        daily_confidence=daily_conf,
        us_open_bias="Neutral",
        us_open_confidence=daily_conf,
        us_open_bias_30="Neutral",
        us_open_confidence_30=daily_conf,
        us_open_bias_60="Neutral",
        us_open_confidence_60=daily_conf,
        explanation=explanation,
        vwap_comment="VWAP posture will update once intraday data is available.",
        news_comment="News signal will update with the live feed.",
    )


def main():
    st.title("Live Analysis")

    now = _now_et()
    today = trading_date_for_timestamp(now)
    symbol = st.sidebar.text_input("Symbol", value="NQH26")
    selected_date = st.sidebar.date_input("Analysis date", value=today)
    auto_advance = st.sidebar.checkbox(
        "Auto-advance after close",
        value=True,
        help="Advance to the next trading day after US close + 1 hour.",
    )

    try:
        selected_windows = get_session_windows_for_date(selected_date)
    except Exception:
        selected_windows = {}

    effective_date = selected_date
    auto_advanced = False
    if (
        auto_advance
        and selected_date == today
        and selected_windows
        and "US" in selected_windows
    ):
        _, trading_end = trading_day_window(selected_date)
        if trading_end and now > trading_end + dt.timedelta(hours=1):
            effective_date = get_next_trading_day(selected_date)
            auto_advanced = True

    if auto_advanced:
        st.info(
            f"Trading day ended. Auto-advanced to {effective_date.isoformat()} for next-day prep."
        )

    # Fetch today's and previous day's data (robust to different return shapes)
    res_today = fetch_intraday_ohlcv(symbol, effective_date)
    if isinstance(res_today, tuple):
        df_today, used_ticker = res_today
    else:
        df_today = res_today
        used_ticker = ""

    prev_date = get_prev_trading_day(effective_date)
    res_prev = fetch_intraday_ohlcv(symbol, prev_date)
    if isinstance(res_prev, tuple):
        df_prev, _ = res_prev
    else:
        df_prev = res_prev

    st.sidebar.write(f"Data source ticker: {used_ticker}")

    has_today_data = df_today is not None and not df_today.empty
    if not has_today_data and (df_prev is None or df_prev.empty):
        st.warning("No intraday data available for the selected symbol/date.")
        return

    # Ensure timestamp column exists and is datetime
    day_high = None
    day_low = None

    if has_today_data:
        if "timestamp" in df_today.columns:
            df_today["timestamp"] = pd.to_datetime(df_today["timestamp"])
        else:
            df_today = df_today.reset_index().rename(columns={df_today.index.name or "index": "timestamp"})
            df_today["timestamp"] = pd.to_datetime(df_today["timestamp"])

    if df_prev is not None and not df_prev.empty and "timestamp" in df_prev.columns:
        df_prev["timestamp"] = pd.to_datetime(df_prev["timestamp"])

    # Session windows and stats
    try:
        windows = get_session_windows_for_date(effective_date)
    except Exception:
        windows = {}

    if has_today_data and df_prev is not None and not df_prev.empty:
        df_sessions_source = pd.concat([df_prev, df_today], ignore_index=True).sort_values("timestamp")
    else:
        df_sessions_source = df_today if has_today_data else df_prev

    sessions = compute_session_stats(df_sessions_source, effective_date) if df_sessions_source is not None else {}
    prev_sessions = compute_session_stats(df_prev, prev_date) if df_prev is not None and not df_prev.empty else {}

    htf_zones = build_htf_zones(df_sessions_source) if df_sessions_source is not None else []

    if has_today_data:
        # Price chart
        st.subheader(f"Price Chart ({effective_date.isoformat()})")
        chart_df = df_today[["timestamp", "close"]].copy()
        zoom_bars = min(200, len(chart_df))
        chart_view = chart_df.tail(zoom_bars)
        y_min = float(chart_view["close"].min())
        y_max = float(chart_view["close"].max())
        pad = max((y_max - y_min) * 0.05, 1e-6)
        y_domain = [y_min - pad, y_max + pad]

        base = alt.Chart(chart_df)

        brush = alt.selection_interval(encodings=["x"])

        price_layer = (
            base.mark_line()
            .encode(
                x=alt.X("timestamp:T", title="Time", scale=alt.Scale(domain=brush)),
                y=alt.Y("close:Q", title="Price", scale=alt.Scale(zero=False, domain=y_domain)),
                tooltip=[
                    alt.Tooltip("timestamp:T", title="Close Time", format="%Y-%m-%d %H:%M:%S"),
                    alt.Tooltip("close:Q", title="Close", format=".2f"),
                ],
            )
            .properties(height=420)
        )

        main = price_layer

        overview = (
            base.mark_area(opacity=0.25)
            .encode(
                x=alt.X("timestamp:T", title=""),
                y=alt.Y("close:Q", title=""),
                tooltip=[
                    alt.Tooltip("timestamp:T", title="Close Time", format="%Y-%m-%d %H:%M:%S"),
                    alt.Tooltip("close:Q", title="Close", format=".2f"),
                ],
            )
            .properties(height=80)
            .add_params(brush)
        )

        chart = alt.vconcat(main, overview).resolve_scale(y="independent")

        st.altair_chart(chart, use_container_width=True)
        st.caption(f"Showing last {zoom_bars} bars with a tighter y-axis; drag below to zoom.")
    else:
        st.subheader(f"Next Trading Day Prep ({effective_date.isoformat()})")
        st.info("No intraday data yet. Biases will update automatically as sessions complete.")

    # Quick numeric summary
    if has_today_data:
        try:
            last_price = float(df_today["close"].iloc[-1])
            open_price = float(df_today["open"].iloc[0])
            high_price = float(df_today["high"].max())
            low_price = float(df_today["low"].min())
            total_vol = float(df_today["volume"].sum()) if "volume" in df_today.columns else 0.0
            bars = len(df_today)
        except Exception:
            st.error("Malformed price data.")
            return

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Last Price", f"{last_price:.2f}", delta=f"{(last_price - open_price):.2f}")
        c2.metric("Day Range", f"{high_price - low_price:.2f}", delta=f"High {high_price:.2f} / Low {low_price:.2f}")
        c3.metric("Volume (sum)", f"{total_vol:.0f}", delta=f"Bars: {bars}")
        c4.metric("Bars", f"{bars}")
    elif df_prev is not None and not df_prev.empty:
        prev_open = float(df_prev["open"].iloc[0])
        prev_close = float(df_prev["close"].iloc[-1])
        prev_high = float(df_prev["high"].max())
        prev_low = float(df_prev["low"].min())
        prev_vol = float(df_prev["volume"].sum()) if "volume" in df_prev.columns else 0.0
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Prev Close", f"{prev_close:.2f}", delta=f"{(prev_close - prev_open):.2f}")
        p2.metric("Prev Range", f"{prev_high - prev_low:.2f}", delta=f"High {prev_high:.2f} / Low {prev_low:.2f}")
        p3.metric("Prev Volume", f"{prev_vol:.0f}")
        p4.metric("Prev Date", prev_date.isoformat())

    # Session Stats
    st.markdown("### Session Stats")
    cols = st.columns(3)
    if windows and has_today_data and df_sessions_source is not None:
        missing_sessions = []
        for name in ["Asia", "London", "US"]:
            win = windows.get(name)
            if not win:
                continue
            count = df_sessions_source[
                (df_sessions_source["timestamp"] >= win["start"])
                & (df_sessions_source["timestamp"] <= win["end"])
            ].shape[0]
            if count == 0:
                missing_sessions.append(name)
        if missing_sessions:
            st.caption(
                f"Data feed is missing bars for: {', '.join(missing_sessions)} session(s) on this date."
            )
    for i, name in enumerate(["Asia", "London", "US"]):
        with cols[i]:
            win = windows.get(name)
            sess = sessions.get(name)
            prev = prev_sessions.get(name)
            st.subheader(name)
            if win:
                if now < win.get("start", now):
                    st.info("Session not started yet.")
                elif now < win.get("end", now):
                    st.warning("Session not finished — data may be partial.")
            if sess:
                try:
                    rng_std = (
                        session_range_std(df_sessions_source, win["start"], win["end"]) if win else None
                    )
                    st.metric("Open", f"{sess.open:.2f}")
                    st.metric("Close", f"{sess.close:.2f}")
                    st.write(f"High: {sess.high:.2f}")
                    st.write(f"Low: {sess.low:.2f}")
                    st.write(f"Range: {sess.range:.2f} ({classify_range_size(getattr(sess,'range',0), getattr(prev,'range', None))})")
                    st.write(f"Range Std: {rng_std:.2f}" if rng_std is not None else "Range Std: n/a")
                    st.write(f"Volume: {getattr(sess,'volume',0):.0f}")
                    st.write(f"Price action: {session_trend_label(sess)}")
                except Exception:
                    st.write("Session data present but could not be displayed.")
            else:
                st.write("No data for this session yet.")

    # Patterns
    patterns = detect_patterns(sessions, df_today if has_today_data else None, df_prev)
    st.markdown("### Structural Patterns")
    st.write(f"London Breakout: {getattr(patterns,'london_breakout', False)}")
    st.write(f"Whipsaw: {getattr(patterns,'whipsaw', False)}")
    trend_label = ""
    if getattr(patterns, "trend_day", False):
        trend_label = f" ({trend_day_direction(sessions)})"
    st.write(f"Trend Day: {getattr(patterns,'trend_day', False)}{trend_label}")
    st.write(f"Volatility Expansion: {getattr(patterns,'volatility_expansion', False)}")
    st.write(f"Asia Range Hold: {getattr(patterns,'asia_range_hold', False)}")
    st.write(f"Asia Range Sweep: {getattr(patterns,'asia_range_sweep', False)} ({getattr(patterns,'asia_range_sweep_bias', 'n/a')})")
    st.write(f"London Continuation: {getattr(patterns,'london_continuation', False)} ({getattr(patterns,'london_continuation_bias', 'n/a')})")
    st.write(f"US Open Gap Fill: {getattr(patterns,'us_open_gap_fill', False)} ({getattr(patterns,'us_open_gap_fill_bias', 'n/a')})")
    st.write(f"ORB 30m: {getattr(patterns,'orb_30', False)} ({getattr(patterns,'orb_30_bias', 'n/a')})")
    st.write(f"ORB 60m: {getattr(patterns,'orb_60', False)} ({getattr(patterns,'orb_60_bias', 'n/a')})")
    st.write(f"Failed ORB 30m: {getattr(patterns,'failed_orb_30', False)} ({getattr(patterns,'failed_orb_30_bias', 'n/a')})")
    st.write(f"Failed ORB 60m: {getattr(patterns,'failed_orb_60', False)} ({getattr(patterns,'failed_orb_60_bias', 'n/a')})")
    st.write(f"Power Hour Trend: {getattr(patterns,'power_hour_trend', False)} ({getattr(patterns,'power_hour_bias', 'n/a')})")
    st.write(f"VWAP Reclaim/Reject: {getattr(patterns,'vwap_reclaim_reject', False)} ({getattr(patterns,'vwap_reclaim_reject_bias', 'n/a')})")
    if getattr(patterns, "notes", None):
        st.info(patterns.notes)

    # HTF zones and confluence
    last_ref_price = None
    if has_today_data:
        last_ref_price = last_price
    elif df_prev is not None and not df_prev.empty:
        last_ref_price = prev_close

    zone_confluence = summarize_zone_confluence(htf_zones, last_ref_price)

    zone_rejection_hits = 0
    for z in htf_zones:
        zone_rejection_hits += len(find_rejection_candles(df_sessions_source, z))

    trend_bias = "Neutral"
    try:
        if df_sessions_source is not None and len(df_sessions_source) >= 20:
            trend_val = float(trend_strength(df_sessions_source["close"]))
            if trend_val > 0:
                trend_bias = "Bullish"
            elif trend_val < 0:
                trend_bias = "Bearish"
    except Exception:
        trend_bias = "Neutral"

    st.markdown("### HTF Zones")
    st.write(
        "Order blocks, fair value gaps, and breaker blocks can act as supply/demand zones. "
        "When price revisits them, resting orders and liquidity can encourage reaction or continuation. "
        "Liquidity can be estimated two ways: swing-line density (more prior highs/lows nearby) "
        "and volume intensity (more traded volume inside the zone)."
    )
    st.write(
        f"Confluence bias: {zone_confluence.bias} (points {zone_confluence.score:.2f})"
    )
    st.caption(zone_confluence.notes)
    st.caption(
        f"Rejection candles after zone formation: {zone_rejection_hits}."
    )

    zone_rows = []
    for z in htf_zones:
        touched = is_zone_touched(df_sessions_source, z)
        inversed = is_fvg_inversed(df_sessions_source, z) if z.kind == "fvg" else None
        density, vol_score = zone_liquidity_scores(df_sessions_source, z)
        setup_score = score_zone_setup(
            z,
            last_ref_price,
            touched=touched,
            liquidity_density=density,
            volume_score=vol_score,
        )
        zone_rows.append(
            {
                "Kind": z.kind,
                "Side": z.side,
                "TF": z.timeframe,
                "Range": f"{z.low:.2f} - {z.high:.2f}",
                "Size (pts)": zone_size_points(z),
                "Touched": "Yes" if touched else "No",
                "Inversed": "Yes" if inversed is True else "No" if inversed is False else "n/a",
                "Liquidity Lines": float(density),
                "Volume Intensity": float(vol_score),
                "Setup Score": float(setup_score),
                "Formed": z.start.strftime("%Y-%m-%d %H:%M"),
            }
        )

    if zone_rows:
        zones_df = pd.DataFrame(zone_rows)
        if trend_bias == "Bullish":
            retracement = zones_df[zones_df["Side"] == "bearish"]
            continuation = zones_df[zones_df["Side"] == "bullish"]
        elif trend_bias == "Bearish":
            retracement = zones_df[zones_df["Side"] == "bullish"]
            continuation = zones_df[zones_df["Side"] == "bearish"]
        else:
            retracement = zones_df
            continuation = zones_df

        st.subheader("Top Retracement Zones")
        st.caption(f"Trend context: {trend_bias}. Retracement zones oppose the trend.")
        st.dataframe(retracement.sort_values("Setup Score", ascending=False).head(6), use_container_width=True)

        st.subheader("Top Continuation Zones")
        st.caption(f"Trend context: {trend_bias}. Continuation zones align with the trend.")
        st.dataframe(continuation.sort_values("Setup Score", ascending=False).head(6), use_container_width=True)

        st.subheader("Highest Liquidity Zones")
        st.caption(
            "Liquidity lines show where prior highs/lows cluster; volume intensity shows where more trading occurred."
        )
        liquidity_sorted = zones_df.copy()
        liquidity_sorted["Liquidity Score"] = (
            liquidity_sorted["Liquidity Lines"] * 0.2
            + liquidity_sorted["Volume Intensity"]
        )
        st.dataframe(liquidity_sorted.sort_values("Liquidity Score", ascending=False).head(6), use_container_width=True)
    else:
        st.info("No HTF zones detected yet.")

    # Bias and trade suggestion
    p_low = float(st.session_state.get("vol_p_low", 0.30))
    p_high = float(st.session_state.get("vol_p_high", 0.70))
    if has_today_data:
        bias = build_bias(
            df_today,
            df_prev,
            sessions,
            zone_confluence=zone_confluence,
            vol_thresholds=(p_low, p_high),
        )
    else:
        bias = build_preview_bias(df_prev, sessions, prev_sessions, zone_confluence=zone_confluence)
    st.markdown("### Bias")
    st.write(f"Daily Bias: {getattr(bias,'daily_bias', 'n/a')} ({getattr(bias,'daily_confidence', 0):.0%})")
    us_30 = getattr(bias, "us_open_bias_30", None) or getattr(bias, "us_open_bias", "n/a")
    us_60 = getattr(bias, "us_open_bias_60", None) or getattr(bias, "us_open_bias", "n/a")
    us_30_conf = getattr(bias, "us_open_confidence_30", None)
    us_60_conf = getattr(bias, "us_open_confidence_60", None)
    if us_30_conf is None:
        us_30_conf = getattr(bias, "us_open_confidence", 0)
    if us_60_conf is None:
        us_60_conf = getattr(bias, "us_open_confidence", 0)
    st.write(f"US Open Bias 30m: {us_30} ({us_30_conf:.0%})")
    st.write(f"US Open Bias 60m: {us_60} ({us_60_conf:.0%})")
    st.caption("Finalized at 09:10 ET using overnight range, gap, VWAP, and premarket trend.")
    st.write(f"VWAP Posture: {getattr(bias,'vwap_comment', '')}")
    st.write(f"News Effect: {getattr(bias,'news_comment', '')}")
    if getattr(bias, "amd_summary", None):
        st.write(f"AMD: {bias.amd_summary}")
    if getattr(bias, "explanation", None):
        st.info(bias.explanation)

    suggestion = build_trade_suggestion(bias)
    st.markdown("### Trade Suggestion")
    st.write(f"**Action:** {getattr(suggestion,'action','HOLD')} ")
    st.write(getattr(suggestion, "rationale", ""))

    if has_today_data:
        st.markdown("### Market Structure (BOS)")
        structure = detect_market_structure(df_today)
        bos_time = structure.bos_time.strftime("%Y-%m-%d %H:%M") if structure.bos_time is not None else "n/a"
        bos_level = f"{structure.bos_level:.2f}" if structure.bos_level is not None else "n/a"
        bos_price = f"{structure.bos_price:.2f}" if structure.bos_price is not None else "n/a"
        if structure.bos_1m == "Bullish":
            bos_side = "Upside"
        elif structure.bos_1m == "Bearish":
            bos_side = "Downside"
        else:
            bos_side = "None"
        structure_rows = [
            {
                "Timeframe": "1m",
                "BOS Direction": bos_side,
                "BOS Time": bos_time,
                "BOS Level": bos_level,
                "BOS Close": bos_price,
                "15m Trend": structure.trend_15m,
                "1m vs 15m": structure.alignment,
            }
        ]
        st.dataframe(pd.DataFrame(structure_rows), use_container_width=True)

    if has_today_data:
        # Break detection
        st.markdown("### Range Breaks")
        prev_high = df_prev["high"].max() if df_prev is not None and not df_prev.empty else None
        prev_low = df_prev["low"].min() if df_prev is not None and not df_prev.empty else None
        asia_high = getattr(sessions.get("Asia"), "high", None)
        asia_low = getattr(sessions.get("Asia"), "low", None)

        day_start = windows.get("Asia", {}).get("start") if windows else None
        day_end = windows.get("US", {}).get("end") if windows else None
        london_start = windows.get("London", {}).get("start") if windows else None
        london_end = windows.get("London", {}).get("end") if windows else None

        ph = (
            first_break_in_window(df_sessions_source, prev_high, day_start, day_end, above=True)
            if day_start and day_end
            else None
        )
        pl = (
            first_break_in_window(df_sessions_source, prev_low, day_start, day_end, above=False)
            if day_start and day_end
            else None
        )

        lh = (
            first_break_in_window(df_sessions_source, asia_high, london_start, london_end, above=True)
            if london_start and london_end
            else None
        )
        ll = (
            first_break_in_window(df_sessions_source, asia_low, london_start, london_end, above=False)
            if london_start and london_end
            else None
        )

        st.write(f"Prev day high break: {format_break(ph)}")
        st.write(f"Prev day low break: {format_break(pl)}")
        st.write(f"London broke Asia high: {format_break(lh)}")
        st.write(f"London broke Asia low: {format_break(ll)}")

    if has_today_data:
        day_high, day_low, day_start, day_end = compute_trading_day_extremes(
            df_sessions_source, effective_date
        )
        _, cutoff = trading_day_window(effective_date)
        st.markdown("### Daily High/Low (End-of-Day)")
        if now >= cutoff or effective_date < today:
            if day_high is not None and day_low is not None:
                st.write(f"High: {day_high:.2f}")
                st.write(f"Low: {day_low:.2f}")
                if day_start and day_end:
                    st.caption(f"Window: {day_start:%Y-%m-%d %H:%M} to {day_end:%Y-%m-%d %H:%M} (ET)")
            else:
                st.info("Not enough data to compute the trading-day high/low yet.")
        else:
            st.info("Trading day in progress — daily high/low will finalize at 17:00 ET.")

    if has_today_data:
        # VWAP / moving averages
        st.markdown("### VWAP and Moving Averages")
        dvwap = compute_daily_vwap(df_today)
        wvwap = compute_weekly_vwap(df_today)
        last_dvwap = float(dvwap.iloc[-1]) if len(dvwap) else None
        last_wvwap = float(wvwap.iloc[-1]) if len(wvwap) else None
        sma_20 = df_today["close"].rolling(20).mean()
        sma_50 = df_today["close"].rolling(50).mean()
        last_sma20 = float(sma_20.iloc[-1]) if len(sma_20) else None
        last_sma50 = float(sma_50.iloc[-1]) if len(sma_50) else None

        v1, v2, v3, v4 = st.columns(4)
        with v1:
            v1.metric("Daily VWAP", f"{last_dvwap:.2f}" if last_dvwap is not None else "n/a")
            if last_dvwap is not None:
                stance = "bullish" if last_price > last_dvwap else "bearish"
                st.caption(f"Price is {'above' if last_price > last_dvwap else 'below'} daily VWAP ({stance}).")
                st.caption(
                    f"Key: open {'above' if open_price > last_dvwap else 'below'} VWAP = {'bullish' if open_price > last_dvwap else 'bearish'} bias."
                )
        with v2:
            v2.metric("Weekly VWAP", f"{last_wvwap:.2f}" if last_wvwap is not None else "n/a")
            if last_wvwap is not None:
                stance = "bullish" if last_price > last_wvwap else "bearish"
                st.caption(f"Price is {'above' if last_price > last_wvwap else 'below'} weekly VWAP ({stance}).")
                st.caption(
                    f"Key: open {'above' if open_price > last_wvwap else 'below'} VWAP = {'bullish' if open_price > last_wvwap else 'bearish'} bias."
                )
        with v3:
            v3.metric("SMA 20", f"{last_sma20:.2f}" if last_sma20 is not None else "n/a")
            if last_sma20 is not None:
                stance = "bullish" if last_price > last_sma20 else "bearish"
                st.caption(f"Price is {'above' if last_price > last_sma20 else 'below'} SMA20 ({stance}).")
                st.caption("Key: above SMA20 = bullish bias; below = bearish; near = neutral.")
        with v4:
            v4.metric("SMA 50", f"{last_sma50:.2f}" if last_sma50 is not None else "n/a")
            if last_sma50 is not None:
                stance = "bullish" if last_price > last_sma50 else "bearish"
                st.caption(f"Price is {'above' if last_price > last_sma50 else 'below'} SMA50 ({stance}).")
                st.caption("Key: above SMA50 = bullish bias; below = bearish; near = neutral.")

        st.markdown("#### Anchored VWAP")
        df_all = None
        if df_prev is not None and not df_prev.empty:
            df_all = pd.concat([df_prev, df_today], ignore_index=True).sort_values("timestamp")
        else:
            df_all = df_today.copy()
        last_price = float(df_today["close"].iloc[-1]) if len(df_today) else None
        atr_series = atr_like(df_today, length=20)
        atr_value = float(atr_series.iloc[-1]) if len(atr_series) and pd.notna(atr_series.iloc[-1]) else None
        anchor_times = anchored_vwap_anchor_times(df_prev, df_today, df_all)
        rows = build_anchored_vwap_rows(df_all, anchor_times, last_price, atr_value)
        if rows:
            avwap_df = pd.DataFrame(rows)
            avwap_df["Anchored VWAP"] = avwap_df["Anchored VWAP"].map(lambda x: f"{x:.2f}")
            avwap_df["Distance (ATR)"] = avwap_df["Distance (ATR)"].map(lambda x: f"{x:.2f}")
            st.dataframe(avwap_df, use_container_width=True)
        else:
            st.info("Anchored VWAP stats unavailable (need enough intraday data).")

    if has_today_data:
        # Indicators / statistics
        st.markdown("### Indicator Numbers")
        try:
            z = float(zscore(df_today["close"]).iloc[-1])
        except Exception:
            z = None
        vol = float(rolling_volatility(df_today["close"]).iloc[-1]) if len(df_today) >= 20 else None
        roc_series = roc(df_today["close"]) if len(df_today) >= 10 else pd.Series(dtype=float)
        roc_last = float(roc_series.iloc[-1]) if len(roc_series) else None
        mom = float(trend_strength(df_today["close"])) if len(df_today) >= 20 else None
        rvol_series = rvol(df_today) if "volume" in df_today.columns else pd.Series(dtype=float)
        rvol_last = float(rvol_series.iloc[-1]) if len(rvol_series) else None

        ic1, ic2, ic3, ic4 = st.columns(4)
        with ic1:
            ic1.metric("Z-score", f"{z:.3f}" if z is not None else "n/a")
            st.caption("Z-score = (last close - 20-bar mean) / 20-bar std dev.")
            st.latex(r"z = (x_t - \mu_{20}) / \sigma_{20}")
            if z is not None:
                st.caption("Key: z > 0 bullish, z < 0 bearish; |z| > 2 is stretched/mean-revert risk.")
        with ic2:
            ic2.metric("Std Dev (20)", f"{vol:.4f}" if vol is not None else "n/a")
            if vol is not None:
                st.caption("Key: rising std dev = expanding risk; falling std dev = compression/breakout potential.")
        with ic3:
            ic3.metric("ROC (10)", f"{roc_last:.4f}" if roc_last is not None else "n/a")
            if roc_last is not None:
                st.caption("Key: ROC > 0 bullish momentum; ROC < 0 bearish momentum; near 0 = flat.")
        with ic4:
            ic4.metric("RVOL (20)", f"{rvol_last:.2f}" if rvol_last is not None else "n/a")
            if rvol_last is not None:
                st.caption("Key: > 1.2 strong participation; < 0.8 thin volume; 0.8-1.2 normal.")

        st.markdown("#### Momentum / Trend")
        st.write(f"Trend strength (slope): {mom:.4f}" if mom is not None else "n/a")
        if mom is not None:
            st.caption("Key: slope > 0 bullish bias; slope < 0 bearish bias; near 0 = range.")

        st.markdown("### Rejection Signals")
        last_row = df_today.iloc[-1]
        rejection_levels = []
        if last_dvwap is not None:
            rejection_levels.append(("Daily VWAP", last_dvwap))
        if last_wvwap is not None:
            rejection_levels.append(("Weekly VWAP", last_wvwap))
        if last_sma20 is not None:
            rejection_levels.append(("SMA 20", last_sma20))
        if last_sma50 is not None:
            rejection_levels.append(("SMA 50", last_sma50))
        prev_high = df_prev["high"].max() if df_prev is not None and not df_prev.empty else None
        prev_low = df_prev["low"].min() if df_prev is not None and not df_prev.empty else None
        if prev_high is not None:
            rejection_levels.append(("Prev High", float(prev_high)))
        if prev_low is not None:
            rejection_levels.append(("Prev Low", float(prev_low)))
        asia_high = getattr(sessions.get("Asia"), "high", None)
        asia_low = getattr(sessions.get("Asia"), "low", None)
        london_high = getattr(sessions.get("London"), "high", None)
        london_low = getattr(sessions.get("London"), "low", None)
        if asia_high is not None:
            rejection_levels.append(("Asia High", float(asia_high)))
        if asia_low is not None:
            rejection_levels.append(("Asia Low", float(asia_low)))
        if london_high is not None:
            rejection_levels.append(("London High", float(london_high)))
        if london_low is not None:
            rejection_levels.append(("London Low", float(london_low)))

        rejection_rows = []
        for name, level in rejection_levels:
            bias = _level_rejection(last_row, level)
            if bias:
                rejection_rows.append({"Level": name, "Price": f"{level:.2f}", "Bias": bias})

        if rejection_rows:
            st.dataframe(pd.DataFrame(rejection_rows), use_container_width=True)
        else:
            st.info("No rejection candles detected on the latest bar.")

        st.markdown("### Indicator and Confluence Summary")
        def _bias_from_level(price: float, level: float) -> str:
            if price > level:
                return "Bullish"
            if price < level:
                return "Bearish"
            return "Neutral"

        summary_rows = []
        if last_dvwap is not None:
            summary_rows.append(
                {"Signal": "Daily VWAP", "Value": f"{last_dvwap:.2f}", "Bias": _bias_from_level(last_price, last_dvwap)}
            )
        if last_wvwap is not None:
            summary_rows.append(
                {"Signal": "Weekly VWAP", "Value": f"{last_wvwap:.2f}", "Bias": _bias_from_level(last_price, last_wvwap)}
            )
        if last_sma20 is not None:
            summary_rows.append(
                {"Signal": "SMA 20", "Value": f"{last_sma20:.2f}", "Bias": _bias_from_level(last_price, last_sma20)}
            )
        if last_sma50 is not None:
            summary_rows.append(
                {"Signal": "SMA 50", "Value": f"{last_sma50:.2f}", "Bias": _bias_from_level(last_price, last_sma50)}
            )
        if z is not None:
            summary_rows.append(
                {"Signal": "Z-score", "Value": f"{z:.3f}", "Bias": "Bullish" if z > 0 else "Bearish" if z < 0 else "Neutral"}
            )
        if vol is not None:
            summary_rows.append(
                {"Signal": "Std Dev (20)", "Value": f"{vol:.4f}", "Bias": "Neutral"}
            )
        if roc_last is not None:
            summary_rows.append(
                {"Signal": "ROC (10)", "Value": f"{roc_last:.4f}", "Bias": "Bullish" if roc_last > 0 else "Bearish" if roc_last < 0 else "Neutral"}
            )
        if rvol_last is not None:
            rvol_bias = "Neutral"
            if rvol_last > 1.2:
                rvol_bias = "Bullish"
            elif rvol_last < 0.8:
                rvol_bias = "Bearish"
            summary_rows.append(
                {"Signal": "RVOL (20)", "Value": f"{rvol_last:.2f}", "Bias": rvol_bias}
            )
        if mom is not None:
            summary_rows.append(
                {"Signal": "Trend Strength", "Value": f"{mom:.4f}", "Bias": "Bullish" if mom > 0 else "Bearish" if mom < 0 else "Neutral"}
            )
        summary_rows.append(
            {
                "Signal": "HTF Zones",
                "Value": f"{zone_confluence.score:.2f} pts",
                "Bias": zone_confluence.bias,
            }
        )

        if rejection_rows:
            bull = len([r for r in rejection_rows if r["Bias"] == "Bullish"])
            bear = len([r for r in rejection_rows if r["Bias"] == "Bearish"])
            rej_bias = "Bullish" if bull > bear else "Bearish" if bear > bull else "Neutral"
            summary_rows.append(
                {"Signal": "Rejection Candles", "Value": f"{len(rejection_rows)} levels", "Bias": rej_bias}
            )

        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

        if summary_rows:
            bias_votes = [row["Bias"] for row in summary_rows if row["Bias"] in ("Bullish", "Bearish")]
            if bias_votes:
                overall_bias = max(set(bias_votes), key=bias_votes.count)
            else:
                overall_bias = "Neutral"
            st.info(
                f"Based on these confluences and indicators, price looks {overall_bias.lower()}."
            )

    st.markdown("### Regime & Manifold Similarity")
    explain = describe_regimes(2)
    st.info(explain["regime_meaning"])
    st.caption(explain["isomap"])
    st.caption(explain["pca"])
    st.caption(explain["spd"])
    summaries = load_all_summaries()
    history = [s for s in summaries if s.symbol == symbol]
    current_vec = build_session_feature_vector(sessions)
    if not history or current_vec is None:
        st.info("Need saved history and session stats to compute regime clustering.")
    else:
        X_hist, labels = build_feature_matrix(history)
        if X_hist.size == 0:
            st.info("No usable historical session vectors found for regime clustering.")
        else:
            X_hist_scaled, mean, std = standardize_features(X_hist)
            current_scaled = apply_standardization(current_vec.reshape(1, -1), mean, std)
            k = int(max(2, min(4, round(np.sqrt(len(X_hist_scaled))))))
            explain = describe_regimes(k)
            st.caption(explain["regime_labels"])
            centroids = kmeans_fit(X_hist_scaled, k)
            if centroids.size:
                hist_clusters = kmeans_predict(X_hist_scaled, centroids)
                current_cluster = int(kmeans_predict(current_scaled, centroids)[0])
                cluster_size = int((hist_clusters == current_cluster).sum())
                st.metric("Regime Cluster", f"Regime {current_cluster + 1}")
                st.caption(f"Cluster size: {cluster_size}/{len(X_hist_scaled)} saved days.")
                cluster_map = cluster_labels(labels, hist_clusters)
                same_regime = cluster_map.get(current_cluster, [])
                if same_regime:
                    with st.expander("Days in the same regime"):
                        st.write(", ".join(same_regime))

            labels_all = labels + ["Current"]
            X_all = np.vstack([X_hist_scaled, current_scaled])

            isomap_emb = isomap_embedding(X_all, n_components=2, n_neighbors=5)
            isomap_neighbors = nearest_neighbors(isomap_emb, labels_all, target_index=len(labels_all) - 1, k=3)
            if isomap_neighbors:
                st.markdown("**Isomap Nearest Days**")
                st.dataframe(
                    pd.DataFrame(
                        [{"Day": n.label, "Distance": f"{n.distance:.4f}"} for n in isomap_neighbors]
                    ),
                    use_container_width=True,
                )
            else:
                st.info("Isomap neighbors unavailable (need scikit-learn and multiple saved days).")

            pca_emb = pca_embedding(X_all, n_components=2)
            pca_neighbors = nearest_neighbors(pca_emb, labels_all, target_index=len(labels_all) - 1, k=3)
            if pca_neighbors:
                st.markdown("**Linear Autoencoder (PCA) Nearest Days**")
                st.dataframe(
                    pd.DataFrame(
                        [{"Day": n.label, "Distance": f"{n.distance:.4f}"} for n in pca_neighbors]
                    ),
                    use_container_width=True,
                )
            else:
                st.info("PCA neighbors unavailable (need scikit-learn and multiple saved days).")

    spd_curr = spd_covariance_from_intraday(df_today) if has_today_data else None
    spd_prev = spd_covariance_from_intraday(df_prev) if df_prev is not None else None
    if spd_curr is not None and spd_prev is not None:
        spd_dist = spd_log_euclidean_distance(spd_curr, spd_prev)
        if spd_dist is not None:
            st.write(f"SPD distance vs prior day: {spd_dist:.4f}")
            st.caption("Lower distance = more similar intraday covariance structure.")
        else:
            st.info("SPD distance unavailable due to shape mismatch.")
    else:
        st.info("SPD distance unavailable (need enough intraday bars for both days).")

    # Accuracy: only compute for past dates
    if bias is None:
        if has_today_data:
            p_low = float(st.session_state.get("vol_p_low", 0.30))
            p_high = float(st.session_state.get("vol_p_high", 0.70))
            bias = build_bias(
                df_today,
                df_prev,
                sessions,
                zone_confluence=zone_confluence,
                vol_thresholds=(p_low, p_high),
            )
        else:
            bias = build_preview_bias(df_prev, sessions, zone_confluence=zone_confluence)

    _, trading_end = trading_day_window(effective_date)
    is_trading_day_done = now >= trading_end
    if (effective_date < today or (effective_date == today and is_trading_day_done)) and has_today_data:
        source_df = df_sessions_source if df_sessions_source is not None else df_today
        accuracy = evaluate_bias_accuracy(source_df, bias, trading_date=effective_date)
        st.markdown("### Daily Bias Accuracy (End-of-Day)")
        st.write(f"Actual Direction: {getattr(accuracy,'actual_direction', 'n/a')}")
        st.write(f"Bias Correct: {getattr(accuracy,'bias_correct', 'n/a')}")
        st.write(f"Used Bias: {getattr(accuracy,'used_bias', 'n/a')}")
        st.write(f"US Open Bias Correct (30m): {getattr(accuracy,'us_open_bias_correct_30', 'n/a')}")
        st.write(f"US Open Bias Correct (60m): {getattr(accuracy,'us_open_bias_correct_60', 'n/a')}")
        if getattr(accuracy, "explanation", None):
            st.info(accuracy.explanation)
    else:
        st.markdown("### Daily Bias Accuracy")
        st.info("Trading day in progress — accuracy will be available after the day ends.")

    # Manual save
    if st.button("Save Today to History"):
        try:
            if effective_date < today and has_today_data:
                source_df = df_sessions_source if df_sessions_source is not None else df_today
                acc = evaluate_bias_accuracy(source_df, bias, trading_date=effective_date)
            else:
                acc = None
            day_summary = DaySummary(
                date=str(effective_date),
                symbol=symbol,
                sessions=sessions,
                patterns=patterns,
                bias=bias,
                trade_suggestion=suggestion,
                accuracy=acc,
                day_high=day_high if has_today_data else None,
                day_low=day_low if has_today_data else None,
            )
            save_day_summary(day_summary)
            st.success("Saved current day summary to history.")
        except Exception as e:
            st.error(f"Failed to save summary: {e}")

    # Automatic archiving: if today is past and more than 1 hour since US close, save to history
    try:
        if (
            selected_date == today
            and not auto_advanced
            and selected_windows
            and "US" in selected_windows
        ):
            us_end = selected_windows["US"].get("end") if isinstance(selected_windows["US"], dict) else None
            if us_end and now > us_end + dt.timedelta(hours=1):
                saved = [s for s in load_all_summaries() if s.date == str(today) and s.symbol == symbol]
                if not saved and has_today_data:
                    source_df = df_sessions_source if df_sessions_source is not None else df_today
                    acc = evaluate_bias_accuracy(source_df, bias, trading_date=today)
                    day_summary = DaySummary(
                        date=str(today),
                        symbol=symbol,
                        sessions=sessions,
                        patterns=patterns,
                        bias=bias,
                        trade_suggestion=suggestion,
                        accuracy=acc,
                        day_high=day_high if has_today_data else None,
                        day_low=day_low if has_today_data else None,
                    )
                    save_day_summary(day_summary)
                    st.success("Trading day archived to history automatically.")
    except Exception:
        pass


def render_live_analysis():
    return main()

if __name__ == "__main__":
    main()
