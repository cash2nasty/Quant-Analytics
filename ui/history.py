import datetime as dt
from typing import Dict, Optional
import numpy as np
import pandas as pd
import streamlit as st
from data.data_fetcher import fetch_intraday_ohlcv, filter_date
from engines.sessions import compute_session_stats
from engines.patterns import detect_patterns
from engines.bias import build_bias, analyze_vwap_posture, anchored_vwap_anchor_times, build_anchored_vwap_rows
from engines.accuracy import evaluate_bias_accuracy
from engines.zones import build_htf_zones, summarize_zone_confluence, resample_ohlcv
from engines.trade_suggestions import build_trade_suggestion
from storage.history_manager import DaySummary, load_all_summaries, save_day_summary
from indicators.volatility import atr_like
from indicators.momentum import roc, trend_strength
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


def _get_prev_day_df(df, date: dt.date):
    prev = date - dt.timedelta(days=1)
    mask = df["timestamp"].dt.date == prev
    return df.loc[mask].copy()


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


def trading_days_between(start_date: dt.date, end_date: dt.date):
    days = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:
            days.append(current)
        current += dt.timedelta(days=1)
    return days


def _direction_label(value: float, threshold: float = 0.0) -> str:
    if value > threshold:
        return "Bullish"
    if value < -threshold:
        return "Bearish"
    return "Neutral"


def _atr_direction_threshold(df_today: pd.DataFrame, df_prev: pd.DataFrame) -> float:
    ref = df_today if df_today is not None and not df_today.empty else df_prev
    if ref is None or ref.empty:
        return 0.0
    atr_series = atr_like(ref, length=20)
    if atr_series.empty or pd.isna(atr_series.iloc[-1]):
        return 0.0
    return float(atr_series.iloc[-1]) * 0.10


def _compute_bias_diagnostics(
    df_day: pd.DataFrame,
    df_prev: pd.DataFrame,
    sessions,
    bias,
    zone_confluence=None,
) -> Optional[Dict[str, object]]:
    if df_day is None or df_day.empty:
        return None

    actual_open = float(df_day["open"].iloc[0])
    actual_close = float(df_day["close"].iloc[-1])
    actual_move = actual_close - actual_open
    actual_dir = _direction_label(actual_move)

    prev_raw_label = "n/a"
    prev_atr_label = "n/a"
    if df_prev is not None and not df_prev.empty:
        prev_open = float(df_prev["open"].iloc[0])
        prev_close = float(df_prev["close"].iloc[-1])
        prev_raw_label = _direction_label(prev_close - prev_open)
        atr_threshold = _atr_direction_threshold(df_day, df_prev)
        prev_atr_label = _direction_label(prev_close - prev_open, threshold=atr_threshold)

    asia = sessions.get("Asia") if sessions else None
    london = sessions.get("London") if sessions else None
    asia_label = _direction_label(asia.close - asia.open) if asia else "Neutral"
    london_label = _direction_label(london.close - london.open) if london else "Neutral"

    us_open_bias = getattr(bias, "us_open_bias_60", None) or getattr(bias, "us_open_bias", "Neutral")

    vwap_posture = analyze_vwap_posture(df_day)
    if vwap_posture.daily_above and vwap_posture.weekly_above:
        vwap_label = "Bullish"
    elif (not vwap_posture.daily_above) and (not vwap_posture.weekly_above):
        vwap_label = "Bearish"
    else:
        vwap_label = "Neutral"

    htf_bias = "Neutral"
    rs_1h = resample_ohlcv(df_day, "1H")
    rs_4h = resample_ohlcv(df_day, "4H")
    if len(rs_1h) >= 10 and len(rs_4h) >= 6:
        htf_1h = float(trend_strength(rs_1h["close"], length=10))
        htf_4h = float(trend_strength(rs_4h["close"], length=6))
        if htf_1h > 0 and htf_4h > 0:
            htf_bias = "Bullish"
        elif htf_1h < 0 and htf_4h < 0:
            htf_bias = "Bearish"

    roc_series = roc(df_day["close"], length=10) if len(df_day) >= 10 else pd.Series(dtype=float)
    last_roc = float(roc_series.iloc[-1]) if len(roc_series) else 0.0
    roc_bias = "Bullish" if last_roc > 0 else "Bearish" if last_roc < 0 else "Neutral"

    zone_bias = "Neutral"
    zone_score = 0.0
    if zone_confluence is not None:
        zone_bias = zone_confluence.bias
        zone_score = float(zone_confluence.score)

    signals = [
        ("Previous day (raw)", prev_raw_label),
        ("Previous day (ATR)", prev_atr_label),
        ("Asia", asia_label),
        ("London", london_label),
        ("US Open", us_open_bias),
        ("VWAP posture", vwap_label),
        ("HTF slope", htf_bias),
        ("ROC", roc_bias),
        ("Zones", zone_bias),
    ]

    def _is_opposite(a: str, b: str) -> bool:
        return (a == "Bullish" and b == "Bearish") or (a == "Bearish" and b == "Bullish")

    support = [name for name, label in signals if label == actual_dir]
    conflict = [name for name, label in signals if _is_opposite(label, actual_dir)]

    rationale = (
        f"The actual move was {actual_dir.lower()}, while the model predicted {getattr(bias, 'daily_bias', 'n/a').lower()}. "
        f"Conflicting signals included {', '.join(conflict) if conflict else 'none'}. "
        f"Supporting signals included {', '.join(support) if support else 'none'}. "
        "This suggests the bias likely missed a reversal or liquidity-driven move where opposing signals outweighed the dominant inputs."
    )

    return {
        "actual_dir": actual_dir,
        "actual_open": actual_open,
        "actual_close": actual_close,
        "predicted": getattr(bias, "daily_bias", "n/a"),
        "support": support,
        "conflict": conflict,
        "zone_score": zone_score,
        "rationale": rationale,
    }


def render_history():
    st.header("History")

    # Allow user to pick symbol and date to view historical analysis
    symbol = st.text_input("Symbol", value="NQH26", help="Futures symbol to analyze")
    today = dt.datetime.now().date()
    selected_date = st.date_input("Select date to view", value=today)

    # First try to load saved summary if it exists
    summaries = load_all_summaries()
    symbol_summaries = [s for s in summaries if s.symbol == symbol]
    if symbol_summaries:
        earliest = min(dt.date.fromisoformat(s.date) for s in symbol_summaries)
        latest = max(dt.date.fromisoformat(s.date) for s in symbol_summaries)
        st.markdown("### Backtest Summary")
        st.caption("Summary can use saved days or recompute from raw data.")
        st.caption("Recompute uses raw data even if the day was never saved.")
        c_start, c_end = st.columns(2)
        with c_start:
            start_date = st.date_input("Start date", value=earliest)
        with c_end:
            end_date = st.date_input("End date", value=latest)
        use_recompute = st.checkbox("Recompute from raw data", value=True)
        if start_date > end_date:
            st.warning("Start date must be on or before end date.")
        else:
            trading_days = trading_days_between(start_date, end_date)
            in_range = [
                s for s in symbol_summaries
                if start_date <= dt.date.fromisoformat(s.date) <= end_date
            ]
            saved_days = {dt.date.fromisoformat(s.date) for s in in_range}
            missing_saved_days = [d for d in trading_days if d not in saved_days]

            scored = []
            used_ticker = ""
            missing_data_days = []
            missing_session_days = []
            if use_recompute:
                max_lookback_days = 59
                limited_start = start_date
                if (end_date - start_date).days > max_lookback_days:
                    limited_start = end_date - dt.timedelta(days=max_lookback_days)
                    st.warning(
                        "Intraday data provider only supports the last 60 days. "
                        f"Recomputing from {limited_start.isoformat()} to {end_date.isoformat()}."
                    )
                    missing_data_days.extend([d for d in trading_days if d < limited_start])

                fetch_start = limited_start - dt.timedelta(days=1)
                df, used_ticker = fetch_intraday_ohlcv(symbol, lookback_days=(fetch_start, end_date))
                if df is None or df.empty:
                    st.warning("No data available to recompute this range.")
                else:
                    if "timestamp" in df.columns:
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                    p_low = float(st.session_state.get("vol_p_low", 0.30))
                    p_high = float(st.session_state.get("vol_p_high", 0.70))
                    for day in trading_days:
                        if day < limited_start:
                            continue
                        df_day = filter_date(df, day)
                        if df_day.empty:
                            missing_data_days.append(day)
                            continue
                        df_prev = _get_prev_day_df(df, day)
                        df_sessions_source = df_day
                        if df_prev is not None and not df_prev.empty:
                            df_sessions_source = (
                                pd.concat([df_prev, df_day], ignore_index=True)
                                .sort_values("timestamp")
                                .reset_index(drop=True)
                            )
                        sessions = compute_session_stats(df_sessions_source, day)
                        if not all(k in sessions for k in ("Asia", "London")):
                            missing_session_days.append((day, "Missing Asia/London session data"))
                            continue
                        patterns = detect_patterns(sessions, df_day, df_prev)
                        zone_confluence = None
                        if df_sessions_source is not None and not df_sessions_source.empty:
                            htf_zones = build_htf_zones(df_sessions_source)
                            last_price = float(df_day["close"].iloc[-1]) if not df_day.empty else None
                            zone_confluence = summarize_zone_confluence(htf_zones, last_price)
                        bias = build_bias(
                            df_day,
                            df_prev,
                            sessions,
                            zone_confluence=zone_confluence,
                            vol_thresholds=(p_low, p_high),
                        )
                        accuracy = evaluate_bias_accuracy(df_sessions_source, bias, trading_date=day)
                        scored.append({"date": day, "accuracy": accuracy, "bias": bias, "patterns": patterns})
            else:
                scored = [
                    {"date": dt.date.fromisoformat(s.date), "accuracy": s.accuracy, "bias": s.bias, "patterns": s.patterns}
                    for s in in_range
                    if s.accuracy
                    and s.accuracy.actual_direction != "n/a"
                    and s.sessions is not None
                    and all(k in s.sessions for k in ("Asia", "London"))
                ]
                missing_session_days.extend(
                    [
                        (dt.date.fromisoformat(s.date), "Missing Asia/London session data")
                        for s in in_range
                        if s.sessions is None or not all(k in s.sessions for k in ("Asia", "London"))
                    ]
                )

            acc_list = [item["accuracy"] for item in scored]
            excluded_daily = [
                item["date"]
                for item in scored
                if item["accuracy"].actual_direction in (None, "n/a")
            ]
            excluded_30 = []
            excluded_60 = []
            for item in scored:
                acc = item["accuracy"]
                bias = item["bias"]
                if acc.us_open_bias_correct_30 is None:
                    reason = "Neutral bias"
                    if getattr(bias, "us_open_bias_30", "Neutral") not in ("Bullish", "Bearish"):
                        reason = "Neutral bias"
                    excluded_30.append((item["date"], reason))
                if acc.us_open_bias_correct_60 is None:
                    reason = "Neutral bias"
                    if getattr(bias, "us_open_bias_60", "Neutral") not in ("Bullish", "Bearish"):
                        reason = "Neutral bias"
                    excluded_60.append((item["date"], reason))
            daily_total = len(acc_list)
            daily_correct = sum(1 for a in acc_list if a.bias_correct)
            us30_total = sum(1 for a in acc_list if a.us_open_bias_correct_30 is not None)
            us30_correct = sum(1 for a in acc_list if a.us_open_bias_correct_30)
            us60_total = sum(1 for a in acc_list if a.us_open_bias_correct_60 is not None)
            us60_correct = sum(1 for a in acc_list if a.us_open_bias_correct_60)

            c1, c2, c3 = st.columns(3)
            with c1:
                if daily_total:
                    st.metric("Daily Bias Accuracy", f"{daily_correct / daily_total:.0%}")
                    st.caption(f"{daily_correct} / {daily_total} days")
                else:
                    st.metric("Daily Bias Accuracy", "n/a")
            with c2:
                if us30_total:
                    st.metric("US Open 30m Accuracy", f"{us30_correct / us30_total:.0%}")
                    st.caption(f"{us30_correct} / {us30_total} days")
                else:
                    st.metric("US Open 30m Accuracy", "n/a")
            with c3:
                if us60_total:
                    st.metric("US Open 60m Accuracy", f"{us60_correct / us60_total:.0%}")
                    st.caption(f"{us60_correct} / {us60_total} days")
                else:
                    st.metric("US Open 60m Accuracy", "n/a")
            st.caption("US Open accuracy uses direction over the first 30/60 minutes after 09:30 ET.")

            c4, c5 = st.columns(2)
            with c4:
                st.metric("Days in Range", f"{len(trading_days)}")
            with c5:
                st.metric("Scored Days", f"{daily_total}")
                if use_recompute:
                    st.metric("Computed Days", f"{daily_total} / {len(trading_days)}")
                    if missing_data_days:
                        missing_preview = ", ".join(d.isoformat() for d in missing_data_days[:5])
                        st.caption(f"Missing data days: {missing_preview}")
                    if used_ticker:
                        st.caption(f"Data source ticker: {used_ticker}")
                else:
                    st.metric("Saved Days", f"{len(in_range)} / {len(trading_days)}")
                    if missing_saved_days:
                        missing_preview = ", ".join(d.isoformat() for d in missing_saved_days[:5])
                        st.caption(f"Missing saved days: {missing_preview}")

            wrong_rows = []
            for item in scored:
                acc = item["accuracy"]
                day = item["date"]
                flags = []
                if acc.bias_correct is False:
                    flags.append("Daily")
                if acc.us_open_bias_correct_30 is False:
                    flags.append("US Open 30m")
                if acc.us_open_bias_correct_60 is False:
                    flags.append("US Open 60m")
                if flags:
                    wrong_rows.append({"Date": day.isoformat(), "Wrong": ", ".join(flags)})

            if wrong_rows:
                st.subheader("Wrong Days")
                st.dataframe(pd.DataFrame(wrong_rows), use_container_width=True)

            # Pattern impact report (flip/boost candidates)
            pattern_rows = []
            pattern_hits = {}
            for item in scored:
                acc = item["accuracy"]
                bias = item["bias"]
                patterns = item["patterns"]
                actual = acc.actual_direction
                pattern_biases = [
                    getattr(patterns, "asia_range_sweep_bias", None),
                    getattr(patterns, "london_continuation_bias", None),
                    getattr(patterns, "us_open_gap_fill_bias", None),
                    getattr(patterns, "orb_30_bias", None),
                    getattr(patterns, "orb_60_bias", None),
                    getattr(patterns, "failed_orb_30_bias", None),
                    getattr(patterns, "failed_orb_60_bias", None),
                    getattr(patterns, "power_hour_bias", None),
                    getattr(patterns, "vwap_reclaim_reject_bias", None),
                ]
                pattern_biases = [p for p in pattern_biases if p in ("Bullish", "Bearish")]
                if not pattern_biases:
                    continue
                pattern_bias = max(set(pattern_biases), key=pattern_biases.count)
                support = sum(1 for p in pattern_biases if p == pattern_bias)

                if pattern_bias != bias.daily_bias and actual == pattern_bias and support >= 2:
                    pattern_rows.append(
                        {
                            "Date": item["date"].isoformat(),
                            "Pattern Bias": pattern_bias,
                            "Outcome": "Could Flip",
                            "Support": support,
                        }
                    )
                if bias.daily_bias == actual and bias.daily_confidence < 0.6 and pattern_bias == bias.daily_bias:
                    pattern_rows.append(
                        {
                            "Date": item["date"].isoformat(),
                            "Pattern Bias": pattern_bias,
                            "Outcome": "Confidence Boost",
                            "Support": support,
                        }
                    )

                for p in pattern_biases:
                    pattern_hits.setdefault(p, {"total": 0, "correct": 0})
                    pattern_hits[p]["total"] += 1
                    if actual == p:
                        pattern_hits[p]["correct"] += 1

            if pattern_rows:
                st.subheader("Pattern Impact Report")
                st.dataframe(pd.DataFrame(pattern_rows), use_container_width=True)

            if pattern_hits:
                weight_rows = []
                for bias_dir, stats in pattern_hits.items():
                    hit_rate = stats["correct"] / max(1, stats["total"])
                    weight_rows.append({"Bias": bias_dir, "Hit Rate": f"{hit_rate:.0%}"})
                st.subheader("Pattern Effectiveness")
                st.dataframe(pd.DataFrame(weight_rows), use_container_width=True)

                weights = {
                    "default": 0.5,
                }
                for bias_dir, stats in pattern_hits.items():
                    hit_rate = stats["correct"] / max(1, stats["total"])
                    weights[bias_dir.lower()] = round(hit_rate, 3)
                try:
                    path = Path("data") / "pattern_weights.json"
                    with open(path, "w") as f:
                        json.dump({"weights": weights}, f, indent=2)
                    st.caption("Updated pattern weights for future bias calculations.")
                except Exception:
                    st.caption("Pattern weights could not be saved.")

            if excluded_daily or excluded_30 or excluded_60 or missing_session_days:
                st.subheader("Excluded Days")
                excluded_rows = []
                for day in excluded_daily:
                    excluded_rows.append({"Date": day.isoformat(), "Excluded From": "Daily", "Reason": "No actual direction"})
                for day, reason in excluded_30:
                    excluded_rows.append({"Date": day.isoformat(), "Excluded From": "US Open 30m", "Reason": reason})
                for day, reason in excluded_60:
                    excluded_rows.append({"Date": day.isoformat(), "Excluded From": "US Open 60m", "Reason": reason})
                for day, reason in missing_session_days:
                    excluded_rows.append({"Date": day.isoformat(), "Excluded From": "Daily", "Reason": reason})
                if excluded_rows:
                    st.dataframe(pd.DataFrame(excluded_rows), use_container_width=True)

            if use_recompute and missing_data_days:
                st.subheader("Missing Feed Data")
                st.caption("These dates have no bars for one or more sessions in the data feed.")
                missing_rows = [{"Date": d.isoformat()} for d in missing_data_days]
                st.dataframe(pd.DataFrame(missing_rows), use_container_width=True)
    else:
        st.info("No saved history for this symbol yet.")
    key = f"{selected_date.isoformat()} | {symbol}"
    saved = None
    for s in summaries:
        if s.date == selected_date.isoformat() and s.symbol == symbol:
            saved = s
            break

    df_day = None
    df_prev = None

    if saved:
        s = saved
        st.subheader(f"Saved Summary for {s.date} ({s.symbol})")
    else:
        # Fetch data covering the requested date and compute on the fly
        df, used_ticker = fetch_intraday_ohlcv(symbol, lookback_days=selected_date)
        df_day = filter_date(df, selected_date)
        if df_day.empty:
            st.info("No live market data available for the selected date.")
            if used_ticker and used_ticker != symbol:
                st.caption(f"Attempted data ticker: {used_ticker}")
            return

        df_prev = _get_prev_day_df(df, selected_date)
        df_sessions_source = df_day
        if df_prev is not None and not df_prev.empty:
            df_sessions_source = (
                pd.concat([df_prev, df_day], ignore_index=True)
                .sort_values("timestamp")
                .reset_index(drop=True)
            )
        sessions = compute_session_stats(df_sessions_source, selected_date)
        patterns = detect_patterns(sessions, df_day, df_prev)
        p_low = float(st.session_state.get("vol_p_low", 0.30))
        p_high = float(st.session_state.get("vol_p_high", 0.70))
        bias = build_bias(df_day, df_prev, sessions, vol_thresholds=(p_low, p_high))
        suggestion = build_trade_suggestion(bias)
        # Only compute accuracy for fully completed historical days
        if selected_date < today:
            accuracy = evaluate_bias_accuracy(df_sessions_source, bias, trading_date=selected_date)
        else:
            accuracy = None

        # Build a lightweight object to reuse existing display code
        class _S:
            pass

        s = _S()
        s.date = selected_date.isoformat()
        s.symbol = symbol
        s.sessions = sessions
        s.patterns = patterns
        s.bias = bias
        s.trade_suggestion = suggestion
        s.accuracy = accuracy
        day_high, day_low, _, _ = compute_trading_day_extremes(df_sessions_source, selected_date)
        s.day_high = day_high
        s.day_low = day_low

        st.subheader(f"Computed Summary for {s.date} ({s.symbol})")
        if used_ticker:
            st.caption(f"Data source ticker: {used_ticker}")

        if st.button("Save This Day to History"):
            if accuracy is None:
                st.warning("Accuracy is only available after the trading day ends.")
            else:
                day_high, day_low, _, _ = compute_trading_day_extremes(df_sessions_source, selected_date)
                day_summary = DaySummary(
                    date=str(selected_date),
                    symbol=symbol,
                    sessions=sessions,
                    patterns=patterns,
                    bias=bias,
                    trade_suggestion=suggestion,
                    accuracy=accuracy,
                    day_high=day_high,
                    day_low=day_low,
                )
                save_day_summary(day_summary)
                st.success("Saved day summary to history.")

    st.markdown("### Sessions")
    if s.sessions:
        cols = st.columns(len(s.sessions))
        for i, (name, sess) in enumerate(s.sessions.items()):
            with cols[i]:
                st.subheader(name)
                st.metric("Open", f"{sess.open:.2f}")
                st.metric("Close", f"{sess.close:.2f}")
                st.write(f"High: {sess.high:.2f}")
                st.write(f"Low: {sess.low:.2f}")
                st.write(f"Range: {sess.range:.2f}")
                st.write(f"Volume: {sess.volume:.0f}")
    else:
        st.write("No session data available for this date.")

    st.markdown("### Patterns")
    p = s.patterns
    st.write(f"London Breakout: {p.london_breakout}")
    st.write(f"Whipsaw: {p.whipsaw}")
    st.write(f"Trend Day: {p.trend_day}")
    st.write(f"Volatility Expansion: {p.volatility_expansion}")
    st.write(f"Asia Range Hold: {getattr(p,'asia_range_hold', False)}")
    st.write(f"Asia Range Sweep: {getattr(p,'asia_range_sweep', False)} ({getattr(p,'asia_range_sweep_bias', 'n/a')})")
    st.write(f"London Continuation: {getattr(p,'london_continuation', False)} ({getattr(p,'london_continuation_bias', 'n/a')})")
    st.write(f"US Open Gap Fill: {getattr(p,'us_open_gap_fill', False)} ({getattr(p,'us_open_gap_fill_bias', 'n/a')})")
    st.write(f"ORB 30m: {getattr(p,'orb_30', False)} ({getattr(p,'orb_30_bias', 'n/a')})")
    st.write(f"ORB 60m: {getattr(p,'orb_60', False)} ({getattr(p,'orb_60_bias', 'n/a')})")
    st.write(f"Failed ORB 30m: {getattr(p,'failed_orb_30', False)} ({getattr(p,'failed_orb_30_bias', 'n/a')})")
    st.write(f"Failed ORB 60m: {getattr(p,'failed_orb_60', False)} ({getattr(p,'failed_orb_60_bias', 'n/a')})")
    st.write(f"Power Hour Trend: {getattr(p,'power_hour_trend', False)} ({getattr(p,'power_hour_bias', 'n/a')})")
    st.write(f"VWAP Reclaim/Reject: {getattr(p,'vwap_reclaim_reject', False)} ({getattr(p,'vwap_reclaim_reject_bias', 'n/a')})")
    st.info(p.notes)

    st.markdown("### Regime & Manifold Similarity")
    explain = describe_regimes(2)
    st.info(explain["regime_meaning"])
    st.caption(explain["isomap"])
    st.caption(explain["pca"])
    st.caption(explain["spd"])
    history = [h for h in summaries if h.symbol == symbol]
    current_vec = build_session_feature_vector(s.sessions)
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

    spd_curr = spd_covariance_from_intraday(df_day) if df_day is not None else None
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

    # Ensure we have prev-day data for bias context display
    if df_day is None or df_prev is None:
        df, _ = fetch_intraday_ohlcv(symbol, lookback_days=selected_date)
        df_day = filter_date(df, selected_date)
        df_prev = _get_prev_day_df(df, selected_date)

    df_sessions_source = None
    zone_confluence = None
    if df_day is not None and not df_day.empty:
        df_sessions_source = df_day
        if df_prev is not None and not df_prev.empty:
            df_sessions_source = (
                pd.concat([df_prev, df_day], ignore_index=True)
                .sort_values("timestamp")
                .reset_index(drop=True)
            )
        htf_zones = build_htf_zones(df_sessions_source)
        last_price = float(df_sessions_source["close"].iloc[-1]) if not df_sessions_source.empty else None
        zone_confluence = summarize_zone_confluence(htf_zones, last_price)

    st.markdown("### Bias")
    b = s.bias
    if df_prev is not None and not df_prev.empty:
        prev_open = float(df_prev["open"].iloc[0])
        prev_close = float(df_prev["close"].iloc[-1])
        prev_move = prev_close - prev_open
        prev_raw_label = _direction_label(prev_move)
        atr_threshold = _atr_direction_threshold(df_day, df_prev)
        prev_atr_label = _direction_label(prev_move, threshold=atr_threshold)
        atr_reason = (
            f"Move {prev_move:+.2f} vs ATR threshold {atr_threshold:.2f} -> {prev_atr_label}"
        )
        st.write(f"Previous Day Direction: {prev_raw_label}")
        st.write(f"Previous Day Direction (ATR-adjusted): {prev_atr_label} ({atr_reason})")
    else:
        st.write("Previous Day Direction: n/a")
        st.write("Previous Day Direction (ATR-adjusted): n/a")
    st.write(f"Daily Bias: {b.daily_bias} ({b.daily_confidence:.0%})")
    us30 = getattr(b, "us_open_bias_30", None) or b.us_open_bias
    us60 = getattr(b, "us_open_bias_60", None) or b.us_open_bias
    us30_conf = getattr(b, "us_open_confidence_30", None)
    us60_conf = getattr(b, "us_open_confidence_60", None)
    if us30_conf is None:
        us30_conf = b.us_open_confidence
    if us60_conf is None:
        us60_conf = b.us_open_confidence
    st.write(f"US Open Bias 30m: {us30} ({us30_conf:.0%})")
    st.write(f"US Open Bias 60m: {us60} ({us60_conf:.0%})")
    st.caption("Finalized at 09:10 ET using overnight range, gap, VWAP, and premarket trend.")
    if getattr(b, "amd_summary", None):
        st.write(f"AMD: {b.amd_summary}")
    # vwap_comment/news_comment may not exist for older saved objects; guard
    if hasattr(b, "vwap_comment"):
        st.write(f"VWAP Comment: {b.vwap_comment}")
    if hasattr(b, "news_comment"):
        st.write(f"News Comment: {b.news_comment}")
    st.info(b.explanation)

    st.markdown("### Anchored VWAP")
    df_all = None
    if df_prev is not None and not df_prev.empty and df_day is not None and not df_day.empty:
        df_all = pd.concat([df_prev, df_day], ignore_index=True).sort_values("timestamp")
    elif df_day is not None and not df_day.empty:
        df_all = df_day.copy()
    last_price = float(df_day["close"].iloc[-1]) if df_day is not None and not df_day.empty else None
    atr_series = atr_like(df_day, length=20) if df_day is not None else pd.Series(dtype=float)
    atr_value = float(atr_series.iloc[-1]) if len(atr_series) and pd.notna(atr_series.iloc[-1]) else None
    anchor_times = anchored_vwap_anchor_times(df_prev, df_day, df_all)
    rows = build_anchored_vwap_rows(df_all, anchor_times, last_price, atr_value)
    if rows:
        avwap_df = pd.DataFrame(rows)
        avwap_df["Anchored VWAP"] = avwap_df["Anchored VWAP"].map(lambda x: f"{x:.2f}")
        avwap_df["Distance (ATR)"] = avwap_df["Distance (ATR)"].map(lambda x: f"{x:.2f}")
        st.dataframe(avwap_df, use_container_width=True)
    else:
        st.info("Anchored VWAP stats unavailable (need enough intraday data).")


    diagnostics = _compute_bias_diagnostics(df_day, df_prev, s.sessions, b, zone_confluence=zone_confluence)
    st.markdown("### Diagnostics")
    if diagnostics is None:
        st.info("Diagnostics unavailable (missing data).")
    else:
        st.write(
            f"Actual direction: {diagnostics['actual_dir']} "
            f"(open {diagnostics['actual_open']:.2f} -> close {diagnostics['actual_close']:.2f})"
        )
        st.write(f"Predicted daily bias: {diagnostics['predicted']}")
        if diagnostics["conflict"]:
            st.write("Conflicting signals: " + ", ".join(diagnostics["conflict"]))
        if diagnostics["support"]:
            st.write("Supporting signals: " + ", ".join(diagnostics["support"]))
        if zone_confluence is not None:
            st.write(
                f"Zone confluence: {zone_confluence.bias} "
                f"(score {zone_confluence.score:.2f}; {zone_confluence.notes})"
            )
        st.write(diagnostics["rationale"])

    st.markdown("### Daily High/Low (End-of-Day)")
    if getattr(s, "day_high", None) is not None and getattr(s, "day_low", None) is not None:
        st.write(f"High: {s.day_high:.2f}")
        st.write(f"Low: {s.day_low:.2f}")
        if selected_date < today and df_day is not None and not df_day.empty:
            day_open = float(df_day["open"].iloc[0])
            day_close = float(df_day["close"].iloc[-1])
            st.write(f"Open: {day_open:.2f}")
            st.write(f"Close: {day_close:.2f}")
    elif isinstance(s, DaySummary):
        st.info("Daily high/low not stored for this day.")

    st.markdown("### Trade Suggestion")
    t = s.trade_suggestion
    st.write(f"Action: {t.action}")
    st.write(t.rationale)

    st.markdown("### Bias Accuracy")
    a = s.accuracy
    if a:
        st.write(f"Actual Direction: {a.actual_direction}")
        st.write(f"Bias Correct: {a.bias_correct}")
        st.write(f"Used Bias: {getattr(a,'used_bias', 'n/a')}")
        st.write(f"US Open Bias Correct (30m): {getattr(a,'us_open_bias_correct_30', 'n/a')}")
        st.write(f"US Open Bias Correct (60m): {getattr(a,'us_open_bias_correct_60', 'n/a')}")
        st.info(a.explanation)
    else:
        st.info("Trading day in progress — accuracy will be available after the day ends.")