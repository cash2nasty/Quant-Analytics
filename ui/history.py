import datetime as dt
import pandas as pd
import streamlit as st
from data.data_fetcher import fetch_intraday_ohlcv, filter_date
from engines.sessions import compute_session_stats
from engines.patterns import detect_patterns
from engines.bias import build_bias
from engines.accuracy import evaluate_bias_accuracy
from engines.zones import build_htf_zones, summarize_zone_confluence
from engines.trade_suggestions import build_trade_suggestion
from storage.history_manager import DaySummary, load_all_summaries, save_day_summary


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
                        accuracy = evaluate_bias_accuracy(df_day, bias)
                        scored.append({"date": day, "accuracy": accuracy, "bias": bias, "patterns": patterns})
            else:
                scored = [
                    {"date": dt.date.fromisoformat(s.date), "accuracy": s.accuracy, "bias": s.bias, "patterns": s.patterns}
                    for s in in_range
                    if s.accuracy and s.accuracy.actual_direction != "n/a"
                ]

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

            if excluded_daily or excluded_30 or excluded_60:
                st.subheader("Excluded Days")
                excluded_rows = []
                for day in excluded_daily:
                    excluded_rows.append({"Date": day.isoformat(), "Excluded From": "Daily", "Reason": "No actual direction"})
                for day, reason in excluded_30:
                    excluded_rows.append({"Date": day.isoformat(), "Excluded From": "US Open 30m", "Reason": reason})
                for day, reason in excluded_60:
                    excluded_rows.append({"Date": day.isoformat(), "Excluded From": "US Open 60m", "Reason": reason})
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
            accuracy = evaluate_bias_accuracy(df_day, bias)
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

    st.markdown("### Bias")
    b = s.bias
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

    st.markdown("### Daily High/Low (End-of-Day)")
    if getattr(s, "day_high", None) is not None and getattr(s, "day_low", None) is not None:
        st.write(f"High: {s.day_high:.2f}")
        st.write(f"Low: {s.day_low:.2f}")
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