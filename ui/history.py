import datetime as dt
import pandas as pd
import streamlit as st
from data.data_fetcher import fetch_intraday_ohlcv, filter_date
from engines.sessions import compute_session_stats
from engines.patterns import detect_patterns
from engines.bias import build_bias
from engines.accuracy import evaluate_bias_accuracy
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


def render_history():
    st.header("History")

    # Allow user to pick symbol and date to view historical analysis
    symbol = st.text_input("Symbol", value="NQH26", help="Futures symbol to analyze")
    selected_date = st.date_input("Select date to view", value=dt.datetime.now().date())

    # First try to load saved summary if it exists
    summaries = load_all_summaries()
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
        today = dt.datetime.now().date()
        days_needed = max(1, (today - selected_date).days + 1)
        df, used_ticker = fetch_intraday_ohlcv(symbol, lookback_days=days_needed + 1)
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
        patterns = detect_patterns(sessions)
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
    st.info(p.notes)

    st.markdown("### Bias")
    b = s.bias
    st.write(f"Daily Bias: {b.daily_bias} ({b.daily_confidence:.2f})")
    st.write(f"US Open Bias: {b.us_open_bias} ({b.us_open_confidence:.2f})")
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
        st.write(f"US Open Bias Correct: {getattr(a,'us_open_bias_correct', 'n/a')}")
        st.info(a.explanation)
    else:
        st.info("Trading day in progress — accuracy will be available after the day ends.")