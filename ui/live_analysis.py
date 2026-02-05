import datetime as dt
import pandas as pd
import altair as alt
import streamlit as st

from data.data_fetcher import fetch_intraday_ohlcv
from data.session_reference import get_session_windows_for_date
from engines.sessions import compute_session_stats
from engines.patterns import detect_patterns
from engines.bias import build_bias
from engines.accuracy import evaluate_bias_accuracy
from indicators.moving_averages import compute_daily_vwap, compute_weekly_vwap
from indicators.statistics import zscore
from indicators.volatility import rolling_volatility
from indicators.momentum import roc, trend_strength
from indicators.volume import rvol
from storage.history_manager import DaySummary, save_day_summary, load_all_summaries

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


def main():
    st.title("Live Analysis")

    today = dt.date.today()
    symbol = st.sidebar.text_input("Symbol", value="NQH26")
    analysis_date = st.sidebar.date_input("Analysis date", value=today)

    # Fetch today's and previous day's data (robust to different return shapes)
    res_today = fetch_intraday_ohlcv(symbol, analysis_date)
    if isinstance(res_today, tuple):
        df_today, used_ticker = res_today
    else:
        df_today = res_today
        used_ticker = ""

    prev_date = analysis_date - dt.timedelta(days=1)
    res_prev = fetch_intraday_ohlcv(symbol, prev_date)
    if isinstance(res_prev, tuple):
        df_prev, _ = res_prev
    else:
        df_prev = res_prev

    st.sidebar.write(f"Data source ticker: {used_ticker}")

    if df_today is None or df_today.empty:
        st.warning("No intraday data available for the selected symbol/date.")
        return

    # Ensure timestamp column exists and is datetime
    if "timestamp" in df_today.columns:
        df_today["timestamp"] = pd.to_datetime(df_today["timestamp"])
    else:
        df_today = df_today.reset_index().rename(columns={df_today.index.name or "index": "timestamp"})
        df_today["timestamp"] = pd.to_datetime(df_today["timestamp"])

    if df_prev is not None and not df_prev.empty and "timestamp" in df_prev.columns:
        df_prev["timestamp"] = pd.to_datetime(df_prev["timestamp"])

    now = dt.datetime.now()

    # Session windows and stats
    try:
        windows = get_session_windows_for_date(analysis_date)
    except Exception:
        windows = {}

    df_sessions_source = df_today
    if df_prev is not None and not df_prev.empty:
        df_sessions_source = pd.concat([df_prev, df_today], ignore_index=True).sort_values("timestamp")

    sessions = compute_session_stats(df_sessions_source, analysis_date)
    prev_sessions = (
        compute_session_stats(df_prev, prev_date) if df_prev is not None and not df_prev.empty else {}
    )

    # Price chart
    st.subheader(f"Price Chart ({analysis_date.isoformat()})")
    chart_df = df_today[["timestamp", "close"]].copy()
    zoom_bars = min(200, len(chart_df))
    chart_view = chart_df.tail(zoom_bars)
    y_min = float(chart_view["close"].min())
    y_max = float(chart_view["close"].max())
    pad = max((y_max - y_min) * 0.05, 1e-6)
    y_domain = [y_min - pad, y_max + pad]

    base = alt.Chart(chart_df)

    brush = alt.selection_interval(encodings=["x"])

    main = (
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

    # Quick numeric summary
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

    # Session Stats
    st.markdown("### Session Stats")
    cols = st.columns(3)
    for i, name in enumerate(["Asia", "London", "US"]):
        with cols[i]:
            win = windows.get(name)
            sess = sessions.get(name)
            prev = prev_sessions.get(name)
            st.subheader(name)
            if win and now < win.get("end", now):
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
    patterns = detect_patterns(sessions)
    st.markdown("### Structural Patterns")
    st.write(f"London Breakout: {getattr(patterns,'london_breakout', False)}")
    st.write(f"Whipsaw: {getattr(patterns,'whipsaw', False)}")
    st.write(f"Trend Day: {getattr(patterns,'trend_day', False)}")
    st.write(f"Volatility Expansion: {getattr(patterns,'volatility_expansion', False)}")
    if getattr(patterns, "notes", None):
        st.info(patterns.notes)

    # Bias and trade suggestion
    bias = build_bias(df_today, df_prev, sessions)
    st.markdown("### Bias")
    st.write(f"Daily Bias: {getattr(bias,'daily_bias', 'n/a')} ({getattr(bias,'daily_confidence', 0):.2f})")
    st.write(f"US Open Bias: {getattr(bias,'us_open_bias', 'n/a')} ({getattr(bias,'us_open_confidence', 0):.2f})")
    st.write(f"VWAP Posture: {getattr(bias,'vwap_comment', '')}")
    st.write(f"News Effect: {getattr(bias,'news_comment', '')}")
    if getattr(bias, "explanation", None):
        st.info(bias.explanation)

    suggestion = build_trade_suggestion(bias)
    st.markdown("### Trade Suggestion")
    st.write(f"**Action:** {getattr(suggestion,'action','HOLD')} ")
    st.write(getattr(suggestion, "rationale", ""))

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

    ph = first_break_in_window(df_sessions_source, prev_high, day_start, day_end, above=True) if day_start and day_end else None
    pl = first_break_in_window(df_sessions_source, prev_low, day_start, day_end, above=False) if day_start and day_end else None

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

    # Accuracy: only compute for past dates
    if analysis_date < today:
        accuracy = evaluate_bias_accuracy(df_today, bias)
        st.markdown("### Daily Bias Accuracy (End-of-Day)")
        st.write(f"Actual Direction: {getattr(accuracy,'actual_direction', 'n/a')}")
        st.write(f"Bias Correct: {getattr(accuracy,'bias_correct', 'n/a')}")
        st.write(f"Used Bias: {getattr(accuracy,'used_bias', 'n/a')}")
        st.write(f"US Open Bias Correct: {getattr(accuracy,'us_open_bias_correct', 'n/a')}")
        if getattr(accuracy, "explanation", None):
            st.info(accuracy.explanation)
    else:
        st.markdown("### Daily Bias Accuracy")
        st.info("Trading day in progress — accuracy will be available after the day ends.")

    # Manual save
    if st.button("Save Today to History"):
        try:
            acc = evaluate_bias_accuracy(df_today, bias) if analysis_date < today else None
            day_summary = DaySummary(
                date=str(analysis_date),
                symbol=symbol,
                sessions=sessions,
                patterns=patterns,
                bias=bias,
                trade_suggestion=suggestion,
                accuracy=acc,
            )
            save_day_summary(day_summary)
            st.success("Saved current day summary to history.")
        except Exception as e:
            st.error(f"Failed to save summary: {e}")

    # Automatic archiving: if today is past and more than 1 hour since US close, save to history
    try:
        if analysis_date == today and windows and "US" in windows:
            us_end = windows["US"].get("end") if isinstance(windows["US"], dict) else None
            if us_end and now > us_end + dt.timedelta(hours=1):
                saved = [s for s in load_all_summaries() if s.date == str(today) and s.symbol == symbol]
                if not saved:
                    acc = evaluate_bias_accuracy(df_today, bias)
                    day_summary = DaySummary(
                        date=str(today),
                        symbol=symbol,
                        sessions=sessions,
                        patterns=patterns,
                        bias=bias,
                        trade_suggestion=suggestion,
                        accuracy=acc,
                    )
                    save_day_summary(day_summary)
                    st.success("Trading day archived to history automatically.")
    except Exception:
        pass


def render_live_analysis():
    return main()

if __name__ == "__main__":
    main()
