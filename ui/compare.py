import streamlit as st

from storage.history_manager import load_all_summaries


def render_compare():
    st.header("Compare Days")

    summaries = load_all_summaries()
    if len(summaries) < 2:
        st.info("Need at least two saved days to compare.")
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

    c1, c2 = st.columns(2)

    with c1:
        st.subheader(f"{left.date} ({left.symbol})")
        st.markdown("**Bias**")
        st.write(f"Daily: {left.bias.daily_bias} ({left.bias.daily_confidence:.2f})")
        st.write(f"US: {left.bias.us_open_bias} ({left.bias.us_open_confidence:.2f})")
        st.markdown("**Patterns**")
        st.write(f"London Breakout: {left.patterns.london_breakout}")
        st.write(f"Whipsaw: {left.patterns.whipsaw}")
        st.write(f"Trend Day: {left.patterns.trend_day}")
        st.write(f"Vol Expansion: {left.patterns.volatility_expansion}")
        st.markdown("**Accuracy**")
        st.write(f"Actual: {left.accuracy.actual_direction}")
        st.write(f"Correct: {left.accuracy.bias_correct}")
        st.write(f"Used Bias: {getattr(left.accuracy,'used_bias', 'n/a')}")
        st.write(f"US Open Correct: {getattr(left.accuracy,'us_open_bias_correct', 'n/a')}")

    with c2:
        st.subheader(f"{right.date} ({right.symbol})")
        st.markdown("**Bias**")
        st.write(f"Daily: {right.bias.daily_bias} ({right.bias.daily_confidence:.2f})")
        st.write(f"US: {right.bias.us_open_bias} ({right.bias.us_open_confidence:.2f})")
        st.markdown("**Patterns**")
        st.write(f"London Breakout: {right.patterns.london_breakout}")
        st.write(f"Whipsaw: {right.patterns.whipsaw}")
        st.write(f"Trend Day: {right.patterns.trend_day}")
        st.write(f"Vol Expansion: {right.patterns.volatility_expansion}")
        st.markdown("**Accuracy**")
        st.write(f"Actual: {right.accuracy.actual_direction}")
        st.write(f"Correct: {right.accuracy.bias_correct}")
        st.write(f"Used Bias: {getattr(right.accuracy,'used_bias', 'n/a')}")
        st.write(f"US Open Correct: {getattr(right.accuracy,'us_open_bias_correct', 'n/a')}")