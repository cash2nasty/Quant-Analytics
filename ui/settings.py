import streamlit as st


def render_settings():
    st.header("Settings")

    st.subheader("Volatility Regime")
    st.write(
        "Volatility regime labels how fast the market is moving right now compared to its "
        "recent history. If current volatility is much higher than normal, it is 'expanded'. "
        "If it is much lower, it is 'compressed'. This helps set expectations for range and risk."
    )
    p_low, p_high = st.slider(
        "Compression/Expansion thresholds (percentiles)",
        min_value=0.10,
        max_value=0.90,
        value=(0.30, 0.70),
        step=0.01,
        help="Lower = compressed threshold, Upper = expanded threshold.",
    )
    st.session_state["vol_p_low"] = float(p_low)
    st.session_state["vol_p_high"] = float(p_high)
    st.caption(
        "Example: 30/70 means the market is 'compressed' below the 30th percentile of recent "
        "volatility and 'expanded' above the 70th percentile."
    )

    st.markdown("---")
    st.write("This page will later allow:")
    st.write("- Session overrides")
    st.write("- Data source configuration (Nasdaq-style, etc.)")
    st.write("- Indicator parameters")
    st.write("- Refresh behavior")

    st.info(
        "Core quant logic (sessions, VWAP, bias, accuracy, news modifiers) "
        "is implemented in the engines and indicators modules."
    )