import streamlit as st


def render_settings():
    st.header("Settings")

    st.write("This page will later allow:")
    st.write("- Session overrides")
    st.write("- Data source configuration (Nasdaq-style, etc.)")
    st.write("- Indicator parameters")
    st.write("- Refresh behavior")

    st.info(
        "Core quant logic (sessions, VWAP, bias, accuracy, news modifiers) "
        "is implemented in the engines and indicators modules."
    )