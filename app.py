import streamlit as st

from ui.live_analysis import render_live_analysis
from ui.history import render_history
from ui.compare import render_compare
from ui.settings import render_settings


APP_TITLE = "NQ Quant Terminal"


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(
        "Session-based, VWAP-aware, news-informed quant terminal with daily bias verification."
    )

    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Page",
            ["Live Analysis", "History", "Compare Days", "Settings"],
        )

        st.markdown("---")
        st.subheader("External")
        import webbrowser
        if st.button("Open TopstepX"):
            webbrowser.open("https://topstepx.com")

    if page == "Live Analysis":
        render_live_analysis()
    elif page == "History":
        render_history()
    elif page == "Compare Days":
        render_compare()
    elif page == "Settings":
        render_settings()


if __name__ == "__main__":
    main()