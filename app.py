import streamlit as st

from ui.live_analysis import render_live_analysis
from ui.history import render_history
from ui.compare import render_compare
from ui.settings import render_settings
from ui.definitions_explanations import render_definitions_explanations
from ui.strategy_playbook import render_strategy_playbook


APP_TITLE = "NQ Quant Terminal"
PAGES = [
    "Live Analysis",
    "Strategy Playbook",
    "History",
    "Compare Days",
    "Definitions/Explanatinos",
    "Settings",
]


def _get_query_page() -> str:
    try:
        qp = st.query_params
        page = qp.get("page", "")
        if isinstance(page, list):
            page = page[0] if page else ""
        return str(page)
    except Exception:
        try:
            qp = st.experimental_get_query_params()
            vals = qp.get("page", [""])
            return str(vals[0]) if vals else ""
        except Exception:
            return ""


def _set_query_page(page: str) -> None:
    try:
        st.query_params["page"] = page
        return
    except Exception:
        pass
    try:
        st.experimental_set_query_params(page=page)
    except Exception:
        pass


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(
        "Session-based, VWAP-aware, news-informed quant terminal with daily bias verification."
    )

    with st.sidebar:
        st.header("Navigation")
        query_page = _get_query_page()
        default_index = PAGES.index(query_page) if query_page in PAGES else 0
        page = st.selectbox(
            "Page",
            PAGES,
            index=default_index,
        )
        if query_page != page:
            _set_query_page(page)

        st.markdown("---")
        st.subheader("External")
        import webbrowser
        if st.button("Open TopstepX"):
            webbrowser.open("https://topstepx.com")

    if page == "Live Analysis":
        render_live_analysis()
    elif page == "Strategy Playbook":
        render_strategy_playbook()
    elif page == "History":
        render_history()
    elif page == "Compare Days":
        render_compare()
    elif page == "Settings":
        render_settings()
    elif page == "Definitions/Explanatinos":
        render_definitions_explanations()


if __name__ == "__main__":
    main()