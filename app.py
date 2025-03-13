import pathlib
import streamlit as st


st.set_page_config(
    page_title="Kitpipat J. | 6604062610217",
    page_icon="👋",
)


def load_css(file_name: str) -> None:
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


css_path = pathlib.Path("./styles/index.css")
load_css(css_path)

st.markdown(
    """
      # Intellenge Project

      ## 1. Character Stats Prediction
    """
)


st.markdown(
    """
    <a href="/StatsPrediction" target="_self" class="nav-button">
            Model Information
    </a>
    <a href="/StatsPredictionModel" target="_self" class="nav-button">
            Model
    </a>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

st.markdown(
    """
      ## 2. Train Station Finder
    """
)

st.markdown(
    """
    <a href="/" target="_self" class="nav-button">
            Model Information
    </a>
    <a href="/" target="_self" class="nav-button">
            Model
    </a>
    """,
    unsafe_allow_html=True,
)
