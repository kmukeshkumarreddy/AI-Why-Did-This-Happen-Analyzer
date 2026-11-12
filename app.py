"""AI "Why Did This Happen?" Analyzer - Main Application.

Entry point for the explainable AI data investigation platform.
"""

import sys
from pathlib import Path
import streamlit as st

# Ensure project root is in Python path for absolute imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings
from config.constants import APP_NAME, APP_ICON, NavigationSection
from components.sidebar import render_sidebar
from components.dashboard import (
    render_welcome_screen,
    render_placeholder_section,
    render_dataset_understanding_page,
)
from components.preprocessing_ui import render_preprocessing_page
from components.unexpected_detection_ui import render_unexpected_detection_page
from components.eda_ui import render_eda_page
from components.statistical_ui import render_statistical_page
from components.ml_studio_ui import render_ml_studio_page
from components.shap_ui import render_shap_studio_page
from components.lime_ui import render_lime_studio_page
from components.root_cause_ui import render_root_cause_studio_page
from components.reporting_ui import render_reporting_page
from utils.helpers import render_custom_css


def main() -> None:
    """Initialize application state, apply styles, and route navigation."""
    # 1. Page Configuration (Must be invoked before other Streamlit calls)
    st.set_page_config(
        page_title=APP_NAME,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 2. Inject Enterprise Dark Styling
    render_custom_css()

    # 3. Initialize Settings & Storage
    settings = get_settings()

    # 4. Session State Initialization
    if "current_dataset" not in st.session_state:
        st.session_state["current_dataset"] = None
    if "uploaded_dataset" not in st.session_state:
        st.session_state["uploaded_dataset"] = None
    if "active_section" not in st.session_state:
        st.session_state["active_section"] = NavigationSection.WELCOME

    # 5. Sidebar Navigation
    try:
        selected_section = render_sidebar()
    except Exception as e:
        st.error(f"Error rendering sidebar: {str(e)}")
        selected_section = NavigationSection.WELCOME

    # 6. Central Page Routing
    try:
        if selected_section == NavigationSection.WELCOME:
            render_welcome_screen()
        elif selected_section in (NavigationSection.DATASET_OVERVIEW, NavigationSection.DATA_QUALITY):
            render_dataset_understanding_page()
        elif selected_section == NavigationSection.DATA_PREPROCESSING:
            render_preprocessing_page()
        elif selected_section == NavigationSection.EDA:
            render_eda_page()
        elif selected_section == NavigationSection.OUTCOME_INVESTIGATION:
            render_unexpected_detection_page()
        elif selected_section == NavigationSection.STATISTICAL_ANALYSIS:
            render_statistical_page()
        elif selected_section == NavigationSection.MACHINE_LEARNING:
            render_ml_studio_page()
        elif selected_section == NavigationSection.EXPLAINABLE_AI:
            xai_mode = st.radio(
                "Select XAI Interpretability Engine",
                ["💡 SHAP Explainability Studio", "🍋 LIME Explainability Studio"],
                horizontal=True,
                label_visibility="collapsed",
                key="xai_engine_switcher",
            )
            if "SHAP" in xai_mode:
                render_shap_studio_page()
            else:
                render_lime_studio_page()
        elif selected_section == NavigationSection.ROOT_CAUSE:
            render_root_cause_studio_page()
        elif selected_section == NavigationSection.INVESTIGATION_REPORTS:
            render_reporting_page()
        else:
            render_placeholder_section(selected_section)
    except Exception as err:
        st.error(
            f"An error occurred while loading section '{selected_section.value}': {str(err)}"
        )
        if settings.debug_mode:
            st.exception(err)


if __name__ == "__main__":
    main()
