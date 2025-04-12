"""
Watchdog AI - Main Streamlit Application

This is the entry point for the Watchdog AI application.
It sets up the Streamlit UI and routes to the appropriate pages.
"""
import streamlit as st
import os
from pathlib import Path

# Import configuration
from config.config import (
    APP_NAME, DEBUG, validate_config, get_config, 
    DATA_DICTIONARY_PATH, UI_CONFIG
)

# Import UI components
from src.ui.pages.upload import render_upload_page
from src.ui.pages.chat import render_chat_page
from src.ui.pages.dashboard.metrics import render_metrics_dashboard
from src.ui.pages.dashboard.insights import render_insights_dashboard

# Validate configuration (will raise an error if required vars are missing)
try:
    validate_config()
except Exception as e:
    st.error(f"Configuration error: {str(e)}")
    st.stop()

# App setup
st.set_page_config(
    page_title=UI_CONFIG["page_title"],
    page_icon=UI_CONFIG["page_icon"],
    layout=UI_CONFIG["layout"],
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main .block-container {padding-top: 2rem;}
    .stApp {max-width: 1200px; margin: 0 auto;}
    .upload-header {margin-bottom: 1.5rem !important;}
    .info-box {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .main-title {
        font-size: 2.5rem !important;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F3F4F6;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #DBEAFE;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    if UI_CONFIG["show_logo"]:
        st.title(f"{UI_CONFIG['page_icon']} {APP_NAME}")
    else:
        st.title(APP_NAME)
    
    st.write("Turn dealership data into actionable insights.")
    
    # Navigation options
    page = st.radio(
        "Navigation",
        options=["Upload Files", "Copilot Chat", "Dashboard"]
    )
    
    # Company branding if set
    if UI_CONFIG.get("company_name"):
        st.sidebar.markdown("---")
        st.sidebar.caption(f"Powered by {UI_CONFIG['company_name']}")
    
    # Version info
    st.sidebar.markdown("---")
    st.sidebar.caption("Version: MVP 1.0")
    if DEBUG:
        st.sidebar.caption("Debug mode: ON")

# Check if data dictionary exists
if not Path(DATA_DICTIONARY_PATH).exists():
    st.error(f"Data dictionary not found at {DATA_DICTIONARY_PATH}")
    st.stop()

# Routing based on user selection
if page == "Upload Files":
    render_upload_page()
elif page == "Copilot Chat":
    render_chat_page()
elif page == "Dashboard":
    # Use the consolidated dashboard page from the module
    from src.ui.pages.dashboard import render_dashboard_page
    render_dashboard_page()

if __name__ == "__main__":
    if DEBUG:
        st.write("Debug information:")
        config = get_config()
        # Remove sensitive information before displaying
        if "api_keys" in config:
            for key in config["api_keys"]:
                if config["api_keys"][key]:
                    config["api_keys"][key] = "****"
        st.json(config)