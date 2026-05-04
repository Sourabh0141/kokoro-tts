# =============================================================================
# Sidebar Component
# =============================================================================

import streamlit as st
from src.core.config import AppConfig

def render_sidebar(config: AppConfig, fetch_health_status):
    """
    Renders the sidebar with system status and about information.
    """
    with st.sidebar:
        st.title(f"{config.APP_ICON} {config.APP_TITLE}")
        st.markdown("---")
        st.subheader("System Status")

        health = fetch_health_status()
        if health and health.get("status") == "ok":
            st.markdown(
                """
                <div style='display: flex; align-items: center; color: #28a745;'>
                    <span class='status-indicator status-ok'></span> System Online
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("Details", expanded=False):
                st.caption(f"Loaded Languages: {health.get('loaded_languages', 'N/A')}")
                st.caption(f"Loaded Voices: {health.get('loaded_voices', 'N/A')}")
                st.caption(f"Memory Usage: {health.get('memory_usage_mb', 'N/A'):.1f} MB")
                st.caption(f"Device: {health.get('device', 'N/A')}")
        else:
            st.markdown(
                """
                <div style='display: flex; align-items: center; color: #dc3545;'>
                    <span class='status-indicator status-error'></span> System Offline or Error
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Retry Connection"):
                st.cache_data.clear()
                st.rerun()

        st.markdown("---")
        st.info(config.ABOUT_TEXT)
