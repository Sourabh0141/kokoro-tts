# =============================================================================
# UI Helper Functions for Kokoro TTS Frontend
# =============================================================================

import streamlit as st
from src.services.api_client import TTSClient
from src.core.config import AppConfig

def load_css(file_path: str):
    """
    Loads and injects custom CSS for styling the application.
    """
    try:
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"CSS file not found at {file_path}")

def get_fetchers(client: TTSClient, config: AppConfig):
    """
    Returns cached fetcher functions for health status and voices.
    """
    
    @st.cache_data(ttl=config.HEALTH_CHECK_TTL)
    def fetch_health_status():
        return client.get_health()

    @st.cache_data(ttl=config.VOICES_TTL)
    def fetch_voices():
        try:
            return client.get_voices()
        except Exception as e:
            st.error(f"Failed to load voices: {e}")
            return None
            
    return fetch_health_status, fetch_voices
