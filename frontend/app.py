# =============================================================================
# Kokoro TTS - Streamlit Frontend (Modular Entrypoint)
# =============================================================================

import time
import streamlit as st

# Local Application Imports
from src.core.config import get_config
from src.services.api_client import TTSClient
from src.utils.ui_helpers import load_css, get_fetchers
from src.components.sidebar import render_sidebar
from src.components.tts_form import render_tts_form
from src.components.audio_player import render_audio_player

# -----------------------------------------------------------------------------
# Configuration and Initialization
# -----------------------------------------------------------------------------
config = get_config()
client = TTSClient(config)

st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout=config.PAGE_LAYOUT,
    initial_sidebar_state=config.SIDEBAR_STATE,
)

# Initialize Fetchers
fetch_health_status, fetch_voices = get_fetchers(client, config)

# -----------------------------------------------------------------------------
# Main Application UI
# -----------------------------------------------------------------------------

# Load custom CSS styles
load_css(config.STYLE_PATH)

# Initialize session state variables
if "audio_data" not in st.session_state:
    st.session_state.audio_data = None
if "last_text" not in st.session_state:
    st.session_state.last_text = ""

# Render Sidebar
render_sidebar(config, fetch_health_status)

# Render Main Content
st.title("Text to Speech Generation")

# Fetch available voices
voices_data = fetch_voices()

# Render TTS Form
submitted, text_input, selected_voice_name, selected_language, speed = render_tts_form(
    config, voices_data
)

# Handle Form Submission
if submitted:
    if not text_input.strip():
        st.warning("Please enter some text to generate audio.")
    else:
        try:
            with st.spinner("Generating audio... This may take a moment."):
                start_time = time.time()

                # Call the API to generate audio
                audio_bytes = client.generate_audio(
                    text=text_input,
                    voice=selected_voice_name,
                    language=selected_language,
                    speed=speed,
                )

                generation_time = time.time() - start_time

                st.session_state.audio_data = audio_bytes
                st.session_state.last_text = text_input

            st.success(f"Audio generated in {generation_time:.2f} seconds!")

        except Exception as e:
            st.error(f"An error occurred while generating audio: {e}")

# Render Audio Player (if audio has been generated)
render_audio_player(config, st.session_state.audio_data, st.session_state.last_text)

st.markdown("---")
st.caption(f"Connected to TTS Service at `{config.API_URL}`")
