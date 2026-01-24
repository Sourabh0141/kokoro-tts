# =============================================================================
# Kokoro TTS - Streamlit Frontend
# =============================================================================
# This file defines the main user interface for the Kokoro TTS service,
# built with Streamlit. It provides controls for language and voice selection,
# text input, and audio playback.

# -----------------------------------------------------------------------------
# Standard Library Imports
# -----------------------------------------------------------------------------
import time

# -----------------------------------------------------------------------------
# Third-Party Imports
# -----------------------------------------------------------------------------
import streamlit as st

# -----------------------------------------------------------------------------
# Local Application Imports
# -----------------------------------------------------------------------------
from config import get_config
from services.api_client import TTSClient

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


# -----------------------------------------------------------------------------
# Function Definitions
# -----------------------------------------------------------------------------
def load_css(file_path: str):
    """
    Loads and injects custom CSS for styling the application.

    Args:
        file_path (str): The path to the CSS file.
    """
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_data(ttl=config.HEALTH_CHECK_TTL)
def fetch_health_status():
    """
    Fetches the health status of the backend TTS service.

    The result is cached to prevent excessive requests.

    Returns:
        dict: The health status information from the API.
    """
    return client.get_health()


@st.cache_data(ttl=config.VOICES_TTL)
def fetch_voices():
    """
    Fetches the available voices from the backend TTS service.

    The result is cached for a longer duration as voices change infrequently.

    Returns:
        dict: The available languages and voices from the API.
    """
    try:
        return client.get_voices()
    except Exception as e:
        st.error(f"Failed to load voices: {e}")
        return None


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

# --- Sidebar ---
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

# --- Main Content ---
st.title("Text to Speech Generation")

voices_data = fetch_voices()

if not voices_data or not voices_data.get("languages"):
    st.warning("No voices available. Please check the backend service.")
    st.stop()

languages_map = voices_data.get("languages", {})
available_languages = list(languages_map.keys())

with st.form("tts_form"):
    col1, col2 = st.columns([1, 1])

    with col1:
        selected_language = st.selectbox(
            "Language",
            options=available_languages,
            index=0,
            help="Select the language for the speech.",
        )

    with col2:
        language_voices = languages_map.get(selected_language, {})
        voice_names = list(language_voices.keys())

        selected_voice_name = st.selectbox(
            "Voice",
            options=voice_names,
            index=0,
            help="Select the voice for the speech.",
        )

        selected_voice_id = language_voices.get(selected_voice_name)

    text_input = st.text_area(
        "Text to Speech",
        height=config.TEXT_AREA_HEIGHT,
        placeholder=config.TEXT_AREA_PLACEHOLDER,
        max_chars=config.MAX_TEXT_LENGTH,
        help=f"Maximum {config.MAX_TEXT_LENGTH} characters per request.",
    )

    col3, col4 = st.columns([1, 3])
    with col3:
        speed = st.slider(
            "Speed",
            min_value=config.MIN_SPEED,
            max_value=config.MAX_SPEED,
            value=config.DEFAULT_SPEED,
            step=config.SPEED_STEP,
            help="Adjust the speed of the generated speech.",
        )

    with col4:
        # This column is intentionally left empty for layout purposes
        pass

    submitted = st.form_submit_button(
        "Generate Audio", type="primary", use_container_width=True
    )

if submitted:
    if not text_input.strip():
        st.warning("Please enter some text to generate audio.")
    elif not selected_voice_id:
        st.error("Invalid voice selection. Please try again.")
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

if st.session_state.audio_data:
    st.markdown("### Generated Audio")

    st.audio(st.session_state.audio_data, format=config.AUDIO_FORMAT)

    st.download_button(
        label="Download WAV",
        data=st.session_state.audio_data,
        file_name=config.DEFAULT_FILENAME,
        mime=config.AUDIO_FORMAT,
    )

    with st.expander("Show Synthesized Text"):
        st.text(st.session_state.last_text)

st.markdown("---")
st.caption(f"Connected to TTS Service at `{config.API_URL}`")
