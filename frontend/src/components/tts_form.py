# =============================================================================
# TTS Form Component
# =============================================================================

import streamlit as st
from src.core.config import AppConfig

def render_tts_form(config: AppConfig, voices_data: dict):
    """
    Renders the main TTS configuration form.
    
    Returns:
        tuple: (submitted, text_input, selected_voice_name, selected_language, speed)
    """
    if not voices_data or not voices_data.get("languages"):
        st.warning("No voices available. Please check the backend service.")
        return False, None, None, None, None

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

        submitted = st.form_submit_button(
            "Generate Audio", type="primary", use_container_width=True
        )
        
    return submitted, text_input, selected_voice_name, selected_language, speed
