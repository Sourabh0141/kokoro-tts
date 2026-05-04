# =============================================================================
# Audio Player Component
# =============================================================================

import streamlit as st
from src.core.config import AppConfig

def render_audio_player(config: AppConfig, audio_data, last_text: str):
    """
    Renders the audio playback and download controls.
    """
    if audio_data:
        st.markdown("### Generated Audio")

        st.audio(audio_data, format=config.AUDIO_FORMAT)

        st.download_button(
            label="Download WAV",
            data=audio_data,
            file_name=config.DEFAULT_FILENAME,
            mime=config.AUDIO_FORMAT,
        )

        with st.expander("Show Synthesized Text"):
            st.text(last_text)
