# =============================================================================
# Streamlit Configuration - Pydantic Settings
# =============================================================================
# This module provides centralized configuration management for the Streamlit
# application using Pydantic Settings. It loads configuration from environment
# variables and a .env file.

# -----------------------------------------------------------------------------
# Standard Library Imports
# -----------------------------------------------------------------------------
from functools import lru_cache

# -----------------------------------------------------------------------------
# Third-Party Imports
# -----------------------------------------------------------------------------
from pydantic_settings import BaseSettings, SettingsConfigDict


# -----------------------------------------------------------------------------
# Class Definitions
# -----------------------------------------------------------------------------
class AppConfig(BaseSettings):
    """
    Main application settings for the Streamlit frontend.

    Attributes:
        API_BASE_URL (str): The base URL of the backend TTS service.
        API_KEY (str): The API key for authenticating with the backend.
        API_V1_PREFIX (str): The versioned API prefix.
        APP_TITLE (str): The title of the Streamlit application.
        APP_ICON (str): The icon for the Streamlit application.
        PAGE_LAYOUT (str): The page layout mode (e.g., "wide").
        SIDEBAR_STATE (str): The initial state of the sidebar (e.g., "expanded").
        ABOUT_TEXT (str): The informational text displayed in the sidebar.
        STYLE_PATH (str): The path to the custom CSS file.
        HEALTH_CHECK_TTL (int): The cache TTL for the health check in seconds.
        VOICES_TTL (int): The cache TTL for the voice list in seconds.
        TEXT_AREA_HEIGHT (int): The height of the text input area.
        TEXT_AREA_PLACEHOLDER (str): The placeholder text for the input area.
        MAX_TEXT_LENGTH (int): The maximum allowed character length for input.
        MIN_SPEED (float): The minimum speech speed.
        MAX_SPEED (float): The maximum speech speed.
        DEFAULT_SPEED (float): The default speech speed.
        SPEED_STEP (float): The step increment for the speed slider.
        AUDIO_FORMAT (str): The audio format for playback and download.
        DEFAULT_FILENAME (str): The default filename for downloaded audio.
    """

    # API Configuration
    API_BASE_URL: str
    API_KEY: str
    API_V1_PREFIX: str

    # UI Configuration
    APP_TITLE: str
    APP_ICON: str
    PAGE_LAYOUT: str
    SIDEBAR_STATE: str
    ABOUT_TEXT: str
    STYLE_PATH: str

    # Caching
    HEALTH_CHECK_TTL: int
    VOICES_TTL: int

    # Text Input
    TEXT_AREA_HEIGHT: int
    TEXT_AREA_PLACEHOLDER: str
    MAX_TEXT_LENGTH: int

    # Speech Controls
    MIN_SPEED: float
    MAX_SPEED: float
    DEFAULT_SPEED: float
    SPEED_STEP: float

    # Audio Output
    AUDIO_FORMAT: str
    DEFAULT_FILENAME: str

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def API_URL(self) -> str:
        """
        Constructs the full V1 API URL from base and prefix.

        Returns:
            str: The complete, formatted API URL.
        """
        return f"{self.API_BASE_URL.rstrip('/')}{self.API_V1_PREFIX}"


# -----------------------------------------------------------------------------
# Singleton Instance
# -----------------------------------------------------------------------------
@lru_cache
def get_config() -> AppConfig:
    """
    Returns a cached instance of the AppConfig settings.

    Using lru_cache ensures that the .env file is read and settings are
    parsed only once.

    Returns:
        AppConfig: The application configuration object.
    """
    return AppConfig()
