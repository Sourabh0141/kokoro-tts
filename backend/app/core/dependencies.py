# =============================================================================
# FastAPI Dependencies - Dependency Injection and Authentication
# =============================================================================
# This module provides FastAPI dependency functions for the TTS service.
# It manages TTS engine singleton access and API key-based authentication
# for securing the text-to-speech endpoints.

# -----------------------------------------------------------------------------
# Standard Library Imports
# -----------------------------------------------------------------------------
from typing import Optional

# -----------------------------------------------------------------------------
# Third-Party Imports
# -----------------------------------------------------------------------------
from fastapi import Header, HTTPException, status

# -----------------------------------------------------------------------------
# Local Application Imports
# -----------------------------------------------------------------------------
from app.core.config import get_settings
from app.services.tts import TTSEngine

# -----------------------------------------------------------------------------
# Global TTS Engine Instance
# -----------------------------------------------------------------------------
# Global singleton instance for TTS engine access
# Initialized during application startup in main.py lifespan handler
_tts_engine: Optional[TTSEngine] = None


def set_tts_engine(engine: TTSEngine):
    """
    Set the global TTS engine instance.

    This function is called during application startup to initialize
    the global TTS engine singleton. The engine instance is created
    once and reused throughout the application lifetime.

    Args:
        engine: Initialized TTSEngine instance ready for audio generation

    Note:
        This function modifies global state and should only be called
        during application initialization.
    """
    global _tts_engine
    _tts_engine = engine


def get_tts_engine() -> TTSEngine:
    """
    FastAPI dependency to get the TTS engine instance.

    Provides access to the global TTS engine singleton for endpoint functions.
    This dependency ensures the engine is initialized before any TTS operations.

    Returns:
        TTSEngine: The initialized TTS engine instance

    Raises:
        RuntimeError: If TTS engine has not been initialized yet

    Usage:
        Use as a FastAPI dependency parameter in endpoint functions:

        @app.post("/audio")
        async def generate_audio(request: TTSRequest, engine: TTSEngine = Depends(get_tts_engine)):
            return engine.generate(request.text, request.language, request.voice, request.speed)
    """
    if _tts_engine is None:
        raise RuntimeError("TTSEngine is not initialized.")
    return _tts_engine


async def api_key_auth(
    x_api_key: str = Header(..., description="API Key for authentication"),
):
    """
    FastAPI dependency for API key authentication.

    Validates the X-API-Key header against the configured service API key.
    This dependency secures TTS endpoints by requiring valid authentication.

    Args:
        x_api_key: API key from X-API-Key header (automatically extracted by FastAPI)

    Raises:
        HTTPException: 401 Unauthorized if API key is invalid or missing

    Usage:
        Add as a dependency to secure endpoints:

        @app.post("/audio", dependencies=[Depends(api_key_auth)])
        async def generate_audio(request: TTSRequest):
            # Only reachable with valid API key
            pass

    Security Notes:
        - API key is configured via SERVICE__API_KEY environment variable
        - Default value is "dev_api_key" for development
        - Should be changed to a secure value in production
    """
    settings = get_settings()
    if x_api_key != settings.service.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key"
        )
