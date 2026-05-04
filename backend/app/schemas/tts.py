# =============================================================================
# TTS API Schemas - Pydantic Models for Request/Response Validation
# =============================================================================
# This module defines Pydantic models for API request/response validation and
# serialization in the Kokoro TTS service. These schemas ensure type safety,
# automatic validation, and API documentation generation.

# -----------------------------------------------------------------------------
# Standard Library Imports
# -----------------------------------------------------------------------------
from typing import List, Dict

# -----------------------------------------------------------------------------
# Third-Party Imports
# -----------------------------------------------------------------------------
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# Local Application Imports
# -----------------------------------------------------------------------------
from app.core.config import get_settings

# Load settings for schema validation limits
settings = get_settings()


class TTSRequest(BaseModel):
    """
    Request model for text-to-speech audio generation.

    Defines the structure and validation rules for TTS API requests.
    All fields are required except speed which has a default value.

    Attributes:
        text: The input text to synthesize into speech (minimum 1 character)
        language: Target language name (e.g., "American English", "Japanese")
        voice: Specific voice to use (e.g., "Bella (Female)", "Adam (Male)")
        speed: Playback speed multiplier (range: 0.5 to 2.0, default: 1.0)
    """

    text: str = Field(
        ..., min_length=settings.limits.min_text_length, description="Text to synthesize"
    )
    language: str = Field(
        ...,
        description="Language name (e.g., 'American English', 'British English', 'Japanese')",
    )
    voice: str = Field(
        ..., description="Voice name (e.g., 'Bella (Female)', 'Adam (Male)')"
    )
    speed: float = Field(
        default=settings.limits.default_speed,
        ge=settings.limits.min_speed,
        le=settings.limits.max_speed,
        description="Speed multiplier for playback",
    )


class HealthResponse(BaseModel):
    """
    Response model for service health check endpoint.

    Provides comprehensive health information about the TTS service status,
    loaded resources, and memory usage for monitoring and debugging.

    Attributes:
        status: Service status string (typically "ok")
        loaded_languages: Number of language pipelines currently loaded
        loaded_voices: Number of voice embeddings currently in memory
        total_voices: Total number of voices available across all languages
        memory_usage_mb: Memory used by loaded voices in megabytes
        model_memory_mb: Estimated memory used by the model and pipelines
        device: PyTorch device being used (cpu/cuda/mps)
        is_ready: Whether the service is fully initialized and ready
    """

    status: str
    loaded_languages: int
    loaded_voices: int
    total_voices: int
    memory_usage_mb: float
    model_memory_mb: float
    device: str
    is_ready: bool


class VoiceStatusDetail(BaseModel):
    """
    Detailed information about a single loaded voice.

    Used in status responses to provide per-voice memory and usage statistics
    for monitoring voice cache performance and TTL management.

    Attributes:
        language: Language name this voice belongs to
        voice: Display name of the voice (e.g., "Bella (Female)")
        voice_id: Internal file identifier (e.g., "af_bella")
        size_mb: Memory size of the voice tensor in megabytes
        age_seconds: Seconds since this voice was last accessed
        ttl_remaining_seconds: Seconds until automatic unloading (TTL expiry)
    """

    language: str
    voice: str
    voice_id: str
    size_mb: float
    age_seconds: float
    ttl_remaining_seconds: int


class StatusResponse(BaseModel):
    """
    Response model for detailed voice status and memory information.

    Provides comprehensive statistics about voice loading, memory usage,
    and cleanup operations for service monitoring and optimization.

    Attributes:
        voices: List of detailed information for each loaded voice
        total_memory_mb: Total memory used by all loaded voices
        cleanup_checks_performed: Number of TTL checks performed
        voices_unloaded_total: Total voices unloaded since service start
    """

    voices: List[VoiceStatusDetail]
    total_memory_mb: float
    cleanup_checks_performed: int
    voices_unloaded_total: int


class VoicesResponse(BaseModel):
    """
    Response model for available voices and languages list.

    Returns the complete catalog of supported languages and their available
    voices for client applications to present voice selection options.

    Attributes:
        languages: Dictionary mapping language names to voice dictionaries
                   Format: {"Language Name": {"Voice Name": "voice_id", ...}, ...}
        total_voices: Total number of voices across all languages
        total_languages: Number of supported languages
    """

    languages: Dict[str, Dict[str, str]]
    total_voices: int
    total_languages: int
