# =============================================================================
# API Endpoints V1 - FastAPI Route Handlers for TTS Service
# =============================================================================
# This module defines all V1 API endpoints for the Kokoro text-to-speech service.
# Endpoints provide TTS generation, health monitoring, voice management, and
# service information with proper error handling and response formatting.

# -----------------------------------------------------------------------------
# Standard Library Imports
# -----------------------------------------------------------------------------
import asyncio

# -----------------------------------------------------------------------------
# Third-Party Imports
# -----------------------------------------------------------------------------
import psutil
from fastapi import APIRouter, Depends, HTTPException, Response, status

# -----------------------------------------------------------------------------
# Local Application Imports
# -----------------------------------------------------------------------------
from app.schemas.tts import (
    TTSRequest,
    HealthResponse,
    VoicesResponse,
    StatusResponse,
    VoiceStatusDetail,
)
from app.services.tts import TTSEngine
from app.core.dependencies import get_tts_engine, api_key_auth
from app.core.exceptions import LanguageNotSupportedError, VoiceNotFoundError
from app.core.logging import logger

# -----------------------------------------------------------------------------
# API Router
# -----------------------------------------------------------------------------
# Create the V1 API router for all TTS endpoints
router = APIRouter()


@router.post("/audio", response_class=Response, dependencies=[Depends(api_key_auth)])
async def generate_audio(
    request: TTSRequest, service: TTSEngine = Depends(get_tts_engine)
):
    """
    Generate audio from text using the specified voice and language.

    This is the main TTS endpoint that converts input text to speech audio.
    The endpoint requires API key authentication and returns WAV audio data.

    Args:
        request: TTS request containing text, language, voice, and speed parameters
        service: TTS engine instance (injected via FastAPI dependency)

    Returns:
        Response: WAV audio file with appropriate headers for download

    Raises:
        HTTPException:
            - 400: Invalid language, voice, or malformed request
            - 401: Missing or invalid API key
            - 500: Internal server error during audio generation

    Authentication:
        Requires X-API-Key header with valid service API key.

    Example:
        POST /v1/audio
        Headers: X-API-Key: your_api_key
        Body: {
            "text": "Hello world",
            "language": "American English",
            "voice": "Bella (Female)",
            "speed": 1.0
        }

        Returns: WAV audio file download
    """
    try:
        # Run TTS generation in thread pool to avoid blocking event loop
        wav_bytes = await asyncio.to_thread(
            service.generate,
            request.text,
            request.language,
            request.voice,
            request.speed,
        )

        # Return audio as downloadable WAV file
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=audio.wav"},
        )

    except (LanguageNotSupportedError, VoiceNotFoundError, ValueError) as e:
        # Client errors - log as warning and return 400
        logger.warning(f"Client error in audio generation: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Server errors - log with stack trace and return 500
        logger.error(
            f"Internal Server Error during audio generation: {str(e)}", exc_info=True
        )
        raise HTTPException(
            status_code=500, detail="Internal Server Error during audio generation."
        )


@router.get("/health", response_model=HealthResponse)
async def health_check(service: TTSEngine = Depends(get_tts_engine)):
    """
    Service health check endpoint with detailed status information.

    Provides comprehensive health monitoring including service readiness,
    loaded resources, memory usage, and system information. Used by
    load balancers, monitoring systems, and deployment orchestration.

    Args:
        service: TTS engine instance (injected via FastAPI dependency)

    Returns:
        HealthResponse: Detailed service health information

    Raises:
        HTTPException: 503 Service Unavailable if TTS engine is not ready

    Response Fields:
        - status: "ok" if service is healthy
        - loaded_languages: Number of language pipelines currently cached
        - loaded_voices: Number of voice embeddings in memory
        - total_voices: Total voices available across all languages
        - memory_usage_mb: Memory used by loaded voices
        - model_memory_mb: Estimated memory used by model and pipelines
        - device: PyTorch device (cpu/cuda/mps)
        - is_ready: Whether service has completed initialization

    Usage:
        GET /v1/health

        Used by:
        - Docker health checks
        - Load balancers for service discovery
        - Monitoring dashboards
        - Deployment systems for readiness probes
    """
    logger.debug("Health check requested")

    # Check if service is ready for requests
    if not service.is_ready:
        logger.warning("Health check failed: Service not ready")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is initializing or not ready",
        )

    # Gather voice manager statistics
    stats = service.voice_manager.get_stats()

    # Get process memory information using psutil
    process = psutil.Process()
    total_rss_mb = process.memory_info().rss / (1024 * 1024)

    # Estimate model memory as total RSS minus voice memory
    model_memory_mb = max(0, total_rss_mb - stats["total_memory_mb"])

    return HealthResponse(
        status="ok",
        loaded_languages=len(service.pipelines_cache),
        loaded_voices=stats["loaded_voices"],
        total_voices=sum(len(v) for v in service.available_voices.values()),
        memory_usage_mb=stats["total_memory_mb"],  # Voice memory
        model_memory_mb=model_memory_mb,          # Model + overhead memory
        device=service.device,
        is_ready=service.is_ready,
    )


@router.get("/status", response_model=StatusResponse)
async def voice_status(service: TTSEngine = Depends(get_tts_engine)):
    """
    Detailed voice loading status and memory management information.

    Provides comprehensive information about currently loaded voices,
    their memory usage, age, and TTL status. Useful for monitoring
    voice cache performance and memory management effectiveness.

    Args:
        service: TTS engine instance (injected via FastAPI dependency)

    Returns:
        StatusResponse: Detailed voice status information

    Raises:
        HTTPException: 503 Service Unavailable if TTS engine is not ready

    Response Fields:
        - voices: List of VoiceStatusDetail for each loaded voice
        - total_memory_mb: Total memory used by all loaded voices
        - cleanup_checks_performed: Number of TTL checks performed
        - voices_unloaded_total: Total voices unloaded since startup

    Voice Details Include:
        - language: Language name
        - voice: Display name (e.g., "Bella (Female)")
        - voice_id: Internal identifier (e.g., "af_bella")
        - size_mb: Memory size in megabytes
        - age_seconds: Seconds since last access
        - ttl_remaining_seconds: Seconds until automatic cleanup

    Usage:
        GET /v1/status

        Useful for:
        - Memory usage monitoring
        - Cache performance analysis
        - Voice loading diagnostics
        - TTL policy tuning
    """
    logger.debug("Voice status requested")

    if not service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is initializing or not ready",
        )

    # Get comprehensive voice manager statistics
    stats = service.voice_manager.get_stats()

    # Transform raw stats into structured response objects
    voice_details = [
        VoiceStatusDetail(
            language=v["language"],
            voice=v["voice"],
            voice_id=v["voice_id"],
            size_mb=v["size_mb"],
            age_seconds=v["age_seconds"],
            ttl_remaining_seconds=max(0, 600 - int(v["age_seconds"])),  # 10 min TTL
        )
        for v in stats["voices_detail"]
    ]

    return StatusResponse(
        voices=voice_details,
        total_memory_mb=stats["total_memory_mb"],
        cleanup_checks_performed=len(stats["voices_detail"]),  # Simplified metric
        voices_unloaded_total=stats["total_unloaded"],
    )


@router.get("/voices", response_model=VoicesResponse)
async def list_voices(service: TTSEngine = Depends(get_tts_engine)):
    """
    List all available voices organized by language.

    Returns the complete catalog of supported languages and their available
    voices for client applications to present voice selection options.
    This endpoint is essential for building user interfaces that allow
    voice selection.

    Args:
        service: TTS engine instance (injected via FastAPI dependency)

    Returns:
        VoicesResponse: Available voices organized by language

    Raises:
        HTTPException: 503 Service Unavailable if TTS engine is not ready

    Response Fields:
        - languages: Dictionary mapping language names to voice dictionaries
        - total_voices: Total number of voices across all languages
        - total_languages: Number of supported languages

    Language Structure:
        {
            "American English": {
                "Bella (Female)": "af_bella",
                "Adam (Male)": "am_adam"
            },
            "Japanese": {
                "Sakura (Female)": "jf_sakura"
            }
        }

    Usage:
        GET /v1/voices

        Used by client applications to:
        - Populate voice selection dropdowns
        - Validate voice availability
        - Build dynamic user interfaces
        - Show available options to users
    """
    logger.debug("Voice list requested")

    if not service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is initializing or not ready",
        )

    return VoicesResponse(
        languages=service.available_voices,
        total_voices=sum(len(voices) for voices in service.available_voices.values()),
        total_languages=len(service.available_voices),
    )


@router.get("/")
async def root():
    """
    V1 API root endpoint providing service information and navigation.

    Returns basic service metadata and links to documentation and other endpoints.
    This endpoint is excluded from API schema but provides useful information
    for API exploration and integration.

    Returns:
        dict: Service information with navigation links
            - service: Service name
            - version: API version
            - docs: Link to auto-generated API documentation

    Usage:
        GET /v1/

        Useful for:
        - API discovery and exploration
        - Integration testing
        - Documentation links for developers
        - Basic service identification
    """
    return {
        "service": "Kokoro TTS",
        "version": "1.0.0",
        "docs": "/docs"  # Link to FastAPI auto-generated documentation
    }
