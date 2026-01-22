import asyncio
import sys
import psutil
from fastapi import APIRouter, Depends, HTTPException, Response, status
from app.schemas.tts import TTSRequest, HealthResponse, VoicesResponse, StatusResponse, VoiceStatusDetail
from app.services.tts import TTSEngine
from app.core.dependencies import get_tts_engine, api_key_auth
from app.core.exceptions import LanguageNotSupportedError, VoiceNotFoundError
from app.core.logging import logger

router = APIRouter()

@router.post("/audio", response_class=Response, dependencies=[Depends(api_key_auth)])
async def generate_audio(
    request: TTSRequest,
    service: TTSEngine = Depends(get_tts_engine)
):
    """
    Generate audio from text using the Kokoro TTS engine.
    """
    # Middleware logs the request entry/exit and timing.
    # Service logs the generation details.
    try:
        # Run the blocking inference in a separate thread to avoid blocking the event loop
        wav_bytes = await asyncio.to_thread(
            service.generate,
            request.text,
            request.language,
            request.voice,
            request.speed
        )
        return Response(
            content=wav_bytes, 
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=audio.wav"}
        )
    
    except (LanguageNotSupportedError, VoiceNotFoundError, ValueError) as e:
        # Client errors - log as warning
        logger.warning(f"Client error in audio generation: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Server errors - log full trace
        logger.error(f"Internal Server Error during audio generation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error during audio generation.")

@router.get("/health", response_model=HealthResponse)
async def health_check(service: TTSEngine = Depends(get_tts_engine)):
    """
    Health check endpoint to verify service status and voice availability.
    """
    logger.debug("Health check requested")
    if not service.is_ready:
        logger.warning("Health check failed: Service not ready")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is initializing or not ready"
        )
    
    stats = service.voice_manager.get_stats()
    
    # Get actual memory usage using psutil
    process = psutil.Process()
    # rss (Resident Set Size) in MB
    total_rss_mb = process.memory_info().rss / (1024 * 1024)
    
    # Estimate model memory as (Total RSS - Voice Memory)
    # This is rough but better than hardcoded
    # Ensure non-negative
    model_memory_mb = max(0, total_rss_mb - stats['total_memory_mb'])
    
    return HealthResponse(
        status="ok",
        loaded_languages=len(service.pipelines_cache),
        loaded_voices=stats['loaded_voices'],
        total_voices=sum(len(v) for v in service.available_voices.values()),
        memory_usage_mb=stats['total_memory_mb'], # Loaded voices memory
        model_memory_mb=model_memory_mb, # Remaining process memory (Model + overhead)
        device=service.device,
        is_ready=service.is_ready
    )

@router.get("/status", response_model=StatusResponse)
async def voice_status(service: TTSEngine = Depends(get_tts_engine)):
    """
    Get detailed status of currently loaded voices and memory usage.
    """
    logger.debug("Voice status requested")
    if not service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is initializing or not ready"
        )
    
    stats = service.voice_manager.get_stats()
    
    voice_details = [
        VoiceStatusDetail(
            language=v['language'],
            voice=v['voice'],
            voice_id=v['voice_id'],
            size_mb=v['size_mb'],
            age_seconds=v['age_seconds'],
            ttl_remaining_seconds=max(0, 600 - int(v['age_seconds']))
        )
        for v in stats['voices_detail']
    ]
    
    return StatusResponse(
        voices=voice_details,
        total_memory_mb=stats['total_memory_mb'],
        cleanup_checks_performed=len(stats['voices_detail']),  # Simplified placeholder
        voices_unloaded_total=stats['total_unloaded']
    )

@router.get("/voices", response_model=VoicesResponse)
async def list_voices(service: TTSEngine = Depends(get_tts_engine)):
    """
    Get list of all available voices and languages.
    """
    logger.debug("Voice list requested")
    if not service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is initializing or not ready"
        )
    
    return VoicesResponse(
        languages=service.available_voices,
        total_voices=sum(len(voices) for voices in service.available_voices.values()),
        total_languages=len(service.available_voices)
    )

@router.get("/")
async def root():
    """
    Root endpoint providing service information.
    """
    return {
        "service": "Kokoro TTS",
        "version": "1.0.0",
        "docs": "/docs"
    }