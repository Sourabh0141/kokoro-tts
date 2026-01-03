import asyncio
import traceback
import sys
from fastapi import APIRouter, Depends, HTTPException, Response, status
from app.schemas.tts import TTSRequest, HealthResponse, VoicesResponse, StatusResponse, VoiceStatusDetail
from app.services.tts import TTSEngine
from app.core.dependencies import get_tts_engine, api_key_auth
from app.core.exceptions import LanguageNotSupportedError, VoiceNotFoundError

router = APIRouter()

@router.post("/audio", response_class=Response, dependencies=[Depends(api_key_auth)])
async def generate_audio(
    request: TTSRequest,
    service: TTSEngine = Depends(get_tts_engine)
):
    """
    Generate audio from text using the Kokoro TTS engine.
    
    Parameters:
    - text: Text to synthesize (required)
    - language: Language name, e.g., "American English", "British English", "Japanese" (required)
    - voice: Voice name, e.g., "Bella (Female)", "Adam (Male)" (required)
    - speed: Speech speed multiplier, 0.5-2.0 (default: 1.0)
    
    Returns:
    - WAV audio file
    
    Note: First request for a voice may be slower (~100-200ms) due to disk loading.
    Subsequent requests are cached (~50ms).
    
    Example:
    {
        "text": "Hello world",
        "language": "American English",
        "voice": "Bella (Female)",
        "speed": 1.0
    }
    """
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
    
    except LanguageNotSupportedError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except VoiceNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Log the full stack trace for debugging
        print(f"ERROR: Internal Server Error during audio generation: {str(e)}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise HTTPException(status_code=500, detail="Internal Server Error during audio generation.")

@router.get("/health", response_model=HealthResponse)
async def health_check(service: TTSEngine = Depends(get_tts_engine)):
    """
    Health check endpoint to verify service status and voice availability.
    
    Returns:
    - status: "ok" if ready
    - loaded_languages: Number of languages with cached pipelines
    - loaded_voices: Number of voices currently in memory
    - total_voices: Total available voices (57)
    - memory_usage_mb: Memory used by loaded voices (not including model)
    - model_memory_mb: Estimated model size (~1500MB for CPU)
    - device: Computation device (cpu, cuda, mps)
    - is_ready: Whether service is ready to process requests
    """
    if not service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is initializing or not ready"
        )
    
    stats = service.voice_manager.get_stats()
    
    # Estimate model memory (rough estimate, actual depends on dtype/device)
    if service.device == "cuda":
        model_memory_mb = 1500  # GPU optimization
    else:
        model_memory_mb = 2000  # CPU mode
    
    return HealthResponse(
        status="ok",
        loaded_languages=len(service.pipelines_cache),
        loaded_voices=stats['loaded_voices'],
        total_voices=sum(len(v) for v in service.available_voices.values()),
        memory_usage_mb=stats['total_memory_mb'],
        model_memory_mb=model_memory_mb,
        device=service.device,
        is_ready=service.is_ready
    )

@router.get("/status", response_model=StatusResponse)
async def voice_status(service: TTSEngine = Depends(get_tts_engine)):
    """
    Get detailed status of currently loaded voices and memory usage.
    
    Returns:
    - voices: List of loaded voices with TTL info
    - total_memory_mb: Total memory used by voices
    - cleanup_checks_performed: Number of TTL checks performed
    - voices_unloaded_total: Total voices unloaded since startup
    """
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
        cleanup_checks_performed=len(stats['voices_detail']),  # Simplified
        voices_unloaded_total=stats['total_unloaded']
    )

@router.get("/voices", response_model=VoicesResponse)
async def list_voices(service: TTSEngine = Depends(get_tts_engine)):
    """
    Get list of all available voices and languages.
    
    Returns:
    - languages: Dictionary mapping language names to voice names and IDs
    - total_voices: Total number of available voices across all languages
    - total_languages: Total number of supported languages
    """
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
