import asyncio
from fastapi import APIRouter, Depends, HTTPException, Response, status
from app.schemas.tts import TTSRequest, HealthResponse
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
    """
    try:
        # Run the blocking inference in a separate thread to avoid blocking the event loop
        wav_bytes = await asyncio.to_thread(
            service.generate,
            request.text,
            request.lang_code,
            request.speed
        )
        return Response(content=wav_bytes, media_type="audio/wav")
    
    except LanguageNotSupportedError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (ValueError, VoiceNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # In a real app, we would log the full stack trace here
        raise HTTPException(status_code=500, detail="Internal Server Error during audio generation.")

@router.get("/health", response_model=HealthResponse)
async def health_check(service: TTSEngine = Depends(get_tts_engine)):
    """
    Health check endpoint to verify service status.
    """
    if not service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is initializing or not ready"
        )
    
    return HealthResponse(
        status="ok",
        loaded_languages=list(service.pipelines.keys()),
        device=service.device
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
