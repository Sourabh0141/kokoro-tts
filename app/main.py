import os
# Suppress Hugging Face symlink warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from app.core.config import get_settings
from app.api.v1.router import api_router
from app.services.tts import TTSEngine
from app.core.dependencies import set_tts_engine
from app.core.logging import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up Kokoro TTS Service...")
    settings = get_settings()
    
    # Check if voices are loaded
    if not settings.all_voices:
        logger.warning(
            f"No voices found in {settings.model.local_voices_dir}. "
            "Did you run models/download_models.py? "
            "The service will start but voice generation will fail until voices are available."
        )
    else:
        logger.info(f"Discovered {sum(len(v) for v in settings.all_voices.values())} voices across {len(settings.all_voices)} languages.")

    try:
        engine = TTSEngine(settings)
        engine.initialize()
        set_tts_engine(engine)
    except Exception as e:
        logger.critical(f"Failed to initialize TTSEngine: {e}", exc_info=True)
        raise e
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    if 'engine' in locals():
        engine.voice_manager.stop_cleanup_loop()

app = FastAPI(
    title=get_settings().service.name,
    version=get_settings().service.version,
    lifespan=lifespan
)

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled Global Exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
    )

# Mount V1 API
app.include_router(api_router, prefix="/v1")

# Optional: Convenience root endpoint pointing to V1 info or Docs
@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": get_settings().service.name,
        "version": get_settings().service.version,
        "docs": "/docs",
        "api_v1": "/v1/"
    }

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.service.host, port=settings.service.port, reload=True)