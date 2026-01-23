# =============================================================================
# Kokoro TTS Service - FastAPI Application Entry Point
# =============================================================================
# Main application file for the Kokoro text-to-speech service.
# Configures FastAPI application with TTS engine initialization,
# request logging, error handling, and API routing.

# -----------------------------------------------------------------------------
# Standard Library Imports
# -----------------------------------------------------------------------------
import os
import time
from contextlib import asynccontextmanager

# -----------------------------------------------------------------------------
# Third-Party Imports
# -----------------------------------------------------------------------------
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

# -----------------------------------------------------------------------------
# Local Application Imports
# -----------------------------------------------------------------------------
from app.core.config import get_settings
from app.api.v1.router import api_router
from app.services.tts import TTSEngine
from app.core.dependencies import set_tts_engine
from app.core.logging import logger

# -----------------------------------------------------------------------------
# Environment Configuration
# -----------------------------------------------------------------------------
# Disable HuggingFace Hub symlink warnings to reduce console noise
# This prevents warnings about symlinks when downloading models
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for application startup and shutdown.

    Handles the complete lifecycle of the TTS service:
    - Startup: Initialize TTS engine, load models and voices
    - Shutdown: Clean up resources and stop background processes

    Args:
        app: The FastAPI application instance

    Yields:
        None: Allows the application to run between startup and shutdown
    """
    # Application Startup Phase
    logger.info("Starting up Kokoro TTS Service...")
    settings = get_settings()

    # Check for available voices and warn if none found
    if not settings.all_voices:
        logger.warning(
            f"No voices found in {settings.model.local_voices_dir}. "
            "Did you run models/download_models.py? "
            "The service will start but voice generation will fail until voices are available."
        )
    else:
        logger.info(
            f"Discovered {sum(len(v) for v in settings.all_voices.values())} voices across {len(settings.all_voices)} languages."
        )

    # Initialize TTS Engine with error handling
    try:
        engine = TTSEngine(settings)
        engine.initialize()
        set_tts_engine(engine)
    except Exception as e:
        logger.critical(f"Failed to initialize TTSEngine: {e}", exc_info=True)
        raise e

    # Yield control to the running application
    yield

    # Application Shutdown Phase
    logger.info("Shutting down Kokoro TTS Service...")
    if "engine" in locals():
        engine.voice_manager.stop_cleanup_loop()


# -----------------------------------------------------------------------------
# FastAPI Application Configuration
# -----------------------------------------------------------------------------
app = FastAPI(
    title=get_settings().service.name,
    version=get_settings().service.version,
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    HTTP middleware for request logging and timing.

    Logs all incoming requests with method, path, response status, and processing time.
    Provides observability into API usage patterns and performance.

    Args:
        request: The incoming HTTP request
        call_next: The next middleware/request handler in the chain

    Returns:
        Response: The HTTP response from the request handler
    """
    start_time = time.time()

    # Extract request details for logging
    path = request.url.path
    method = request.method

    # Process the request
    response = await call_next(request)

    # Calculate and log processing time
    process_time = (time.time() - start_time) * 1000
    status_code = response.status_code

    logger.info(f"{method} {path} - {status_code} - {process_time:.2f}ms")

    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled errors.

    Catches any unhandled exceptions that bubble up through the application,
    logs them with full stack traces, and returns a generic error response.

    Args:
        request: The HTTP request that caused the exception
        exc: The unhandled exception

    Returns:
        JSONResponse: Generic 500 Internal Server Error response
    """
    logger.error(f"Unhandled Global Exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
    )


# -----------------------------------------------------------------------------
# API Router Configuration
# -----------------------------------------------------------------------------
# Mount the V1 API router with /v1 prefix
# This includes all TTS endpoints (audio generation, health checks, etc.)
app.include_router(api_router, prefix="/v1")


@app.get("/", include_in_schema=False)
async def root():
    """
    Root endpoint providing service information and navigation links.

    Returns basic service metadata and links to documentation and API endpoints.
    This endpoint is excluded from the OpenAPI schema to keep the docs focused
    on actual TTS functionality.

    Returns:
        dict: Service information including name, version, and endpoint links
    """
    return {
        "service": get_settings().service.name,
        "version": get_settings().service.version,
        "docs": "/docs",
        "api_v1": "/v1/",
    }


# -----------------------------------------------------------------------------
# Development Server Configuration
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Development server entry point.

    Runs the FastAPI application with Uvicorn when this script is executed directly.
    Enables auto-reload for development and uses configuration from settings.
    """
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.service.host,
        port=settings.service.port,
        reload=True,
    )
