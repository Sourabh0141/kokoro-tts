# =============================================================================
# API Router V1 - FastAPI Route Configuration for TTS Service
# =============================================================================
# This module configures the V1 API router for the Kokoro TTS service.
# It organizes and tags all TTS-related endpoints under a unified router
# that can be mounted in the main FastAPI application.

# -----------------------------------------------------------------------------
# Third-Party Imports
# -----------------------------------------------------------------------------
from fastapi import APIRouter

# -----------------------------------------------------------------------------
# Local Application Imports
# -----------------------------------------------------------------------------
from app.api.v1 import endpoints

# -----------------------------------------------------------------------------
# V1 API Router Configuration
# -----------------------------------------------------------------------------
# Create the main V1 API router for TTS endpoints
# This router will be mounted at /v1 in the main application
api_router = APIRouter()

# Include the TTS endpoints router with appropriate tags
# Tags help organize endpoints in the auto-generated API documentation
api_router.include_router(endpoints.router, tags=["tts"])
