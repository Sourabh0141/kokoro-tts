from typing import List, Dict
from pydantic import BaseModel, Field

class TTSRequest(BaseModel):
    """
    Request model for text-to-speech generation.
    """
    text: str = Field(..., min_length=1, description="Text to synthesize")
    language: str = Field(..., description="Language name (e.g., 'American English', 'British English', 'Japanese')")
    voice: str = Field(..., description="Voice name (e.g., 'Bella (Female)', 'Adam (Male)')")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speed multiplier for playback")

class HealthResponse(BaseModel):
    """
    Response model for the service health check.
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
    """Detail about a loaded voice"""
    language: str
    voice: str
    voice_id: str
    size_mb: float
    age_seconds: float
    ttl_remaining_seconds: int


class StatusResponse(BaseModel):
    """
    Response model for detailed voice status.
    """
    voices: List[VoiceStatusDetail]
    total_memory_mb: float
    cleanup_checks_performed: int
    voices_unloaded_total: int


class VoicesResponse(BaseModel):
    """
    Response model for the voices list endpoint.
    """
    languages: Dict[str, Dict[str, str]]  # Map of language to voice names and IDs
    total_voices: int
    total_languages: int
