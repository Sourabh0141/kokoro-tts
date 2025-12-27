from typing import List
from pydantic import BaseModel, Field

class TTSRequest(BaseModel):
    """
    Request model for text-to-speech generation.
    """
    text: str = Field(..., min_length=1, description="Text to synthesize")
    lang_code: str = Field(..., description="Language code (e.g., 'a', 'b', 'j')")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speed multiplier for playback")

class HealthResponse(BaseModel):
    """
    Response model for the service health check.
    """
    status: str
    loaded_languages: List[str]
    device: str
