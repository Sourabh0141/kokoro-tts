from typing import Optional
from fastapi import Header, HTTPException, status
from app.core.config import get_settings
from app.services.tts import TTSEngine

# Global singleton instance, to be initialized by main.py
_tts_engine: Optional[TTSEngine] = None

def set_tts_engine(engine: TTSEngine):
    global _tts_engine
    _tts_engine = engine

def get_tts_engine() -> TTSEngine:
    if _tts_engine is None:
        raise RuntimeError("TTSEngine is not initialized.")
    return _tts_engine

async def api_key_auth(x_api_key: str = Header(..., description="API Key for authentication")):
    settings = get_settings()
    if x_api_key != settings.service.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
