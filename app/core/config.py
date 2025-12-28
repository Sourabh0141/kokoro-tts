from functools import lru_cache
from typing import Dict
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

class ServiceSettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8880
    name: str = "Kokoro TTS Service"
    version: str = "1.0.0"
    environment: str = "development"
    api_key: str = "dev_api_key"

class ModelSettings(BaseModel):
    # repo_id is optional now, used only if local path not found or for reference
    repo_id: str = "hexgrad/Kokoro-82M" 
    device: str = "cpu"  # 'cpu' or 'cuda'
    dtype: str = "fp32"  # 'fp32', 'fp16', 'bf16'
    local_model_dir: str = "models/model"
    local_voices_dir: str = "models/voices"

class Settings(BaseSettings):
    service: ServiceSettings = ServiceSettings()
    model: ModelSettings = ModelSettings()
    # Default Default mappings (can be overridden by env vars)
    languages: Dict[str, str] = {
        "a": "af_bella",  # Default American
        "b": "bf_emma",   # Default British
        "j": "jf_alpha",  # Default Japanese
        "z": "zf_xiaobei",# Default Chinese
        "e": "ef_dora",   # Default Spanish
        "f": "ff_siwis",  # Default French
        "h": "hf_alpha",  # Default Hindi
        "i": "if_sara",   # Default Italian
        "p": "pf_dora",   # Default Portuguese
    }

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        case_sensitive=False
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()
