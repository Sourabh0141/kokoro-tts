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
    repo_id: str = "hexgrad/Kokoro-82M"
    device: str = "cpu"  # 'cpu' or 'cuda'
    dtype: str = "fp32"  # 'fp32', 'fp16', 'bf16'

class Settings(BaseSettings):
    service: ServiceSettings = ServiceSettings()
    model: ModelSettings = ModelSettings()
    languages: Dict[str, str] = {
        "a": "af_heart",  # American English
        "b": "bf_emma",   # British English
        "j": "jf_alpha",  # Japanese
        "z": "zf_xiaobei",# Chinese
        "e": "ef_dora",   # Spanish
        "f": "ff_siwis",  # French
        "h": "hf_alpha",  # Hindi
        "i": "if_sara",   # Italian
        "p": "pf_dora",   # Portuguese
    }

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        case_sensitive=False
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()
