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
    
    # All available voices per language (full names to voice IDs)
    all_voices: Dict[str, Dict[str, str]] = {
        "American English": {
            "Alloy (Female)": "af_alloy",
            "Aoede (Female)": "af_aoede",
            "Bella (Female)": "af_bella",
            "Heart (Female)": "af_heart",
            "Jessica (Female)": "af_jessica",
            "Kore (Female)": "af_kore",
            "Nicole (Female)": "af_nicole",
            "Nova (Female)": "af_nova",
            "River (Female)": "af_river",
            "Sarah (Female)": "af_sarah",
            "Sky (Female)": "af_sky",
            "Adam (Male)": "am_adam",
            "Echo (Male)": "am_echo",
            "Eric (Male)": "am_eric",
            "Fenrir (Male)": "am_fenrir",
            "Liam (Male)": "am_liam",
            "Michael (Male)": "am_michael",
            "Onyx (Male)": "am_onyx",
            "Puck (Male)": "am_puck",
            "Santa (Male)": "am_santa",
        },
        "British English": {
            "Alice (Female)": "bf_alice",
            "Emma (Female)": "bf_emma",
            "Isabella (Female)": "bf_isabella",
            "Lily (Female)": "bf_lily",
            "Daniel (Male)": "bm_daniel",
            "Fable (Male)": "bm_fable",
            "George (Male)": "bm_george",
            "Lewis (Male)": "bm_lewis",
        },
        "Spanish": {
            "Dora (Female)": "ef_dora",
            "Alex (Male)": "em_alex",
            "Santa (Male)": "em_santa",
        },
        "French": {
            "Siwis (Female)": "ff_siwis",
        },
        "Hindi": {
            "Alpha (Female)": "hf_alpha",
            "Beta (Female)": "hf_beta",
            "Omega (Male)": "hm_omega",
            "Psi (Male)": "hm_psi",
        },
        "Italian": {
            "Sara (Female)": "if_sara",
            "Nicola (Male)": "im_nicola",
        },
        "Japanese": {
            "Alpha (Female)": "jf_alpha",
            "Gongitsune (Female)": "jf_gongitsune",
            "Nezumi (Female)": "jf_nezumi",
            "Tebukuro (Female)": "jf_tebukuro",
            "Kumo (Male)": "jm_kumo",
        },
        "Portuguese": {
            "Dora (Female)": "pf_dora",
            "Alex (Male)": "pm_alex",
            "Santa (Male)": "pm_santa",
        },
        "Chinese": {
            "Xiaobei (Female)": "zf_xiaobei",
            "Xiaoni (Female)": "zf_xiaoni",
            "Xiaoxiao (Female)": "zf_xiaoxiao",
            "Xiaoyi (Female)": "zf_xiaoyi",
            "Yunjian (Male)": "zm_yunjian",
            "Yunxi (Male)": "zm_yunxi",
            "Yunxia (Male)": "zm_yunxia",
            "Yunyang (Male)": "zm_yunyang",
        }
    }

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        case_sensitive=False
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()
