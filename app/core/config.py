from functools import lru_cache
from typing import Dict, Any
import os
import glob
from pathlib import Path
from pydantic import BaseModel, model_validator
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
    # This will be populated dynamically from disk
    all_voices: Dict[str, Dict[str, str]] = {}

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        case_sensitive=False
    )

    @model_validator(mode='after')
    def load_dynamic_voices(self) -> 'Settings':
        """
        Dynamically populate all_voices by scanning the local_voices_dir.
        """
        voices_dir = self.model.local_voices_dir
        
        # If directory doesn't exist yet (e.g. before download), keep empty
        if not os.path.exists(voices_dir):
            return self

        # Map prefix char to Language Name
        # Based on Kokoro convention: a=American, b=British, j=Japanese, z=Chinese, etc.
        lang_map = {
            'a': "American English",
            'b': "British English",
            'e': "Spanish", # e for Espanol
            'f': "French",
            'h': "Hindi",
            'i': "Italian",
            'j': "Japanese",
            'p': "Portuguese",
            'z': "Chinese",
        }

        # Map gender char
        gender_map = {
            'f': "Female",
            'm': "Male"
        }

        discovered_voices: Dict[str, Dict[str, str]] = {}

        # Scan for .pt files
        pattern = os.path.join(voices_dir, "*.pt")
        for file_path in glob.glob(pattern):
            filename = os.path.basename(file_path) # e.g., af_bella.pt
            voice_id = os.path.splitext(filename)[0] # e.g., af_bella
            
            parts = voice_id.split('_')
            if len(parts) < 2:
                continue # Skip invalid format
            
            prefix = parts[0] # e.g., af
            name_part = parts[1] # e.g., bella
            
            if len(prefix) < 2:
                continue
                
            lang_code = prefix[0]
            gender_code = prefix[1]
            
            language = lang_map.get(lang_code, "Unknown Language")
            gender = gender_map.get(gender_code, "")
            
            # Format display name: "Bella (Female)" or just "Bella"
            display_name = name_part.title()
            if gender:
                display_name = f"{display_name} ({gender})"
            
            if language not in discovered_voices:
                discovered_voices[language] = {}
            
            discovered_voices[language][display_name] = voice_id

        # Update the field
        # Sort keys for consistent output
        sorted_voices = {}
        for lang in sorted(discovered_voices.keys()):
            sorted_voices[lang] = dict(sorted(discovered_voices[lang].items()))
            
        self.all_voices = sorted_voices
        return self

@lru_cache
def get_settings() -> Settings:
    return Settings()