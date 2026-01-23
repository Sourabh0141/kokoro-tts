# =============================================================================
# Configuration Management - Pydantic Settings for TTS Service
# =============================================================================
# This module provides centralized configuration management using Pydantic Settings.
# It handles environment variables, nested configuration, and dynamic voice discovery
# from the local filesystem for the Kokoro TTS service.

# -----------------------------------------------------------------------------
# Standard Library Imports
# -----------------------------------------------------------------------------
from functools import lru_cache
from typing import Dict, Any
import os
import glob
from pathlib import Path

# -----------------------------------------------------------------------------
# Third-Party Imports
# -----------------------------------------------------------------------------
from pydantic import BaseModel, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServiceSettings(BaseModel):
    """
    Service-level configuration settings.

    Defines the core service parameters for hosting, identification, and security.
    These settings control how the TTS service operates and is accessed.

    Attributes:
        host: Network interface to bind to (default: "0.0.0.0" for all interfaces)
        port: Port number for the service (default: 8880)
        name: Human-readable service name
        version: Service version string
        environment: Deployment environment ("development", "production", etc.)
        api_key: API key for service authentication
    """

    host: str = "0.0.0.0"
    port: int = 8880
    name: str = "Kokoro TTS Service"
    version: str = "1.0.0"
    environment: str = "development"
    api_key: str = "dev_api_key"


class ModelSettings(BaseModel):
    """
    Model and inference configuration settings.

    Defines parameters for the Kokoro TTS model, including repository information,
    compute device preferences, and local file paths for models and voices.

    Attributes:
        repo_id: Hugging Face repository identifier for the model
        device: Preferred compute device ("cpu", "cuda", "mps")
        dtype: Data type for model weights ("fp32", "fp16", "bf16")
        local_model_dir: Local directory path for model files
        local_voices_dir: Local directory path for voice embedding files
    """

    repo_id: str = "hexgrad/Kokoro-82M"
    device: str = "cpu"
    dtype: str = "fp32"
    local_model_dir: str = "models/model"
    local_voices_dir: str = "models/voices"


class Settings(BaseSettings):
    """
    Main application settings with dynamic voice discovery.

    The root settings class that combines service and model configurations.
    Includes automatic discovery of available voices from the local filesystem
    and environment variable support with nested configuration.

    Attributes:
        service: Service-level configuration settings
        model: Model and inference configuration settings
        all_voices: Dynamically discovered voice catalog

    Environment Variables:
        Configuration can be overridden via environment variables using double
        underscore notation (e.g., SERVICE__PORT=9000, MODEL__DEVICE=cuda).

        Examples:
        - SERVICE__API_KEY=your_secure_key
        - MODEL__DEVICE=cuda
        - MODEL__LOCAL_MODEL_DIR=/path/to/models
    """

    service: ServiceSettings = ServiceSettings()
    model: ModelSettings = ModelSettings()

    # Dynamically populated from local voice files
    all_voices: Dict[str, Dict[str, str]] = {}

    # Pydantic settings configuration
    model_config = SettingsConfigDict(
        env_file=".env",  # Load from .env file if present
        env_nested_delimiter="__",  # Use __ for nested settings
        case_sensitive=False,  # Case-insensitive environment variables
    )

    @model_validator(mode="after")
    def load_dynamic_voices(self) -> "Settings":
        """
        Dynamically discover and catalog available voices from the local filesystem.

        Scans the voices directory for .pt files and parses their filenames to build
        a structured catalog of available voices organized by language. This method
        implements the Kokoro voice naming convention for automatic discovery.

        Voice Filename Convention:
            {lang_code}{gender_code}_{voice_name}.pt
            - lang_code: Single character (a=American, b=British, j=Japanese, etc.)
            - gender_code: Single character (f=Female, m=Male)
            - voice_name: Descriptive name (bella, adam, etc.)

        Examples:
            - af_bella.pt → American English, Bella (Female)
            - bm_adam.pt → British English, Adam (Male)
            - jf_sakura.pt → Japanese, Sakura (Female)

        Returns:
            Settings: Self-reference for Pydantic model validation chaining

        Note:
            If voices directory doesn't exist, returns early with empty catalog.
            Unknown language codes default to "Unknown Language".
        """
        voices_dir = self.model.local_voices_dir

        # Early return if voices directory doesn't exist yet
        if not os.path.exists(voices_dir):
            return self

        # Language code to full name mapping (Kokoro convention)
        lang_map = {
            "a": "American English",
            "b": "British English",
            "e": "Spanish",  # e for Español
            "f": "French",
            "h": "Hindi",
            "i": "Italian",
            "j": "Japanese",
            "p": "Portuguese",
            "z": "Chinese",
        }

        # Gender code mapping
        gender_map = {"f": "Female", "m": "Male"}

        # Dictionary to collect discovered voices: language -> {display_name -> voice_id}
        discovered_voices: Dict[str, Dict[str, str]] = {}

        # Scan for all .pt voice files in the voices directory
        pattern = os.path.join(voices_dir, "*.pt")
        for file_path in glob.glob(pattern):
            filename = os.path.basename(file_path)  # e.g., "af_bella.pt"
            voice_id = os.path.splitext(filename)[0]  # e.g., "af_bella"

            # Parse voice ID components
            parts = voice_id.split("_")
            if len(parts) < 2:
                continue  # Skip malformed filenames

            prefix = parts[0]  # e.g., "af"
            name_part = parts[1]  # e.g., "bella"

            if len(prefix) < 2:
                continue  # Skip if prefix too short

            # Extract language and gender codes
            lang_code = prefix[0]  # e.g., "a"
            gender_code = prefix[1]  # e.g., "f"

            # Map codes to human-readable names
            language = lang_map.get(lang_code, "Unknown Language")
            gender = gender_map.get(gender_code, "")

            # Format display name: "Bella (Female)" or just "Bella"
            display_name = name_part.title()  # Capitalize first letter
            if gender:
                display_name = f"{display_name} ({gender})"

            # Add to discovered voices catalog
            if language not in discovered_voices:
                discovered_voices[language] = {}

            discovered_voices[language][display_name] = voice_id

        # Sort results for consistent output
        sorted_voices = {}
        for lang in sorted(discovered_voices.keys()):
            sorted_voices[lang] = dict(sorted(discovered_voices[lang].items()))

        self.all_voices = sorted_voices
        return self


@lru_cache
def get_settings() -> Settings:
    """
    Get the application settings singleton with caching.

    Creates and returns the main Settings instance, cached for performance.
    The LRU cache ensures the expensive voice discovery process only happens once,
    while still allowing environment variable changes to be picked up on restart.

    Returns:
        Settings: Configured application settings with dynamic voice catalog

    Performance:
        The @lru_cache decorator ensures this function returns the same Settings
        instance on subsequent calls, avoiding repeated file system scans and
        Pydantic model validation. The cache is cleared when the application restarts.

    Usage:
        settings = get_settings()
        print(f"Service name: {settings.service.name}")
        print(f"Available voices: {len(settings.all_voices)} languages")
    """
    return Settings()
