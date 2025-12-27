import sys
import os
from typing import Dict, Optional, Any
import torch

# Ensure kokoro_tts is in sys.path
kokoro_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'kokoro_tts'))
if kokoro_path not in sys.path:
    sys.path.append(kokoro_path)

try:
    from kokoro.model import KModel
    from kokoro.pipeline import KPipeline
except ImportError as e:
    print(f"Warning: Could not import kokoro modules: {e}")
    # Fallback types for static analysis if imports fail
    KModel = Any
    KPipeline = Any

from app.core.config import Settings
from app.core.exceptions import ModelLoadError, VoiceNotFoundError

class TTSEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = self._determine_device(settings.model.device)
        self.model: Optional[KModel] = None
        self.pipelines: Dict[str, KPipeline] = {}
        self.voice_map: Dict[str, str] = settings.languages
        self.is_ready = False

    def _determine_device(self, config_device: str) -> str:
        """
        Determine the actual device to use based on config and hardware availability.
        """
        if config_device == "cuda" and torch.cuda.is_available():
            return "cuda"
        if config_device == "mps" and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def initialize(self):
        """
        1. Load shared KModel to self.device.
        2. Iterate through self.settings.languages:
           a. Create KPipeline(lang_code, model=self.model)
           b. Load Voice: pipeline.load_voice(voice_id)
           c. Store pipeline in self.pipelines[lang_code]
        3. Set self.is_ready = True
        """
        try:
            print(f"Initializing TTS Engine on device: {self.device}")
            
            # 1. Load shared KModel
            print(f"Loading Model: {self.settings.model.repo_id}")
            self.model = KModel(
                repo_id=self.settings.model.repo_id,
                disable_complex=(self.settings.model.dtype != 'fp32') # approximate logic, adjust if needed
            )
            self.model.eval()
            self.model.to(self.device)
            print("Model loaded successfully.")

            # 2. Initialize Pipelines for each language
            for lang_code, voice_id in self.settings.languages.items():
                print(f"Initializing pipeline for language: {lang_code} with voice: {voice_id}")
                
                # Create pipeline sharing the same model instance to save memory
                pipeline = KPipeline(
                    lang_code=lang_code, 
                    model=self.model,
                    device=self.device
                )
                
                # Pre-load the assigned voice
                # KPipeline.load_voice returns the voice embedding tensor, but also caches it internally
                try:
                    pipeline.load_voice(voice_id)
                except Exception as e:
                    print(f"Failed to load voice {voice_id}: {e}")
                    raise VoiceNotFoundError(f"Voice '{voice_id}' could not be loaded.")
                
                self.pipelines[lang_code] = pipeline
            
            self.is_ready = True
            print("TTS Engine initialized and ready.")

        except Exception as e:
            print(f"Critical Error initializing TTS Engine: {e}")
            self.is_ready = False
            raise ModelLoadError(f"Failed to initialize TTSEngine: {str(e)}")

    def validate_request(self, text: str, lang_code: str):
        """
        Checks if lang_code exists.
        Checks if text length is within limits (optional safety).
        """
        pass

    def generate(self, text: str, lang_code: str, speed: float) -> bytes:
        """
        1. Get pipeline = self.pipelines[lang_code]
        2. Get voice_id = self.voice_map[lang_code]
        3. result_generator = pipeline(text, voice=voice_id, speed=speed)
        4. audio_segments = []
        5. For result in result_generator:
               if result.audio is not None:
                   audio_segments.append(result.audio)
        6. full_tensor = torch.cat(audio_segments, dim=0)
        7. Convert full_tensor to WAV bytes using soundfile (Save to BytesIO)
        8. Return bytes
        """
        pass
