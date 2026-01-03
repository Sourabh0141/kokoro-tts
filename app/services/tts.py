import sys
import os
import io
from typing import Dict, Optional, Any, List
from unittest.mock import patch
import torch
import soundfile as sf

# Import directly from the installed package
try:
    from kokoro import KPipeline
    # KModel might be internal or part of pipeline, but let's assume KPipeline handles most or we import if needed
    # For advanced usage we might need KModel, let's keep the import if available or rely on pipeline
    # The original code imported KModel from kokoro.model
    # Checking the pip package structure is hard without running, but usually:
    from kokoro import KModel # Attempt top level import first, or submodules
except ImportError:
    # Fallback/Retry standard paths if top-level fails (some packages structure differently)
    try:
        from kokoro.model import KModel
        from kokoro.pipeline import KPipeline
    except ImportError:
        print("Warning: Could not import kokoro modules. Ensure 'kokoro' is installed.")
        KModel = Any
        KPipeline = Any

from app.core.config import Settings
from app.core.exceptions import ModelLoadError, VoiceNotFoundError, LanguageNotSupportedError
from app.services.voice_manager import VoiceManager

class TTSEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = self._determine_device(settings.model.device)
        self.model: Optional[KModel] = None
        # Voice manager for lazy loading
        self.voice_manager = VoiceManager(
            voices_dir=settings.model.local_voices_dir,
            device=self.device,
            ttl_seconds=600  # 10 minutes
        )
        # Pipelines cache: language_name -> KPipeline
        self.pipelines_cache: Dict[str, KPipeline] = {}
        # Language code mapping: language_name -> language_code
        self.language_codes: Dict[str, str] = {}
        # Available voices metadata (config only, no tensors)
        self.available_voices: Dict[str, Dict[str, str]] = settings.all_voices
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
        Initialize TTS Engine:
        1. Load shared KModel from LOCAL files only
        2. Build language code mapping (lazy load voices)
        3. Start voice manager cleanup loop
        4. Set self.is_ready = True
        
        Voices are loaded on-demand in generate() method.
        """
        try:
            print(f"Initializing TTS Engine on device: {self.device}")
            
            # Step 1: Resolve Model Path
            local_model_path = os.path.join(self.settings.model.local_model_dir, "kokoro-v1_0.pth")
            if not os.path.exists(local_model_path):
                local_model_path = os.path.join(self.settings.model.local_model_dir, "kokoro-v0_19.pth")
            
            print(f"Loading Model from: {local_model_path}")
            
            # Ensure model file exists (or let it fail later if strict)
            # We check here to provide a clear error before attempting complex loading
            if not os.path.exists(local_model_path):
                 # If not found locally, we will try to let the library download it via the fallback
                 # in mock_download, but we need to know WHICH file to load for our cleaning logic.
                 # If we rely on download, we can't pre-clean. 
                 # Production Assumption: Models are present or will be downloaded to this path.
                 print(f"Model not found at {local_model_path}. KModel will attempt download.")

            # Step 2: Intelligent Single-Load & Clean
            # We load the weights MANUALLY once to clean the 'module.' prefixes.
            # Then we feed this cleaned dict to KModel via interception.
            
            state_dict = None
            if os.path.exists(local_model_path):
                print("  • Reading state dict from disk...")
                raw_state_dict = torch.load(local_model_path, map_location=self.device)
                
                # Check for "state_dict" wrapper key commonly used in PyTorch Lightning/etc
                if "state_dict" in raw_state_dict and isinstance(raw_state_dict["state_dict"], dict):
                     print("  • Detected 'state_dict' wrapper, unwrapping...")
                     raw_state_dict = raw_state_dict["state_dict"]

                print("  • Cleaning state dict keys...")
                state_dict = {}
                for key, value in raw_state_dict.items():
                    # Strip 'module.' prefix if present (DataParallel artifact)
                    clean_key = key[7:] if key.startswith("module.") else key
                    state_dict[clean_key] = value
                
                # Free raw memory immediately
                del raw_state_dict
                print(f"  • State dict ready in memory ({len(state_dict)} keys).")

            # MonkeyPatch hf_hub_download in kokoro.model to use local files with fallback
            import kokoro.model
            original_download = kokoro.model.hf_hub_download
            
            def mock_download(repo_id, filename, **kwargs):
                # Priority 1: Check Local Storage
                local_file = os.path.join(self.settings.model.local_model_dir, filename)
                if os.path.exists(local_file):
                    return local_file
                
                # Priority 2: Fallback to Hugging Face
                print(f"  • File {filename} not found locally, downloading from HF...")
                return original_download(repo_id, filename, **kwargs)
            
            kokoro.model.hf_hub_download = mock_download

            # Intercept torch.load to use our in-memory state_dict
            def load_injector(f, *args, **kwargs):
                # If we have a cleaned state_dict and KModel asks for the model file
                if state_dict is not None and isinstance(f, str) and (
                    "kokoro-v0_19.pth" in f or "kokoro-v1_0.pth" in f
                ):
                    print("  • Injecting cleaned weights from memory into KModel")
                    return state_dict
                
                # Otherwise pass through to real torch.load
                return torch.load(f, *args, **kwargs)

            try:
                # Apply the injection patch
                with patch('torch.load', side_effect=load_injector):
                    self.model = KModel(
                        repo_id=self.settings.model.repo_id, 
                        disable_complex=(self.settings.model.dtype != 'fp32') 
                    )
            finally:
                # Restore original download just in case
                kokoro.model.hf_hub_download = original_download            
            
            # Note: We do NOT need self.model.load_state_dict(state_dict) 
            # because KModel already initialized with it via load_injector!

            self.model.eval()
            self.model.to(self.device)
            print("✓ Model loaded successfully.")

            # Step 2: Build language code mapping (no voice loading)
            language_codes = {}
            for language, voices_dict in self.available_voices.items():
                for voice_name, voice_id in voices_dict.items():
                    lang_code = voice_id[0]
                    if lang_code not in language_codes:
                        language_codes[lang_code] = language
                        print(f"  Mapped language: {language} ({lang_code})")
            
            self.language_codes = language_codes

            # Step 3: Start voice manager cleanup loop
            self.voice_manager.start_cleanup_loop(check_interval_seconds=5)

            # Mark ready immediately (no voice loading blocking)
            self.is_ready = True
            print(f"\n✓ TTS Engine initialized and ready (lazy voice loading enabled)!")
            print(f"  Languages: {len(self.language_codes)}")
            print(f"  Total Available Voices: {sum(len(v) for v in self.available_voices.values())}")
            print(f"  Device: {self.device}")
            print(f"  Voice TTL: 10 minutes")

        except Exception as e:
            if isinstance(e, ModelLoadError):
                raise e
            print(f"Critical Error initializing TTS Engine: {e}")
            self.is_ready = False
            raise ModelLoadError(f"Failed to initialize TTSEngine: {str(e)}")

    def _get_or_create_pipeline(self, lang_code: str) -> KPipeline:
        """
        Get cached pipeline for language or create it lazily.
        
        Args:
            lang_code: Language code (2 chars, e.g., 'af', 'jf')
        
        Returns:
            KPipeline for the language
        """
        # Find language name from code
        language_name = None
        for code, lang_name in self.language_codes.items():
            if code == lang_code:
                language_name = lang_name
                break
        
        if language_name is None:
            raise LanguageNotSupportedError(lang_code, list(self.language_codes.values()))
        
        # Check if already cached
        if language_name in self.pipelines_cache:
            return self.pipelines_cache[language_name]
        
        # Create and cache pipeline
        try:
            pipeline = KPipeline(
                lang_code=lang_code,
                model=self.model,
                device=self.device,
                repo_id=self.settings.model.repo_id
            )
            self.pipelines_cache[language_name] = pipeline
            print(f"✓ Created pipeline for language: {language_name} ({lang_code})")
            return pipeline
        except Exception as e:
            raise ModelLoadError(f"Failed to create pipeline for {language_name}: {str(e)}")

    def _ensure_voice_loaded(self, language: str, voice: str, voice_id: str) -> torch.Tensor:
        """
        Ensure voice is loaded in memory (lazy load from disk if needed).
        
        Args:
            language: Language name
            voice: Voice name
            voice_id: Voice file ID
        
        Returns:
            Voice tensor
        """
        # Try to get from cache
        voice_tensor = self.voice_manager.get_voice(language, voice)
        if voice_tensor is not None:
            return voice_tensor
        
        # Load from disk
        voice_tensor = self.voice_manager.load_voice(language, voice, voice_id)
        return voice_tensor

    def validate_request(self, text: str, language: str, voice: str) -> str:
        """
        Validates request parameters.
        Returns the voice_id (file name) to use.
        
        Args:
            text: Text to synthesize
            language: Language name (e.g., "American English")
            voice: Voice name (e.g., "Bella (Female)")
        
        Returns:
            str: The voice_id (file name) to use
            
        Raises:
            LanguageNotSupportedError: If language not supported
            VoiceNotFoundError: If voice not found for language
            ValueError: If text invalid
        """
        # Check language exists
        if language not in self.available_voices:
            raise LanguageNotSupportedError(language, list(self.available_voices.keys()))
        
        # Check text validity
        if not text.strip():
            raise ValueError("Text cannot be empty.")
        
        if len(text) > 5000:
            raise ValueError("Text length exceeds maximum limit of 5000 characters.")
        
        # Check voice exists for language
        if voice not in self.available_voices[language]:
            available = ", ".join(self.available_voices[language].keys())
            raise VoiceNotFoundError(
                f"Voice '{voice}' not available for language '{language}'. "
                f"Available voices: {available}"
            )
        
        # Return the voice_id (file name)
        return self.available_voices[language][voice]

    def generate(self, text: str, language: str, voice: str, speed: float) -> bytes:
        """
        Generate audio from text using specified voice and language.
        
        Voices are loaded on-demand from disk (first request may be slower).
        
        Args:
            text: Text to synthesize
            language: Language name (e.g., "American English")
            voice: Voice name (e.g., "Bella (Female)")
            speed: Speed multiplier
            
        Returns:
            bytes: WAV audio data
            
        Raises:
            LanguageNotSupportedError: If language not supported
            VoiceNotFoundError: If voice not found
            ValueError: If text invalid or no audio generated
        """
        # Validate and get voice_id (file name)
        voice_id = self.validate_request(text, language, voice)
        
        # Get language code
        lang_code = None
        for code, name in self.language_codes.items():
            if name == language:
                lang_code = code
                break
        
        if lang_code is None:
            raise LanguageNotSupportedError(language, list(self.available_voices.keys()))
        
        # Get or create pipeline (lazy load)
        pipeline = self._get_or_create_pipeline(lang_code)
        
        # Ensure voice is loaded (lazy load from disk)
        voice_tensor = self._ensure_voice_loaded(language, voice, voice_id)
        
        # Register voice with pipeline
        pipeline.voices[voice_id] = voice_tensor
        
        # Generator that yields Result objects
        result_generator = pipeline(text, voice=voice_id, speed=speed)
        
        audio_segments = []
        for result in result_generator:
            if result.audio is not None:
                audio_segments.append(result.audio)
        
        if not audio_segments:
            raise ValueError("No audio generated from the input text.")
            
        # Concatenate all audio segments
        full_tensor = torch.cat(audio_segments, dim=0)
        
        # Move to CPU and convert to numpy
        audio_array = full_tensor.cpu().numpy()
        
        # Encode to WAV in memory
        buffer = io.BytesIO()
        sf.write(buffer, audio_array, 24000, format='WAV', subtype='PCM_16')
        buffer.seek(0)
        
        return buffer.read()
