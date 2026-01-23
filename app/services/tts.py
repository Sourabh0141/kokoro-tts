# =============================================================================
# TTS Engine - Text-to-Speech Service Implementation
# =============================================================================
# Core TTS engine that manages Kokoro model loading, voice management,
# and audio generation with lazy loading and memory optimization.

# -----------------------------------------------------------------------------
# Standard Library Imports
# -----------------------------------------------------------------------------
import os
import io
import time
from typing import Dict, Optional, Any

# -----------------------------------------------------------------------------
# Third-Party Imports
# -----------------------------------------------------------------------------
from unittest.mock import patch
import torch
import soundfile as sf

# Kokoro TTS Model and Pipeline Imports
# Handles different installation scenarios and import paths
try:
    from kokoro import KPipeline
    from kokoro import KModel
    import kokoro.model  # For monkey patching hf_hub_download
except ImportError:
    try:
        from kokoro.model import KModel
        from kokoro.pipeline import KPipeline
        import kokoro.model  # For monkey patching hf_hub_download
    except ImportError:
        KModel = Any
        KPipeline = Any

        # Dummy module for monkey patching when kokoro not available
        class DummyModel:
            hf_hub_download = None

        kokoro = type("module", (), {"model": DummyModel()})()

# -----------------------------------------------------------------------------
# Local Application Imports
# -----------------------------------------------------------------------------
from app.core.config import Settings
from app.core.exceptions import (
    ModelLoadError,
    VoiceNotFoundError,
    LanguageNotSupportedError,
)
from app.services.voice_manager import VoiceManager
from app.core.logging import logger


class TTSEngine:
    """
    Text-to-Speech Engine using Kokoro model with lazy loading and memory management.

    Manages the complete TTS pipeline including model loading, voice management,
    language processing, and audio generation. Features intelligent memory management
    with lazy loading of voices and pipelines to optimize resource usage.

    Key Features:
    - Lazy loading of voices and language pipelines
    - Intelligent model loading with state dict preprocessing
    - Memory-efficient voice management with TTL-based cleanup
    - Multi-language support with automatic pipeline creation
    - Thread-safe audio generation
    - Comprehensive error handling and validation

    Attributes:
        settings: Application configuration settings
        device: PyTorch device for computation (cpu/cuda/mps)
        model: Loaded Kokoro model instance
        voice_manager: Voice loading and memory management
        pipelines_cache: Cache of language-specific pipelines
        language_codes: Mapping from language codes to names
        available_voices: Available voices organized by language
        is_ready: Flag indicating if engine is fully initialized
    """

    def __init__(self, settings: Settings):
        """
        Initialize the TTS engine with configuration.

        Sets up the basic engine structure and initializes the voice manager.
        The engine is not ready for use until initialize() is called.

        Args:
            settings: Application configuration containing model and voice settings
        """
        self.settings = settings
        self.device = self._determine_device(settings.model.device)
        self.model: Optional[KModel] = None

        # Initialize voice manager for lazy loading and memory management
        self.voice_manager = VoiceManager(
            voices_dir=settings.model.local_voices_dir,
            device=self.device,
            ttl_seconds=settings.cache.ttl_seconds,
            shutdown_timeout_seconds=settings.cache.shutdown_timeout_seconds,
        )

        # Cache for language-specific pipelines (lazy loaded)
        self.pipelines_cache: Dict[str, KPipeline] = {}

        # Mapping from language codes to language names
        self.language_codes: Dict[str, str] = {}

        # Available voices metadata (from config)
        self.available_voices: Dict[str, Dict[str, str]] = settings.all_voices

        # Engine readiness flag
        self.is_ready = False

    def _determine_device(self, config_device: str) -> str:
        """
        Determine the actual device to use based on configuration and hardware availability.

        Checks for CUDA and MPS availability in order of preference, falling back to CPU.

        Args:
            config_device: Preferred device from configuration ('cpu', 'cuda', 'mps')

        Returns:
            str: Actual device to use ('cpu', 'cuda', or 'mps')
        """
        if config_device == "cuda" and torch.cuda.is_available():
            return "cuda"
        if config_device == "mps" and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def initialize(self):
        """
        Initialize the TTS engine with model loading and pipeline setup.

        Performs complete engine initialization including:
        1. Model file resolution and loading with state dict preprocessing
        2. Language code mapping from available voices
        3. Voice manager cleanup loop startup
        4. Readiness flag setting

        This method handles various model file formats and provides fallback
        to HuggingFace download if local files are not available.

        Raises:
            ModelLoadError: If model loading or initialization fails
        """
        try:
            logger.info(f"Initializing TTS Engine on device: {self.device}")

            # Resolve model file path with version fallback
            # Try v1.0 first, then fall back to v0.19 for compatibility
            local_model_path = os.path.join(
                self.settings.model.local_model_dir, self.settings.model.file_v1
            )
            if not os.path.exists(local_model_path):
                local_model_path = os.path.join(
                    self.settings.model.local_model_dir, self.settings.model.file_v0_19
                )

            logger.info(f"Loading Model from: {local_model_path}")

            # Warn if model file doesn't exist (will fallback to HF download)
            if not os.path.exists(local_model_path):
                logger.warning(
                    f"Model not found at {local_model_path}. KModel will attempt download."
                )

            # Preprocess model state dict for clean loading
            state_dict = None
            if os.path.exists(local_model_path):
                logger.info("Reading state dict from disk...")
                raw_state_dict = torch.load(local_model_path, map_location=self.device)

                # Handle PyTorch Lightning checkpoint format
                if "state_dict" in raw_state_dict and isinstance(
                    raw_state_dict["state_dict"], dict
                ):
                    logger.info("Detected 'state_dict' wrapper, unwrapping...")
                    raw_state_dict = raw_state_dict["state_dict"]

                # Clean DataParallel artifacts (remove 'module.' prefixes)
                logger.debug("Cleaning state dict keys...")
                state_dict = {}
                for key, value in raw_state_dict.items():
                    # Remove 'module.' prefix if present (from DataParallel training)
                    clean_key = key[7:] if key.startswith("module.") else key
                    state_dict[clean_key] = value

                # Free memory immediately
                del raw_state_dict
                logger.info(f"State dict ready in memory ({len(state_dict)} keys).")

            # Setup intelligent model loading with local file preference
            # Monkey patch HuggingFace download to check local files first
            original_download = kokoro.model.hf_hub_download

            def mock_download(repo_id, filename, **kwargs):
                """Mock download function that prioritizes local files."""
                local_file = os.path.join(self.settings.model.local_model_dir, filename)
                if os.path.exists(local_file):
                    return local_file

                logger.info(
                    f"File {filename} not found locally, downloading from HF..."
                )
                return original_download(repo_id, filename, **kwargs)

            kokoro.model.hf_hub_download = mock_download

            # Torch.load interceptor to inject preprocessed state dict
            def load_injector(f, *args, **kwargs):
                """Intercept torch.load to use preprocessed state dict."""
                if (
                    state_dict is not None
                    and isinstance(f, str)
                    and (self.settings.model.file_v0_19 in f or self.settings.model.file_v1 in f)
                ):
                    logger.debug("Injecting cleaned weights from memory into KModel")
                    return state_dict

                # Fall back to normal torch.load for other files
                return torch.load(f, *args, **kwargs)

            try:
                with patch("torch.load", side_effect=load_injector):
                    self.model = KModel(
                        repo_id=self.settings.model.repo_id,
                        disable_complex=(self.settings.model.dtype != "fp32"),
                    )
            finally:
                kokoro.model.hf_hub_download = original_download

            self.model.eval()
            self.model.to(self.device)
            logger.info("Model loaded successfully.")

            language_codes = {}
            for language, voices_dict in self.available_voices.items():
                for voice_name, voice_id in voices_dict.items():
                    lang_code = voice_id[0]
                    if lang_code not in language_codes:
                        language_codes[lang_code] = language
                        logger.debug(f"Mapped language: {language} ({lang_code})")

            self.language_codes = language_codes

            self.voice_manager.start_cleanup_loop(
                check_interval_seconds=self.settings.cache.cleanup_interval_seconds
            )

            self.is_ready = True
            logger.info(
                "TTS Engine initialized and ready (lazy voice loading enabled)!"
            )
            logger.info(
                f"Languages: {len(self.language_codes)} | Total Available Voices: {sum(len(v) for v in self.available_voices.values())} | Device: {self.device}"
            )

        except Exception as e:
            if isinstance(e, ModelLoadError):
                raise e
            logger.critical(f"Critical Error initializing TTS Engine: {e}")
            self.is_ready = False
            raise ModelLoadError(f"Failed to initialize TTSEngine: {str(e)}")

    def _get_or_create_pipeline(self, lang_code: str) -> KPipeline:
        """
        Get cached pipeline for language or create it lazily.

        Pipelines are expensive to create and memory-intensive, so they're cached
        per language. This method ensures we have a pipeline ready for the requested
        language code.

        Args:
            lang_code: Two-character language code (e.g., 'af', 'jf')

        Returns:
            KPipeline: Ready-to-use pipeline for the language

        Raises:
            LanguageNotSupportedError: If language code is not configured
            ModelLoadError: If pipeline creation fails
        """
        # Resolve language name from code
        language_name = None
        for code, lang_name in self.language_codes.items():
            if code == lang_code:
                language_name = lang_name
                break

        if language_name is None:
            raise LanguageNotSupportedError(
                lang_code, list(self.language_codes.values())
            )

        # Return cached pipeline if available
        if language_name in self.pipelines_cache:
            return self.pipelines_cache[language_name]

        # Create new pipeline for this language
        try:
            logger.info(
                f"Creating new pipeline for language: {language_name} ({lang_code})"
            )
            pipeline = KPipeline(
                lang_code=lang_code,
                model=self.model,
                device=self.device,
                repo_id=self.settings.model.repo_id,
            )
            self.pipelines_cache[language_name] = pipeline
            return pipeline
        except Exception as e:
            logger.error(f"Failed to create pipeline for {language_name}: {e}")
            raise ModelLoadError(
                f"Failed to create pipeline for {language_name}: {str(e)}"
            )

    def _ensure_voice_loaded(
        self, language: str, voice: str, voice_id: str
    ) -> torch.Tensor:
        """
        Ensure voice embedding is loaded in memory (lazy load from disk if needed).

        First checks the voice manager cache for an existing loaded voice.
        If not found, triggers disk loading through the voice manager.

        Args:
            language: Language name (e.g., "American English")
            voice: Voice display name (e.g., "Bella (Female)")
            voice_id: Voice file identifier (e.g., "af_bella")

        Returns:
            torch.Tensor: The loaded voice embedding tensor
        """
        # Try to get from cache first (fast path)
        voice_tensor = self.voice_manager.get_voice(language, voice)
        if voice_tensor is not None:
            return voice_tensor

        # Cache miss - load from disk (slow path)
        logger.debug(f"Voice miss in cache, loading from disk: {voice} ({voice_id})")
        voice_tensor = self.voice_manager.load_voice(language, voice, voice_id)
        return voice_tensor

    def validate_request(self, text: str, language: str, voice: str) -> str:
        """
        Validate TTS request parameters and return voice identifier.

        Performs comprehensive validation of input parameters including:
        - Language availability
        - Text content and length limits
        - Voice availability for the selected language

        Args:
            text: Input text to synthesize
            language: Requested language name
            voice: Requested voice name

        Returns:
            str: Voice file identifier (e.g., "af_bella")

        Raises:
            LanguageNotSupportedError: If language is not available
            ValueError: If text is empty or exceeds length limit
            VoiceNotFoundError: If voice is not available for the language
        """
        # Validate language availability
        if language not in self.available_voices:
            logger.warning(f"Validation failed: Language '{language}' not supported.")
            raise LanguageNotSupportedError(
                language, list(self.available_voices.keys())
            )

        # Validate text content
        if not text.strip():
            logger.warning("Validation failed: Empty text.")
            raise ValueError("Text cannot be empty.")

        if len(text) > self.settings.limits.max_text_length:
            logger.warning(f"Validation failed: Text too long ({len(text)} chars).")
            raise ValueError(f"Text length exceeds maximum limit of {self.settings.limits.max_text_length} characters.")

        # Validate voice availability for language
        if voice not in self.available_voices[language]:
            available = ", ".join(self.available_voices[language].keys())
            logger.warning(
                f"Validation failed: Voice '{voice}' not found for '{language}'."
            )
            raise VoiceNotFoundError(
                f"Voice '{voice}' not available for language '{language}'. "
                f"Available voices: {available}"
            )

        # Return the voice file identifier
        return self.available_voices[language][voice]

    def generate(self, text: str, language: str, voice: str, speed: float) -> bytes:
        """
        Generate audio from text using specified voice and language.

        Main TTS generation method that orchestrates the complete audio synthesis pipeline:
        1. Input validation and voice resolution
        2. Language code mapping
        3. Pipeline acquisition (lazy loading)
        4. Voice tensor loading (lazy loading)
        5. Audio generation through Kokoro pipeline
        6. Audio concatenation and WAV encoding

        Args:
            text: Input text to synthesize (max 5000 characters)
            language: Language name (e.g., "American English")
            voice: Voice name (e.g., "Bella (Female)")
            speed: Playback speed multiplier (0.5 to 2.0)

        Returns:
            bytes: WAV audio data as bytes

        Raises:
            LanguageNotSupportedError: If language is not supported
            VoiceNotFoundError: If voice is not available
            ValueError: If text is invalid or generation fails
            Exception: For other processing errors
        """
        start_time = time.time()

        # Validate inputs and get voice identifier
        voice_id = self.validate_request(text, language, voice)

        # Resolve language code from language name
        lang_code = None
        for code, name in self.language_codes.items():
            if name == language:
                lang_code = code
                break

        if lang_code is None:
            logger.error(f"Language code map inconsistency for {language}")
            raise LanguageNotSupportedError(
                language, list(self.available_voices.keys())
            )

        try:
            # Get or create language-specific pipeline
            pipeline = self._get_or_create_pipeline(lang_code)

            # Ensure voice is loaded in memory
            voice_tensor = self._ensure_voice_loaded(language, voice, voice_id)

            # Register voice with pipeline for this generation
            pipeline.voices[voice_id] = voice_tensor

            # Generate audio through Kokoro pipeline
            result_generator = pipeline(text, voice=voice_id, speed=speed)

            # Collect audio segments from generator
            audio_segments = []
            for result in result_generator:
                if result.audio is not None:
                    audio_segments.append(result.audio)

            # Validate that audio was actually generated
            if not audio_segments:
                logger.warning(
                    f"No audio segments generated for text of length {len(text)}"
                )
                raise ValueError("No audio generated from the input text.")

            # Concatenate all audio segments
            full_tensor = torch.cat(audio_segments, dim=0)

            # Convert to numpy array for WAV encoding
            audio_array = full_tensor.cpu().numpy()

            # Encode to WAV format in memory
            buffer = io.BytesIO()
            sf.write(
                buffer,
                audio_array,
                self.settings.audio.sample_rate,
                format=self.settings.audio.format,
                subtype=self.settings.audio.subtype,
            )
            buffer.seek(0)

            # Get final audio data
            wav_data = buffer.read()
            duration_ms = (time.time() - start_time) * 1000

            # Log generation statistics
            logger.info(
                f"Generated audio: {len(text)} chars | {language}/{voice} | Speed: {speed} | Size: {len(wav_data)} bytes | Time: {duration_ms:.2f}ms"
            )

            return wav_data

        except Exception as e:
            # Log unexpected errors with full stack trace
            if not isinstance(
                e, (LanguageNotSupportedError, VoiceNotFoundError, ValueError)
            ):
                logger.error(f"Error during generation: {e}", exc_info=True)
            raise e
