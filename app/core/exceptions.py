class TTSException(Exception):
    """Base exception for TTS service errors."""
    pass

class LanguageNotSupportedError(TTSException):
    """Raised when a requested language code is not configured or supported."""
    def __init__(self, lang_code: str):
        self.lang_code = lang_code
        super().__init__(f"Language code '{lang_code}' is not supported.")

class ModelLoadError(TTSException):
    """Raised when the TTS model configuration or weights fail to load."""
    pass

class VoiceNotFoundError(TTSException):
    """Raised when a specific voice ID cannot be found."""
    pass
