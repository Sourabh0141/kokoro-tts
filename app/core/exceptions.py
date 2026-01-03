class TTSException(Exception):
    """Base exception for TTS service errors."""
    pass

class LanguageNotSupportedError(TTSException):
    """Raised when a requested language is not configured or supported."""
    def __init__(self, language: str, available_languages: list = None):
        self.language = language
        if available_languages:
            msg = f"Language '{language}' is not supported. Available languages: {', '.join(available_languages)}"
        else:
            msg = f"Language '{language}' is not supported."
        super().__init__(msg)

class ModelLoadError(TTSException):
    """Raised when the TTS model configuration or weights fail to load."""
    pass

class VoiceNotFoundError(TTSException):
    """Raised when a specific voice cannot be found for the given language."""
    def __init__(self, message: str):
        super().__init__(message)
