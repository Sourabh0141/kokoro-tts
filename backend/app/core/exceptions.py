# =============================================================================
# Custom Exceptions - TTS Service Error Handling
# =============================================================================
# This module defines custom exception classes for the Kokoro TTS service.
# These exceptions provide specific error types for different failure scenarios,
# enabling better error handling and user feedback throughout the application.


class TTSException(Exception):
    """
    Base exception class for all TTS service errors.

    This is the root exception class for the TTS service error hierarchy.
    All TTS-specific exceptions should inherit from this class to enable
    consistent error handling and logging across the application.

    Usage:
        All TTS service exceptions inherit from this base class, allowing
        code to catch TTSException to handle any TTS-related error.
    """

    pass


class LanguageNotSupportedError(TTSException):
    """
    Exception raised when a requested language is not supported.

    This exception is raised when a client requests TTS generation for a language
    that is not configured or available in the current TTS service setup.

    Attributes:
        language: The requested language name that is not supported

    Args:
        language: The unsupported language name
        available_languages: Optional list of supported languages for error message

    Example:
        raise LanguageNotSupportedError("French", ["English", "Spanish", "Japanese"])
    """

    def __init__(self, language: str, available_languages: list = None):
        self.language = language
        if available_languages:
            msg = f"Language '{language}' is not supported. Available languages: {', '.join(available_languages)}"
        else:
            msg = f"Language '{language}' is not supported."
        super().__init__(msg)


class ModelLoadError(TTSException):
    """
    Exception raised when TTS model loading or initialization fails.

    This exception is raised when the Kokoro model cannot be loaded from disk,
    initialized properly, or when pipeline creation fails. This typically indicates
    missing model files, corrupted data, or incompatible model versions.

    Usage:
        Raised during TTS engine initialization when model loading fails.
        Indicates a critical service configuration issue that prevents operation.
    """

    pass


class VoiceNotFoundError(TTSException):
    """
    Exception raised when a requested voice is not available.

    This exception is raised when a client requests a specific voice that does not
    exist for the selected language, or when voice loading fails due to missing
    files or other issues.

    Args:
        message: Detailed error message explaining the voice availability issue

    Example:
        raise VoiceNotFoundError("Voice 'John (Male)' not available for language 'English'")
    """

    def __init__(self, message: str):
        super().__init__(message)
