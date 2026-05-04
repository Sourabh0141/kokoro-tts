# =============================================================================
# Logging Configuration - Centralized Logging Setup for TTS Service
# =============================================================================
# This module provides centralized logging configuration for the Kokoro TTS service.
# It sets up structured console logging with configurable levels and reduces noise
# from third-party libraries for cleaner application logs.

# -----------------------------------------------------------------------------
# Standard Library Imports
# -----------------------------------------------------------------------------
import logging
import sys

# -----------------------------------------------------------------------------
# Local Application Imports
# -----------------------------------------------------------------------------
from app.core.config import get_settings


def setup_logging():
    """
    Configure the root logger with structured console output and noise reduction.

    Sets up logging with the following features:
    - Configurable log level via configuration
    - Structured format with timestamps, levels, and logger names
    - Console output to stdout for containerized deployment
    - Automatic handler cleanup to prevent duplicate logs
    - Third-party library noise reduction (uvicorn, transformers, torch)

    The function configures the root logger based on application settings.

    Returns:
        logging.Logger: Configured root logger instance
    """
    settings = get_settings()

    # Get log level from settings
    log_level_str = settings.service.log_level.upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Create console handler for structured output
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt=settings.logging.date_format,
    )
    handler.setFormatter(formatter)

    # Clear existing handlers to prevent duplicate logging
    if logger.hasHandlers():
        logger.handlers.clear()

    # Add our handler to the root logger
    logger.addHandler(handler)

    # Reduce noise from third-party libraries
    # Uvicorn access logs are handled by our custom middleware
    logging.getLogger("uvicorn.access").setLevel(settings.logging.uvicorn_level)

    # Suppress verbose logging from ML libraries during inference
    logging.getLogger("transformers").setLevel(settings.logging.transformers_level)
    logging.getLogger("torch").setLevel(settings.logging.torch_level)

    return logger


# -----------------------------------------------------------------------------
# Global Logger Instance
# -----------------------------------------------------------------------------
# Initialize the root logger with our configuration
# This is called once at module import time
logger = setup_logging()
