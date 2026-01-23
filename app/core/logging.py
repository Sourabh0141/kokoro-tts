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
import os


def setup_logging():
    """
    Configure the root logger with structured console output and noise reduction.

    Sets up logging with the following features:
    - Configurable log level via LOG_LEVEL environment variable
    - Structured format with timestamps, levels, and logger names
    - Console output to stdout for containerized deployment
    - Automatic handler cleanup to prevent duplicate logs
    - Third-party library noise reduction (uvicorn, transformers, torch)

    The function configures the root logger to use INFO level by default,
    but this can be overridden with LOG_LEVEL environment variable.

    Returns:
        logging.Logger: Configured root logger instance
    """
    # Get log level from environment, default to INFO
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Create console handler for structured output
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)

    # Clear existing handlers to prevent duplicate logging
    if logger.hasHandlers():
        logger.handlers.clear()

    # Add our handler to the root logger
    logger.addHandler(handler)

    # Reduce noise from third-party libraries
    # Uvicorn access logs are handled by our custom middleware
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    # Suppress verbose logging from ML libraries during inference
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)

    return logger


# -----------------------------------------------------------------------------
# Global Logger Instance
# -----------------------------------------------------------------------------
# Initialize the root logger with our configuration
# This is called once at module import time
logger = setup_logging()
