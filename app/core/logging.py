import logging
import sys
import os

def setup_logging():
    """
    Configure the root logger to output to stdout in a structured format.
    Reads LOG_LEVEL from environment, defaults to INFO.
    """
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    logger = logging.getLogger()
    logger.setLevel(log_level)

    handler = logging.StreamHandler(sys.stdout)
    # detailed format with timestamps
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    # Remove existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
    
    logger.addHandler(handler)
    
    # Set levels for libraries
    # uvicorn.access is handled by our own middleware usually, or we keep it enabled
    # If we add custom request logging, we might want to silence uvicorn.access
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    
    # Set third-party noise to warning
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)

    return logger

# Initialize logging immediately on import
logger = setup_logging()

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name."""
    return logging.getLogger(name)
