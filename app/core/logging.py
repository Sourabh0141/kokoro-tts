import logging
import sys

def setup_logging():
    """
    Configure the root logger to output to stdout in a structured format.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    logger.addHandler(handler)
    
    # Set levels for some noisy libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    return logger

logger = setup_logging()
