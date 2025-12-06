"""Simple logging configuration for Drug Assistant AI Agent."""

import logging
import sys


# Log format template with timestamp and file path
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(pathname)s:%(lineno)d - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a simple configured logger for a module.

    Args:
        name: Logger name (typically __name__ of the module)
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Only configure if no handlers exist (prevent duplicate handlers)
    if not logger.handlers:
        logger.setLevel(level)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        # Formatter with timestamp and file path
        formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.propagate = False

    return logger


def configure_root_logger(debug: bool = False):
    """
    Configure the root logger for the entire application.

    Args:
        debug: Whether to enable debug logging
    """
    level = logging.DEBUG if debug else logging.INFO

    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    root_logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Formatter with timestamp and file path
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    console_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)

    # Reduce noise from external libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('azure').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.INFO)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
