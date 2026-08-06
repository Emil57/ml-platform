"""
Centralized logging configuration for the ML Platform.
"""

import logging

from ml_platform.config import settings

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

_configured = False


def configure_logging() -> None:
    """Configure the logging system."""
    global _configured
    if _configured:
        # Avoid configuring logging more than once.
        return

    formatter = logging.Formatter(LOG_FORMAT)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(settings.log_level)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger instance."""
    configure_logging()
    logger = logging.getLogger(name)

    return logger
