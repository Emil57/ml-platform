import logging

from ml_platform.config import settings
from ml_platform.utils import configure_logging, get_logger


def test_get_logger_returns_logger():
    logger = get_logger("test")

    assert isinstance(logger, logging.Logger)


def test_logger_name():
    logger = get_logger("training")

    assert logger.name == "training"


def test_configure_logger():
    configure_logging()
    root_logger = logging.getLogger()

    assert len(root_logger.handlers) > 0


def test_root_logger_level():
    configure_logging()

    root_logger = logging.getLogger()

    assert root_logger.level == getattr(logging, settings.log_level)


def test_get_logger_returns_same_instance():
    logger1 = get_logger("training")
    logger2 = get_logger("training")

    assert logger1 is logger2


def test_logger_logs_info_message(caplog):
    logger = get_logger("training")

    with caplog.at_level(logging.INFO):
        logger.info("Training started")

    assert "Training started" in caplog.text
