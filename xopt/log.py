import logging
import sys

from logging.handlers import RotatingFileHandler


stdout_log_format = "%(message)s"
file_log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
iso8601_datefmt = "%Y-%m-%dT%H:%M:%S%z"


current_handler = None


def validate_level(level) -> int:
    """
    Return a int for level comparison
    """
    if isinstance(level, int):
        levelno = level
    elif isinstance(level, str):
        levelno = logging.getLevelName(level)

    if isinstance(levelno, int):
        return levelno
    else:
        raise ValueError(f"Bad logging level: {levelno}")


def set_handler_with_logger(logger_name="xopt", file=sys.stdout, level="WARNING"):
    logger = logging.getLogger(logger_name)

    if isinstance(file, str):
        # handler = logging.FileHandler(file)
        handler = RotatingFileHandler(file, maxBytes=100000, backupCount=10)
        format = file_log_format

    else:
        handler = logging.StreamHandler(file)
        format = stdout_log_format

    levelno = validate_level(level)
    handler.setLevel(levelno)

    handler.setFormatter(logging.Formatter(format, datefmt=iso8601_datefmt))

    # Finally add the handler
    logger.addHandler(handler)

    if logger.getEffectiveLevel() > levelno:
        logger.setLevel(levelno)

    return handler


def configure_logger(logger_name="xopt", file=sys.stdout, level="INFO"):
    logger = logging.getLogger(logger_name)

    handler = set_handler_with_logger(logger_name=logger_name, file=file, level=level)

    global current_handler

    if current_handler in logger.handlers:
        logger.removeHandler(current_handler)
    logger.addHandler(handler)

    current_handler = handler
