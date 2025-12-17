import io
import logging
import tempfile
import os
import pytest
from xopt import log


def test_validate_level():
    assert log.validate_level(logging.INFO) == logging.INFO
    assert log.validate_level("INFO") == logging.INFO
    with pytest.raises(ValueError):
        log.validate_level("NOTALEVEL")


def test_set_handler_with_logger_stdout():
    logger_name = "xopt.test.stdout"
    stream = io.StringIO()
    handler = log.set_handler_with_logger(
        logger_name=logger_name, file=stream, level="INFO"
    )
    logger = logging.getLogger(logger_name)
    logger.info("hello world")
    handler.flush()
    assert "hello world" in stream.getvalue()
    logger.removeHandler(handler)


def test_set_handler_with_logger_file():
    logger_name = "xopt.test.file"
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        handler = log.set_handler_with_logger(
            logger_name=logger_name, file=tmp.name, level="INFO"
        )
        logger = logging.getLogger(logger_name)
        logger.info("file test message")
        handler.flush()
        logger.removeHandler(handler)
        with open(tmp.name) as f:
            contents = f.read()
        assert "file test message" in contents
    os.remove(tmp.name)


def test_configure_logger_replaces_handler():
    logger_name = "xopt.test.configure"
    stream1 = io.StringIO()
    stream2 = io.StringIO()
    # First configure
    _ = log.configure_logger(logger_name=logger_name, file=stream1, level="INFO")
    logger = logging.getLogger(logger_name)
    logger.info("first message")
    # Reconfigure with a new stream
    log.configure_logger(logger_name=logger_name, file=stream2, level="INFO")
    logger.info("second message")
    # Get the last handler attached to the logger
    handler2 = logger.handlers[-1]
    handler2.flush()
    assert "second message" in stream2.getvalue()
    # The old handler should not receive new logs
    assert "second message" not in stream1.getvalue()
    logger.removeHandler(handler2)
