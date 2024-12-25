"""This module sets up a logger with a specified name and logging level."""

from logging import INFO, Formatter, Logger, StreamHandler, getLogger


def setup_logger(name: str, level: int = INFO) -> Logger:
    """Set up a logger with the specified name and level.

    Args:
        name (str): The name of the logger.
        level (int): The logging level.
    """
    logger = getLogger(name)
    fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
    formatter = Formatter(fmt)
    handler = StreamHandler()
    handler.setLevel(level)
    logger.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
