import logging
from datetime import datetime
from pathlib import Path
from typing import Optional


def get_logger(log_dir: Optional[Path] = None) -> logging.Logger:
    """
    Create and configure a logger for logging to a file or terminal.

    Args:
        log_dir (Optional[Path]): The path to the directory where log files will be stored.

    Returns:
        logging.Logger: A configured logger instance.
    """

    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    # Set time zone
    formatter.converter = lambda *args: datetime.now().astimezone().timetuple()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers
    handler_types = {type(h) for h in logger.handlers}

    if log_dir is not None and logging.FileHandler not in handler_types:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "log.txt", mode="a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if log_dir is None and logging.StreamHandler not in handler_types:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger
