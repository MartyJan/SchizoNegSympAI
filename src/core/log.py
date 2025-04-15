import logging
from datetime import datetime
from pathlib import Path


def get_logger(log_folder: Path) -> logging.Logger:
    """
    Create and configure a logger for logging to a file.

    Args:
        log_folder (Path): The path to the directory where log files will be stored.

    Returns:
        logging.Logger: A configured logger instance.
    """
    # Ensure directory exists
    log_folder.mkdir(exist_ok=True)

    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    # Set time zone
    formatter.converter = lambda *args: datetime.now().astimezone().timetuple()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_folder / "log.txt", mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
