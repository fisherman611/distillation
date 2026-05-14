import logging
import sys
from pathlib import Path


def setup_logger(
    name: str, level: int = logging.INFO, log_file: str = "log.txt"
) -> logging.Logger:
    """
    Sets up and returns a logger with a specified name and level.
    Logs to both console and file. Avoids adding duplicate handlers.

    Args:
        name: Logger name
        level: Logging level (default: INFO)
        log_file: Path to log file (default: "log.txt")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        # Formatter for both handlers
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def clear_log_file(log_file: str = "log.txt"):
    """
    Clear the contents of the log file.

    Args:
        log_file: Path to log file to clear (default: "log.txt")
    """
    log_path = Path(log_file)
    if log_path.exists():
        log_path.write_text("", encoding="utf-8")
        print(f"Cleared log file: {log_file}")
    else:
        print(f"Log file does not exist: {log_file}")
