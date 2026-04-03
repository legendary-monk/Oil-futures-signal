"""
logger.py — Unified Logging
"""
import logging
import sys
import config


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(config.LOG_LEVEL)
    formatter = logging.Formatter(fmt=config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT)

    try:
        fh = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except (IOError, PermissionError) as e:
        print(f"[WARNING] Cannot write log file: {e}")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger
