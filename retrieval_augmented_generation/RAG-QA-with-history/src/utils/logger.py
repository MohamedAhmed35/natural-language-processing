# src/utils/logger.py
"""
logger.py
------------
Configures logging for the RAG-QA-with-History project.
- Logs go to console and rotating files.
- app.log keeps all logs.
- Logs stored under /src/runtime/logs.
"""

import logging
import os
from logging.handlers import TimedRotatingFileHandler
from config import settings


def setup_logging():
    """
    Sets up the logging configuration for the application.
    Logs will be written to a file in the runtime directory and to the console.
    """
    # Ensure log directory exists
    log_dir = os.path.join(settings.PROJECT_ROOT, "runtime", "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, 'app.log')

    
    # Resolve log level from settings
    log_level =  getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
   
    # Create root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)


    # Clear existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()


    # Create formatter
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)


    # App log handlers
    fh_app = TimedRotatingFileHandler(
        log_file, when="midnight", backupCount=7, encoding="utf-8"
        )
    fh_app.setLevel(log_level)
    fh_app.setFormatter(fmt)
    logger.addHandler(fh_app)
    


def get_logger(name: str):
    return logging.getLogger(name)