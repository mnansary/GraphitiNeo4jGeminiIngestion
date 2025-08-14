# gemini_api_manager/utils/logging_config.py

import sys
from pathlib import Path
from loguru import logger

def setup_logging(
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    log_dir: str = "logs",
    retention: str = "10 days",
    rotation: str = "10 MB",
):
    """
    Sets up a comprehensive, multi-sink logger for the application.

    This function configures two log handlers:
    1. A console logger with rich formatting and colors for immediate feedback.
    2. A file logger that saves logs to a rotating file for persistence and auditing.

    Args:
        console_level (str, optional): The minimum log level to display on the console.
                                       Defaults to "INFO". Can be "DEBUG", "INFO", "WARNING", "ERROR".
        file_level (str, optional): The minimum log level to write to the log file.
                                    Defaults to "DEBUG" to capture all details.
        log_dir (str, optional): The directory where log files will be stored.
                                 Defaults to "logs". Will be created if it doesn't exist.
        retention (str, optional): How long to keep old log files.
                                   Defaults to "10 days".
        rotation (str, optional): The condition for creating a new log file.
                                  Defaults to "10 MB". Can also be time-based (e.g., "1 week").
    """
    # 1. Start with a clean slate by removing any default handlers.
    logger.remove()

    # 2. Configure the Console Logger for interactive, real-time output.
    # This format is verbose and colorized for easy reading during development.
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    logger.add(
        sys.stderr,  # Send logs to standard error, a common practice for application logs.
        level=console_level.upper(),
        format=console_format,
        colorize=True,  # Enable color-coded output.
        backtrace=True, # Show the full stack trace on exceptions.
        diagnose=True,  # Add exception variable values for easier debugging.
    )

    # 3. Configure the File Logger for persistent storage.
    # This is crucial for auditing, and debugging issues after the fact.
    log_file_path = Path(log_dir) / "app_{time}.log"
    
    # Ensure the log directory exists.
    Path(log_dir).mkdir(exist_ok=True)
    
    # This format is plain text, suitable for file storage and log parsers.
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} | "
        "{message}"
    )
    logger.add(
        log_file_path,
        level=file_level.upper(),
        format=file_format,
        colorize=False, # Colors are not needed in a file.
        rotation=rotation, # Automatically rotate to a new file when the size/time limit is reached.
        retention=retention, # Automatically clean up old log files.
        compression="zip", # Compress rotated log files to save space.
        enqueue=True, # Make logging calls non-blocking, important for high-performance apps.
        backtrace=True,
        diagnose=True, # Also log full exception details to the file.
    )

    logger.info("Logging configured. Console level: {}, File level: {}.", console_level, file_level)