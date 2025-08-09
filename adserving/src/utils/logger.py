"""Enhanced logger implementation with config-based level and colored output."""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for different log levels."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green 
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def __init__(self, fmt: Optional[str] = None, use_colors: bool = True) -> None:
        """Initialize colored formatter.

        Args:
            fmt: Log format string
            use_colors: Whether to use colors in output
        """
        super().__init__(fmt)
        self.use_colors = use_colors and self._supports_color()
        # Add line number and filename to the log format if not already present
        if fmt is None:
            self._fmt = "%(asctime)s [%(name)s:%(lineno)d] [%(levelname)s] %(message)s"

    def _supports_color(self) -> bool:
        """Check if terminal supports color output.

        Returns:
            True if colors are supported
        """
        # Check if running in terminal and supports colors 
        if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
            return False

        # Check environment variables
        if os.getenv("NO_COLOR"):
            return False

        if os.getenv("FORCE_COLOR"):
            return True

        # Check TERM environment variable
        term = os.getenv("TERM", "")
        if "color" in term or term in ["xterm", "xterm-256color", "screen"]:
            return True

        return False

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors.

        Args:
            record: Log record to format

        Returns:
            Formatted log message
        """
        if self.use_colors:
            level_name = record.levelname
            color = self.COLORS.get(level_name, "")
            reset = self.COLORS["RESET"]

            # Format the message first without colors
            formatted = super().format(record)

            # Add line number and filename
            filename = record.filename
            lineno = record.lineno
            formatted = f"{color}{formatted} [{filename}:{lineno}]{reset}"

            return formatted

        # Add line number and filename without colors
        formatted = super().format(record)
        filename = record.filename
        lineno = record.lineno
        formatted = f"{formatted} [{filename}:{lineno}]"

        return formatted


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance with enhanced configuration.
    
    Args:
        name: Logger name. If None, uses the calling module name.
        
    Returns:
        Configured logger instance
    """
    if name is None:
        # Get the caller's module name
        import inspect
        frame = inspect.currentframe()
        try:
            if frame and frame.f_back:
                caller_module = frame.f_back.f_globals.get('__name__', 'unknown')
                name = caller_module
            else:
                name = 'adserving'
        finally:
            del frame
    
    # Get or create logger
    logger = logging.getLogger(name)
    
    # Don't configure if already configured
    if logger.handlers:
        return logger
    
    # Set default level
    logger.setLevel(logging.INFO)
    
    # Create console handler with colored formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = ColoredFormatter(
        fmt="%(asctime)s [%(name)s] [%(levelname)s] %(message)s",
        use_colors=True
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Prevent duplicate logs
    logger.propagate = False
    
    return logger


def setup_file_logging(
    logger: logging.Logger,
    log_file: str,
    max_bytes: int = 100 * 1024 * 1024,  # 100MB
    backup_count: int = 5
) -> None:
    """Setup file logging for a logger.
    
    Args:
        logger: Logger instance to configure
        log_file: Path to log file
        max_bytes: Maximum file size before rotation
        backup_count: Number of backup files to keep
    """
    # Ensure log directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create rotating file handler
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    
    # Use plain formatter for file (no colors)
    file_formatter = logging.Formatter(
        fmt="%(asctime)s [%(name)s:%(lineno)d] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
