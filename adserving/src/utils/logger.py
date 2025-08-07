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

            # Apply color to the entire log line
            return f"{color}{formatted}{reset}"
        return super().format(record)


class FrameworkLogger:
    """Enhanced logger class with config-based level and colored output."""

    _instance: Optional["FrameworkLogger"] = None
    _logger: Optional[logging.Logger] = None
    _current_config: Optional[Dict[str, Any]] = None

    def __new__(cls) -> "FrameworkLogger":
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the logger if not already initialized."""
        if self._logger is None:
            self._setup_logger()

    def _setup_logger(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Setup the logger configuration.

        Args:
            config: Logger configuration dictionary
        """
        self._current_config = config or self._get_default_config()

        self._logger = logging.getLogger("adtrainingmodel")

        # Set log level from config
        log_level = self._parse_log_level(self._current_config.get("level", "INFO"))
        self._logger.setLevel(log_level)

        # Remove existing handlers to avoid duplicates
        for handler in self._logger.handlers[:]:
            self._logger.removeHandler(handler)

        # Create a console handler
        self._setup_console_handler()

        # Setup file handler if configured
        if self._current_config.get("file_logging", {}).get("enabled", False):
            self._setup_file_handler()

        # Prevent propagation to root logger
        self._logger.propagate = False

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default logger configuration.

        Returns:
            Default configuration dictionary
        """
        return {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "date_format": "%Y-%m-%d %H:%M:%S",
            "use_colors": True,
            "file_logging": {
                "enabled": False,
                "path": "logs/framework.log",
                "max_bytes": 10485760,  # 10MB
                "backup_count": 5,
                "level": "DEBUG",
            },
        }

    def _parse_log_level(self, level: str) -> int:
        """Parse log level string to numeric level.

        Args:
            level: Log level string

        Returns:
            Numeric log level
        """
        level_mapping = {
            "CRITICAL": logging.CRITICAL,
            "ERROR": logging.ERROR,
            "WARNING": logging.WARNING,
            "WARN": logging.WARNING,
            "INFO": logging.INFO,
            "DEBUG": logging.DEBUG,
            "NOTSET": logging.NOTSET,
        }

        normalized_level = str(level).upper()
        return level_mapping.get(normalized_level, logging.INFO)

    def _setup_console_handler(self) -> None:
        """Setup console handler with colored output."""
        if self._current_config is None or self._logger is None:
            return

        console_handler = logging.StreamHandler(sys.stdout)

        # Set console handler level
        console_level = self._parse_log_level(
            self._current_config.get(
                "console_level", self._current_config.get("level", "INFO")
            )
        )
        console_handler.setLevel(console_level)

        # Create colored formatter
        use_colors = self._current_config.get("use_colors", True)
        log_format = self._current_config.get(
            "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        date_format = self._current_config.get("date_format", "%Y-%m-%d %H:%M:%S")

        formatter = ColoredFormatter(fmt=log_format, use_colors=use_colors)
        formatter.datefmt = date_format

        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)

    def _setup_file_handler(self) -> None:
        """Setup file handler for logging to file."""
        if self._current_config is None or self._logger is None:
            return

        try:
            file_config = self._current_config.get("file_logging", {})
            log_file = file_config.get("path", "logs/framework.log")

            # Create a log directory if it doesn't exist
            log_dir = Path(log_file).parent
            log_dir.mkdir(parents=True, exist_ok=True)

            # Create a rotating file handler
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=file_config.get("max_bytes", 10485760),
                backupCount=file_config.get("backup_count", 5),
            )

            # Set file handler level
            file_level = self._parse_log_level(file_config.get("level", "DEBUG"))
            file_handler.setLevel(file_level)

            # Create plain formatter for file (no colors)
            log_format = self._current_config.get(
                "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            date_format = self._current_config.get("date_format", "%Y-%m-%d %H:%M:%S")

            file_formatter = logging.Formatter(log_format, date_format)
            file_handler.setFormatter(file_formatter)

            self._logger.addHandler(file_handler)

        except Exception as e:
            # Fallback to console logging if file setup fails
            if self._logger:
                self._logger.warning("Failed to setup file logging: %s", e)

    def configure_from_config(self, config: Dict[str, Any]) -> None:
        """Configure logger from config dictionary.

        Args:
            config: Configuration dictionary containing logger settings
        """
        # Extract logger configuration from main config
        logger_config = config.get("logging", {})

        # Merge with defaults
        merged_config = {**self._get_default_config(), **logger_config}

        # Reconfigure logger
        self._setup_logger(merged_config)

    def update_level(self, level: str) -> None:
        """Update logger level dynamically.

        Args:
            level: New log level
        """
        numeric_level = self._parse_log_level(level)
        if self._logger:
            self._logger.setLevel(numeric_level)

            # Update current config
            if self._current_config:
                self._current_config["level"] = level.upper()

    @property
    def logger(self) -> logging.Logger:
        """Get the logger instance."""
        if self._logger is None:
            self._setup_logger()

        # Đảm bảo _logger không còn None sau khi setup
        if self._logger is None:
            raise RuntimeError("Failed to initialize logger")

        return self._logger

    @property
    def current_level(self) -> str:
        """Get current log level as string."""
        if self._logger:
            return logging.getLevelName(self._logger.level)
        return "INFO"

    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)

    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)

    def critical(self, message: str) -> None:
        """Log critical message."""
        self.logger.critical(message)

    def exception(self, message: str) -> None:
        """Log an error message with exception information."""
        self.logger.exception(message)

    def set_level(self, level: str) -> None:
        """Set logging level (legacy method for compatibility).

        Args:
            level: Log level string
        """
        self.update_level(level)


def get_logger(config: Optional[Dict[str, Any]] = None) -> FrameworkLogger:
    """Get the singleton logger instance with optional configuration.

    Args:
        config: Optional configuration dictionary

    Returns:
        Configured logger instance
    """
    logger_instance = FrameworkLogger()

    # Configure if config is provided
    if config is not None:
        logger_instance.configure_from_config(config)

    return logger_instance
