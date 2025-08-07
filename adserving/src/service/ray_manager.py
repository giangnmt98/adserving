"""
Ray Serve initialization and management
Handles Ray Serve startup with proper configuration
"""

import os
import logging
from ray import serve
from ..config.config import Config
from .port_manager import PortManager
from adserving.src.utils.logger import get_logger

logger = get_logger()


class RayManager:
    """Manages Ray Serve initialization"""

    def __init__(self) -> None:
        """Initialize Ray Serve manager."""
        self.port_manager = PortManager()

    def initialize(self, config: Config) -> None:
        """Initialize Ray Serve with custom configuration."""
        logger.info("Initializing Ray Serve...")

        try:
            # Configure Ray logging
            self._configure_ray_logging(config)

            # Find available port for Ray Serve
            api_port = getattr(config, "api_port", 8000)
            ray_port = self.port_manager.find_available_port(api_port + 1, "127.0.0.1")

            logger.info(f"Ray Serve HTTP proxy using port {ray_port}")

            # Start Ray Serve
            serve.start(http_options={"host": "127.0.0.1", "port": ray_port})

            logger.info(f"Ray Serve initialized on port {ray_port}")

        except Exception as e:
            logger.warning(f"Ray Serve init failed/already running: {e}")
            logger.info("Continuing with existing Ray Serve instance...")

    def _configure_ray_logging(self, config: Config) -> None:
        """Configure Ray logging levels."""
        ray_log_level = config.ray.log_level
        logger.info(f"Setting Ray log level to: {ray_log_level}")

        # Set environment variable
        os.environ["RAY_LOG_LEVEL"] = ray_log_level

        # Configure Ray loggers directly
        ray_logger = logging.getLogger("ray")
        ray_logger.setLevel(getattr(logging, ray_log_level.upper()))

        ray_serve_logger = logging.getLogger("ray.serve")
        ray_serve_logger.setLevel(getattr(logging, ray_log_level.upper()))
