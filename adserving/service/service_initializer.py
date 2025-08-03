"""
Service initialization utilities
Handles FastAPI app setup and configuration
"""

import logging

from ..api.api_routes import initialize_app_with_config
from ..config.config import Config

logger = logging.getLogger(__name__)


class ServiceInitializer:
    """Handles service initialization"""

    def initialize_app(self, config: Config):
        """Initialize FastAPI application with configuration."""
        logger.info(f"Initializing FastAPI app with prefix: " f"{config.api_prefix}")
        return initialize_app_with_config(config.api_prefix)
