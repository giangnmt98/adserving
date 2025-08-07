"""
Main service class for Anomaly Detection Serve
"""

import os
import signal
import sys
from pathlib import Path
from typing import Optional

import uvicorn

from ..config.config import Config, create_sample_config
from .service_components import ServiceComponents
from .existing_warmup_integration import ExistingWarmupIntegration
from adserving.src.utils.logger import get_logger

logger = get_logger()


class AnomalyDetectionServe:
    """Main service class for Anomaly Detection Serve"""

    def __init__(self) -> None:
        """Initialize the service."""
        # Initialize config as None first
        self.config: Optional[Config] = None
        self.components = ServiceComponents()
        self.warmup_integration: Optional[ExistingWarmupIntegration] = None

        # Service settings - defaults that will be updated from config
        self.host = "0.0.0.0"
        self.port = 8000
        self.config_file = "config.yaml"

        # Service state tracking
        self._service_ready = False
        self._loaded_model_count = 0
        self._failed_model_count = 0

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("Anomaly Detection Serve initialized")

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.cleanup()
        sys.exit(0)

    def load_configuration(self) -> None:
        """Load or create configuration."""
        try:
            if not Path(self.config_file).exists():
                logger.info(f"Creating default config: {self.config_file}")
                create_sample_config(self.config_file)

            logger.info(f"Loading configuration from: {self.config_file}")
            self.config = Config.from_file(self.config_file)

            # Update settings from config
            self._update_settings_from_config()

            logger.info(f"Configuration loaded - API: {self.host}:{self.port}")

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    def _update_settings_from_config(self) -> None:
        """Update service settings from configuration."""
        if not self.config:
            return

        # Update from config
        if hasattr(self.config, "api_host"):
            self.host = self.config.api_host
        if hasattr(self.config, "api_port"):
            self.port = self.config.api_port

        # Override with environment variables
        if os.getenv("MLOPS_HOST"):
            self.host = os.getenv("MLOPS_HOST")
        if os.getenv("MLOPS_PORT"):
            self.port = int(os.getenv("MLOPS_PORT"))

    def run(self) -> None:
        """Main service execution."""
        try:
            logger.info("Starting Anomaly Detection Serve...")

            # 1. Load configuration FIRST - CRITICAL STEP
            self.load_configuration()

            # 2. Verify config is loaded
            if not self.config:
                raise RuntimeError("Configuration not loaded")

            # 3. Configure existing warmup mechanisms AFTER config is loaded
            logger.info("Configuring warmup mechanisms...")
            self.warmup_integration = ExistingWarmupIntegration(self.config)
            self.warmup_integration.ensure_warmup_enabled()

            # 4. Log warmup status
            warmup_status = self.warmup_integration.get_warmup_status()
            logger.info(f"Warmup mechanisms active: {warmup_status}")

            # 5. Initialize all services
            logger.info("Initializing services...")
            self.components.initialize_all(self.config)

            # 6. Deploy production models
            logger.info("Deploying production models...")
            import asyncio

            model_stats = asyncio.run(self.components.deploy_production_models())
            self._loaded_model_count = model_stats["loaded"]
            self._failed_model_count = model_stats["failed"]

            # 7. Start background services
            logger.info("Starting background services...")
            self.components.start_background_services()

            # 8. Mark service as ready
            self._service_ready = True
            self.components.update_readiness_state(
                ready=True,
                models_loaded=self._loaded_model_count,
                models_failed=self._failed_model_count,
            )

            # 9. Start FastAPI server
            logger.info(f"Starting server on {self.host}:{self.port}")
            from .service_initializer import ServiceInitializer

            initializer = ServiceInitializer()
            app_instance = initializer.initialize_app(self.config)

            logger.info("Anomaly Detection Serve ready with warmup mechanisms!")

            uvicorn.run(
                app_instance,
                host=self.host,
                port=self.port,
                log_level="info",
                access_log=True,
            )

        except KeyboardInterrupt:
            logger.info("Service interrupted by user")
        except Exception as e:
            logger.error(f"Service failed to start: {e}")
            raise
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Cleanup resources on shutdown."""
        try:
            logger.info("Cleaning up resources...")
            self.components.cleanup()
            logger.info("Cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
