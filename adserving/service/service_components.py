"""
Service components management
Handles initialization and coordination of all service components
"""
import logging
from typing import Dict, Optional

from ..api.api_routes import (
    initialize_dependencies,
    update_service_readiness
)
from ..config.config import Config
from ..core.model_manager import ModelManager
from ..datahandler.data_handler import DataHandler
from ..deployment.pooled_deployment import PooledModelDeployment
from ..monitoring.model_monitor import ModelMonitor
from ..router.model_router import ModelRouter
from .ray_manager import RayManager
from .model_deployment_handler import ModelDeploymentHandler

logger = logging.getLogger(__name__)


class ServiceComponents:
    """Manages all service components"""

    def __init__(self) -> None:
        """Initialize component container."""
        self.model_manager: Optional[ModelManager] = None
        self.model_router: Optional[ModelRouter] = None
        self.monitor: Optional[ModelMonitor] = None
        self.input_handler: Optional[DataHandler] = None
        self.deployment_manager: Optional[PooledModelDeployment] = None
        self.ray_manager = RayManager()
        self.deployment_handler = ModelDeploymentHandler()

    def initialize_all(self, config: Config) -> None:
        """Initialize all service components."""
        try:
            logger.info("Initializing service components...")

            # Initialize Ray Serve
            self.ray_manager.initialize(config)

            # Initialize core components
            self._initialize_core_components(config)

            # Initialize FastAPI dependencies
            self._initialize_fastapi_dependencies()

            logger.info("All services initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            raise

    def _initialize_core_components(self, config: Config) -> None:
        """Initialize all core components."""
        self._initialize_model_manager(config)
        self._initialize_model_router(config)
        self._initialize_monitor(config)
        self._initialize_input_handler()
        self._initialize_deployment_manager(config)

    def _initialize_model_manager(self, config: Config) -> None:
        """Initialize Model Manager."""
        logger.info("Setting up Model Manager...")
        self.model_manager = ModelManager(
            mlflow_tracking_uri=config.mlflow.tracking_uri,
            hot_cache_size=config.tiered_loading.hot_cache_size,
            warm_cache_size=config.tiered_loading.warm_cache_size,
            cold_cache_size=config.tiered_loading.cold_cache_size,
            max_workers=config.max_workers,
            enable_model_warming=config.tiered_loading.enable_model_warming,
        )

    def _initialize_model_router(self, config: Config) -> None:
        """Initialize Model Router."""
        logger.info("Setting up Model Router...")
        self.model_router = ModelRouter(
            model_manager=self.model_manager,
            pooled_deployment=config.pooled_deployment,
            routing_strategy=config.routing.strategy,
            enable_request_queuing=config.routing.enable_request_queuing,
            max_queue_size=config.routing.max_queue_size
        )

    def _initialize_monitor(self, config: Config) -> None:
        """Initialize Monitor."""
        logger.info("Setting up Monitor...")
        self.monitor = ModelMonitor(
            collection_interval=config.monitoring.collection_interval,
            optimization_interval=config.monitoring.optimization_interval,
            enable_prometheus=config.monitoring.enable_prometheus_export,
            prometheus_port=config.monitoring.prometheus_port
        )

    def _initialize_input_handler(self) -> None:
        """Initialize Input Handler."""
        logger.info("Setting up Input Handler...")
        self.input_handler = DataHandler()

    def _initialize_deployment_manager(self, config: Config) -> None:
        """Initialize Deployment Manager."""
        logger.info("Setting up Deployment Manager...")
        self.deployment_manager = PooledModelDeployment(
            self.model_manager, config
        )

    def _initialize_fastapi_dependencies(self) -> None:
        """Initialize FastAPI dependencies."""
        logger.info("Configuring FastAPI dependencies...")
        initialize_dependencies(
            self.model_manager,
            self.model_router,
            self.monitor,
            self.input_handler
        )

    def deploy_production_models(self) -> Dict[str, int]:
        """Deploy production models with parallel loading."""
        if not self.model_manager or not self.deployment_manager:
            return {'loaded': 0, 'failed': 0}

        return self.deployment_handler.deploy_models(
            self.model_manager,
            self.deployment_manager,
            self.model_router
        )

    def start_background_services(self) -> None:
        """Start background monitoring services."""
        try:
            logger.info("Starting background services...")

            if self.model_manager:
                self.model_manager.start_monitoring()

            if self.monitor:
                self.monitor.start_monitoring()

            logger.info("Background services started")

        except Exception as e:
            logger.error(f"Failed to start background services: {e}")
            raise

    def update_readiness_state(self, ready: bool, models_loaded: int,
                               models_failed: int) -> None:
        """Update service readiness state."""
        update_service_readiness(
            ready=ready,
            models_loaded=models_loaded,
            models_failed=models_failed,
            initialization_complete=True
        )

    def cleanup(self) -> None:
        """Cleanup all components."""
        components = [
            ('deployment_manager', self.deployment_manager),
            ('model_manager', self.model_manager),
            ('monitor', self.monitor),
        ]

        for name, component in components:
            if component and hasattr(component, 'cleanup'):
                try:
                    logger.info(f"Cleaning up {name}...")
                    component.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up {name}: {e}")