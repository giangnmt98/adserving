"""
Service components management
Handles initialization and coordination of all service components
Enhanced with Unified Error Handling
"""

import asyncio
import logging
from typing import Dict, Optional

from ..api.api_dependencies import (
    initialize_dependencies,
    update_service_readiness,
    enable_enhanced_error_handling,
)
from ..config.config import Config
from ..core.model_manager import ModelManager
from ..core.tier_deployment_orchestrator import TierDeploymentOrchestrator
from ..datahandler.data_handler import DataHandler
from ..deployment.pooled_deployment import PooledModelDeployment
from ..monitoring.model_monitor import ModelMonitor
from ..router.model_router import ModelRouter
from .ray_manager import RayManager
from .model_deployment_handler import ModelDeploymentHandler
from ..api import api_dependencies

logger = logging.getLogger(__name__)


class ServiceComponents:
    """Manages all service components with enhanced error handling"""

    def __init__(self) -> None:
        """Initialize component container."""
        self.model_manager: Optional[ModelManager] = None
        self.model_router: Optional[ModelRouter] = None
        self.monitor: Optional[ModelMonitor] = None
        self.input_handler: Optional[DataHandler] = None
        self.deployment_manager: Optional[PooledModelDeployment] = None
        self.tier_orchestrator: Optional[TierDeploymentOrchestrator] = None
        self.ray_manager = RayManager()
        self.deployment_handler = ModelDeploymentHandler()
        self.use_tier_based_deployment = True
        self.enhanced_error_handling_enabled = True

    def initialize_all(self, config: Config) -> None:
        """Initialize all service components."""
        try:
            logger.info("Initializing service components...")

            # Check enhanced error handling configuration
            self.enhanced_error_handling_enabled = getattr(
                config, "enable_enhanced_error_handling", True
            )

            # Check if tier-based deployment is enabled
            self.use_tier_based_deployment = hasattr(
                config, "tier_based_deployment"
            ) and getattr(
                config.tier_based_deployment, "enable_tier_based_deployment", False
            )

            logger.info(
                f"Tier-based deployment: "
                f"{'enabled' if self.use_tier_based_deployment else 'disabled'}"
            )
            logger.info(
                f"Enhanced error handling: "
                f"{'enabled' if self.enhanced_error_handling_enabled else 'disabled'}"
            )

            # Initialize Ray Serve
            self.ray_manager.initialize(config)

            # Initialize core components
            self._initialize_core_components(config)

            # Initialize tier orchestrator if enabled
            if self.use_tier_based_deployment:
                self._initialize_tier_orchestrator(config)

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
            max_queue_size=config.routing.max_queue_size,
        )

    def _initialize_monitor(self, config: Config) -> None:
        """Initialize Monitor."""
        logger.info("Setting up Monitor...")
        self.monitor = ModelMonitor(
            collection_interval=config.monitoring.collection_interval,
            optimization_interval=config.monitoring.optimization_interval,
            enable_prometheus=config.monitoring.enable_prometheus_export,
            prometheus_port=config.monitoring.prometheus_port,
        )

    def _initialize_input_handler(self) -> None:
        """Initialize Input Handler with config parameter."""
        logger.info("Setting up Input Handler...")
        self.input_handler = DataHandler()

    def _initialize_deployment_manager(self, config: Config) -> None:
        """Initialize Deployment Manager."""
        logger.info("Setting up Deployment Manager...")
        self.deployment_manager = PooledModelDeployment(self.model_manager, config)

    def _initialize_tier_orchestrator(self, config: Config) -> None:
        """Initialize Tier-based Deployment Orchestrator."""
        logger.info("Setting up Tier-based Deployment Orchestrator...")
        self.tier_orchestrator = TierDeploymentOrchestrator(self.model_manager, config)

    def _initialize_fastapi_dependencies(self) -> None:
        """Initialize FastAPI dependencies with enhanced error handling."""
        logger.info("Configuring FastAPI dependencies...")

        # Initialize regular dependencies
        initialize_dependencies(
            self.model_manager,
            self.model_router,
            self.monitor,
            self.input_handler,
            self.tier_orchestrator,
            self.use_tier_based_deployment,
        )

        # Cập nhật global variable
        api_dependencies.use_tier_based_deployment = self.use_tier_based_deployment

        # Set tier orchestrator if available
        if self.tier_orchestrator and self.use_tier_based_deployment:
            api_dependencies.tier_orchestrator = self.tier_orchestrator
            logger.info("Tier-based deployment enabled and configured")
        else:
            logger.warning("Tier-based deployment is disabled")

        # Enable enhanced error handling if configured
        if self.enhanced_error_handling_enabled:
            try:
                enable_enhanced_error_handling(True)
                logger.info("Enhanced error handling enabled successfully")
            except Exception as e:
                logger.warning(f"Could not enable enhanced error handling: {e}")
                logger.warning("Continuing with standard error handling")

    # Các methods khác giữ nguyên như code gốc...
    async def deploy_production_models(self) -> Dict[str, int]:
        """Deploy production models with parallel loading or tier-based deployment with HOT tier pre-loading."""
        if not self.model_manager:
            return {"loaded": 0, "failed": 0}

        if self.use_tier_based_deployment and self.tier_orchestrator:
            # Use tier-based deployment with HOT tier pre-loading
            logger.info("Deploying models using tier-based deployment strategy...")
            try:
                success = await self.tier_orchestrator.initialize()
                if success:
                    # Get tier statistics
                    tier_stats = self.tier_orchestrator.get_tier_statistics()
                    tier_management = tier_stats.get("tier_management", {})

                    # Pre-load HOT tier models to ensure they're truly ready
                    hot_tier_models = tier_management.get("hot", {}).get("models", [])
                    loaded_count = 0
                    failed_count = 0

                    if hot_tier_models:
                        logger.info(
                            f"Pre-loading {len(hot_tier_models)} HOT tier models..."
                        )
                        loaded_count, failed_count = await self._load_hot_tier_models(
                            hot_tier_models
                        )
                        logger.info(
                            f"HOT tier models loaded: {loaded_count} successful, {failed_count} failed"
                        )

                    # Warm up tier-based deployment pools
                    if loaded_count > 0:
                        await self._warm_up_tier_deployments()

                    total_models = sum(
                        tier_data.get("model_count", 0)
                        for tier_data in tier_management.values()
                    )
                    logger.info(
                        f"Tier-based deployment completed: {total_models} models registered, {loaded_count} HOT models pre-loaded"
                    )
                    return {"loaded": loaded_count, "failed": failed_count}
                else:
                    logger.error("Failed to initialize tier-based deployment")
                    return {"loaded": 0, "failed": 1}
            except Exception as e:
                logger.error(f"Error in tier-based deployment: {e}")
                return {"loaded": 0, "failed": 1}
        else:
            # Use traditional deployment
            if not self.deployment_manager:
                return {"loaded": 0, "failed": 0}

            return self.deployment_handler.deploy_models(
                self.model_manager, self.deployment_manager, self.model_router
            )

    async def _load_hot_tier_models(self, hot_tier_models: list) -> tuple[int, int]:
        """Load HOT tier models into ModelManager and synchronize TierManager state."""
        loaded_count = 0
        failed_count = 0

        logger.info(
            f"Loading {len(hot_tier_models)} HOT tier models into ModelManager..."
        )

        # Load models sequentially to avoid overwhelming the system
        for model_name in hot_tier_models:
            try:
                logger.debug(f"Loading HOT tier model: {model_name}")
                # Use async model loading to load the model into ModelManager
                model_info = await self.model_manager.load_model_async(model_name)

                if model_info:
                    loaded_count += 1
                    logger.debug(f"Successfully loaded HOT tier model: {model_name}")

                    # CRITICAL FIX: Synchronize TierManager state with ModelManager
                    if self.tier_orchestrator and self.tier_orchestrator.tier_manager:
                        # Update tier manager tracking to reflect that model is loaded and deployed
                        self.tier_orchestrator.tier_manager.loaded_models.add(
                            model_name
                        )
                        self.tier_orchestrator.tier_manager.deployed_models.add(
                            model_name
                        )
                        logger.debug(
                            f"Synchronized TierManager state for HOT model: {model_name}"
                        )
                else:
                    failed_count += 1
                    logger.warning(f"Failed to load HOT tier model: {model_name}")

            except Exception as e:
                failed_count += 1
                logger.error(f"Error loading HOT tier model {model_name}: {e}")

        logger.info(
            f"HOT tier model loading completed: {loaded_count} loaded, {failed_count} failed"
        )
        logger.info(
            f"TierManager state synchronized for {loaded_count} HOT tier models"
        )
        return loaded_count, failed_count

    async def _warm_up_tier_deployments(self):
        """Warm up tier-based deployment pools to prevent cold starts."""
        try:
            logger.info("Warming up tier-based deployment pools...")

            import ray
            from ray import serve

            # Get all Ray Serve deployments
            deployments = serve.list_deployments()
            if not deployments:
                logger.warning("No Ray Serve deployments found for warmup")
                return

            # Find tier-based deployments (they have tier names in them)
            tier_deployments = []
            for deployment_name in deployments:
                if any(
                    tier in deployment_name.lower()
                    for tier in ["hot", "warm", "cold", "pooled"]
                ):
                    tier_deployments.append(deployment_name)

            if not tier_deployments:
                logger.warning("No tier-based deployments found for warmup")
                return

            # Warm up each tier deployment with a test request
            warmup_payload = {
                "ma_don_vi": "UBND.0019",
                "ma_bao_cao": "10628953_CT",
                "ky_du_lieu": "2024-01-01",
                "data": [{"ma_tieu_chi": "TONGCONG", "FN01": 1000000}],
            }

            for deployment_name in tier_deployments[:3]:  # Limit to first 3 deployments
                try:
                    logger.debug(f"Warming up deployment: {deployment_name}")
                    deployment_handle = serve.get_app_handle(deployment_name)

                    # Make a warmup request with timeout
                    result_ref = await deployment_handle.remote(warmup_payload)
                    await asyncio.wait_for(ray.get(result_ref), timeout=10.0)

                    logger.debug(
                        f"Successfully warmed up deployment: {deployment_name}"
                    )

                except asyncio.TimeoutError:
                    logger.warning(f"Warmup timeout for deployment: {deployment_name}")
                except Exception as e:
                    logger.warning(
                        f"Warmup failed for deployment {deployment_name}: {e}"
                    )

            logger.info("Tier-based deployment warmup completed")

        except Exception as e:
            logger.warning(f"Tier deployment warmup failed (non-critical): {e}")

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

    def update_readiness_state(
        self, ready: bool, models_loaded: int, models_failed: int
    ) -> None:
        """Update service readiness state."""
        update_service_readiness(
            ready=ready,
            models_loaded=models_loaded,
            models_failed=models_failed,
            initialization_complete=True,
        )

    def cleanup(self) -> None:
        """Cleanup all components."""
        components = [
            ("deployment_manager", self.deployment_manager),
            ("model_manager", self.model_manager),
            ("monitor", self.monitor),
        ]

        for name, component in components:
            if component and hasattr(component, "cleanup"):
                try:
                    logger.info(f"Cleaning up {name}...")
                    component.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up {name}: {e}")
