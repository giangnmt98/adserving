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
from ..core.ray_model_deployment_manager import RayModelDeploymentManager
from ..datahandler.data_handler import DataHandler
from ..core.manager.ray_deployment_manager import RayDeploymentManager
from ..core.manager.deployment_manager import DeploymentManager
from ..monitoring.model_monitor import ModelMonitor
from ..router.model_router import ModelRouter
from .ray_manager import RayManager
from ..api import api_dependencies
from ..deployment.deployment_orchestrator import UltraScaleDeploymentOrchestrator
from ..deployment.deployment_config import DeploymentConfig, ConfigurationManager

logger = logging.getLogger(__name__)


class ServiceComponents:
    """Manages all service components with enhanced error handling"""

    def __init__(self) -> None:
        """Initialize component container."""
        self.model_manager: Optional[ModelManager] = None
        self.model_router: Optional[ModelRouter] = None
        self.monitor: Optional[ModelMonitor] = None
        self.input_handler: Optional[DataHandler] = None
        self.deployment_manager: Optional[DeploymentManager] = None
        self.ray_deployment_manager = None
        self.ray_manager = RayManager()
        self.enhanced_error_handling_enabled = True
        
        # Ultra-scale deployment components
        self.ultra_scale_orchestrator: Optional[UltraScaleDeploymentOrchestrator] = None
        self.deployment_config_manager: Optional[ConfigurationManager] = None
        self.use_ultra_scale_deployment = False

    def initialize_all(self, config: Config) -> None:
        """Initialize all service components."""
        try:
            logger.info("Initializing service components...")

            # Check enhanced error handling configuration
            self.enhanced_error_handling_enabled = getattr(
                config, "enable_enhanced_error_handling", True
            )


            # Check if ultra-scale deployment is enabled
            self.use_ultra_scale_deployment = hasattr(
                config, "ultra_scale_deployment"
            ) and getattr(
                config.ultra_scale_deployment, "enable_ultra_scale_deployment", False
            )

            logger.info(
                f"Ultra-scale deployment: "
                f"{'enabled' if self.use_ultra_scale_deployment else 'disabled'}"
            )
            logger.info(
                f"Enhanced error handling: "
                f"{'enabled' if self.enhanced_error_handling_enabled else 'disabled'}"
            )

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

    def initialize_minimal(self, config: Config) -> None:
        """Initialize minimal components for fast server startup - defer Ray initialization."""
        try:
            logger.info("Initializing minimal service components for fast startup...")

            # Check enhanced error handling configuration
            self.enhanced_error_handling_enabled = getattr(
                config, "enable_enhanced_error_handling", True
            )

            # Check if ultra-scale deployment is enabled
            self.use_ultra_scale_deployment = hasattr(
                config, "ultra_scale_deployment"
            ) and getattr(
                config.ultra_scale_deployment, "enable_ultra_scale_deployment", False
            )

            logger.info(
                f"Ultra-scale deployment: "
                f"{'enabled' if self.use_ultra_scale_deployment else 'disabled'}"
            )
            logger.info(
                f"Enhanced error handling: "
                f"{'enabled' if self.enhanced_error_handling_enabled else 'disabled'}"
            )

            # Initialize only basic components (no Ray dependencies)
            self._initialize_basic_components(config)

            # Initialize minimal FastAPI dependencies
            self._initialize_minimal_fastapi_dependencies()

            logger.info("Minimal services initialized successfully - Ray initialization deferred")

        except Exception as e:
            logger.error(f"Failed to initialize minimal services: {e}")
            raise

    async def complete_initialization(self, config: Config) -> None:
        """Complete full initialization with Ray and model deployment in background."""
        try:
            logger.info("Starting background initialization of Ray and model components...")

            # Initialize Ray Serve (this is the slow part)
            self.ray_manager.initialize(config)

            # Initialize Ray-dependent components
            self._initialize_ray_dependent_components(config)

            # Initialize full FastAPI dependencies
            self._initialize_fastapi_dependencies()

            logger.info("Background initialization completed successfully")

        except Exception as e:
            logger.error(f"Failed to complete background initialization: {e}")
            raise

    def _initialize_core_components(self, config: Config) -> None:
        """Initialize all core components."""
        self._initialize_model_manager(config)
        self._initialize_model_router(config)
        self._initialize_monitor(config)
        self._initialize_input_handler()
        self._initialize_deployment_manager(config)
        
        # Initialize ultra-scale deployment if enabled
        if self.use_ultra_scale_deployment:
            self._initialize_ultra_scale_deployment(config)

    def _initialize_basic_components(self, config: Config) -> None:
        """Initialize basic components that don't depend on Ray."""
        # Input handler can work without Ray
        self._initialize_input_handler()
        
        # Monitor can work without Ray for basic health checks
        self._initialize_monitor(config)
        
        logger.info("Basic components initialized (no Ray dependencies)")

    def _initialize_ray_dependent_components(self, config: Config) -> None:
        """Initialize components that depend on Ray being ready."""
        self._initialize_model_manager(config)
        self._initialize_model_router(config)
        self._initialize_deployment_manager(config)
        
        # Initialize ultra-scale deployment if enabled
        if self.use_ultra_scale_deployment:
            self._initialize_ultra_scale_deployment(config)
            
        logger.info("Ray-dependent components initialized")

    def _initialize_minimal_fastapi_dependencies(self) -> None:
        """Initialize minimal FastAPI dependencies for basic server functionality."""
        # For now, defer all FastAPI dependencies until full initialization
        # This ensures basic health check endpoints can work without complex dependencies
        logger.info("Minimal FastAPI dependencies deferred until full initialization")

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
        """Initialize Ray-based Deployment Manager (eliminates all worker mechanisms)."""
        logger.info("Setting up Ray-based Deployment Manager (no workers, pure Ray parallel processing)...")
        
        # Provide model loading function from model_manager
        load_model_func = None
        if self.model_manager:
            load_model_func = self.model_manager._load_model_sync
        
        # Get number of Ray actors from config (replaces worker count)
        num_actors = getattr(config, 'ray_deployment_actors', 4)  # Default 4 Ray actors
        logger.info(f"Using Ray parallel deployment with {num_actors} Ray actors (no workers)")
        
        self.deployment_manager = RayDeploymentManager(
            batch_size=5,
            batch_interval=2.0,
            load_model_func=load_model_func,
            num_actors=num_actors
        )
        
        # Initialize Ray Model Deployment Manager for request routing
        logger.info("Setting up Ray Model Deployment Manager for request handling...")
        self.ray_deployment_manager = RayModelDeploymentManager(self.model_manager)


    def _initialize_ultra_scale_deployment(self, config: Config) -> None:
        """Initialize Ultra-Scale Deployment components."""
        logger.info("Setting up Ultra-Scale Deployment...")
        
        # Initialize configuration manager
        self.deployment_config_manager = ConfigurationManager()
        
        # Create deployment configuration from existing config
        deployment_config = self._create_deployment_config_from_adserving_config(config)
        
        # Initialize ultra-scale deployment orchestrator
        self.ultra_scale_orchestrator = UltraScaleDeploymentOrchestrator(deployment_config)
        
        logger.info("Ultra-Scale Deployment components initialized successfully")

    def _create_deployment_config_from_adserving_config(self, config: Config) -> DeploymentConfig:
        """Create DeploymentConfig from existing adserving Config."""
        from ..deployment.deployment_config import (
            DeploymentConfig, DeploymentMode, ResourceLimits, 
            ModelTierConfig, RayConfig, MLflowConfig
        )
        
        # Map adserving config to deployment config
        deployment_config = DeploymentConfig(
            mode=DeploymentMode.PRODUCTION,
            environment_name=getattr(config, 'environment', 'production'),
            target_model_count=getattr(config, 'max_models', 2000),
            
            # Resource configuration
            resources=ResourceLimits(
                max_cpu_cores=getattr(config, 'max_workers', 16),
                max_memory_mb=16000,
                max_disk_gb=100
            ),
            
            # Model tier configuration
            model_tiers=ModelTierConfig(
                hot_tier_max_models=getattr(config.tiered_loading, 'hot_cache_size', 500),
                warm_tier_max_models=getattr(config.tiered_loading, 'warm_cache_size', 1000),
                cold_tier_max_models=getattr(config.tiered_loading, 'cold_cache_size', 10000)
            ),
            
            # Ray configuration
            ray=RayConfig(
                enabled=True,
                num_cpus=getattr(config.ray, 'num_cpus', None) if hasattr(config, 'ray') else None,
                num_gpus=getattr(config.ray, 'num_gpus', None) if hasattr(config, 'ray') else None,
                object_store_memory_gb=8
            ),
            
            # MLflow configuration
            mlflow=MLflowConfig(
                enabled=True,
                tracking_uri=config.mlflow.tracking_uri
            )
        )
        
        return deployment_config

    def _initialize_fastapi_dependencies(self) -> None:
        """Initialize FastAPI dependencies with enhanced error handling."""
        logger.info("Configuring FastAPI dependencies...")

        # Initialize regular dependencies
        initialize_dependencies(
            self.model_manager,
            self.model_router,
            self.monitor,
            self.input_handler,
            self.ray_deployment_manager,
            None,  # tier_orchestrator removed
            False,  # use_tier_based_deployment removed
            self.ultra_scale_orchestrator,  # ultra_orchestrator
            self.use_ultra_scale_deployment,  # ultra_scale
        )

        if self.use_ultra_scale_deployment:
            logger.info("Ultra-scale deployment enabled and configured")
        else:
            logger.info("Ultra-scale deployment is disabled")

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
        """Deploy production models using parallel deployment strategy with Ray Model Router System."""
        if not self.model_manager:
            return {"loaded": 0, "failed": 0}

        # Use ultra-scale deployment if enabled and available
        if self.use_ultra_scale_deployment and self.ultra_scale_orchestrator:
            logger.info("Deploying models using ultra-scale deployment strategy...")
            return await self._deploy_with_ultra_scale()

        # Use ModelDeploymentHandler with parallel Ray deployment (FIXED from sequential fallback)
        logger.info("Using parallel Ray Model Router deployment...")
        
        try:
            # Import and use the parallel ModelDeploymentHandler we implemented
            from ..service.model_deployment_handler import ModelDeploymentHandler
            
            deployment_handler = ModelDeploymentHandler()
            
            # Deploy all models in parallel using Ray Model Router workflow
            logger.info("Starting parallel Load > Deploy > Route workflow for all production models")
            deployment_stats = deployment_handler.deploy_models(
                self.model_manager,
                self.deployment_manager, 
                self.model_router
            )
            
            loaded = deployment_stats.get("loaded", 0)
            failed = deployment_stats.get("failed", 0)
            
            logger.info(f"Parallel deployment completed: {loaded} successful, {failed} failed")
            return {"loaded": loaded, "failed": failed}
                
        except Exception as e:
            logger.error(f"Parallel deployment failed: {e}")
            
            # Emergency fallback: Try basic sequential loading if parallel fails
            logger.warning("Falling back to sequential deployment due to parallel deployment failure")
            try:
                if hasattr(self.model_manager, 'get_production_models'):
                    production_models = self.model_manager.get_production_models()
                    loaded = 0
                    failed = 0
                    
                    # Limit to reasonable number for fallback
                    hot_cache_capacity = min(50, len(production_models))  
                    
                    for model_name in production_models[:hot_cache_capacity]:
                        try:
                            # Just load to cache in fallback mode
                            model_info = await self.model_manager.load_model_async(model_name)
                            if model_info:
                                loaded += 1
                                logger.info(f"Fallback: Model {model_name} loaded to cache")
                            else:
                                failed += 1
                        except Exception as model_error:
                            failed += 1
                            logger.error(f"Fallback: Error loading model {model_name}: {model_error}")
                    
                    return {"loaded": loaded, "failed": failed}
                else:
                    logger.warning("No production models method available in fallback")
                    return {"loaded": 0, "failed": 0}
            except Exception as fallback_error:
                logger.error(f"Emergency fallback deployment also failed: {fallback_error}")
                return {"loaded": 0, "failed": 0}

    async def _deploy_with_ultra_scale(self) -> Dict[str, int]:
        """Deploy models using ultra-scale deployment system."""
        try:
            # Initialize the ultra-scale deployment system
            await self.ultra_scale_orchestrator.initialize()
            
            # Get production models from existing model manager
            production_models = []
            if hasattr(self.model_manager, 'get_production_models'):
                model_names = self.model_manager.get_production_models()
                
                # Create model configurations for ultra-scale deployment
                for model_name in model_names:
                    model_config = {
                        "name": model_name,
                        "type": "mlflow",
                        "version": "latest"
                    }
                    production_models.append(model_config)
            
            if not production_models:
                logger.warning("No production models found for ultra-scale deployment")
                return {"loaded": 0, "failed": 0}
            
            # Deploy models using ultra-scale deployment
            logger.info(f"Deploying {len(production_models)} models with ultra-scale system...")
            deployment_result = await self.ultra_scale_orchestrator.deploy_models_ultra_scale(
                production_models,
                deployment_strategy="blue_green"
            )
            
            if deployment_result.get("success", False):
                logger.info("Ultra-scale deployment completed successfully")
                return {
                    "loaded": deployment_result.get("deployed", 0),
                    "failed": deployment_result.get("failed", 0)
                }
            else:
                logger.error(f"Ultra-scale deployment failed: {deployment_result.get('error', 'Unknown error')}")
                return {"loaded": 0, "failed": len(production_models)}
                
        except Exception as e:
            logger.error(f"Ultra-scale deployment error: {e}")
            return {"loaded": 0, "failed": len(production_models) if 'production_models' in locals() else 0}


    def start_background_services(self) -> None:
        """Start background monitoring services."""
        try:
            logger.info("Starting background services...")

            if self.model_manager:
                self.model_manager.start_monitoring()

            if self.monitor:
                self.monitor.start_monitoring()

            # Start ultra-scale deployment background services if enabled
            if self.use_ultra_scale_deployment and self.ultra_scale_orchestrator:
                try:
                    # The ultra-scale orchestrator's background services are started
                    # during initialization, so we just log that they're available
                    logger.info("Ultra-scale deployment background services are active")
                except Exception as e:
                    logger.warning(f"Failed to start ultra-scale deployment services: {e}")

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
            ("ray_serve_deployment_service", self.ray_serve_deployment_service),
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
