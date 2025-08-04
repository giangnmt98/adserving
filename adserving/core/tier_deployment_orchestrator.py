"""
Tier-based deployment orchestrator that coordinates tier management,
deployment scaling, and intelligent routing
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any

from ..config.base_types import ModelTier
from ..deployment.resource_config import TierBasedDeploymentConfig
from .tier_manager import TierManager
from .tier_aware_router import TierAwareRouter
from ..deployment.pooled_deployment import PooledModelDeployment

logger = logging.getLogger(__name__)


class TierDeploymentOrchestrator:
    """Orchestrates tier-based deployment
    with dynamic scaling and intelligent routing"""

    def __init__(self, model_manager, config: Any):
        self.model_manager = model_manager
        self.config = config

        # Initialize tier-based configuration
        self.tier_config = TierBasedDeploymentConfig({})

        # Initialize components
        self.tier_manager = TierManager(self.tier_config, config)
        self.router = TierAwareRouter(self.tier_config, self.tier_manager)
        self.deployment_manager = PooledModelDeployment(
            model_manager, config, self.tier_manager
        )

        # Connect components
        self.tier_manager.set_router(self.router)
        self.tier_manager.set_deployment_manager(self.deployment_manager)
        self.tier_manager.set_model_manager(self.model_manager)

        self._initialized = False
        self._running = False

    async def initialize(self) -> bool:
        """Initialize the tier-based deployment system"""
        try:
            logger.info("Initializing tier-based deployment orchestrator...")

            # Start tier manager
            await self.tier_manager.start()

            # Create tier-based deployment pools
            created_pools = self.deployment_manager.create_tier_based_pools(
                self.tier_config
            )

            # Register deployments with router
            for tier, deployments in created_pools.items():
                for deployment_name in deployments:
                    self.router.register_deployment(deployment_name, tier)

            # Load and classify initial models
            await self._initialize_model_tiers()

            self._initialized = True
            self._running = True

            logger.info("Tier-based deployment orchestrator initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize tier deployment orchestrator: {e}")
            return False

    async def shutdown(self):
        """Shutdown the orchestrator"""
        try:
            self._running = False

            # Stop tier manager
            await self.tier_manager.stop()

            # Cleanup deployments
            self.deployment_manager.cleanup()

            logger.info("Tier deployment orchestrator shutdown complete")

        except Exception as e:
            logger.error(f"Error during orchestrator shutdown: {e}")

    async def _initialize_model_tiers(self):
        """Initialize model tiers with lazy loading strategy"""
        try:
            # Get production models
            production_models = self.model_manager.get_production_models()

            if not production_models:
                logger.info("No production models found for tier initialization")
                return

            # Initialize models based on lazy loading strategy
            hot_models = []
            warm_models = []
            cold_models = []

            # For now, classify models based on simple rules
            # In production, this could be based on historical usage,
            # configuration, etc.
            for i, model_name in enumerate(production_models):
                if i < len(production_models) // 3:  # First 1/3 as HOT
                    hot_models.append(model_name)
                    self.tier_manager.register_model(model_name, ModelTier.HOT)
                elif i < 2 * len(production_models) // 3:  # Next 1/3 as WARM
                    warm_models.append(model_name)
                    self.tier_manager.register_model(model_name, ModelTier.WARM)
                else:  # Last 1/3 as COLD
                    cold_models.append(model_name)
                    self.tier_manager.register_model(model_name, ModelTier.COLD)

            # Apply lazy loading strategy
            await self._apply_lazy_loading_strategy(
                hot_models, warm_models, cold_models
            )

            logger.info(
                f"Initialized models with lazy loading: "
                f"{len(hot_models)} HOT, "
                f"{len(warm_models)} WARM, {len(cold_models)} COLD"
            )

        except Exception as e:
            logger.error(f"Error initializing model tiers: {e}")

    async def _apply_lazy_loading_strategy(
        self, hot_models: List[str], warm_models: List[str], cold_models: List[str]
    ):
        """Apply lazy loading strategy for different tiers"""
        try:
            # HOT tier: Load models and create routers immediately
            logger.info("Loading HOT tier models immediately...")
            for model_name in hot_models:
                try:
                    result = await self.tier_manager._load_and_deploy_model(model_name)
                    if result["status"] == "success":
                        logger.info(f"HOT model {model_name} loaded and deployed")
                    else:
                        logger.warning(
                            f"Failed to initialize HOT model {model_name}: {result.get('message')}"
                        )
                except Exception as e:
                    logger.error(f"Error initializing HOT model {model_name}: {e}")

            # WARM tier: Load models only (no routers)
            logger.info("Loading WARM tier models (no routers)...")
            for model_name in warm_models:
                try:
                    result = await self.tier_manager._load_model_only(model_name)
                    if result["status"] == "success":
                        logger.info(
                            f"WARM model {model_name} loaded (router will be created on first request)"
                        )
                    else:
                        logger.warning(
                            f"Failed to load WARM model {model_name}: {result.get('message')}"
                        )
                except Exception as e:
                    logger.error(f"Error loading WARM model {model_name}: {e}")

            # COLD tier: Do nothing initially
            logger.info(
                f"COLD tier models ({len(cold_models)}) will be loaded on first request"
            )

        except Exception as e:
            logger.error(f"Error applying lazy loading strategy: {e}")

    async def route_request(
        self, model_name: str, request_data: Any = None
    ) -> Optional[str]:
        """Route a request using tier-aware routing with lazy loading support"""
        if not self._initialized:
            logger.warning("Orchestrator not initialized, cannot route request")
            return None

        # Check if model needs lazy loading
        if await self._needs_lazy_loading(model_name):
            logger.info(f"Triggering lazy loading for model {model_name}")
            lazy_result = await self.tier_manager.handle_lazy_loading_request(
                model_name
            )

            if lazy_result["status"] == "loading":
                # Model is being loaded, wait a bit and retry
                await asyncio.sleep(1)
                return await self.route_request(model_name, request_data)
            elif lazy_result["status"] == "error":
                logger.error(
                    f"Lazy loading failed for {model_name}: {lazy_result.get('message')}"
                )
                return None
            elif lazy_result["status"] == "success":
                logger.info(
                    f"Lazy loading completed for {model_name}: {lazy_result.get('action')}"
                )

        return self.router.route_request(model_name, request_data)

    async def _needs_lazy_loading(self, model_name: str) -> bool:
        """Check if a model needs lazy loading with ModelManager fallback verification"""
        tier = self.tier_manager.get_model_tier(model_name)

        # HOT tier models should already be loaded and deployed
        if tier == ModelTier.HOT:
            tier_loaded = self.tier_manager.is_model_loaded(model_name)
            tier_deployed = self.tier_manager.is_model_deployed(model_name)

            # CRITICAL FIX: Add ModelManager fallback check for robustness
            if not tier_loaded and hasattr(self.model_manager, "is_model_loaded"):
                try:
                    # Check if model is actually loaded in ModelManager
                    model_manager_loaded = await self.model_manager.is_model_loaded(
                        model_name
                    )
                    if model_manager_loaded:
                        # Synchronize TierManager state if out of sync
                        self.tier_manager.loaded_models.add(model_name)
                        self.tier_manager.deployed_models.add(model_name)
                        logger.info(
                            f"Synchronized TierManager state for {model_name} from ModelManager"
                        )
                        return False  # No lazy loading needed
                except Exception as e:
                    logger.debug(
                        f"ModelManager fallback check failed for {model_name}: {e}"
                    )

            return not (tier_loaded and tier_deployed)

        # WARM tier models should be loaded but may need deployment
        elif tier == ModelTier.WARM:
            return not self.tier_manager.is_model_deployed(model_name)

        # COLD tier models need both loading and deployment
        elif tier == ModelTier.COLD:
            return not (
                self.tier_manager.is_model_loaded(model_name)
                and self.tier_manager.is_model_deployed(model_name)
            )

        return False

    def record_request_metrics(
        self,
        model_name: str,
        deployment_name: str,
        response_time: float,
        is_error: bool = False,
    ):
        """Record request metrics for both tier management and routing"""
        try:
            # Record for tier management (tier promotion/demotion)
            self.tier_manager.record_request(model_name, response_time, is_error)

            # Record for routing optimization
            self.router.update_request_metrics(deployment_name, response_time, is_error)

        except Exception as e:
            logger.error(f"Error recording request metrics: {e}")

    def get_model_tier(self, model_name: str) -> str:
        """Get the current tier of a model"""
        return self.tier_manager.get_model_tier(model_name)

    def force_tier_change(self, model_name: str, new_tier: str) -> bool:
        """Force a model to change tier"""
        return self.tier_manager.force_tier_change(model_name, new_tier)

    def get_tier_statistics(self) -> Dict[str, Any]:
        """Get comprehensive tier statistics"""
        try:
            tier_stats = self.tier_manager.get_tier_statistics()
            routing_stats = self.router.get_routing_statistics()

            return {
                "tier_management": tier_stats,
                "routing": routing_stats,
                "system_status": {
                    "initialized": self._initialized,
                    "running": self._running,
                    "total_models": sum(len(models) for models in tier_stats.values()),
                },
            }

        except Exception as e:
            logger.error(f"Error getting tier statistics: {e}")
            return {}

    def set_routing_strategy(self, strategy: str):
        """Set the routing strategy"""
        self.router.set_routing_strategy(strategy)

    async def handle_model_deployment(self, model_name: str) -> bool:
        """Handle deployment of a new model"""
        try:
            # Register model with tier manager
            self.tier_manager.register_model(model_name, ModelTier.COLD)

            # Load model in appropriate deployment
            # This would integrate with the actual model loading logic
            logger.info(f"Handled deployment of new model: {model_name}")
            return True

        except Exception as e:
            logger.error(f"Error handling model deployment for {model_name}: {e}")
            return False

    async def handle_model_removal(self, model_name: str) -> bool:
        """Handle removal of a model"""
        try:
            # Remove from tier manager
            current_tier = self.tier_manager.get_model_tier(model_name)
            if current_tier:
                self.tier_manager.tier_models[current_tier].discard(model_name)
                if model_name in self.tier_manager.model_metrics:
                    del self.tier_manager.model_metrics[model_name]

            # Remove from router
            if model_name in self.router.model_to_deployment:
                del self.router.model_to_deployment[model_name]

            logger.info(f"Handled removal of model: {model_name}")
            return True

        except Exception as e:
            logger.error(f"Error handling model removal for {model_name}: {e}")
            return False

    def get_deployment_for_model(self, model_name: str) -> Optional[str]:
        """Get the deployment that should handle a specific model"""
        model_tier = self.get_model_tier(model_name)
        return self.deployment_manager.get_tier_deployment(model_tier, model_name)

    async def rebalance_tiers(self) -> Dict[str, int]:
        """Manually trigger tier rebalancing"""
        try:
            # Force tier evaluation
            await self.tier_manager._evaluate_tier_changes()

            # Get current tier distribution
            tier_stats = self.tier_manager.get_tier_statistics()
            tier_distribution = {
                tier: stats["model_count"] for tier, stats in tier_stats.items()
            }

            logger.info(f"Manual tier rebalancing completed: {tier_distribution}")
            return tier_distribution

        except Exception as e:
            logger.error(f"Error during manual tier rebalancing: {e}")
            return {}

    def get_tier_capacity_status(self) -> Dict[str, Dict[str, Any]]:
        """Get capacity status for each tier"""
        try:
            capacity_status = {}

            for tier in [ModelTier.HOT, ModelTier.WARM, ModelTier.COLD]:
                tier_config = self.tier_config.tier_configs.get(tier)
                tier_models = self.tier_manager.get_tier_models(tier)

                if tier_config:
                    models_per_replica = tier_config.max_models_per_replica
                    current_models = len(tier_models)
                    max_capacity = tier_config.max_replicas * models_per_replica
                    utilization = (
                        (current_models / max_capacity) if max_capacity > 0 else 0
                    )

                    capacity_status[tier] = {
                        "current_models": current_models,
                        "max_capacity": max_capacity,
                        "utilization": utilization,
                        "deployments": len(self.router.tier_deployments.get(tier, [])),
                        "needs_scaling": utilization > 0.8,
                    }

            return capacity_status

        except Exception as e:
            logger.error(f"Error getting tier capacity status: {e}")
            return {}

    def is_healthy(self) -> bool:
        """Check if the orchestrator is healthy"""
        try:
            if not self._initialized or not self._running:
                return False

            # Check if we have deployments for each tier
            for tier in [ModelTier.HOT, ModelTier.WARM, ModelTier.COLD]:
                if not self.router.tier_deployments.get(tier):
                    logger.warning(f"No deployments available for {tier} tier")
                    return False

            return True

        except Exception as e:
            logger.error(f"Error checking orchestrator health: {e}")
            return False
