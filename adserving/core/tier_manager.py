"""
Tier management system for dynamic model tier classification and transitions
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

from ..config.base_types import ModelTier
from ..deployment.resource_config import TierBasedDeploymentConfig, TierResourceConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Metrics for a model to determine tier placement"""

    model_name: str
    request_count: int = 0
    last_request_time: float = 0
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    error_count: int = 0
    current_tier: str = ModelTier.COLD
    tier_change_time: float = 0

    def add_request(self, response_time: float, is_error: bool = False):
        """Add a request metric"""
        self.request_count += 1
        self.last_request_time = time.time()
        self.response_times.append(response_time)
        if is_error:
            self.error_count += 1

    def get_average_response_time(self) -> float:
        """Get average response time"""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)

    def get_error_rate(self) -> float:
        """Get error rate"""
        if self.request_count == 0:
            return 0.0
        return self.error_count / self.request_count


class TierManager:
    """Manages model tier classification and transitions with lazy loading support"""

    def __init__(self, config: TierBasedDeploymentConfig, main_config: Any = None):
        self.config = config
        self.main_config = main_config  # Store main config for business-critical models
        self.model_metrics: Dict[str, ModelMetrics] = {}
        self.tier_models: Dict[str, Set[str]] = {
            ModelTier.HOT: set(),
            ModelTier.WARM: set(),
            ModelTier.COLD: set(),
        }
        self.tier_request_windows: Dict[str, Dict[str, deque]] = {
            ModelTier.HOT: defaultdict(lambda: deque(maxlen=1000)),
            ModelTier.WARM: defaultdict(lambda: deque(maxlen=1000)),
            ModelTier.COLD: defaultdict(lambda: deque(maxlen=1000)),
        }

        # Manual tier assignment tracking
        self.manual_tier_assignments: Dict[str, str] = {}  # model_name -> tier
        self.business_critical_models: Set[str] = (
            set()
        )  # Models with guaranteed tier placement
        self.manual_assignment_patterns: List[str] = (
            []
        )  # Patterns for automatic business-critical detection
        self.pattern_tier_assignments: Dict[str, str] = {}  # pattern -> tier mapping
        self.default_business_critical_tier: str = ModelTier.HOT
        self.prevent_automatic_changes: bool = True

        # Lazy loading state tracking
        self.loaded_models: Set[str] = set()  # Models that have been loaded
        self.deployed_models: Set[str] = set()  # Models that have deployments/routers
        self.loading_in_progress: Set[str] = set()  # Models currently being loaded

        self._running = False
        self._tier_monitor_task = None

        # Load business-critical models configuration
        self._load_business_critical_config()

    def _load_business_critical_config(self):
        """Load business-critical models configuration from main config"""
        try:
            if not self.main_config:
                logger.info(
                    "No main config provided, skipping business-critical models configuration"
                )
                return

            # Get tier-based deployment config
            tier_config = getattr(self.main_config, "tier_based_deployment", None)
            if not tier_config:
                logger.info("No tier_based_deployment config found")
                return

            # Get business-critical models config
            business_critical_config = getattr(
                tier_config, "business_critical_models", None
            )
            if not business_critical_config:
                logger.info("No business_critical_models config found")
                return

            # Load configuration settings
            self.default_business_critical_tier = getattr(
                business_critical_config,
                "default_business_critical_tier",
                ModelTier.HOT,
            )
            self.prevent_automatic_changes = getattr(
                business_critical_config, "prevent_automatic_changes", True
            )

            # Load explicit model assignments
            explicit_assignments = getattr(
                business_critical_config, "explicit_assignments", {}
            )
            if explicit_assignments:
                for model_name, tier in explicit_assignments.items():
                    if tier in [ModelTier.HOT, ModelTier.WARM, ModelTier.COLD]:
                        self.manual_tier_assignments[model_name] = tier
                        self.business_critical_models.add(model_name)
                        logger.info(
                            f"Loaded explicit business-critical assignment: {model_name} -> {tier}"
                        )
                    else:
                        logger.warning(
                            f"Invalid tier '{tier}' for model '{model_name}', skipping"
                        )

            # Load pattern-based assignments
            patterns = getattr(business_critical_config, "patterns", {})
            if patterns:
                for pattern, tier in patterns.items():
                    if tier in [ModelTier.HOT, ModelTier.WARM, ModelTier.COLD]:
                        self.manual_assignment_patterns.append(pattern)
                        self.pattern_tier_assignments[pattern] = tier
                        logger.info(
                            f"Loaded business-critical pattern: {pattern} -> {tier}"
                        )
                    else:
                        logger.warning(
                            f"Invalid tier '{tier}' for pattern '{pattern}', skipping"
                        )

            logger.info(
                f"Loaded business-critical configuration: "
                f"{len(self.manual_tier_assignments)} explicit assignments, "
                f"{len(self.manual_assignment_patterns)} patterns, "
                f"default_tier={self.default_business_critical_tier}, "
                f"prevent_automatic_changes={self.prevent_automatic_changes}"
            )

        except Exception as e:
            logger.error(f"Error loading business-critical models configuration: {e}")

    async def start(self):
        """Start the tier management system"""
        if self._running:
            return

        self._running = True
        self._tier_monitor_task = asyncio.create_task(self._tier_monitoring_loop())
        logger.info("Tier management system started")

    async def stop(self):
        """Stop the tier management system"""
        self._running = False
        if self._tier_monitor_task:
            self._tier_monitor_task.cancel()
            try:
                await self._tier_monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Tier management system stopped")

    def register_model(self, model_name: str, initial_tier: str = ModelTier.COLD):
        """Register a new model with initial tier, applying business-critical assignments if applicable"""
        if model_name not in self.model_metrics:
            # Check for business-critical assignments
            assigned_tier = initial_tier

            # Check explicit assignments first
            if model_name in self.manual_tier_assignments:
                assigned_tier = self.manual_tier_assignments[model_name]
                logger.info(
                    f"Applying explicit business-critical assignment: {model_name} -> {assigned_tier}"
                )

            # Check pattern-based assignments
            elif self._matches_business_critical_pattern(model_name):
                pattern_tier = self._get_pattern_tier_assignment(model_name)
                if pattern_tier:
                    assigned_tier = pattern_tier
                    # Add to manual assignments to prevent automatic changes
                    self.manual_tier_assignments[model_name] = assigned_tier
                    self.business_critical_models.add(model_name)
                    logger.info(
                        f"Applying pattern-based business-critical assignment: {model_name} -> {assigned_tier}"
                    )
                else:
                    assigned_tier = self.default_business_critical_tier
                    self.manual_tier_assignments[model_name] = assigned_tier
                    self.business_critical_models.add(model_name)
                    logger.info(
                        f"Applying default business-critical assignment: {model_name} -> {assigned_tier}"
                    )

            self.model_metrics[model_name] = ModelMetrics(
                model_name=model_name,
                current_tier=assigned_tier,
                tier_change_time=time.time(),
            )
            self.tier_models[assigned_tier].add(model_name)

            if assigned_tier != initial_tier:
                logger.info(
                    f"Registered business-critical model {model_name} in {assigned_tier} tier (overrode {initial_tier})"
                )
            else:
                logger.info(f"Registered model {model_name} in {assigned_tier} tier")

    def record_request(
        self, model_name: str, response_time: float, is_error: bool = False
    ):
        """Record a request for tier calculation"""
        if model_name not in self.model_metrics:
            self.register_model(model_name)

        metrics = self.model_metrics[model_name]
        metrics.add_request(response_time, is_error)

        # Add to tier-specific request window
        current_time = time.time()
        tier = metrics.current_tier
        self.tier_request_windows[tier][model_name].append(current_time)

    def get_model_tier(self, model_name: str) -> str:
        """Get current tier of a model"""
        if model_name not in self.model_metrics:
            return ModelTier.COLD
        return self.model_metrics[model_name].current_tier

    def get_tier_models(self, tier: str) -> Set[str]:
        """Get all models in a specific tier"""
        return self.tier_models.get(tier, set()).copy()

    def get_tier_resource_config(self, tier: str) -> Optional[TierResourceConfig]:
        """Get resource configuration for a tier"""
        return self.config.tier_configs.get(tier)

    def force_tier_change(self, model_name: str, new_tier: str) -> bool:
        """Force a model to change tier"""
        if model_name not in self.model_metrics:
            return False

        old_tier = self.model_metrics[model_name].current_tier
        if old_tier == new_tier:
            return True

        # Remove from old tier
        self.tier_models[old_tier].discard(model_name)

        # Add to new tier
        self.tier_models[new_tier].add(model_name)
        self.model_metrics[model_name].current_tier = new_tier
        self.model_metrics[model_name].tier_change_time = time.time()

        logger.info(f"Forced tier change: {model_name} from {old_tier} to {new_tier}")
        return True

    async def _tier_monitoring_loop(self):
        """Main tier monitoring loop"""
        while self._running:
            try:
                await self._evaluate_tier_changes()
                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in tier monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error

    async def _evaluate_tier_changes(self):
        """Evaluate and execute tier changes"""
        current_time = time.time()
        changes_made = []

        for model_name, metrics in self.model_metrics.items():
            # Skip models with manual tier assignments
            if self._is_manually_assigned(model_name):
                logger.debug(
                    f"Skipping automatic tier evaluation for manually assigned model: {model_name}"
                )
                continue

            new_tier = self._calculate_optimal_tier(model_name, current_time)

            if new_tier != metrics.current_tier:
                # Check if enough time has passed since last tier change
                time_since_change = current_time - metrics.tier_change_time
                min_time_between_changes = 300  # 5 minutes minimum

                if time_since_change >= min_time_between_changes:
                    old_tier = metrics.current_tier

                    # Remove from old tier
                    self.tier_models[old_tier].discard(model_name)

                    # Add to new tier
                    self.tier_models[new_tier].add(model_name)
                    metrics.current_tier = new_tier
                    metrics.tier_change_time = current_time

                    changes_made.append((model_name, old_tier, new_tier))

        if changes_made:
            for model_name, old_tier, new_tier in changes_made:
                logger.info(
                    f"Automatic tier change: {model_name} from {old_tier} to {new_tier}"
                )

            # Notify deployment system about tier changes
            await self._notify_deployment_changes(changes_made)

    def _calculate_optimal_tier(self, model_name: str, current_time: float) -> str:
        """Calculate optimal tier for a model based on metrics"""
        metrics = self.model_metrics[model_name]
        current_tier = metrics.current_tier

        # Calculate request rate in different time windows
        hot_window_requests = self._count_requests_in_window(
            model_name, current_time, self.config.promotion_time_window
        )
        warm_window_requests = self._count_requests_in_window(
            model_name, current_time, self.config.promotion_time_window * 2
        )

        # Check for promotion to HOT tier
        if (
            current_tier != ModelTier.HOT
            and hot_window_requests >= self.config.promotion_threshold
        ):
            return ModelTier.HOT

        # Check for promotion to WARM tier
        if (
            current_tier == ModelTier.COLD
            and warm_window_requests >= self.config.promotion_threshold // 4
        ):
            return ModelTier.WARM

        # Check for demotion
        time_since_last_request = current_time - metrics.last_request_time

        # Demote from HOT to WARM
        if (
            current_tier == ModelTier.HOT
            and hot_window_requests < self.config.demotion_threshold
            and time_since_last_request > self.config.demotion_time_window // 2
        ):
            return ModelTier.WARM

        # Demote from WARM to COLD
        if (
            current_tier == ModelTier.WARM
            and warm_window_requests < self.config.demotion_threshold
            and time_since_last_request > self.config.demotion_time_window
        ):
            return ModelTier.COLD

        return current_tier

    def _count_requests_in_window(
        self, model_name: str, current_time: float, window_seconds: int
    ) -> int:
        """Count requests for a model in a time window"""
        cutoff_time = current_time - window_seconds
        count = 0

        for tier in [ModelTier.HOT, ModelTier.WARM, ModelTier.COLD]:
            if model_name in self.tier_request_windows[tier]:
                requests = self.tier_request_windows[tier][model_name]
                count += sum(1 for req_time in requests if req_time >= cutoff_time)

        return count

    async def _notify_deployment_changes(self, changes: List[Tuple[str, str, str]]):
        """Notify deployment system about tier changes"""
        try:
            # Notify router about tier changes for rebalancing
            if hasattr(self, "_router") and self._router:
                self._router.rebalance_tiers(changes)

            # Notify deployment manager about resource reallocation needs
            if hasattr(self, "_deployment_manager") and self._deployment_manager:
                await self._trigger_resource_reallocation(changes)

            logger.info(f"Notified deployment system about {len(changes)} tier changes")

        except Exception as e:
            logger.error(f"Error notifying deployment system: {e}")

    async def _trigger_resource_reallocation(self, changes: List[Tuple[str, str, str]]):
        """Trigger resource reallocation for tier changes"""
        try:
            # Group changes by new tier
            tier_changes = {ModelTier.HOT: [], ModelTier.WARM: [], ModelTier.COLD: []}

            for model_name, old_tier, new_tier in changes:
                tier_changes[new_tier].append((model_name, old_tier))

            # Check if we need to scale deployments for each tier
            for tier, models_changed in tier_changes.items():
                if models_changed:
                    await self._check_tier_scaling_needs(tier, models_changed)

        except Exception as e:
            logger.error(f"Error triggering resource reallocation: {e}")

    async def _check_tier_scaling_needs(
        self, tier: str, models_changed: List[Tuple[str, str]]
    ):
        """Check if tier deployments need scaling based on model changes"""
        try:
            tier_config = self.config.tier_configs.get(tier)
            if not tier_config:
                return

            # Get current models in tier
            current_models = self.get_tier_models(tier)
            models_added = len(
                [m for m, old_tier in models_changed if old_tier != tier]
            )

            # Calculate if we need more deployment capacity
            models_per_replica = tier_config.max_models_per_replica
            current_capacity_needed = len(current_models) / models_per_replica

            # Log scaling recommendation
            if models_added > 0:
                logger.info(
                    f"Tier {tier}: {models_added} models added, "
                    f"current capacity needed: {current_capacity_needed:.1f} replicas"
                )

                # In a full implementation, this would trigger actual scaling
                # For now, we just log the recommendation
                if current_capacity_needed > tier_config.max_replicas * 0.8:
                    logger.warning(
                        f"Tier {tier} approaching capacity limits, "
                        f"consider scaling up deployments"
                    )

        except Exception as e:
            logger.error(f"Error checking scaling needs for tier {tier}: {e}")

    def set_router(self, router):
        """Set the router for tier change notifications"""
        self._router = router
        logger.info("Router registered with tier manager")

    def set_deployment_manager(self, deployment_manager):
        """Set the deployment manager for resource reallocation"""
        self._deployment_manager = deployment_manager
        logger.info("Deployment manager registered with tier manager")

    # Lazy Loading Methods

    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model has been loaded"""
        return model_name in self.loaded_models

    def is_model_deployed(self, model_name: str) -> bool:
        """Check if a model has deployment/router created"""
        return model_name in self.deployed_models

    def is_model_loading(self, model_name: str) -> bool:
        """Check if a model is currently being loaded"""
        return model_name in self.loading_in_progress

    def should_load_immediately(self, model_name: str) -> bool:
        """Check if a model should be loaded immediately based on its tier"""
        tier = self.get_model_tier(model_name)
        return tier == ModelTier.HOT

    def should_deploy_immediately(self, model_name: str) -> bool:
        """Check if a model should have deployment/router created immediately"""
        tier = self.get_model_tier(model_name)
        return tier == ModelTier.HOT

    def should_load_on_request(self, model_name: str) -> bool:
        """Check if a model should be loaded on first request"""
        tier = self.get_model_tier(model_name)
        return tier in [ModelTier.WARM, ModelTier.COLD]

    def should_deploy_on_request(self, model_name: str) -> bool:
        """Check if a model should have deployment created on first request"""
        tier = self.get_model_tier(model_name)
        return tier == ModelTier.COLD or (
            tier == ModelTier.WARM and not self.is_model_deployed(model_name)
        )

    async def handle_lazy_loading_request(self, model_name: str) -> Dict[str, Any]:
        """Handle lazy loading for a model based on its tier and current state"""
        tier = self.get_model_tier(model_name)

        # Prevent concurrent loading of the same model
        if self.is_model_loading(model_name):
            return {
                "status": "loading",
                "message": f"Model {model_name} is currently being loaded",
                "tier": tier,
            }

        try:
            self.loading_in_progress.add(model_name)

            # HOT tier: Should already be loaded and deployed
            if tier == ModelTier.HOT:
                if not self.is_model_loaded(model_name) or not self.is_model_deployed(
                    model_name
                ):
                    logger.warning(
                        f"HOT tier model {model_name} not properly initialized"
                    )
                    return await self._load_and_deploy_model(model_name)
                return {"status": "ready", "tier": tier}

            # WARM tier: Load if not loaded, deploy if not deployed
            elif tier == ModelTier.WARM:
                if not self.is_model_loaded(model_name):
                    result = await self._load_model_only(model_name)
                    if result["status"] != "success":
                        return result

                if not self.is_model_deployed(model_name):
                    return await self._deploy_model_only(model_name)

                return {"status": "ready", "tier": tier}

            # COLD tier: Load and deploy on demand
            elif tier == ModelTier.COLD:
                if not self.is_model_loaded(model_name) or not self.is_model_deployed(
                    model_name
                ):
                    return await self._load_and_deploy_model(model_name)
                return {"status": "ready", "tier": tier}

            else:
                return {
                    "status": "error",
                    "message": f"Unknown tier {tier} for model {model_name}",
                }

        finally:
            self.loading_in_progress.discard(model_name)

    async def _load_model_only(self, model_name: str) -> Dict[str, Any]:
        """Load a model without creating deployment/router"""
        try:
            if hasattr(self, "_model_manager") and self._model_manager:
                model_info = await self._model_manager.load_model_async(model_name)
                print(f"Model {model_name} loaded with info: {model_info}")
                if model_info:
                    self.loaded_models.add(model_name)
                    logger.info(f"Loaded model {model_name} (no deployment created)")
                    return {"status": "success", "action": "loaded"}
                else:
                    return {
                        "status": "error",
                        "message": f"Failed to load model {model_name}",
                    }
            else:
                return {"status": "error", "message": "Model manager not available"}
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return {"status": "error", "message": str(e)}

    async def _deploy_model_only(self, model_name: str) -> Dict[str, Any]:
        """Create deployment/router for an already loaded model"""
        try:
            if hasattr(self, "_deployment_manager") and self._deployment_manager:
                # Get appropriate deployment for the model's tier
                tier = self.get_model_tier(model_name)
                deployment_name = self._deployment_manager.get_tier_deployment(
                    tier, model_name
                )

                if deployment_name:
                    # For HOT tier models, actually pre-load them into the Ray deployment
                    if tier == ModelTier.HOT:
                        logger.info(
                            f"Pre-loading HOT tier model {model_name} into deployment {deployment_name}"
                        )
                        preload_success = await self._deployment_manager.preload_model_into_deployment(
                            deployment_name, model_name
                        )

                        if preload_success:
                            self.deployed_models.add(model_name)
                            logger.info(
                                f"Successfully pre-loaded HOT model {model_name} into deployment {deployment_name}"
                            )
                            return {
                                "status": "success",
                                "action": "deployed_and_preloaded",
                                "deployment": deployment_name,
                            }
                        else:
                            logger.error(
                                f"Failed to pre-load HOT model {model_name} into deployment {deployment_name}"
                            )
                            return {
                                "status": "error",
                                "message": f"Failed to pre-load HOT model {model_name}",
                            }
                    else:
                        # For WARM and COLD tiers, just mark as deployed (lazy loading)
                        self.deployed_models.add(model_name)
                        logger.info(
                            f"Created deployment for model {model_name} in tier {tier}"
                        )
                        return {
                            "status": "success",
                            "action": "deployed",
                            "deployment": deployment_name,
                        }
                else:
                    return {
                        "status": "error",
                        "message": f"No deployment available for tier {tier}",
                    }
            else:
                return {
                    "status": "error",
                    "message": "Deployment manager not available",
                }
        except Exception as e:
            logger.error(f"Error deploying model {model_name}: {e}")
            return {"status": "error", "message": str(e)}

    async def _load_and_deploy_model(self, model_name: str) -> Dict[str, Any]:
        """Load model and create deployment/router"""
        try:
            # First load the model
            load_result = await self._load_model_only(model_name)
            if load_result["status"] != "success":
                return load_result

            # Then create deployment
            deploy_result = await self._deploy_model_only(model_name)
            if deploy_result["status"] != "success":
                return deploy_result

            return {
                "status": "success",
                "action": "loaded_and_deployed",
                "deployment": deploy_result.get("deployment"),
            }

        except Exception as e:
            logger.error(f"Error loading and deploying model {model_name}: {e}")
            return {"status": "error", "message": str(e)}

    def set_model_manager(self, model_manager):
        """Set the model manager for lazy loading"""
        self._model_manager = model_manager
        logger.info("Model manager registered with tier manager")

    # Manual Tier Assignment Methods

    def _is_manually_assigned(self, model_name: str) -> bool:
        """Check if a model has a manual tier assignment"""
        return (
            model_name in self.manual_tier_assignments
            or self._matches_business_critical_pattern(model_name)
        )

    def _matches_business_critical_pattern(self, model_name: str) -> bool:
        """Check if model name matches any business-critical patterns"""
        import fnmatch

        for pattern in self.manual_assignment_patterns:
            if fnmatch.fnmatch(model_name, pattern):
                return True
        return False

    def _get_pattern_tier_assignment(self, model_name: str) -> Optional[str]:
        """Get tier assignment for a model based on pattern matching"""
        import fnmatch

        for pattern, tier in self.pattern_tier_assignments.items():
            if fnmatch.fnmatch(model_name, pattern):
                return tier
        return None

    def set_manual_tier_assignment(
        self, model_name: str, tier: str, is_business_critical: bool = True
    ) -> bool:
        """
        Set a manual tier assignment for a model

        Args:
            model_name: Name of the model
            tier: Target tier (HOT/WARM/COLD)
            is_business_critical: Whether this is a business-critical model

        Returns:
            bool: True if assignment was successful
        """
        if tier not in [ModelTier.HOT, ModelTier.WARM, ModelTier.COLD]:
            logger.error(f"Invalid tier '{tier}' for manual assignment")
            return False

        # Register model if not exists
        if model_name not in self.model_metrics:
            self.register_model(model_name, tier)

        # Get current tier
        old_tier = self.model_metrics[model_name].current_tier

        # Update manual assignment tracking
        self.manual_tier_assignments[model_name] = tier

        if is_business_critical:
            self.business_critical_models.add(model_name)

        # Perform the tier change if needed
        if old_tier != tier:
            # Remove from old tier
            self.tier_models[old_tier].discard(model_name)

            # Add to new tier
            self.tier_models[tier].add(model_name)
            self.model_metrics[model_name].current_tier = tier
            self.model_metrics[model_name].tier_change_time = time.time()

            logger.info(
                f"Manual tier assignment: {model_name} from {old_tier} to {tier} (business_critical={is_business_critical})"
            )
        else:
            logger.info(
                f"Manual tier assignment confirmed: {model_name} remains in {tier} (business_critical={is_business_critical})"
            )

        return True

    def remove_manual_tier_assignment(self, model_name: str) -> bool:
        """
        Remove manual tier assignment for a model, allowing automatic tier management

        Args:
            model_name: Name of the model

        Returns:
            bool: True if removal was successful
        """
        if model_name not in self.manual_tier_assignments:
            logger.warning(f"No manual tier assignment found for model: {model_name}")
            return False

        # Remove from manual assignments
        del self.manual_tier_assignments[model_name]
        self.business_critical_models.discard(model_name)

        logger.info(f"Removed manual tier assignment for model: {model_name}")
        return True

    def get_manual_tier_assignment(self, model_name: str) -> Optional[str]:
        """Get the manual tier assignment for a model"""
        return self.manual_tier_assignments.get(model_name)

    def is_business_critical(self, model_name: str) -> bool:
        """Check if a model is marked as business-critical"""
        return (
            model_name in self.business_critical_models
            or self._matches_business_critical_pattern(model_name)
        )

    def add_business_critical_pattern(self, pattern: str) -> None:
        """
        Add a pattern for automatic business-critical model detection

        Args:
            pattern: Glob pattern (e.g., "CRITICAL_*", "*_PROD_*")
        """
        if pattern not in self.manual_assignment_patterns:
            self.manual_assignment_patterns.append(pattern)
            logger.info(f"Added business-critical pattern: {pattern}")

    def remove_business_critical_pattern(self, pattern: str) -> bool:
        """Remove a business-critical pattern"""
        if pattern in self.manual_assignment_patterns:
            self.manual_assignment_patterns.remove(pattern)
            logger.info(f"Removed business-critical pattern: {pattern}")
            return True
        return False

    def get_business_critical_models(self) -> Dict[str, Dict[str, Any]]:
        """Get all business-critical models and their assignments"""
        result = {}

        # Explicit business-critical models
        for model_name in self.business_critical_models:
            result[model_name] = {
                "tier": self.manual_tier_assignments.get(
                    model_name, self.get_model_tier(model_name)
                ),
                "assignment_type": "explicit",
                "is_manually_assigned": model_name in self.manual_tier_assignments,
            }

        # Pattern-matched models
        for model_name in self.model_metrics.keys():
            if model_name not in result and self._matches_business_critical_pattern(
                model_name
            ):
                result[model_name] = {
                    "tier": self.get_model_tier(model_name),
                    "assignment_type": "pattern",
                    "is_manually_assigned": model_name in self.manual_tier_assignments,
                }

        return result

    def bulk_assign_business_critical_models(
        self, model_assignments: Dict[str, str]
    ) -> Dict[str, bool]:
        """
        Bulk assign multiple models as business-critical

        Args:
            model_assignments: Dict of model_name -> tier

        Returns:
            Dict of model_name -> success_status
        """
        results = {}
        for model_name, tier in model_assignments.items():
            results[model_name] = self.set_manual_tier_assignment(
                model_name, tier, is_business_critical=True
            )

        logger.info(f"Bulk assigned {len(model_assignments)} business-critical models")
        return results

    def get_tier_statistics(self) -> Dict[str, Dict]:
        """Get statistics for all tiers"""
        stats = {}

        for tier in [ModelTier.HOT, ModelTier.WARM, ModelTier.COLD]:
            tier_models = self.tier_models[tier]
            total_requests = sum(
                self.model_metrics[model].request_count
                for model in tier_models
                if model in self.model_metrics
            )

            avg_response_time = 0.0
            if tier_models:
                response_times = [
                    self.model_metrics[model].get_average_response_time()
                    for model in tier_models
                    if model in self.model_metrics
                ]
                if response_times:
                    avg_response_time = sum(response_times) / len(response_times)

            stats[tier] = {
                "model_count": len(tier_models),
                "total_requests": total_requests,
                "average_response_time": avg_response_time,
                "models": list(tier_models),
            }

        return stats
