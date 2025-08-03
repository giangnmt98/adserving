"""
Tier-aware intelligent load balancer for routing requests based on model tiers
"""

import logging
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from ..config.base_types import ModelTier, RoutingStrategy
from ..deployment.resource_config import TierBasedDeploymentConfig

logger = logging.getLogger(__name__)


@dataclass
class DeploymentMetrics:
    """Metrics for a deployment to make routing decisions"""

    deployment_name: str
    tier: str
    current_load: int = 0
    average_response_time: float = 0.0
    error_rate: float = 0.0
    last_request_time: float = 0
    total_requests: int = 0
    recent_response_times: deque = None

    def __post_init__(self):
        if self.recent_response_times is None:
            self.recent_response_times = deque(maxlen=100)

    def update_metrics(self, response_time: float, is_error: bool = False):
        """Update deployment metrics with new request data"""
        self.current_load = max(0, self.current_load - 1)  # Assume request completed
        self.last_request_time = time.time()
        self.total_requests += 1
        self.recent_response_times.append(response_time)

        # Update average response time
        if self.recent_response_times:
            self.average_response_time = sum(self.recent_response_times) / len(
                self.recent_response_times
            )

        # Update error rate (simple moving average)
        if is_error:
            self.error_rate = min(1.0, self.error_rate * 0.9 + 0.1)
        else:
            self.error_rate = max(0.0, self.error_rate * 0.95)

    def add_request(self):
        """Add a new request to current load"""
        self.current_load += 1


class TierAwareRouter:
    """Intelligent load balancer with tier-aware routing"""

    def __init__(self, config: TierBasedDeploymentConfig, tier_manager=None):
        self.config = config
        self.tier_manager = tier_manager
        self.deployment_metrics: Dict[str, DeploymentMetrics] = {}
        self.tier_deployments: Dict[str, List[str]] = {
            ModelTier.HOT: [],
            ModelTier.WARM: [],
            ModelTier.COLD: [],
        }
        self.model_to_deployment: Dict[str, str] = {}
        self.routing_strategy = RoutingStrategy.LEAST_LOADED
        self._request_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )

    def register_deployment(self, deployment_name: str, tier: str):
        """Register a deployment with the router"""
        if deployment_name not in self.deployment_metrics:
            self.deployment_metrics[deployment_name] = DeploymentMetrics(
                deployment_name=deployment_name, tier=tier
            )

        if deployment_name not in self.tier_deployments[tier]:
            self.tier_deployments[tier].append(deployment_name)

        logger.info(f"Registered deployment {deployment_name} for {tier} tier")

    def unregister_deployment(self, deployment_name: str):
        """Unregister a deployment from the router"""
        if deployment_name in self.deployment_metrics:
            tier = self.deployment_metrics[deployment_name].tier
            self.tier_deployments[tier] = [
                d for d in self.tier_deployments[tier] if d != deployment_name
            ]
            del self.deployment_metrics[deployment_name]

        # Remove from model mappings
        self.model_to_deployment = {
            model: deployment
            for model, deployment in self.model_to_deployment.items()
            if deployment != deployment_name
        }

        logger.info(f"Unregistered deployment {deployment_name}")

    def route_request(self, model_name: str, request_data: Any = None) -> Optional[str]:
        """Route a request to the best available deployment"""
        try:
            # Get model tier
            model_tier = self._get_model_tier(model_name)

            # Get candidate deployments for the tier
            candidates = self._get_candidate_deployments(model_tier, model_name)

            if not candidates:
                # Fallback to other tiers if no deployments available
                candidates = self._get_fallback_deployments(model_tier)

            if not candidates:
                logger.warning(f"No available deployments for model {model_name}")
                return None

            # Select best deployment based on routing strategy
            selected_deployment = self._select_deployment(
                candidates, model_name, request_data
            )

            if selected_deployment:
                # Update metrics
                self.deployment_metrics[selected_deployment].add_request()
                self._record_routing_decision(model_name, selected_deployment)

                logger.debug(f"Routed {model_name} to {selected_deployment}")

            return selected_deployment

        except Exception as e:
            logger.error(f"Error routing request for {model_name}: {e}")
            return None

    def update_request_metrics(
        self, deployment_name: str, response_time: float, is_error: bool = False
    ):
        """Update metrics after request completion"""
        if deployment_name in self.deployment_metrics:
            self.deployment_metrics[deployment_name].update_metrics(
                response_time, is_error
            )

    def _get_model_tier(self, model_name: str) -> str:
        """Get the tier for a model"""
        if self.tier_manager:
            return self.tier_manager.get_model_tier(model_name)
        return ModelTier.COLD  # Default fallback

    def _get_candidate_deployments(self, tier: str, model_name: str) -> List[str]:
        """Get candidate deployments for a tier"""
        candidates = []

        # Primary tier deployments
        if tier in self.tier_deployments:
            candidates.extend(self.tier_deployments[tier])

        # Apply tier-aware filtering
        if self.config.enable_tier_aware_routing:
            # Filter based on deployment health and capacity
            healthy_candidates = []
            for deployment in candidates:
                if deployment in self.deployment_metrics:
                    metrics = self.deployment_metrics[deployment]

                    # Check if deployment is healthy
                    if (
                        metrics.error_rate < 0.1
                        and metrics.current_load < 100  # Less than 10% error rate
                    ):  # Not overloaded
                        healthy_candidates.append(deployment)

            if healthy_candidates:
                candidates = healthy_candidates

        return candidates

    def _get_fallback_deployments(self, original_tier: str) -> List[str]:
        """Get fallback deployments when primary tier is unavailable"""
        fallback_order = []

        if original_tier == ModelTier.HOT:
            fallback_order = [ModelTier.WARM, ModelTier.COLD]
        elif original_tier == ModelTier.WARM:
            fallback_order = [ModelTier.HOT, ModelTier.COLD]
        else:  # COLD
            fallback_order = [ModelTier.WARM, ModelTier.HOT]

        for tier in fallback_order:
            candidates = self.tier_deployments.get(tier, [])
            if candidates:
                logger.info(f"Using fallback tier {tier} for {original_tier} request")
                return candidates

        return []

    def _select_deployment(
        self, candidates: List[str], model_name: str, request_data: Any
    ) -> Optional[str]:
        """Select the best deployment from candidates"""
        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0]

        if self.routing_strategy == RoutingStrategy.LEAST_LOADED:
            return self._select_least_loaded(candidates)
        elif self.routing_strategy == RoutingStrategy.FASTEST_RESPONSE:
            return self._select_fastest_response(candidates)
        elif self.routing_strategy == RoutingStrategy.MODEL_AFFINITY:
            return self._select_model_affinity(candidates, model_name)
        else:  # ROUND_ROBIN
            return self._select_round_robin(candidates)

    def _select_least_loaded(self, candidates: List[str]) -> str:
        """Select deployment with least current load"""
        best_deployment = candidates[0]
        min_load = float("inf")

        for deployment in candidates:
            if deployment in self.deployment_metrics:
                metrics = self.deployment_metrics[deployment]
                # Consider both current load and tier priority
                tier_weight = self.config.tier_routing_weights.get(metrics.tier, 1.0)
                weighted_load = metrics.current_load / tier_weight

                if weighted_load < min_load:
                    min_load = weighted_load
                    best_deployment = deployment

        return best_deployment

    def _select_fastest_response(self, candidates: List[str]) -> str:
        """Select deployment with fastest average response time"""
        best_deployment = candidates[0]
        min_response_time = float("inf")

        for deployment in candidates:
            if deployment in self.deployment_metrics:
                metrics = self.deployment_metrics[deployment]
                if metrics.average_response_time < min_response_time:
                    min_response_time = metrics.average_response_time
                    best_deployment = deployment

        return best_deployment

    def _select_model_affinity(self, candidates: List[str], model_name: str) -> str:
        """Select deployment with model affinity (sticky routing)"""
        # Check if model has been routed before
        if model_name in self.model_to_deployment:
            preferred_deployment = self.model_to_deployment[model_name]
            if preferred_deployment in candidates:
                return preferred_deployment

        # Fallback to least loaded
        return self._select_least_loaded(candidates)

    def _select_round_robin(self, candidates: List[str]) -> str:
        """Select deployment using round-robin"""
        # Simple round-robin based on total requests
        total_requests = sum(
            self.deployment_metrics.get(d, DeploymentMetrics("", "")).total_requests
            for d in candidates
        )
        return candidates[total_requests % len(candidates)]

    def _record_routing_decision(self, model_name: str, deployment_name: str):
        """Record routing decision for model affinity"""
        self.model_to_deployment[model_name] = deployment_name
        self._request_history[model_name].append(
            {"deployment": deployment_name, "timestamp": time.time()}
        )

    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing statistics"""
        stats = {
            "total_deployments": len(self.deployment_metrics),
            "tier_deployments": {
                tier: len(deployments)
                for tier, deployments in self.tier_deployments.items()
            },
            "deployment_metrics": {},
            "routing_strategy": self.routing_strategy,
        }

        for deployment_name, metrics in self.deployment_metrics.items():
            stats["deployment_metrics"][deployment_name] = {
                "tier": metrics.tier,
                "current_load": metrics.current_load,
                "average_response_time": metrics.average_response_time,
                "error_rate": metrics.error_rate,
                "total_requests": metrics.total_requests,
            }

        return stats

    def set_routing_strategy(self, strategy: str):
        """Set the routing strategy"""
        if strategy in [
            RoutingStrategy.LEAST_LOADED,
            RoutingStrategy.ROUND_ROBIN,
            RoutingStrategy.FASTEST_RESPONSE,
            RoutingStrategy.MODEL_AFFINITY,
        ]:
            self.routing_strategy = strategy
            logger.info(f"Routing strategy changed to {strategy}")
        else:
            logger.warning(f"Unknown routing strategy: {strategy}")

    def rebalance_tiers(self, tier_changes: List[Tuple[str, str, str]]):
        """Rebalance routing after tier changes"""
        for model_name, old_tier, new_tier in tier_changes:
            # Update model routing preferences
            if model_name in self.model_to_deployment:
                current_deployment = self.model_to_deployment[model_name]
                current_tier = self.deployment_metrics.get(
                    current_deployment, DeploymentMetrics("", "")
                ).tier

                # If model moved to a different tier, clear affinity to allow re-routing
                if current_tier != new_tier:
                    del self.model_to_deployment[model_name]
                    logger.info(
                        f"Cleared routing affinity for {model_name} due to tier change"
                    )

        logger.info(f"Rebalanced routing for {len(tier_changes)} tier changes")
