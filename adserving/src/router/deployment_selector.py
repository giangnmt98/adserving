"""
Deployment selection strategies
"""

from typing import Dict, List, Optional

from .route_metrics import RouteMetrics
from .routing_strategy import RoutingStrategy
from adserving.src.utils.logger import get_logger


class DeploymentSelector:
    """Deployment selection based on routing strategy"""

    def __init__(self):
        self.logger = get_logger()

    def select_deployment(
        self,
        routing_strategy: RoutingStrategy,
        model_name: str,
        available_deployments: List[str],
        deployment_loads: Dict[str, int],
        route_metrics: Dict[str, RouteMetrics],
        model_affinity: Dict[str, str],
        round_robin_index: int,
    ) -> tuple[Optional[str], int]:
        """Select optimal deployment based on routing strategy"""
        if not available_deployments:
            return None, round_robin_index

        if routing_strategy == RoutingStrategy.ROUND_ROBIN:
            return self._select_round_robin(available_deployments, round_robin_index)

        elif routing_strategy == RoutingStrategy.LEAST_LOADED:
            return (
                self._select_least_loaded(available_deployments, deployment_loads),
                round_robin_index,
            )

        elif routing_strategy == RoutingStrategy.FASTEST_RESPONSE:
            return (
                self._select_fastest_response(
                    model_name, available_deployments, route_metrics, deployment_loads
                ),
                round_robin_index,
            )

        elif routing_strategy == RoutingStrategy.MODEL_AFFINITY:
            return (
                self._select_with_affinity(
                    model_name, available_deployments, model_affinity, deployment_loads
                ),
                round_robin_index,
            )

        else:
            # Default to least loaded
            return (
                self._select_least_loaded(available_deployments, deployment_loads),
                round_robin_index,
            )

    def _select_round_robin(
        self, available_deployments: List[str], round_robin_index: int
    ) -> tuple[str, int]:
        """Round-robin deployment selection"""
        if not available_deployments:
            return "", round_robin_index

        deployment = available_deployments[round_robin_index]
        new_index = (round_robin_index + 1) % len(available_deployments)
        return deployment, new_index

    def _select_least_loaded(
        self, available_deployments: List[str], deployment_loads: Dict[str, int]
    ) -> str:
        """Select deployment with least current load"""
        if not available_deployments:
            return ""

        return min(available_deployments, key=lambda d: deployment_loads.get(d, 0))

    def _select_fastest_response(
        self,
        model_name: str,
        available_deployments: List[str],
        route_metrics: Dict[str, RouteMetrics],
        deployment_loads: Dict[str, int],
    ) -> str:
        """Select deployment with fastest response time for this model"""
        if not available_deployments:
            return ""

        # Find deployments that have served this model before
        model_deployments = [
            dep
            for dep in available_deployments
            if f"{model_name}_{dep}" in route_metrics
        ]

        if model_deployments:
            # Select fastest for this model
            return min(
                model_deployments,
                key=lambda d: route_metrics.get(
                    f"{model_name}_{d}",
                    RouteMetrics(model_name, d, float("inf"), 0, 0, 0),
                ).avg_response_time,
            )
        else:
            # Fallback to least loaded
            return self._select_least_loaded(available_deployments, deployment_loads)

    def _select_with_affinity(
        self,
        model_name: str,
        available_deployments: List[str],
        model_affinity: Dict[str, str],
        deployment_loads: Dict[str, int],
    ) -> str:
        """Select deployment with model affinity (sticky routing)"""
        # Check if model has affinity to a deployment
        if model_name in model_affinity:
            preferred_deployment = model_affinity[model_name]
            if preferred_deployment in available_deployments:
                return preferred_deployment

        # No affinity or preferred deployment unavailable
        selected = self._select_least_loaded(available_deployments, deployment_loads)

        # Create affinity for future requests (this will be updated by caller)
        return selected
