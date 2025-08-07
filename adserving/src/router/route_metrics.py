"""
Metrics for routing decisions
"""

from dataclasses import dataclass


@dataclass
class RouteMetrics:
    """Metrics for routing decisions"""

    model_name: str
    deployment_name: str
    avg_response_time: float
    current_load: int
    success_rate: float
    last_updated: float
