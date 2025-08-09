"""
Tier-aware intelligent load balancer for routing requests based on model tiers
"""

import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from ..config.base_types import ModelTier, RoutingStrategy
from ..deployment.resource_config import TierBasedDeploymentConfig
from adserving.src.utils.logger import get_logger

logger = get_logger()


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
