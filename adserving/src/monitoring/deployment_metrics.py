"""
Deployment metrics data structures.
"""

import time
from dataclasses import dataclass, field


@dataclass
class DeploymentMetrics:
    """Metrics for pooled deployments"""

    deployment_name: str
    replica_count: int = 0
    target_replicas: int = 0

    # Load metrics
    current_load: int = 0
    avg_load: float = 0.0
    peak_load: int = 0

    # Queue metrics
    queue_length: int = 0
    avg_queue_time: float = 0.0
    max_queue_time: float = 0.0

    # Health metrics
    healthy_replicas: int = 0
    unhealthy_replicas: int = 0

    # Resource metrics
    total_cpu_usage: float = 0.0
    total_memory_usage: float = 0.0
    total_gpu_usage: float = 0.0

    # Model distribution
    loaded_models: int = 0
    active_models: int = 0

    last_updated: float = field(default_factory=time.time)
