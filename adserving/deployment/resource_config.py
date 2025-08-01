"""
Resource configuration for pooled deployments
"""

from dataclasses import dataclass


@dataclass
class PooledResourceConfig:
    """Enhanced resource configuration for pooled deployments"""

    num_cpus: float = 4.0  # Increased for multiple models
    num_gpus: float = 1.0  # Shared GPU across models
    memory: int = 8192  # MB - Increased for multiple models
    object_store_memory: int = 4096  # MB
    max_models_per_replica: int = 10  # Maximum models per deployment replica


@dataclass
class AutoscalingSettings:
    """Enhanced autoscaling settings for hundreds of models"""

    min_replicas: int = 5  # Increased minimum
    max_replicas: int = 100  # Significantly increased maximum
    target_num_ongoing_requests_per_replica: int = 10  # Increased capacity
    metrics_interval_s: float = 5.0  # More frequent scaling decisions
    look_back_period_s: float = 30.0
    smoothing_factor: float = 0.8
    scale_up_threshold: float = 0.8  # Scale up when 80% capacity
    scale_down_threshold: float = 0.3  # Scale down when below 30%


@dataclass
class PooledDeploymentConfig:
    """Configuration for pooled model deployment"""

    deployment_name: str
    resource_config: PooledResourceConfig
    autoscaling: AutoscalingSettings
    max_concurrent_queries: int = 2000  # Significantly increased
    health_check_period_s: int = 10
    health_check_timeout_s: int = 30
    model_pool_size: int = 50  # Models per deployment pool
    enable_batching: bool = True
    batch_max_size: int = 32
    batch_timeout_ms: int = 50

    def __post_init__(self):
        if self.resource_config is None:
            self.resource_config = PooledResourceConfig()
        if self.autoscaling is None:
            self.autoscaling = AutoscalingSettings()
