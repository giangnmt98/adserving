"""
Resource configuration for pooled deployments
"""

from dataclasses import dataclass
from typing import Dict, Optional
from ..config.base_types import ModelTier


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


@dataclass
class TierResourceConfig:
    """Tier-specific resource configuration"""
    
    tier: str
    num_cpus: float
    num_gpus: float
    memory: int  # MB
    object_store_memory: int  # MB
    max_models_per_replica: int
    min_replicas: int
    max_replicas: int
    target_requests_per_replica: int
    priority: int  # Higher priority for better tiers


@dataclass
class TierBasedDeploymentConfig:
    """Configuration for tier-based deployment strategy"""
    
    # Tier-specific resource configurations
    tier_configs: Dict[str, TierResourceConfig]
    
    # Dynamic scaling settings
    promotion_threshold: int = 20  # Requests to promote to higher tier
    demotion_threshold: int = 2   # Low requests to demote to lower tier
    promotion_time_window: int = 300  # Time window for promotion (seconds)
    demotion_time_window: int = 1800  # Time window for demotion (seconds)
    
    # Load balancing settings
    enable_tier_aware_routing: bool = True
    prefer_higher_tier: bool = True
    tier_routing_weights: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        if not self.tier_configs:
            self.tier_configs = self._get_default_tier_configs()
        
        if self.tier_routing_weights is None:
            self.tier_routing_weights = {
                ModelTier.HOT: 1.0,
                ModelTier.WARM: 0.7,
                ModelTier.COLD: 0.3
            }
    
    def _get_default_tier_configs(self) -> Dict[str, TierResourceConfig]:
        """Get default tier configurations"""
        return {
            ModelTier.HOT: TierResourceConfig(
                tier=ModelTier.HOT,
                num_cpus=6.0,  # High CPU allocation
                num_gpus=1.0,  # Dedicated GPU
                memory=12288,  # 12GB RAM
                object_store_memory=8192,  # 8GB object store
                max_models_per_replica=5,  # Fewer models per replica for better performance
                min_replicas=3,  # Always keep replicas running
                max_replicas=20,  # High scaling capacity
                target_requests_per_replica=8,  # Lower target for faster response
                priority=100
            ),
            ModelTier.WARM: TierResourceConfig(
                tier=ModelTier.WARM,
                num_cpus=4.0,  # Medium CPU allocation
                num_gpus=0.5,  # Shared GPU
                memory=8192,   # 8GB RAM
                object_store_memory=4096,  # 4GB object store
                max_models_per_replica=15,  # Medium models per replica
                min_replicas=2,  # Keep some replicas running
                max_replicas=15,  # Medium scaling capacity
                target_requests_per_replica=12,  # Medium target
                priority=50
            ),
            ModelTier.COLD: TierResourceConfig(
                tier=ModelTier.COLD,
                num_cpus=2.0,  # Low CPU allocation
                num_gpus=0.0,  # No GPU
                memory=4096,   # 4GB RAM
                object_store_memory=2048,  # 2GB object store
                max_models_per_replica=30,  # Many models per replica
                min_replicas=1,  # Minimal replicas
                max_replicas=10,  # Limited scaling
                target_requests_per_replica=20,  # Higher target acceptable
                priority=10
            )
        }
