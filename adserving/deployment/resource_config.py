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


"""
Resource configuration for tier-based deployment
Enhanced with complete YAML compatibility
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from enum import Enum

from ..config.base_types import ModelTier


@dataclass
class TierResourceConfig:
    """Tier-specific resource configuration"""

    tier: str  # Tier name (hot/warm/cold)
    num_cpus: float  # CPU cores allocation
    num_gpus: float  # GPU cards allocation
    memory: int  # RAM in MB
    object_store_memory: int  # Object store memory in MB
    max_models_per_replica: int  # Max models per replica
    min_replicas: int  # Minimum replicas
    max_replicas: int  # Maximum replicas
    target_requests_per_replica: int  # Target concurrent requests per replica
    priority: int  # Tier priority (higher = more important)


@dataclass
class BusinessCriticalModelsConfig:
    """Configuration for business-critical models"""

    explicit_assignments: Dict[str, str] = field(default_factory=dict)  # model_name -> tier
    patterns: Dict[str, str] = field(default_factory=dict)  # pattern -> tier
    default_business_critical_tier: str = "hot"  # Default tier for business critical
    prevent_automatic_changes: bool = True  # Prevent auto tier changes


@dataclass
class TierBasedDeploymentConfig:
    """Enhanced configuration for tier-based deployment strategy"""

    # Core enable flag
    enable_tier_based_deployment: bool = True

    # Tier-specific resource configurations
    tier_configs: Dict[str, TierResourceConfig] = field(default_factory=dict)

    # Dynamic scaling settings
    promotion_threshold: int = 20  # Requests to promote to higher tier
    demotion_threshold: int = 2  # Low requests to demote to lower tier
    promotion_time_window: int = 300  # Time window for promotion (seconds)
    demotion_time_window: int = 1800  # Time window for demotion (seconds)

    # Load balancing settings
    enable_tier_aware_routing: bool = True  # Enable intelligent tier-aware routing
    prefer_higher_tier: bool = True  # Prefer higher tier deployments
    routing_strategy: str = "least_loaded"  # FIXED: Added routing_strategy field

    # Tier routing weights (higher = more preferred)
    tier_routing_weights: Dict[str, float] = field(default_factory=lambda: {
        "hot": 1.0,
        "warm": 0.7,
        "cold": 0.3
    })

    # Monitoring and health checks
    tier_monitoring_interval: int = 30  # Tier evaluation interval (seconds)
    capacity_check_interval: int = 300  # Capacity monitoring interval (seconds)
    health_check_enabled: bool = True  # Enable tier health monitoring

    # Business-Critical Models - Manual tier assignments
    business_critical_models: BusinessCriticalModelsConfig = field(
        default_factory=BusinessCriticalModelsConfig
    )

    def __post_init__(self):
        """Post-initialization to set up default configurations"""
        # Initialize tier_configs if empty
        if not self.tier_configs:
            self.tier_configs = self._get_default_tier_configs()

        # Convert dict tier_configs to TierResourceConfig objects if needed
        if self.tier_configs and isinstance(next(iter(self.tier_configs.values())), dict):
            converted_configs = {}
            for tier_name, tier_config in self.tier_configs.items():
                if isinstance(tier_config, dict):
                    # Ensure tier field is set
                    if "tier" not in tier_config:
                        tier_config["tier"] = tier_name
                    converted_configs[tier_name] = TierResourceConfig(**tier_config)
                else:
                    converted_configs[tier_name] = tier_config
            self.tier_configs = converted_configs

        # Initialize business_critical_models if it's a dict
        if isinstance(self.business_critical_models, dict):
            self.business_critical_models = BusinessCriticalModelsConfig(**self.business_critical_models)

        # Validate routing_strategy
        valid_strategies = ["least_loaded", "round_robin", "fastest_response", "model_affinity"]
        if self.routing_strategy not in valid_strategies:
            self.routing_strategy = "least_loaded"

    def _get_default_tier_configs(self) -> Dict[str, TierResourceConfig]:
        """Get default tier configurations"""
        return {
            "hot": TierResourceConfig(
                tier="hot",
                num_cpus=1.0,  # High CPU allocation
                num_gpus=0.0,  # Dedicated GPU
                memory=1288,  # 12GB RAM
                object_store_memory=8192,  # 8GB object store
                max_models_per_replica=5,  # Fewer models per replica for better performance
                min_replicas=1,  # Always keep replicas running
                max_replicas=20,  # High scaling capacity
                target_requests_per_replica=8,  # Lower target for faster response
                priority=100,
            ),
            "warm": TierResourceConfig(
                tier="warm",
                num_cpus=1.0,  # Medium CPU allocation
                num_gpus=0.0,  # Shared GPU
                memory=1192,  # 8GB RAM
                object_store_memory=4096,  # 4GB object store
                max_models_per_replica=15,  # Medium models per replica
                min_replicas=1,  # Keep some replicas running
                max_replicas=15,  # Medium scaling capacity
                target_requests_per_replica=12,  # Medium target
                priority=50,
            ),
            "cold": TierResourceConfig(
                tier="cold",
                num_cpus=1.0,  # Low CPU allocation
                num_gpus=0.0,  # No GPU
                memory=1096,  # 4GB RAM
                object_store_memory=2048,  # 2GB object store
                max_models_per_replica=30,  # Many models per replica
                min_replicas=1,  # Minimal replicas
                max_replicas=10,  # Limited scaling
                target_requests_per_replica=20,  # Higher target acceptable
                priority=10,
            ),
        }

    def get_tier_config(self, tier_name: str) -> Optional[TierResourceConfig]:
        """Get configuration for a specific tier"""
        return self.tier_configs.get(tier_name)

    def get_tier_priority(self, tier_name: str) -> int:
        """Get priority for a specific tier"""
        tier_config = self.get_tier_config(tier_name)
        return tier_config.priority if tier_config else 0

    def get_tier_routing_weight(self, tier_name: str) -> float:
        """Get routing weight for a specific tier"""
        return self.tier_routing_weights.get(tier_name, 0.0)

    def is_business_critical_model(self, model_name: str) -> bool:
        """Check if a model is marked as business critical"""
        # Check explicit assignments
        if model_name in self.business_critical_models.explicit_assignments:
            return True

        # Check pattern matches
        for pattern in self.business_critical_models.patterns:
            if self._pattern_matches(model_name, pattern):
                return True

        return False

    def get_business_critical_tier(self, model_name: str) -> Optional[str]:
        """Get tier assignment for business critical model"""
        # Check explicit assignments first
        if model_name in self.business_critical_models.explicit_assignments:
            return self.business_critical_models.explicit_assignments[model_name]

        # Check pattern matches
        for pattern, tier in self.business_critical_models.patterns.items():
            if self._pattern_matches(model_name, pattern):
                return tier

        # Return default if model is business critical but no specific assignment
        if self.is_business_critical_model(model_name):
            return self.business_critical_models.default_business_critical_tier

        return None

    def _pattern_matches(self, model_name: str, pattern: str) -> bool:
        """Check if model name matches pattern (simple wildcard support)"""
        if "*" not in pattern:
            return model_name == pattern

        # Simple wildcard matching
        import re
        regex_pattern = pattern.replace("*", ".*")
        return bool(re.match(f"^{regex_pattern}$", model_name))

    def validate_config(self) -> None:
        """Validate the tier-based deployment configuration"""
        if not self.tier_configs:
            raise ValueError("At least one tier configuration is required")

        # Validate each tier config
        for tier_name, tier_config in self.tier_configs.items():
            if tier_config.min_replicas > tier_config.max_replicas:
                raise ValueError(f"Tier {tier_name}: min_replicas cannot exceed max_replicas")

            if tier_config.num_cpus <= 0:
                raise ValueError(f"Tier {tier_name}: num_cpus must be positive")

            if tier_config.memory <= 0:
                raise ValueError(f"Tier {tier_name}: memory must be positive")

        # Validate time windows
        if self.promotion_time_window <= 0:
            raise ValueError("promotion_time_window must be positive")

        if self.demotion_time_window <= 0:
            raise ValueError("demotion_time_window must be positive")

        # Validate thresholds
        if self.promotion_threshold <= 0:
            raise ValueError("promotion_threshold must be positive")

        if self.demotion_threshold < 0:
            raise ValueError("demotion_threshold cannot be negative")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {}

        # Convert tier configs
        tier_configs_dict = {}
        for tier_name, tier_config in self.tier_configs.items():
            if hasattr(tier_config, '__dict__'):
                tier_configs_dict[tier_name] = tier_config.__dict__
            else:
                tier_configs_dict[tier_name] = tier_config

        result["tier_configs"] = tier_configs_dict
        result["enable_tier_based_deployment"] = self.enable_tier_based_deployment
        result["promotion_threshold"] = self.promotion_threshold
        result["demotion_threshold"] = self.demotion_threshold
        result["promotion_time_window"] = self.promotion_time_window
        result["demotion_time_window"] = self.demotion_time_window
        result["enable_tier_aware_routing"] = self.enable_tier_aware_routing
        result["prefer_higher_tier"] = self.prefer_higher_tier
        result["routing_strategy"] = self.routing_strategy
        result["tier_routing_weights"] = self.tier_routing_weights
        result["tier_monitoring_interval"] = self.tier_monitoring_interval
        result["capacity_check_interval"] = self.capacity_check_interval
        result["health_check_enabled"] = self.health_check_enabled

        # Convert business critical models
        if hasattr(self.business_critical_models, '__dict__'):
            result["business_critical_models"] = self.business_critical_models.__dict__
        else:
            result["business_critical_models"] = self.business_critical_models

        return result

    def __str__(self) -> str:
        """String representation"""
        tier_names = list(self.tier_configs.keys())
        return (
            f"TierBasedDeploymentConfig("
            f"enabled={self.enable_tier_based_deployment}, "
            f"tiers={tier_names}, "
            f"routing={self.routing_strategy})"
        )