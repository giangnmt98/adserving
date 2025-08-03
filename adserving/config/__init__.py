"""
Configuration package initialization
Avoiding circular imports by carefully ordering imports
"""

# Import base types first
from .base_types import ModelTier, ResourceSharingStrategy, RoutingStrategy

# Import deployment types
from .deployment_types import AutoscalingSettings, PooledResourceConfig

# Import core configs
from .core_configs import (
    AnomalyDetectionConfig,
    MLflowConfig,
    PooledDeploymentSettings,
    RayConfig,
    ResourceSharingConfig,
    RoutingConfig,
    TieredLoadingConfig,
)

# Import system configs
from .system_configs import (
    BatchProcessingConfig,
    ConnectionPoolingConfig,
    LoggingConfig,
    MonitoringConfig,
    PerformanceConfig,
    SecurityConfig,
)

# Import config manager (without direct TierBasedDeploymentConfig import)
from .config_manager import (
    Config,
    _global_config,
    create_sample_config,
    get_config,
    load_config,
    set_config,
)

# Export all for backward compatibility
__all__ = [
    # Base types
    "ModelTier",
    "ResourceSharingStrategy",
    "RoutingStrategy",

    # Deployment types
    "AutoscalingSettings",
    "PooledResourceConfig",

    # Core configs
    "AnomalyDetectionConfig",
    "MLflowConfig",
    "PooledDeploymentSettings",
    "RayConfig",
    "ResourceSharingConfig",
    "RoutingConfig",
    "TieredLoadingConfig",

    # System configs
    "BatchProcessingConfig",
    "ConnectionPoolingConfig",
    "LoggingConfig",
    "MonitoringConfig",
    "PerformanceConfig",
    "SecurityConfig",

    # Config manager
    "Config",
    "get_config",
    "load_config",
    "set_config",
    "create_sample_config",
]