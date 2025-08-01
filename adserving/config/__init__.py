"""
Configuration module for Anomaly Detection Serve

This module provides a modular configuration system with backward compatibility.
All classes and functions are imported to maintain existing API.
"""

# Import base types
from .base_types import ModelTier, ResourceSharingStrategy, RoutingStrategy

# Import main config and utilities
from .config_manager import (
    Config,
    create_sample_config,
    get_config,
    load_config,
    set_config,
)

# Import core configurations
from .core_configs import (
    AnomalyDetectionConfig,
    MLflowConfig,
    PooledDeploymentSettings,
    RayConfig,
    ResourceSharingConfig,
    RoutingConfig,
    TieredLoadingConfig,
)

# Import deployment types
from .deployment_types import AutoscalingSettings, PooledResourceConfig

# Import system configurations
from .system_configs import (
    BatchProcessingConfig,
    ConnectionPoolingConfig,
    LoggingConfig,
    MonitoringConfig,
    PerformanceConfig,
    SecurityConfig,
)

# Export all for backward compatibility
__all__ = [
    # Base types
    "ResourceSharingStrategy",
    "ModelTier",
    "RoutingStrategy",
    # Deployment types
    "PooledResourceConfig",
    "AutoscalingSettings",
    # Core configurations
    "MLflowConfig",
    "RayConfig",
    "TieredLoadingConfig",
    "ResourceSharingConfig",
    "PooledDeploymentSettings",
    "RoutingConfig",
    "AnomalyDetectionConfig",
    # System configurations
    "MonitoringConfig",
    "LoggingConfig",
    "SecurityConfig",
    "PerformanceConfig",
    "BatchProcessingConfig",
    "ConnectionPoolingConfig",
    # Main config and utilities
    "Config",
    "load_config",
    "create_sample_config",
    "get_config",
    "set_config",
]
