"""
Main configuration module - Backward compatibility wrapper

This file maintains backward compatibility by importing all configuration
classes and functions from the modular structure.
"""

# Import everything from the new modular structure
from . import *
from .base_types import ModelTier, ResourceSharingStrategy, RoutingStrategy

# Maintain backward compatibility
from .config_manager import (
    Config,
    _global_config,
    create_sample_config,
    get_config,
    load_config,
    set_config,
)
from .core_configs import (
    AnomalyDetectionConfig,
    MLflowConfig,
    PooledDeploymentSettings,
    RayConfig,
    ResourceSharingConfig,
    RoutingConfig,
    TieredLoadingConfig,
)
from .deployment_types import AutoscalingSettings, PooledResourceConfig
from .system_configs import (
    BatchProcessingConfig,
    ConnectionPoolingConfig,
    LoggingConfig,
    MonitoringConfig,
    PerformanceConfig,
    SecurityConfig,
)

# Module docstring for documentation
__doc__ = """
Configuration for Anomaly Detection Serve

This module provides improved configuration management for serving hundreds of models
with scaling, resource sharing, and tiered loading configurations.

Key improvements:
- Scaling limits for hundreds of models
- Resource pooling and sharing configurations
- Tiered model loading settings
- Advanced monitoring and optimization settings
- GPU sharing and memory management

The configuration system is now modular with the following structure:
- base_types.py: Enums and basic types
- deployment_types.py: Deployment-related classes
- core_configs.py: Core configuration dataclasses
- system_configs.py: System configuration dataclasses
- config_manager.py: Main Config class and utilities
"""
