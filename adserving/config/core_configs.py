"""
Core configuration classes
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .base_types import ResourceSharingStrategy, RoutingStrategy
from .deployment_types import AutoscalingSettings, PooledResourceConfig


@dataclass
class MLflowConfig:
    """MLflow configuration"""

    tracking_uri: str = "http://localhost:5000"
    registry_uri: Optional[str] = None
    experiment_name: str = "anomalydetectionserving"
    artifact_location: Optional[str] = None
    model_stage: str = "Production"
    enable_model_versioning: bool = True
    model_cache_ttl: int = 3600  # seconds


@dataclass
class RayConfig:
    """Ray cluster configuration for hundreds of models"""

    address: Optional[str] = None  # null for local mode
    runtime_env: Optional[Dict[str, Any]] = None
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8265
    object_store_memory: Optional[str] = "50GB"  # Increased for hundreds of models
    num_cpus: Optional[int] = None  # Auto-detect
    num_gpus: Optional[int] = None  # Auto-detect
    enable_gpu_sharing: bool = True
    gpu_memory_fraction: float = 0.8  # Reserve 80% GPU memory for models
    plasma_store_socket_name: Optional[str] = None
    raylet_socket_name: Optional[str] = None
    log_level: str = "ERROR"  # Ray logging level


@dataclass
class TieredLoadingConfig:
    """Configuration for tiered model loading strategy"""

    enable_tiered_loading: bool = True
    hot_cache_size: int = 50  # Always loaded models
    warm_cache_size: int = 200  # On-demand loaded models
    cold_cache_size: int = 500  # Rarely used models

    # Tier promotion/demotion thresholds
    hot_promotion_threshold: int = 20  # requests within time window
    warm_promotion_threshold: int = 5
    hot_promotion_time_window: int = 300  # 5 minutes
    warm_promotion_time_window: int = 3600  # 1 hour

    # Model warming settings
    enable_model_warming: bool = True
    warm_popular_models_count: int = 10
    warming_interval: int = 300  # 5 minutes

    # Cleanup settings
    cleanup_interval: int = 3600  # 1 hour
    cold_model_ttl: int = 7200  # 2 hours


@dataclass
class ResourceSharingConfig:
    """Configuration for resource sharing optimization"""

    strategy: ResourceSharingStrategy = ResourceSharingStrategy.GPU_SHARED
    enable_memory_mapping: bool = True
    enable_model_weight_sharing: bool = True
    shared_memory_size: str = "10GB"

    # GPU sharing settings
    max_models_per_gpu: int = 5
    gpu_memory_reserve: float = 0.2  # Reserve 20% for system
    enable_dynamic_gpu_allocation: bool = True

    # CPU sharing settings
    cpu_oversubscription_factor: float = 2.0  # Allow 2x CPU oversubscription
    enable_cpu_affinity: bool = True

    # Memory optimization
    enable_model_compression: bool = True
    compression_ratio: float = 0.7  # Target 70% of original size
    enable_lazy_loading: bool = True


@dataclass
class PooledDeploymentSettings:
    """Settings for pooled deployments"""

    default_pool_count: int = 2  # Number of default pools
    models_per_pool: int = 50
    enable_cross_pool_balancing: bool = True
    pool_rebalancing_interval: int = 600  # 10 minutes

    # Resource allocation per pool
    pool_resource_config: PooledResourceConfig = field(
        default_factory=lambda: PooledResourceConfig(
            num_cpus=8.0,  # Increased for multiple models
            num_gpus=1.0,
            memory=16384,  # 16GB per pool
            object_store_memory=8192,  # 8GB
            max_models_per_replica=20,
        )
    )

    # Autoscaling
    autoscaling_config: AutoscalingSettings = field(
        default_factory=lambda: AutoscalingSettings(
            min_replicas=3,
            max_replicas=50,  # Increased for hundreds of models
            target_num_ongoing_requests_per_replica=15,
            metrics_interval_s=5.0,
            look_back_period_s=30.0,
            smoothing_factor=0.8,
        )
    )


@dataclass
class RoutingConfig:
    """Configuration for intelligent routing"""

    strategy: RoutingStrategy = getattr(RoutingStrategy, "LEAST_LOADED")
    enable_request_queuing: bool = True
    max_queue_size: int = 50000  # Increased for hundreds of models
    queue_timeout: int = 30  # seconds

    # Load balancing settings
    enable_sticky_routing: bool = True
    routing_cache_ttl: int = 300  # 5 minutes
    max_retries: int = 3
    retry_backoff: float = 0.1  # seconds

    # Performance optimization
    enable_request_batching: bool = True
    batch_size: int = 32
    batch_timeout_ms: int = 50
    enable_async_processing: bool = True


@dataclass
class AnomalyDetectionConfig:
    """Anomaly detection configuration"""

    # Default threshold for probability-based models
    default_probability_threshold: float = 0.5

    # Default threshold for label-based models (usually 0 since labels are binary)
    default_label_threshold: float = 0.0

    # Model-specific thresholds (model_name -> threshold)
    model_specific_thresholds: Dict[str, float] = field(default_factory=dict)

    # Model output type configuration (model_name -> output_type)
    # output_type can be "probability" or "label"
    model_output_types: Dict[str, str] = field(default_factory=dict)

    # Enable automatic threshold detection from model metadata
    enable_metadata_threshold: bool = True

    # Fallback behavior when model metadata is not available
    fallback_to_config: bool = True
