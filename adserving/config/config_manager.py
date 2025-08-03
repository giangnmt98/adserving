"""
Configuration management and loading utilities
Fixed circular import issues with lazy loading approach
"""

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union, TYPE_CHECKING

import yaml

from .base_types import ModelTier, ResourceSharingStrategy, RoutingStrategy
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

# LAZY TYPES để tránh circular import
if TYPE_CHECKING:
    from ..deployment.resource_config import TierBasedDeploymentConfig


@dataclass
class Config:
    """Configuration for Anomaly Detection Serve"""

    # Core configurations
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    ray: RayConfig = field(default_factory=RayConfig)

    # Features
    tiered_loading: TieredLoadingConfig = field(default_factory=TieredLoadingConfig)

    # LAZY LOADING cho TierBasedDeploymentConfig để tránh circular import
    tier_based_deployment: Optional[Any] = field(default=None)

    resource_sharing: ResourceSharingConfig = field(
        default_factory=ResourceSharingConfig
    )
    pooled_deployment: PooledDeploymentSettings = field(
        default_factory=PooledDeploymentSettings
    )
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    anomaly_detection: AnomalyDetectionConfig = field(
        default_factory=AnomalyDetectionConfig
    )

    # Performance configurations
    batch_processing: BatchProcessingConfig = field(
        default_factory=BatchProcessingConfig
    )
    connection_pooling: ConnectionPoolingConfig = field(
        default_factory=ConnectionPoolingConfig
    )

    # System configurations
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    # System settings
    max_workers: int = 16  # Increased for hundreds of models
    enable_auto_deployment: bool = True
    deployment_timeout: int = 600  # 10 minutes for large deployments

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v2"  # Updated version
    api_version: str = "v2.1.0"
    enable_docs: bool = True
    enable_openapi: bool = True

    def __post_init__(self) -> None:
        """Post-initialization validation and setup"""
        # Initialize tier_based_deployment if None
        if self.tier_based_deployment is None:
            self.tier_based_deployment = self._create_default_tier_config()

        self._load_from_environment()
        self._validate_config()

    def _create_default_tier_config(self) -> Any:
        """Create default tier-based deployment config with late import"""
        try:
            from ..deployment.resource_config import TierBasedDeploymentConfig
            return TierBasedDeploymentConfig()
        except ImportError as e:
            print(f"Warning: Could not import TierBasedDeploymentConfig: {e}")
            # Return a mock object với basic attributes
            return self._create_mock_tier_config()

    def _create_mock_tier_config(self) -> Any:
        """Create mock tier config object nếu import fails"""
        class MockTierBasedDeploymentConfig:
            def __init__(self):
                self.enable_tier_based_deployment = True
                self.promotion_threshold = 20
                self.demotion_threshold = 2
                self.promotion_time_window = 300
                self.demotion_time_window = 1800
                self.enable_tier_aware_routing = True
                self.prefer_higher_tier = True
                self.routing_strategy = "least_loaded"
                self.tier_routing_weights = {"hot": 1.0, "warm": 0.7, "cold": 0.3}
                self.tier_monitoring_interval = 30
                self.capacity_check_interval = 300
                self.health_check_enabled = True
                self.business_critical_models = None
                self.tier_configs = {}

        return MockTierBasedDeploymentConfig()

    def _load_from_environment(self) -> None:
        """Load configuration from environment variables"""
        # MLflow settings
        if os.getenv("MLFLOW_TRACKING_URI"):
            self.mlflow.tracking_uri = os.getenv(
                "MLFLOW_TRACKING_URI", self.mlflow.tracking_uri
            )

        # Ray settings
        if os.getenv("RAY_ADDRESS"):
            self.ray.address = os.getenv("RAY_ADDRESS")

        # Security settings
        if os.getenv("API_KEY"):
            self.security.api_key = os.getenv("API_KEY")
            self.security.enable_auth = True

        # Performance settings
        if os.getenv("MAX_WORKERS"):
            try:
                self.max_workers = int(os.getenv("MAX_WORKERS", str(self.max_workers)))
            except ValueError:
                pass  # Keep default value if conversion fails

    def _validate_config(self) -> None:
        """Validate configuration settings"""
        # Validate cache sizes
        if self.tiered_loading.hot_cache_size <= 0:
            raise ValueError("Hot cache size must be positive")

        if self.tiered_loading.warm_cache_size <= 0:
            raise ValueError("Warm cache size must be positive")

        # Validate resource settings
        if self.pooled_deployment.pool_resource_config.num_cpus <= 0:
            raise ValueError("CPU count must be positive")

        # Validate autoscaling settings
        autoscaling = self.pooled_deployment.autoscaling_config
        if autoscaling.min_replicas > autoscaling.max_replicas:
            raise ValueError("Min replicas cannot exceed max replicas")

        # Validate monitoring thresholds
        for threshold_name, value in self.monitoring.alert_thresholds.items():
            if not 0 <= value <= 100 and "percent" in threshold_name:
                raise ValueError(
                    f"Percentage threshold {threshold_name} must be between 0 and 100"
                )

    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "Config":
        """Load configuration from file"""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            if config_path.suffix.lower() in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            elif config_path.suffix.lower() == ".json":
                data = json.load(f)
            else:
                raise ValueError(
                    f"Unsupported configuration file format: {config_path.suffix}"
                )

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary"""
        config_data = {}

        for key, value in data.items():
            if not isinstance(value, dict):
                if key not in ["model_cache_size", "model_update_interval"]:
                    config_data[key] = value
                continue

            config_handler = {
                "mlflow": cls._handle_mlflow_config,
                "ray": cls._handle_ray_config,
                "tiered_loading": cls._handle_tiered_loading_config,
                "tier_based_deployment": cls._handle_tier_based_deployment_config,
                "resource_sharing": cls._handle_resource_sharing_config,
                "pooled_deployment": cls._handle_pooled_deployment_config,
                "deployment": cls._handle_legacy_deployment_config,
                "routing": cls._handle_routing_config,
                "monitoring": cls._handle_monitoring_config,
                "logging": cls._handle_logging_config,
                "security": cls._handle_security_config,
                "performance": cls._handle_performance_config,
                "anomaly_detection": cls._handle_anomaly_detection_config,
                "batch_processing": cls._handle_batch_processing_config,
                "connection_pooling": cls._handle_connection_pooling_config,
            }

            if key in config_handler:
                config_data.update(config_handler[key](value))

        return cls(**config_data)

    @classmethod
    def _handle_mlflow_config(cls, value: Dict) -> Dict:
        return {"mlflow": MLflowConfig(**value)}

    @classmethod
    def _handle_ray_config(cls, value: Dict) -> Dict:
        return {"ray": RayConfig(**value)}

    @classmethod
    def _handle_tiered_loading_config(cls, value: Dict) -> Dict:
        return {"tiered_loading": TieredLoadingConfig(**value)}

    @classmethod
    def _handle_tier_based_deployment_config(cls, value: Dict) -> Dict:
        """Handle tier-based deployment configuration with lazy import và error handling"""
        try:
            # LAZY IMPORT inside method
            from ..deployment.resource_config import (
                TierBasedDeploymentConfig,
                TierResourceConfig,
                BusinessCriticalModelsConfig
            )

            # Deep copy để tránh modify original dict
            import copy
            config_data = copy.deepcopy(value)

            # Process tier_configs if present
            if "tier_configs" in config_data:
                tier_configs = {}
                for tier_name, tier_config in config_data["tier_configs"].items():
                    if isinstance(tier_config, dict):
                        # Ensure tier field is set
                        if "tier" not in tier_config:
                            tier_config["tier"] = tier_name

                        try:
                            tier_configs[tier_name] = TierResourceConfig(**tier_config)
                        except Exception as e:
                            print(f"Warning: Error creating TierResourceConfig for {tier_name}: {e}")
                            # Keep original dict nếu TierResourceConfig fails
                            tier_configs[tier_name] = tier_config
                    else:
                        tier_configs[tier_name] = tier_config

                config_data["tier_configs"] = tier_configs

            # Process business_critical_models if present
            if "business_critical_models" in config_data and isinstance(config_data["business_critical_models"], dict):
                try:
                    config_data["business_critical_models"] = BusinessCriticalModelsConfig(**config_data["business_critical_models"])
                except Exception as e:
                    print(f"Warning: Error creating BusinessCriticalModelsConfig: {e}")
                    # Keep original dict nếu BusinessCriticalModelsConfig fails

            # Create TierBasedDeploymentConfig
            try:
                tier_config = TierBasedDeploymentConfig(**config_data)
                return {"tier_based_deployment": tier_config}
            except Exception as e:
                print(f"Warning: Error creating TierBasedDeploymentConfig: {e}")
                # Return mock object nếu creation fails
                return {"tier_based_deployment": cls._create_mock_tier_config_from_dict(config_data)}

        except ImportError as e:
            print(f"Warning: Could not import TierBasedDeploymentConfig: {e}")
            return {"tier_based_deployment": cls._create_mock_tier_config_from_dict(value)}

    @classmethod
    def _create_mock_tier_config_from_dict(cls, config_data: Dict) -> Any:
        """Create mock tier config từ dictionary data"""
        class MockTierBasedDeploymentConfig:
            def __init__(self, data: Dict):
                # Set attributes từ config data
                self.enable_tier_based_deployment = data.get("enable_tier_based_deployment", True)
                self.promotion_threshold = data.get("promotion_threshold", 20)
                self.demotion_threshold = data.get("demotion_threshold", 2)
                self.promotion_time_window = data.get("promotion_time_window", 300)
                self.demotion_time_window = data.get("demotion_time_window", 1800)
                self.enable_tier_aware_routing = data.get("enable_tier_aware_routing", True)
                self.prefer_higher_tier = data.get("prefer_higher_tier", True)
                self.routing_strategy = data.get("routing_strategy", "least_loaded")
                self.tier_routing_weights = data.get("tier_routing_weights", {"hot": 1.0, "warm": 0.7, "cold": 0.3})
                self.tier_monitoring_interval = data.get("tier_monitoring_interval", 30)
                self.capacity_check_interval = data.get("capacity_check_interval", 300)
                self.health_check_enabled = data.get("health_check_enabled", True)
                self.business_critical_models = data.get("business_critical_models", {})
                self.tier_configs = data.get("tier_configs", {})

        return MockTierBasedDeploymentConfig(config_data)

    @classmethod
    def _handle_resource_sharing_config(cls, value: Dict) -> Dict:
        if "strategy" in value and isinstance(value["strategy"], str):
            try:
                value["strategy"] = ResourceSharingStrategy(value["strategy"])
            except ValueError:
                value["strategy"] = ResourceSharingStrategy.GPU_SHARED
        return {"resource_sharing": ResourceSharingConfig(**value)}

    @classmethod
    def _handle_pooled_deployment_config(cls, value: Dict) -> Dict:
        if "pool_resource_config" in value:
            value["pool_resource_config"] = PooledResourceConfig(
                **value["pool_resource_config"]
            )
        if "autoscaling_config" in value:
            value["autoscaling_config"] = AutoscalingSettings(
                **value["autoscaling_config"]
            )
        return {"pooled_deployment": PooledDeploymentSettings(**value)}

    @classmethod
    def _handle_legacy_deployment_config(cls, value: Dict) -> Dict:
        pooled_config = {}

        if "resource_config" in value:
            resource_config = value["resource_config"]
            pooled_config["pool_resource_config"] = PooledResourceConfig(
                num_cpus=resource_config.get("num_cpus", 1.0),
                num_gpus=resource_config.get("num_gpus", 0.0),
                memory=resource_config.get("memory", 1024),
                object_store_memory=resource_config.get("object_store_memory", 512),
            )

        if "autoscaling" in value:
            autoscaling = value["autoscaling"]
            pooled_config["autoscaling_config"] = AutoscalingSettings(
                min_replicas=autoscaling.get("min_replicas", 2),
                max_replicas=autoscaling.get("max_replicas", 50),
                target_num_ongoing_requests_per_replica=autoscaling.get(
                    "target_num_ongoing_requests_per_replica", 2
                ),
                metrics_interval_s=autoscaling.get("metrics_interval_s", 10.0),
                look_back_period_s=autoscaling.get("look_back_period_s", 30.0),
                smoothing_factor=autoscaling.get("smoothing_factor", 1.0),
            )

        return {"pooled_deployment": PooledDeploymentSettings(**pooled_config)}

    @classmethod
    def _handle_routing_config(cls, value: Dict) -> Dict:
        if "strategy" in value and isinstance(value["strategy"], str):
            try:
                strategy_value = value["strategy"]
                if hasattr(RoutingStrategy, strategy_value.upper()):
                    value["strategy"] = getattr(RoutingStrategy, strategy_value.upper())
                else:
                    value["strategy"] = RoutingStrategy.LEAST_LOADED
            except (ValueError, AttributeError):
                value["strategy"] = RoutingStrategy.LEAST_LOADED
        return {"routing": RoutingConfig(**value)}

    @classmethod
    def _handle_monitoring_config(cls, value: Dict) -> Dict:
        return {"monitoring": MonitoringConfig(**value)}

    @classmethod
    def _handle_logging_config(cls, value: Dict) -> Dict:
        return {"logging": LoggingConfig(**value)}

    @classmethod
    def _handle_security_config(cls, value: Dict) -> Dict:
        return {"security": SecurityConfig(**value)}

    @classmethod
    def _handle_performance_config(cls, value: Dict) -> Dict:
        return {"performance": PerformanceConfig(**value)}

    @classmethod
    def _handle_anomaly_detection_config(cls, value: Dict) -> Dict:
        return {"anomaly_detection": AnomalyDetectionConfig(**value)}

    @classmethod
    def _handle_batch_processing_config(cls, value: Dict) -> Dict:
        return {"batch_processing": BatchProcessingConfig(**value)}

    @classmethod
    def _handle_connection_pooling_config(cls, value: Dict) -> Dict:
        return {"connection_pooling": ConnectionPoolingConfig(**value)}

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self)

    def to_file(self, config_path: Union[str, Path], format: str = "yaml") -> None:
        """Save configuration to file"""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict()

        with open(config_path, "w", encoding="utf-8") as f:
            if format.lower() in ["yaml", "yml"]:
                yaml.dump(data, f, default_flow_style=False, indent=2)
            elif format.lower() == "json":
                json.dump(data, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported format: {format}")

    def get_model_tier_config(self, model_name: str) -> str:
        """Get tier configuration for a specific model"""
        # This could be extended to load from a model tier configuration file
        # For now, return default tier
        return ModelTier.WARM

    def get_ray_init_config(self) -> Dict[str, Any]:
        """Get Ray initialization configuration"""
        config: Dict[str, Any] = {
            "address": self.ray.address,
            "dashboard_host": self.ray.dashboard_host,
            "dashboard_port": self.ray.dashboard_port,
        }

        if self.ray.object_store_memory:
            config["object_store_memory"] = self.ray.object_store_memory

        if self.ray.num_cpus is not None:
            config["num_cpus"] = self.ray.num_cpus

        if self.ray.num_gpus is not None:
            config["num_gpus"] = self.ray.num_gpus

        if self.ray.runtime_env:
            config["runtime_env"] = self.ray.runtime_env

        return config

    def has_tier_based_deployment(self) -> bool:
        """Check if tier-based deployment is enabled and properly configured"""
        if self.tier_based_deployment is None:
            return False

        # Check if it's mock object hoặc real object
        return hasattr(self.tier_based_deployment, 'enable_tier_based_deployment') and \
               getattr(self.tier_based_deployment, 'enable_tier_based_deployment', False)

    def get_tier_based_deployment_config(self) -> Any:
        """Get tier-based deployment config với fallback"""
        if self.tier_based_deployment is None:
            self.tier_based_deployment = self._create_default_tier_config()
        return self.tier_based_deployment

    def __str__(self) -> str:
        """String representation of configuration"""
        tier_enabled = self.has_tier_based_deployment()
        return (
            f"Config(tiered_loading={self.tiered_loading.enable_tiered_loading}, "
            f"tier_based_deployment={tier_enabled}, "
            f"resource_sharing={self.resource_sharing.strategy.value}, "
            f"pools={self.pooled_deployment.default_pool_count})"
        )

    def __repr__(self) -> str:
        """Detailed representation of configuration"""
        return self.__str__()


# Global configuration instance
_global_config: Optional[Config] = None


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Load configuration from file or create default"""
    if config_path:
        return Config.from_file(config_path)

    # Try to load from default locations
    default_paths = ["config_v2.yaml", "config_v2.yml", "config.yaml", "config.yml"]

    for path in default_paths:
        if Path(path).exists():
            return Config.from_file(path)

    # Return default configuration
    return Config()


def create_sample_config(output_path: Union[str, Path] = "config.yaml") -> None:
    """Create a sample configuration file"""
    config = Config()
    config.to_file(output_path, "yaml")
    print(f"Sample configuration created at: {output_path}")


def get_config() -> Config:
    """Get global configuration instance"""
    global _global_config
    if _global_config is None:
        _global_config = load_config()
    return _global_config


def set_config(config: Config) -> None:
    """Set global configuration instance"""
    global _global_config
    _global_config = config