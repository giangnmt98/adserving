"""
Flexible Configuration Management for Ultra-Scale Deployments
Provides comprehensive configuration management for different deployment scenarios
"""

import os
import yaml
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import logging

from ..utils.logger import get_logger

logger = get_logger()


class DeploymentMode(Enum):
    """Deployment operation modes"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class ScalingMode(Enum):
    """Auto-scaling modes"""
    FIXED = "fixed"
    AUTO = "auto"
    MANUAL = "manual"


@dataclass
class ResourceLimits:
    """Resource limit configuration"""
    max_cpu_cores: int = 16
    max_memory_mb: int = 16000
    max_disk_gb: int = 100
    max_gpu_count: int = 0
    cpu_limit_per_model: float = 0.5
    memory_limit_per_model_mb: int = 500


@dataclass
class ModelTierConfig:
    """Configuration for model storage tiers"""
    hot_tier_max_models: int = 500
    hot_tier_memory_mb: int = 8000
    warm_tier_max_models: int = 1000
    warm_tier_disk_gb: int = 20
    cold_tier_max_models: int = 10000
    promotion_threshold: float = 0.8
    demotion_threshold: float = 0.3


@dataclass
class LoadBalancingConfig:
    """Load balancing configuration"""
    strategy: str = "health_aware"  # round_robin, least_connections, weighted_round_robin, etc.
    health_check_interval: int = 30
    unhealthy_threshold: float = 0.5
    sticky_sessions: bool = False
    connection_timeout: int = 30
    retry_attempts: int = 3


@dataclass
class BlueGreenConfig:
    """Blue-green deployment configuration"""
    enabled: bool = True
    traffic_switch_steps: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"target_weight": 0.1, "duration": 30},
        {"target_weight": 0.5, "duration": 60},
        {"target_weight": 1.0, "duration": 0}
    ])
    health_check_timeout: int = 120
    rollback_on_failure: bool = True
    canary_analysis_duration: int = 300
    success_rate_threshold: float = 0.95


@dataclass
class HealthCheckConfig:
    """Health monitoring configuration"""
    enabled: bool = True
    check_interval: int = 30
    degraded_threshold: float = 0.8
    unhealthy_threshold: float = 0.5
    critical_threshold: float = 0.2
    system_metrics_interval: int = 10
    custom_checks: List[str] = field(default_factory=list)


@dataclass
class ServiceDiscoveryConfig:
    """Service discovery configuration"""
    enabled: bool = True
    heartbeat_interval: int = 30
    stale_threshold: int = 120
    discovery_port: int = 8500
    registration_timeout: int = 30
    service_tags: List[str] = field(default_factory=list)


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration"""
    enabled: bool = True
    metrics_retention_hours: int = 24
    alert_retention_hours: int = 168
    prometheus_enabled: bool = True
    prometheus_port: int = 9090
    dashboard_enabled: bool = True
    custom_collectors: List[str] = field(default_factory=list)
    alert_channels: List[str] = field(default_factory=list)


@dataclass
class RayConfig:
    """Ray cluster configuration"""
    enabled: bool = True
    num_cpus: Optional[int] = None
    num_gpus: Optional[int] = None
    object_store_memory_gb: int = 8
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8265
    log_level: str = "INFO"
    runtime_env: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MLflowConfig:
    """MLflow integration configuration"""
    enabled: bool = True
    tracking_uri: str = "http://localhost:5000"
    model_registry_uri: Optional[str] = None
    artifact_root: Optional[str] = None
    experiment_name: str = "ultra_scale_deployment"
    auto_log: bool = True


@dataclass
class SecurityConfig:
    """Security configuration"""
    enabled: bool = True
    api_key_required: bool = False
    api_keys: List[str] = field(default_factory=list)
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    rate_limiting: bool = True
    max_requests_per_minute: int = 1000


@dataclass
class DeploymentConfig:
    """Comprehensive deployment configuration"""
    
    # Basic settings
    mode: DeploymentMode = DeploymentMode.DEVELOPMENT
    environment_name: str = "default"
    version: str = "1.0.0"
    target_model_count: int = 2000
    
    # Scaling configuration
    scaling_mode: ScalingMode = ScalingMode.AUTO
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_utilization: float = 0.7
    
    # Component configurations
    resources: ResourceLimits = field(default_factory=ResourceLimits)
    model_tiers: ModelTierConfig = field(default_factory=ModelTierConfig)
    load_balancing: LoadBalancingConfig = field(default_factory=LoadBalancingConfig)
    blue_green: BlueGreenConfig = field(default_factory=BlueGreenConfig)
    health_checks: HealthCheckConfig = field(default_factory=HealthCheckConfig)
    service_discovery: ServiceDiscoveryConfig = field(default_factory=ServiceDiscoveryConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    ray: RayConfig = field(default_factory=RayConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Advanced settings
    batch_size: int = 200
    parallel_workers: int = 20
    cache_directory: str = "ultra_deployment_cache"
    log_level: str = "INFO"
    debug_mode: bool = False
    
    # Environment-specific overrides
    environment_overrides: Dict[str, Any] = field(default_factory=dict)
    
    # Custom configuration
    custom_settings: Dict[str, Any] = field(default_factory=dict)


class ConfigurationManager:
    """Manages deployment configurations across different environments"""
    
    def __init__(self, config_dir: str = "config", default_config_file: str = "deployment.yaml"):
        self.config_dir = Path(config_dir)
        self.default_config_file = default_config_file
        self.config_cache = {}
        self.environment_configs = {}
        
        # Ensure config directory exists
        self.config_dir.mkdir(exist_ok=True)
        
        logger.info("ConfigurationManager initialized")
        logger.info(f"   - Config directory: {self.config_dir}")
        logger.info(f"   - Default config file: {default_config_file}")
    
    def create_default_config(self, config_path: Optional[str] = None) -> DeploymentConfig:
        """Create a default configuration"""
        config = DeploymentConfig()
        
        if config_path:
            self.save_config(config, config_path)
            logger.info(f"Default configuration saved to: {config_path}")
        
        return config
    
    def load_config(self, config_path: str, environment: Optional[str] = None) -> DeploymentConfig:
        """Load configuration from file"""
        try:
            config_file = self.config_dir / config_path
            
            if not config_file.exists():
                logger.warning(f"Config file not found: {config_file}")
                return self.create_default_config(str(config_file))
            
            # Load base configuration
            with open(config_file, 'r') as f:
                if config_file.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                elif config_file.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_file.suffix}")
            
            # Create configuration object
            config = self._dict_to_config(config_data)
            
            # Apply environment-specific overrides
            if environment:
                config = self._apply_environment_overrides(config, environment)
            
            # Apply environment variable overrides
            config = self._apply_env_var_overrides(config)
            
            # Cache configuration
            cache_key = f"{config_path}:{environment or 'default'}"
            self.config_cache[cache_key] = config
            
            logger.info(f"Configuration loaded: {config_path} (env: {environment or 'default'})")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            logger.info("Using default configuration")
            return DeploymentConfig()
    
    def save_config(self, config: DeploymentConfig, config_path: str, format: str = "yaml"):
        """Save configuration to file"""
        try:
            config_file = self.config_dir / config_path
            
            # Ensure parent directory exists
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert config to dictionary
            config_dict = asdict(config)
            
            # Save configuration
            with open(config_file, 'w') as f:
                if format.lower() in ['yaml', 'yml']:
                    yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
                elif format.lower() == 'json':
                    json.dump(config_dict, f, indent=2)
                else:
                    raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Configuration saved: {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_path}: {e}")
            raise
    
    def _dict_to_config(self, config_data: Dict[str, Any]) -> DeploymentConfig:
        """Convert dictionary to DeploymentConfig object"""
        try:
            # Handle nested configurations
            if 'resources' in config_data and isinstance(config_data['resources'], dict):
                config_data['resources'] = ResourceLimits(**config_data['resources'])
            
            if 'model_tiers' in config_data and isinstance(config_data['model_tiers'], dict):
                config_data['model_tiers'] = ModelTierConfig(**config_data['model_tiers'])
            
            if 'load_balancing' in config_data and isinstance(config_data['load_balancing'], dict):
                config_data['load_balancing'] = LoadBalancingConfig(**config_data['load_balancing'])
            
            if 'blue_green' in config_data and isinstance(config_data['blue_green'], dict):
                config_data['blue_green'] = BlueGreenConfig(**config_data['blue_green'])
            
            if 'health_checks' in config_data and isinstance(config_data['health_checks'], dict):
                config_data['health_checks'] = HealthCheckConfig(**config_data['health_checks'])
            
            if 'service_discovery' in config_data and isinstance(config_data['service_discovery'], dict):
                config_data['service_discovery'] = ServiceDiscoveryConfig(**config_data['service_discovery'])
            
            if 'monitoring' in config_data and isinstance(config_data['monitoring'], dict):
                config_data['monitoring'] = MonitoringConfig(**config_data['monitoring'])
            
            if 'ray' in config_data and isinstance(config_data['ray'], dict):
                config_data['ray'] = RayConfig(**config_data['ray'])
            
            if 'mlflow' in config_data and isinstance(config_data['mlflow'], dict):
                config_data['mlflow'] = MLflowConfig(**config_data['mlflow'])
            
            if 'security' in config_data and isinstance(config_data['security'], dict):
                config_data['security'] = SecurityConfig(**config_data['security'])
            
            # Convert string enums
            if 'mode' in config_data and isinstance(config_data['mode'], str):
                config_data['mode'] = DeploymentMode(config_data['mode'])
            
            if 'scaling_mode' in config_data and isinstance(config_data['scaling_mode'], str):
                config_data['scaling_mode'] = ScalingMode(config_data['scaling_mode'])
            
            return DeploymentConfig(**config_data)
            
        except Exception as e:
            logger.error(f"Failed to convert dict to config: {e}")
            return DeploymentConfig()
    
    def _apply_environment_overrides(self, config: DeploymentConfig, environment: str) -> DeploymentConfig:
        """Apply environment-specific configuration overrides"""
        try:
            if environment in config.environment_overrides:
                overrides = config.environment_overrides[environment]
                
                # Apply overrides to config
                for key, value in overrides.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
                        logger.debug(f"   Applied environment override: {key} = {value}")
            
            # Update environment name
            config.environment_name = environment
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to apply environment overrides: {e}")
            return config
    
    def _apply_env_var_overrides(self, config: DeploymentConfig) -> DeploymentConfig:
        """Apply environment variable overrides"""
        try:
            # Define environment variable mappings
            env_mappings = {
                'ULTRA_DEPLOY_MODE': ('mode', lambda x: DeploymentMode(x.lower())),
                'ULTRA_DEPLOY_REPLICAS_MIN': ('min_replicas', int),
                'ULTRA_DEPLOY_REPLICAS_MAX': ('max_replicas', int),
                'ULTRA_DEPLOY_TARGET_MODELS': ('target_model_count', int),
                'ULTRA_DEPLOY_BATCH_SIZE': ('batch_size', int),
                'ULTRA_DEPLOY_PARALLEL_WORKERS': ('parallel_workers', int),
                'ULTRA_DEPLOY_LOG_LEVEL': ('log_level', str),
                'ULTRA_DEPLOY_DEBUG': ('debug_mode', lambda x: x.lower() in ['true', '1', 'yes']),
                
                # Component-specific overrides
                'ULTRA_DEPLOY_MLFLOW_URI': ('mlflow.tracking_uri', str),
                'ULTRA_DEPLOY_RAY_CPUS': ('ray.num_cpus', int),
                'ULTRA_DEPLOY_RAY_GPUS': ('ray.num_gpus', int),
                'ULTRA_DEPLOY_MONITORING_ENABLED': ('monitoring.enabled', lambda x: x.lower() in ['true', '1', 'yes']),
                'ULTRA_DEPLOY_BLUE_GREEN_ENABLED': ('blue_green.enabled', lambda x: x.lower() in ['true', '1', 'yes']),
            }
            
            for env_var, (config_path, converter) in env_mappings.items():
                if env_var in os.environ:
                    try:
                        value = converter(os.environ[env_var])
                        self._set_nested_config_value(config, config_path, value)
                        logger.debug(f"   Applied env var override: {env_var} -> {config_path} = {value}")
                    except Exception as e:
                        logger.warning(f"Failed to apply env var override {env_var}: {e}")
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to apply environment variable overrides: {e}")
            return config
    
    def _set_nested_config_value(self, config: DeploymentConfig, path: str, value: Any):
        """Set nested configuration value using dot notation"""
        parts = path.split('.')
        current = config
        
        for part in parts[:-1]:
            current = getattr(current, part)
        
        setattr(current, parts[-1], value)
    
    def create_environment_config(self, base_config_path: str, environment: str, 
                                overrides: Dict[str, Any]) -> str:
        """Create environment-specific configuration file"""
        try:
            # Load base configuration
            base_config = self.load_config(base_config_path)
            
            # Apply overrides
            for key, value in overrides.items():
                if hasattr(base_config, key):
                    setattr(base_config, key, value)
            
            # Update environment name
            base_config.environment_name = environment
            
            # Save environment-specific config
            env_config_path = f"{environment}_{base_config_path}"
            self.save_config(base_config, env_config_path)
            
            logger.info(f"Created environment config: {env_config_path}")
            return env_config_path
            
        except Exception as e:
            logger.error(f"Failed to create environment config: {e}")
            raise
    
    def validate_config(self, config: DeploymentConfig) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        try:
            # Validate resource limits
            if config.resources.max_memory_mb < 1000:
                issues.append("Maximum memory too low (minimum: 1000MB)")
            
            if config.resources.max_cpu_cores < 1:
                issues.append("Maximum CPU cores too low (minimum: 1)")
            
            # Validate model tier configuration
            if config.model_tiers.hot_tier_max_models <= 0:
                issues.append("Hot tier must support at least 1 model")
            
            if config.model_tiers.hot_tier_memory_mb > config.resources.max_memory_mb:
                issues.append("Hot tier memory exceeds resource limit")
            
            # Validate scaling configuration
            if config.min_replicas > config.max_replicas:
                issues.append("Minimum replicas cannot exceed maximum replicas")
            
            if config.target_cpu_utilization <= 0 or config.target_cpu_utilization > 1:
                issues.append("Target CPU utilization must be between 0 and 1")
            
            # Validate batch processing
            if config.batch_size <= 0:
                issues.append("Batch size must be positive")
            
            if config.parallel_workers <= 0:
                issues.append("Parallel workers must be positive")
            
            # Validate thresholds
            if config.health_checks.degraded_threshold >= config.health_checks.unhealthy_threshold:
                issues.append("Degraded threshold must be less than unhealthy threshold")
            
            # Validate blue-green configuration
            if config.blue_green.enabled:
                if config.blue_green.success_rate_threshold <= 0 or config.blue_green.success_rate_threshold > 1:
                    issues.append("Success rate threshold must be between 0 and 1")
                
                if not config.blue_green.traffic_switch_steps:
                    issues.append("Blue-green deployment requires traffic switch steps")
            
        except Exception as e:
            issues.append(f"Configuration validation error: {e}")
        
        if issues:
            logger.warning(f"Configuration validation found {len(issues)} issues")
            for issue in issues:
                logger.warning(f"   - {issue}")
        else:
            logger.info("Configuration validation passed")
        
        return issues
    
    def get_environment_configs(self) -> Dict[str, str]:
        """Get all available environment configurations"""
        try:
            configs = {}
            
            for config_file in self.config_dir.glob("*.yaml"):
                if config_file.name != self.default_config_file:
                    env_name = config_file.stem
                    configs[env_name] = str(config_file.name)
            
            for config_file in self.config_dir.glob("*.yml"):
                env_name = config_file.stem
                configs[env_name] = str(config_file.name)
            
            for config_file in self.config_dir.glob("*.json"):
                env_name = config_file.stem
                configs[env_name] = str(config_file.name)
            
            return configs
            
        except Exception as e:
            logger.error(f"Failed to get environment configs: {e}")
            return {}
    
    def create_preset_configs(self):
        """Create preset configurations for common scenarios"""
        try:
            # Development configuration
            dev_config = DeploymentConfig(
                mode=DeploymentMode.DEVELOPMENT,
                environment_name="development",
                target_model_count=100,
                min_replicas=1,
                max_replicas=3,
                debug_mode=True,
                resources=ResourceLimits(
                    max_cpu_cores=4,
                    max_memory_mb=4000
                ),
                model_tiers=ModelTierConfig(
                    hot_tier_max_models=50,
                    hot_tier_memory_mb=2000
                )
            )
            self.save_config(dev_config, "development.yaml")
            
            # Staging configuration
            staging_config = DeploymentConfig(
                mode=DeploymentMode.STAGING,
                environment_name="staging",
                target_model_count=500,
                min_replicas=2,
                max_replicas=5,
                resources=ResourceLimits(
                    max_cpu_cores=8,
                    max_memory_mb=8000
                ),
                model_tiers=ModelTierConfig(
                    hot_tier_max_models=200,
                    hot_tier_memory_mb=4000
                )
            )
            self.save_config(staging_config, "staging.yaml")
            
            # Production configuration
            prod_config = DeploymentConfig(
                mode=DeploymentMode.PRODUCTION,
                environment_name="production",
                target_model_count=2000,
                min_replicas=5,
                max_replicas=20,
                resources=ResourceLimits(
                    max_cpu_cores=32,
                    max_memory_mb=32000
                ),
                model_tiers=ModelTierConfig(
                    hot_tier_max_models=1000,
                    hot_tier_memory_mb=16000
                ),
                monitoring=MonitoringConfig(
                    prometheus_enabled=True,
                    dashboard_enabled=True
                )
            )
            self.save_config(prod_config, "production.yaml")
            
            logger.info("Created preset configurations (development, staging, production)")
            
        except Exception as e:
            logger.error(f"Failed to create preset configs: {e}")
            raise