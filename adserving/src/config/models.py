"""Configuration classes for ML model serving system using dataclasses."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ServeHTTPConfig:
    """HTTP server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class ServeAutoscalingConfig:
    """Autoscaling configuration for serving."""

    min_replicas: int = 2
    max_replicas: int = 8
    target_num_ongoing_requests_per_replica: int = 12


@dataclass
class ServeDeploymentConfig:
    """Deployment resource configuration."""

    num_cpus: int = 1
    memory_mb: int = 2048


@dataclass
class ServeConfig:
    """Overall serving configuration."""

    http: ServeHTTPConfig = field(default_factory=ServeHTTPConfig)
    autoscaling: ServeAutoscalingConfig = field(default_factory=ServeAutoscalingConfig)
    deployment: ServeDeploymentConfig = field(default_factory=ServeDeploymentConfig)


@dataclass
class PreloadConfig:
    """Model preloading configuration."""

    max_load_concurrency: int = 8


@dataclass
class WatcherConfig:
    """Model watcher configuration."""

    interval_seconds: int = 60
    sanity_check_enabled: bool = False
    sanity_inputs: List[float] = field(default_factory=list)


@dataclass
class APIGroupConfig:
    """API endpoint group configuration."""

    host: str = "0.0.0.0"
    port: int = 8001
    prefix: str = "/api/v1"
    version: str = "v1.0"
    docs: bool = True
    openapi: bool = True


@dataclass
class MLflowConfig:
    """MLflow integration configuration."""

    tracking_uri: str = "http://localhost:5000"
    registry_uri: Optional[str] = None
    experiment_name: str = "anomalydetectionserving"
    artifact_location: Optional[str] = None
    model_stage: str = "Production"
    enable_model_versioning: bool = True
    model_cache_ttl: int = 3600  # seconds
    production_check_interval: int = 60  # seconds


@dataclass
class RayConfig:
    """Ray cluster configuration."""

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
class MonitoringConfig:
    """Monitoring and metrics configuration."""

    collection_interval: int = 5  # More frequent collection
    optimization_interval: int = 60  # More frequent optimization
    history_size: int = 1000  # Increased history
    enable_gpu_monitoring: bool = True
    enable_model_level_monitoring: bool = True
    update_interval: int = 10

    # Health check settings
    health_check_interval: int = 15
    model_health_check_timeout: int = 5
    deployment_health_check_timeout: int = 10

    # Alert thresholds
    alert_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "cpu_percent": 85.0,  # Slightly lower for better performance
            "memory_percent": 85.0,
            "gpu_percent": 90.0,
            "error_rate": 0.02,  # 2% error rate
            "response_time": 2.0,  # 2 seconds
            "queue_length": 1000,
            "cache_hit_rate": 0.8,  # 80% cache hit rate
        }
    )

    # Metrics export
    enable_prometheus_export: bool = True
    prometheus_port: int = 9090
    enable_grafana_dashboard: bool = True
    metrics_retention_days: int = 30


@dataclass
class LoggingConfig:
    """Logging system configuration."""

    log_level: str = "INFO"
    log_dir: str = "logs"
    max_file_size: int = 209715200  # 200MB
    backup_count: int = 10  # More backups for production
    enable_console: bool = True
    enable_structured: bool = True
    enable_performance_tracking: bool = True
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Logging features
    enable_model_level_logging: bool = True
    enable_request_tracing: bool = True
    log_sampling_rate: float = 0.1  # Log 10% of requests for performance
    enable_error_aggregation: bool = True


@dataclass
class SecurityConfig:
    """Security and authentication configuration."""

    enable_auth: bool = False
    api_key: Optional[str] = None
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    rate_limit_requests_per_minute: int = 10000  # Increased for hundreds of models
    enable_https: bool = False
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None

    # Security features
    enable_request_validation: bool = True
    enable_model_access_control: bool = False
    allowed_model_patterns: List[str] = field(default_factory=lambda: ["*"])
    enable_audit_logging: bool = True
    session_timeout: int = 3600  # 1 hour
