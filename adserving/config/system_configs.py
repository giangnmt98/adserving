"""
System configuration classes
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MonitoringConfig:
    """Monitoring configuration for hundreds of models"""

    collection_interval: int = 5  # More frequent collection
    optimization_interval: int = 60  # More frequent optimization
    history_size: int = 10000  # Increased history
    enable_gpu_monitoring: bool = True
    enable_model_level_monitoring: bool = True

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
    """Logging configuration"""

    log_level: str = "INFO"
    log_dir: str = "logs"
    max_file_size: int = 209715200  # 200MB
    backup_count: int = 10  # More backups for production
    enable_console: bool = True
    enable_structured: bool = True
    enable_performance_tracking: bool = True

    # Logging features
    enable_model_level_logging: bool = True
    enable_request_tracing: bool = True
    log_sampling_rate: float = 0.1  # Log 10% of requests for performance
    enable_error_aggregation: bool = True


@dataclass
class SecurityConfig:
    """Security configuration"""

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


@dataclass
class PerformanceConfig:
    """Performance optimization configuration"""

    enable_performance_optimization: bool = True
    optimization_interval: int = 300  # 5 minutes

    # Batch processing
    enable_dynamic_batching: bool = True
    max_batch_size: int = 64
    batch_timeout_ms: int = 100

    # Caching
    enable_response_caching: bool = True
    response_cache_size: int = 10000
    response_cache_ttl: int = 300  # 5 minutes

    # Connection pooling
    enable_connection_pooling: bool = True
    max_connections: int = 1000
    connection_timeout: int = 30

    # Memory optimization
    enable_memory_optimization: bool = True
    gc_interval: int = 60  # Garbage collection interval
    memory_threshold: float = 0.8  # Trigger cleanup at 80% memory


@dataclass
class BatchProcessingConfig:
    """Batch processing configuration"""

    enable_dynamic_batching: bool = True
    max_batch_size: int = 256
    batch_timeout_ms: int = 50
    adaptive_batch_sizing: bool = True
    enable_response_caching: bool = True
    response_cache_size: int = 10000
    response_cache_ttl: int = 300


@dataclass
class ConnectionPoolingConfig:
    """Connection pooling configuration"""

    enable_connection_pooling: bool = True
    max_connections: int = 1000
    connection_timeout: int = 30
    pool_connections: int = 10
    pool_maxsize: int = 20
