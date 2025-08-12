# Python
"""
Module for handling all future annotations.

This module enables the use of future annotations to ensure compatibility
between different Python versions. It allows using type annotations in a
consistent way regardless of the Python version being used.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .models import (
    APIGroupConfig,
    AuditConfig,
    DatabaseConfig,
    LoggingConfig,
    MLflowConfig,
    MonitoringConfig,
    PreloadConfig,
    RayConfig,
    RedisConfig,
    SecurityConfig,
    ServeAutoscalingConfig,
    ServeConfig,
    ServeDeploymentConfig,
    ServeHTTPConfig,
    WatcherConfig,
)
from .utils import load_data_from_file, process_config_section


@dataclass
class Config:
    """Configuration for simplified single deployment"""

    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    ray: RayConfig = field(default_factory=RayConfig)

    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)

    serve: ServeConfig = field(default_factory=ServeConfig)
    http: ServeHTTPConfig = field(default_factory=ServeHTTPConfig)
    autoscaling: ServeAutoscalingConfig = field(default_factory=ServeAutoscalingConfig)

    server_deployment: ServeDeploymentConfig = field(
        default_factory=ServeDeploymentConfig
    )

    preload: PreloadConfig = field(default_factory=PreloadConfig)
    watcher: WatcherConfig = field(default_factory=WatcherConfig)
    api: APIGroupConfig = field(default_factory=APIGroupConfig)

    # Redis & Database
    audit: AuditConfig = field(default_factory=AuditConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)

    # Back-compat fields
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    api_version: str = "v1.0"
    enable_docs: bool = True
    enable_openapi: bool = True

    def __post_init__(self) -> None:
        """Initialize configuration after dataclass creation and sync API settings."""
        self._load_from_environment()
        # Đồng bộ nhóm api -> các trường back-compat
        if self.api:
            self.api_host = self.api.host or self.api_host
            self.api_port = self.api.port or self.api_port
            self.api_prefix = self.api.prefix or self.api_prefix
            self.api_version = self.api.version or self.api_version
            self.enable_docs = bool(self.api.docs)
            self.enable_openapi = bool(self.api.openapi)

    def _load_from_environment(self) -> None:
        """Load configuration from environment variables (ghi đè file config nếu có)"""
        self._load_mlflow_config()
        self._load_ray_config()
        self._load_api_config()
        self._load_legacy_api_config()
        self._load_serve_config()
        self._load_audit_config()
        self._load_redis_config()
        self._load_database_config()

    def _load_mlflow_config(self) -> None:
        """Load MLflow configuration from environment variables."""
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
        if mlflow_uri:
            self.mlflow.tracking_uri = mlflow_uri

    def _load_ray_config(self) -> None:
        """Load Ray configuration from environment variables."""
        ray_addr = os.getenv("RAY_ADDRESS")
        if ray_addr:
            self.ray.address = ray_addr
        dash_host = os.getenv("RAY_DASHBOARD_HOST")
        if dash_host and hasattr(self.ray, "dashboard_host"):
            self.ray.dashboard_host = dash_host
        dash_port = os.getenv("RAY_DASHBOARD_PORT")
        if dash_port and hasattr(self.ray, "dashboard_port"):
            try:
                self.ray.dashboard_port = int(dash_port)
            except ValueError:
                pass

    def _load_api_config(self) -> None:
        """Load FastAPI configuration from environment variables."""
        api_host = os.getenv("FASTAPI_HOST")
        if api_host:
            self.api.host = api_host
        api_port = os.getenv("FASTAPI_PORT")
        if api_port:
            try:
                self.api.port = int(api_port)
            except ValueError:
                pass

    def _load_legacy_api_config(self) -> None:
        """Load legacy API configuration from environment variables."""
        api_host_legacy = os.getenv("API_HOST")
        if api_host_legacy:
            self.api.host = api_host_legacy
        api_port_legacy = os.getenv("API_PORT")
        if api_port_legacy:
            try:
                self.api.port = int(api_port_legacy)
            except ValueError:
                pass

    def _load_serve_config(self) -> None:
        """Load Serve HTTP configuration from environment variables."""
        serve_http_host = os.getenv("SERVE_HTTP_HOST")
        if serve_http_host:
            self.serve.http.host = serve_http_host
        serve_http_port = os.getenv("SERVE_HTTP_PORT")
        if serve_http_port:
            try:
                self.serve.http.port = int(serve_http_port)
            except ValueError:
                pass

    # [NEW] Audit từ env (tùy chọn)
    def _load_audit_config(self) -> None:
        """Load audit configuration from environment variables."""
        enabled = os.getenv("AUDIT_ENABLED")
        if enabled is not None:
            self.audit.enabled = enabled.lower() in ("1", "true", "yes", "y")
        tr = os.getenv("AUDIT_TRAINING_RATE")
        if tr:
            try:
                self.audit.training_rate = float(tr)
            except ValueError:
                pass
        ir = os.getenv("AUDIT_INFERENCE_RATE")
        if ir:
            try:
                self.audit.inference_rate = float(ir)
            except ValueError:
                pass
        redact = os.getenv("AUDIT_REDACT_PII")
        if redact is not None:
            self.audit.redact_pii = redact.lower() in ("1", "true", "yes", "y")

    # Redis từ env
    def _load_redis_config(self) -> None:
        """Load Redis configuration from environment variables."""
        host = os.getenv("REDIS_HOST")
        if host:
            self.redis.host = host
        port = os.getenv("REDIS_PORT")
        if port:
            try:
                self.redis.port = int(port)
            except ValueError:
                pass
        db = os.getenv("REDIS_DB")
        if db:
            try:
                self.redis.db = int(db)
            except ValueError:
                pass
        pwd = os.getenv("REDIS_PASSWORD")
        if pwd:
            self.redis.password = pwd

    # Database từ env
    def _load_database_config(self) -> None:
        """Load database configuration from environment variables."""
        host = os.getenv("DB_HOST")
        if host:
            self.database.host = host
        port = os.getenv("DB_PORT")
        if port:
            try:
                self.database.port = int(port)
            except ValueError:
                pass
        database_name = os.getenv("DB_NAME")
        if database_name:
            self.database.database_name = database_name
        username = os.getenv("DB_USER")
        if username:
            self.database.username = username
        pwd = os.getenv("DB_PASSWORD")
        if pwd:
            self.database.password = pwd

    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "Config":
        """Load configuration from YAML or JSON file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        data = load_data_from_file(config_path)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary data."""
        config_data: Dict[str, Any] = {}

        config_map = {
            "mlflow": MLflowConfig,
            "ray": RayConfig,
            "monitoring": MonitoringConfig,
            "logging": LoggingConfig,
            "security": SecurityConfig,
            "preload": PreloadConfig,
            "watcher": WatcherConfig,
            "api": APIGroupConfig,
            "audit": AuditConfig,
            "redis": RedisConfig,
            "database": DatabaseConfig,
            "serve": ServeConfig,
            "http": ServeHTTPConfig,
            "autoscaling": ServeAutoscalingConfig,
            "server_deployment": ServeDeploymentConfig,
        }

        allowed_scalar_top_level = {
            "api_host",
            "api_port",
            "api_prefix",
            "api_version",
            "enable_docs",
            "enable_openapi",
        }

        for key, value in (data or {}).items():
            if isinstance(value, dict):
                config_value = process_config_section(key, value, config_map)
                if config_value is not None:
                    config_data[key] = config_value
                continue
            if key in allowed_scalar_top_level:
                config_data[key] = value

        return cls(**config_data)


# Global configuration instance
_global_config: Optional[Config] = None


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Load configuration from a file or create default"""
    if config_path:
        return Config.from_file(config_path)

    # Try to load from default locations
    default_paths = ["config.yaml", "config.yml"]

    for path in default_paths:
        if Path(path).exists():
            return Config.from_file(path)

    # Return default configuration
    return Config()


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
