# Python
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from .models import (
    APIGroupConfig,
    LoggingConfig,
    MLflowConfig,
    MonitoringConfig,
    PreloadConfig,
    RayConfig,
    SecurityConfig,
    ServeAutoscalingConfig,
    ServeConfig,
    ServeDeploymentConfig,
    ServeHTTPConfig,
    WatcherConfig,
    # [NEW]
    RedisConfig,
    RedisStreamsConfig,
    RedisConsumerConfig,
    DatabaseConfig,
    AuditConfig,
    AuditSamplingConfig,

)


@dataclass
class Config:
    """Configuration for simplified single deployment"""

    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    ray: RayConfig = field(default_factory=RayConfig)

    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)

    serve: ServeConfig = field(default_factory=ServeConfig)
    preload: PreloadConfig = field(default_factory=PreloadConfig)
    watcher: WatcherConfig = field(default_factory=WatcherConfig)
    api: APIGroupConfig = field(default_factory=APIGroupConfig)

    # [NEW] Redis & Database
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
        # [NEW] có thể override nhanh các secret/endpoint
        self._load_audit_config()
        self._load_redis_config()
        self._load_database_config()

    def _load_mlflow_config(self) -> None:
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
        if mlflow_uri:
            self.mlflow.tracking_uri = mlflow_uri

    def _load_ray_config(self) -> None:
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
        enabled = os.getenv("AUDIT_ENABLED")
        if enabled is not None:
            self.audit.enabled = enabled.lower() in ("1", "true", "yes", "y")
        tr = os.getenv("AUDIT_TRAINING_RATE")
        if tr:
            try:
                self.audit.sampling.training_rate = float(tr)
            except ValueError:
                pass
        ir = os.getenv("AUDIT_INFERENCE_RATE")
        if ir:
            try:
                self.audit.sampling.inference_rate = float(ir)
            except ValueError:
                pass
        redact = os.getenv("AUDIT_REDACT_PII")
        if redact is not None:
            self.audit.redact_pii = redact.lower() in ("1", "true", "yes", "y")

    # [NEW] Redis từ env
    def _load_redis_config(self) -> None:
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

    # [NEW] Database từ env
    def _load_database_config(self) -> None:
        host = os.getenv("DB_HOST")
        if host:
            self.database.host = host
        port = os.getenv("DB_PORT")
        if port:
            try:
                self.database.port = int(port)
            except ValueError:
                pass
        name = os.getenv("DB_NAME")
        if name:
            self.database.name = name
        user = os.getenv("DB_USER")
        if user:
            self.database.user = user
        pwd = os.getenv("DB_PASSWORD")
        if pwd:
            self.database.password = pwd

    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "Config":
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
        config_data: Dict[str, Any] = {}

        def _wrap(d: Dict[str, Any], t):
            try:
                return t(**(d or {}))
            except Exception:
                return t()

        def _process_serve_config(value: Dict[str, Any]) -> ServeConfig:
            autoscaling = _wrap(value.get("autoscaling", {}), ServeAutoscalingConfig)
            deployment = _wrap(value.get("deployment", {}), ServeDeploymentConfig)
            http = _wrap(value.get("http", {}), ServeHTTPConfig)
            return ServeConfig(http=http, autoscaling=autoscaling, deployment=deployment)

        # [NEW] helper xử lý redis/database
        def _process_redis_config(value: Dict[str, Any]) -> RedisConfig:
            streams = _wrap(value.get("streams", {}), RedisStreamsConfig)
            consumer = _wrap(value.get("consumer", {}), RedisConsumerConfig)
            base = value.copy()
            base.pop("streams", None)
            base.pop("consumer", None)
            rc = _wrap(base, RedisConfig)
            rc.streams = streams
            rc.consumer = consumer
            return rc

        def _process_config(key: str, value: Dict[str, Any]) -> Any:
            config_map = {
                "mlflow": MLflowConfig,
                "ray": RayConfig,
                "monitoring": MonitoringConfig,
                "logging": LoggingConfig,
                "security": SecurityConfig,
                "preload": PreloadConfig,
                "watcher": WatcherConfig,
                "api": APIGroupConfig,
                "database": DatabaseConfig,   # [NEW]
            }
            if key == "serve":
                return _process_serve_config(value)
            if key == "redis":
                return _process_redis_config(value)  # [NEW]
            if key in config_map:
                return _wrap(value, config_map[key])
            return None

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
                config_value = _process_config(key, value)
                if config_value is not None:
                    config_data[key] = config_value
                continue
            if key in allowed_scalar_top_level:
                config_data[key] = value

        return cls(**config_data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_file(self, config_path: Union[str, Path], format: str = "yaml") -> None:
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

    def get_ray_init_config(self) -> Dict[str, Any]:
        config: Dict[str, Any] = {
            "address": self.ray.address,
            "dashboard_host": self.ray.dashboard_host,
            "dashboard_port": self.ray.dashboard_port,
        }
        if getattr(self.ray, "object_store_memory", None):
            config["object_store_memory"] = self.ray.object_store_memory
        if self.ray.num_cpus is not None:
            config["num_cpus"] = self.ray.num_cpus
        if self.ray.num_gpus is not None:
            config["num_gpus"] = self.ray.num_gpus
        if self.ray.runtime_env:
            config["runtime_env"] = self.ray.runtime_env
        return config


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
