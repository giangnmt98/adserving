from __future__ import annotations

import os
import json
import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Union, Optional


from .core_configs import (
    MLflowConfig,
    RayConfig,
)
from .system_configs import (
    LoggingConfig,
    MonitoringConfig,
    SecurityConfig,
)
@dataclass
class ServeHTTPConfig:
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class ServeAutoscalingConfig:
    min_replicas: int = 2
    max_replicas: int = 8
    target_num_ongoing_requests_per_replica: int = 12


@dataclass
class ServeDeploymentConfig:
    num_cpus: int = 1
    memory_mb: int = 2048


@dataclass
class ServeConfig:
    http: ServeHTTPConfig = field(default_factory=ServeHTTPConfig)
    autoscaling: ServeAutoscalingConfig = field(default_factory=ServeAutoscalingConfig)
    deployment: ServeDeploymentConfig = field(default_factory=ServeDeploymentConfig)


@dataclass
class PreloadConfig:
    max_load_concurrency: int = 8


@dataclass
class WatcherConfig:
    interval_seconds: int = 60
    sanity_check_enabled: bool = False
    sanity_inputs: List[float] = field(default_factory=list)


@dataclass
class APIGroupConfig:
    host: str = "0.0.0.0"
    port: int = 8001
    prefix: str = "/api/v1"
    version: str = "v1.0"
    docs: bool = True
    openapi: bool = True


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

        # MLflow
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
        if mlflow_uri:
            self.mlflow.tracking_uri = mlflow_uri

        # Ray
        ray_addr = os.getenv("RAY_ADDRESS")
        if ray_addr:
            self.ray.address = ray_addr
        # Dashboard host/port nếu có trong RayConfig
        dash_host = os.getenv("RAY_DASHBOARD_HOST")
        if dash_host and hasattr(self.ray, "dashboard_host"):
            self.ray.dashboard_host = dash_host
        dash_port = os.getenv("RAY_DASHBOARD_PORT")
        if dash_port and hasattr(self.ray, "dashboard_port"):
            try:
                self.ray.dashboard_port = int(dash_port)
            except ValueError:
                pass

        # FastAPI host/port (nhóm api mới)
        api_host = os.getenv("FASTAPI_HOST")
        if api_host:
            self.api.host = api_host
        api_port = os.getenv("FASTAPI_PORT")
        if api_port:
            try:
                self.api.port = int(api_port)
            except ValueError:
                pass

        # Back-compat env (cũ)
        api_host_legacy = os.getenv("API_HOST")
        if api_host_legacy:
            self.api.host = api_host_legacy
        api_port_legacy = os.getenv("API_PORT")
        if api_port_legacy:
            try:
                self.api.port = int(api_port_legacy)
            except ValueError:
                pass

        # Serve HTTP host/port
        serve_http_host = os.getenv("SERVE_HTTP_HOST")
        if serve_http_host:
            self.serve.http.host = serve_http_host
        serve_http_port = os.getenv("SERVE_HTTP_PORT")
        if serve_http_port:
            try:
                self.serve.http.port = int(serve_http_port)
            except ValueError:
                pass

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
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        config_data: Dict[str, Any] = {}

        allowed_scalar_top_level = {
            "api_host", "api_port", "api_prefix", "api_version", "enable_docs", "enable_openapi",
        }

        def _wrap(d: Dict[str, Any], t):
            try:
                return t(**(d or {}))
            except Exception:
                return t()

        for key, value in (data or {}).items():
            if isinstance(value, dict):
                if key == "mlflow":
                    config_data["mlflow"] = _wrap(value, MLflowConfig)
                elif key == "ray":
                    config_data["ray"] = _wrap(value, RayConfig)
                elif key == "monitoring":
                    config_data["monitoring"] = _wrap(value, MonitoringConfig)
                elif key == "logging":
                    config_data["logging"] = _wrap(value, LoggingConfig)
                elif key == "security":
                    config_data["security"] = _wrap(value, SecurityConfig)
                elif key == "serve":
                    autoscaling = _wrap(value.get("autoscaling", {}), ServeAutoscalingConfig)
                    deployment = _wrap(value.get("deployment", {}), ServeDeploymentConfig)
                    http = _wrap(value.get("http", {}), ServeHTTPConfig)
                    config_data["serve"] = ServeConfig(http=http, autoscaling=autoscaling, deployment=deployment)
                elif key == "preload":
                    config_data["preload"] = _wrap(value, PreloadConfig)
                elif key == "watcher":
                    config_data["watcher"] = _wrap(value, WatcherConfig)
                elif key == "api":
                    config_data["api"] = _wrap(value, APIGroupConfig)
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
    """Load configuration from file or create default"""
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
