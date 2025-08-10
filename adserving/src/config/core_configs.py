"""
Core configuration classes
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


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
    production_check_interval: int = 60  # seconds


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
