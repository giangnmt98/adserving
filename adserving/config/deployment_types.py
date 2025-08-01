"""
Deployment-related configuration types
"""

from typing import Any, Dict


class PooledResourceConfig:
    """Configuration for pooled resources"""

    def __init__(
        self,
        num_cpus: float = 1.0,
        num_gpus: float = 0.0,
        memory: int = 1024,
        object_store_memory: int = 512,
        max_models_per_replica: int = 10,
    ):
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        self.memory = memory
        self.object_store_memory = object_store_memory
        self.max_models_per_replica = max_models_per_replica

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "num_cpus": self.num_cpus,
            "num_gpus": self.num_gpus,
            "memory": self.memory,
            "object_store_memory": self.object_store_memory,
            "max_models_per_replica": self.max_models_per_replica,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PooledResourceConfig":
        """Create from dictionary"""
        return cls(**data)


class AutoscalingSettings:
    """Autoscaling configuration settings"""

    def __init__(
        self,
        min_replicas: int = 2,
        max_replicas: int = 10,
        target_num_ongoing_requests_per_replica: int = 2,
        metrics_interval_s: float = 10.0,
        look_back_period_s: float = 30.0,
        smoothing_factor: float = 1.0,
    ):
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.target_num_ongoing_requests_per_replica = (
            target_num_ongoing_requests_per_replica
        )
        self.metrics_interval_s = metrics_interval_s
        self.look_back_period_s = look_back_period_s
        self.smoothing_factor = smoothing_factor

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "min_replicas": self.min_replicas,
            "max_replicas": self.max_replicas,
            "target_num_ongoing_"
            "requests_per_replica": self.target_num_ongoing_requests_per_replica,
            "metrics_interval_s": self.metrics_interval_s,
            "look_back_period_s": self.look_back_period_s,
            "smoothing_factor": self.smoothing_factor,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AutoscalingSettings":
        """Create from dictionary"""
        return cls(**data)
