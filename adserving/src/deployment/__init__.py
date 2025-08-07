"""
Deployment module for model serving
"""

from adserving.src.deployment.utils.metrics_collector import MetricsCollector
from adserving.src.deployment.utils.model_router import ModelPoolRouter
from adserving.src.deployment.utils.request_processor import RequestProcessor
from adserving.src.deployment.utils.task_processor import TaskProcessor

from .model_pool import ModelPool
from .pooled_actor import PooledModelActor
from .pooled_deployment import PooledModelDeployment
from .resource_config import (
    AutoscalingSettings,
    PooledDeploymentConfig,
    PooledResourceConfig,
)

__all__ = [
    "PooledResourceConfig",
    "AutoscalingSettings",
    "PooledDeploymentConfig",
    "ModelPool",
    "PooledModelActor",
    "PooledModelDeployment",
    "RequestProcessor",
    "ModelPoolRouter",
    "TaskProcessor",
    "MetricsCollector",
]
