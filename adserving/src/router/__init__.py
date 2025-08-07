"""
Router module for intelligent model routing with single endpoint strategy
"""

from .deployment_selector import DeploymentSelector
from .model_name_extractor import ModelNameExtractor
from .model_router import ModelRouter
from .request_queue import RequestQueue
from .route_metrics import RouteMetrics
from .routing_strategy import RoutingStrategy
from .unified_endpoint import UnifiedPredictionEndpoint

__all__ = [
    "RoutingStrategy",
    "RouteMetrics",
    "RequestQueue",
    "ModelNameExtractor",
    "DeploymentSelector",
    "ModelRouter",
    "UnifiedPredictionEndpoint",
]
