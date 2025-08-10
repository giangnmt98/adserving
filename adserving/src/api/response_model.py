from typing import Any, Dict, List

from pydantic import BaseModel


class ServiceInfoResponse(BaseModel):
    """Service information response"""

    service: str
    version: str
    status: str
    uptime: str
    endpoints: Dict[str, str]
    timestamp: str
    description: str
    features: List[str]


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    version: str
    timestamp: str
    models_loaded: int
    uptime: str
    deployment_stats: Dict[str, Any]


class ModelInfoResponse(BaseModel):
    """Individual model information response"""

    model_name: str
    model_version: str
    model_uri: str
    loaded_at: str
    last_accessed: str
    access_count: int
    memory_usage: float
    avg_inference_time: float
    error_count: int
    success_count: int
    success_rate: float
