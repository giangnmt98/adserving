from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel, Field
from adserving.src.utils.logger import get_logger

logger = get_logger()


# Response Models
class ErrorResponse(BaseModel):
    """Standard error response model"""

    error_code: int
    message: str
    detail: str = ""
    request_id: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ServiceInfoResponse(BaseModel):
    """Service information response"""

    service: str
    version: str
    status: str
    uptime: str
    models_loaded: int
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
    cache_stats: Dict[str, Any]
    deployment_stats: Dict[str, Any]


class ReadinessResponse(BaseModel):
    """Readiness check response for cold start optimization"""

    ready: bool
    status: str
    timestamp: str
    models_loaded: int
    models_failed: int
    total_models: int
    initialization_complete: bool
    message: str

class APIDocResponse(BaseModel):
    """API documentation response model"""
    content: str
    timestamp: str
    version: str


class ModelStatsResponse(BaseModel):
    """Model statistics response"""

    total_models: int
    hot_models: int
    warm_models: int
    cold_models: int
    cache_hit_rate: float
    avg_load_time: float
    model_breakdown: Dict[str, int]
    tier_distribution: Dict[str, int]
    hot_models_list: List[str] = Field(..., description="List of models in hot tier")
    warm_models_list: List[str] = Field(..., description="List of models in warm tier")
    cold_models_list: List[str] = Field(..., description="List of models in cold tier")


class ModelInfoResponse(BaseModel):
    """Individual model information response"""

    model_name: str
    model_version: str
    model_uri: str
    loaded_at: str
    last_accessed: str
    access_count: int
    tier: str
    memory_usage: float
    avg_inference_time: float
    error_count: int
    success_count: int
    success_rate: float


class ProductionModelVersionsResponse(BaseModel):
    """Response model for production model versions"""

    production_models: Dict[str, str] = Field(
        ..., description="Model name to version mapping"
    )
    total_models: int = Field(..., description="Total number of production models")
    last_checked: str = Field(
        ..., description="Last time production models were checked"
    )


class ModelSwitchingStatusResponse(BaseModel):
    """Response model for model switching status"""

    monitoring_active: bool = Field(
        ..., description="Whether model monitoring is active"
    )
    last_production_check: str = Field(
        ..., description="Last production model check time"
    )
    production_check_interval: int = Field(
        ..., description="Production check interval in seconds"
    )
    cached_models: List[str] = Field(..., description="Currently cached model names")
    production_models: Dict[str, str] = Field(
        ..., description="Production model versions"
    )


class DeploymentVerificationResponse(BaseModel):
    """Response model for deployment verification status"""

    verification_active: bool = Field(
        ..., description="Whether deployment verification is active"
    )
    last_verification_time: str = Field(..., description="Last verification check time")
    zero_downtime_status: str = Field(..., description="Current zero-downtime status")
    deployment_health: Dict[str, Any] = Field(
        ..., description="Deployment health metrics"
    )
    recent_deployments: List[Dict[str, Any]] = Field(
        ..., description="Recent deployment activities"
    )
    service_availability: float = Field(
        ..., description="Current service availability percentage"
    )
    avg_response_time: float = Field(
        ..., description="Average response time in seconds"
    )
    recommendations: List[str] = Field(..., description="Current recommendations")
