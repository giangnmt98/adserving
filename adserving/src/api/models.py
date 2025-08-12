from typing import Any, Dict, List, Optional

from pydantic import BaseModel


# Request/Response models for parameter management
class ParameterUpdateRequest(BaseModel):
    """Request model for updating model parameters
    with new values and optional comment"""

    parameters: Dict[str, float]
    comment: Optional[str] = None


class AnomalyThresholdUpdateRequest(BaseModel):
    """Request model for updating anomaly detection
    threshold value with optional comment"""

    threshold: float
    comment: Optional[str] = None


class RollbackRequest(BaseModel):
    """Request model for rolling back specific
    model to a target version"""

    model_name: str
    target_version: str


class ParameterValidationResponse(BaseModel):
    """Response model containing parameter validation
    results with errors and warnings"""

    valid: bool
    errors: List[str] = []
    warnings: List[str] = []


class ParameterHistoryResponse(BaseModel):
    """Response model containing history of
    parameter updates for a specific model"""

    model_name: str
    updates: List[Dict]
    total_count: int


class ServiceInfoResponse(BaseModel):
    """Response model containing service metadata,
    status and available endpoints"""

    service: str
    version: str
    status: str
    uptime: str
    endpoints: Dict[str, str]
    timestamp: str
    description: str
    features: List[str]


class HealthResponse(BaseModel):
    """Response model containing service
    health status and operational metrics"""

    status: str
    version: str
    timestamp: str
    models_loaded: int
    uptime: str
    deployment_stats: Dict[str, Any]


class ModelInfoResponse(BaseModel):
    """Response model containing detailed information
    about a specific model's status and performance"""

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
