"""
Core Service Endpoints (Health, Readiness, Service Info)
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException

from adserving.src.config.config_manager import get_config
from adserving.src.core.model_manager import ModelManager
from adserving.src.monitoring.model_monitor import ModelMonitor
from adserving.src.router.model_router import ModelRouter
from adserving.src.utils.logger import get_logger

from .api_dependencies import (
    get_model_manager,
    get_model_router,
    get_monitor,
    service_readiness,
    service_start_time,
)
from .response_model import (
    HealthResponse,
    ServiceInfoResponse,
    APIDocResponse
)

logger = get_logger()

router = APIRouter()


@router.get("/", response_model=ServiceInfoResponse)
async def service_info(
    manager: ModelManager = Depends(get_model_manager),
    model_router: ModelRouter = Depends(get_model_router),
):
    """Get service information and status"""
    try:
        uptime = datetime.now() - service_start_time
        uptime_str = str(uptime).split(".")[0]  # Remove microseconds

        # Get model statistics
        cache_stats = manager.get_cache_stats()
        models_loaded = cache_stats.get("total_models", 0)

        return ServiceInfoResponse(
            service="Anomaly Detection API",
            version=get_config().api_version,
            status="running",
            uptime=uptime_str,
            models_loaded=models_loaded,
            endpoints={
                # Core Service Endpoints
                "service_info": "GET /",
                "health_check": "GET /health",
                "readiness_check": "GET /ready",
                # Prediction Endpoints
                "predict": "POST /predict",
                # Model Management Endpoints (từ model_endpoints.py)
                "model_info": "GET /models/{model_name}/info",
                "prometheus_metrics": "GET /metrics",
                # Parameter Management Endpoints (mới bổ sung)
                "production_models": "GET /models/production",
                "model_parameters": "GET /models/{model_name}/parameters",
                "validate_parameters": "POST /models/{model_name}/parameters/validate",
                "update_threshold": "PUT /models/{model_name}/threshold",
                "update_parameters": "PUT /models/{model_name}/parameters",
                "rollback_model": "POST /models/{model_name}/rollback",
                "parameter_history": "GET /models/{model_name}/history",
                "batch_threshold_update": "POST /models/batch-threshold-update",
                "warm_model": "POST /models/{model_name}/warm",
                "evict_model": "DELETE /models/{model_name}/cache",
                # Tier Management Endpoints (từ tier_management_endpoints.py)
                # "manual_tier_assignment": "POST " "/tier/manual-assignment",
                # "remove_tier_assignment": "DELETE /tier/manual-assignment/{model_name}",
                # "get_tier_assignment": "GET /tier/manual-assignment/{model_name}",
                # "business_critical_models": "GET /tier/business-critical-models",
                # "add_critical_pattern": "POST /tier/business-critical-pattern",
                # "remove_critical_pattern": "DELETE /tier/business-critical-pattern/{pattern_id}",
                # System Endpoints
                "api_documentation": "GET /docs",
                "openapi_schema": "GET /openapi.json",
                "redoc_documentation": "GET /redoc",
            },
            timestamp=datetime.now().isoformat(),
            description=(
                "Enhanced Anomaly Detection serving system for hundreds of models "
                "with tiered loading, parameter management, and intelligent routing"
            ),
            features=[
                "Tiered model loading (Hot/Warm/Cold)",
                "Single endpoint routing with intelligent selection",
                "Real-time parameter updates via API",
                "Model version rollback capabilities",
                "Batch parameter operations",
                "Advanced monitoring and metrics",
                "Model cache management (warm/evict)",
                "Business-critical model prioritization",
                "Manual tier assignment support",
                "Comprehensive health checks",
                "Zero-downtime deployment",
                "Prometheus metrics export",
                "Interactive API documentation",
                "Parameter validation before updates",
                "Audit trail for parameter changes",
            ],
        )
    except Exception as e:
        logger.error(f"Error getting service info: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get service info: {str(e)}"
        )


@router.get("/health", response_model=HealthResponse)
async def health_check(
    manager: ModelManager = Depends(get_model_manager),
    model_router: ModelRouter = Depends(get_model_router),
    monitor: ModelMonitor = Depends(get_monitor),
):
    """Comprehensive health check with proper status codes"""
    try:
        uptime = datetime.now() - service_start_time
        uptime_str = str(uptime).split(".")[0]

        # Get comprehensive stats
        cache_stats = manager.get_cache_stats()
        dashboard_data = monitor.get_dashboard_data()
        models_loaded = cache_stats.get("total_models", 0)

        # Check for partial health conditions
        total_models = (
            service_readiness["models_loaded"] + service_readiness["models_failed"]
        )
        has_failed_models = service_readiness["models_failed"] > 0

        # Determine health status
        if has_failed_models and total_models > 0:
            # Some models failed but service is partially healthy
            status_code = 206  # Partial Content
            status = "partially_healthy"
        else:
            status_code = 200
            status = "healthy"

        response = HealthResponse(
            status=status,
            version=get_config().api_version,
            timestamp=datetime.now().isoformat(),
            models_loaded=models_loaded,
            uptime=uptime_str,
            cache_stats=cache_stats,
            deployment_stats=dashboard_data.get("deployments", {}),
        )

        # Return with appropriate status code
        if status_code == 206:
            from fastapi.responses import JSONResponse

            return JSONResponse(status_code=206, content=response.model_dump())
        return response

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")
