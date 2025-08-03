"""
Core Service Endpoints (Health, Readiness, Service Info)
"""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException

from adserving.config.config import get_config
from adserving.core.model_manager import ModelManager
from adserving.monitoring.model_monitor import ModelMonitor
from adserving.router.model_router import ModelRouter

from .api_dependencies import (
    get_model_manager,
    get_model_router,
    get_monitor,
    service_readiness,
    service_start_time,
)
from .response_model import (
    HealthResponse,
    ReadinessResponse,
    ServiceInfoResponse,
)

logger = logging.getLogger(__name__)

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
                "service_info": "/",
                "health_check": "/health",
                "readiness_check": "/ready",

                # Prediction Endpoints
                "predict": "/predict",
                "prediction_health_check": "/predict/health",

                # Model Management Endpoints
                "model_stats": "/models/stats",
                "model_info": "/models/{model_name}/info",
                "deployment_stats": "/models/deployment-stats",

                # Tier Management Endpoints
                "set_manual_tier_assignment": "POST /tier/manual-assignment",
                "remove_manual_tier_assignment": "DELETE /tier/manual-assignment/{model_name}",
                "get_manual_tier_assignment": "GET /tier/manual-assignment/{model_name}",
                "get_business_critical_models": "GET /tier/business-critical-models",
                "add_business_critical_pattern": "POST /tier/business-critical-pattern",
                "remove_business_critical_pattern": "DELETE /tier/business-critical-pattern/{pattern_id}",
                "bulk_assign_business_critical_models": "POST /tier/bulk-assign-business-critical",

                # Monitoring & Metrics Endpoints
                "routing_stats": "/routing/stats",
                "cache_stats": "/cache/stats",
                "dashboard_data": "/dashboard",
                "prometheus_metrics": "/metrics",

                # MLflow Integration Endpoints (if enabled)
                "mlflow_endpoint": "/deployments/{endpoint_name}",

                # System Endpoints
                "api_documentation": "/docs",
                "openapi_schema": "/openapi.json",
            },
            timestamp=datetime.now().isoformat(),
            description=(
                "Anomaly Detection serving system for hundreds of models "
                "with tiered loading and intelligent routing"
            ),
            features=[
                "Tiered model loading (Hot/Warm/Cold)",
                "Single endpoint routing with intelligent selection",
                "Batch processing capabilities",
                "Advanced monitoring and metrics",
                "Resource sharing and GPU optimization",
                "Prometheus metrics export",
                "Auto-scaling and capacity planning",
                "Zero-downtime deployment",
                "Business-critical model management",
                "MLflow integration",
                "Comprehensive health checks",
                "Real-time dashboard monitoring",
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

            return JSONResponse(status_code=206, content=response.dict())
        return response

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/ready", response_model=ReadinessResponse)
async def readiness_check():
    """
    Readiness check with proper HTTP status codes.

    Returns:
    - 200: Service is ready
    - 503: Service is not ready (initializing or failed)
    """
    try:
        ready = service_readiness["ready"]
        models_loaded = service_readiness["models_loaded"]
        models_failed = service_readiness["models_failed"]
        initialization_complete = service_readiness["initialization_complete"]
        total_models = models_loaded + models_failed

        # Determine status and message
        if ready and initialization_complete:
            status = "ready"
            if models_failed == 0:
                message = (
                    f"Service fully ready. "
                    f"All {models_loaded} models loaded successfully."
                )
            else:
                message = (
                    f"Service ready with warnings. "
                    f"{models_loaded}/{total_models} models loaded successfully."
                )
        elif initialization_complete:
            status = "not_ready"
            message = (
                f"Service initialization complete but not ready. "
                f"{models_loaded}/{total_models} models loaded."
            )
        else:
            status = "initializing"
            message = (
                f"Service still initializing. " f"{models_loaded} models loaded so far."
            )

        response = ReadinessResponse(
            ready=ready,
            status=status,
            timestamp=datetime.now().isoformat(),
            models_loaded=models_loaded,
            models_failed=models_failed,
            total_models=total_models,
            initialization_complete=initialization_complete,
            message=message,
        )

        # Return appropriate HTTP status code
        if not ready:
            from fastapi.responses import JSONResponse

            return JSONResponse(status_code=503, content=response.dict())

        return response

    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        # Return 503 for readiness check failures
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=503,
            content=ReadinessResponse(
                ready=False,
                status="error",
                timestamp=datetime.now().isoformat(),
                models_loaded=0,
                models_failed=0,
                total_models=0,
                initialization_complete=False,
                message=f"Readiness check failed: {str(e)}",
            ).dict(),
        )
