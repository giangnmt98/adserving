import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from adserving.config.config import get_config
from adserving.core.model_manager import ModelManager
from adserving.datahandler.data_handler import DataHandler
from adserving.datahandler.models import APIResponse, PredictionRequest
from adserving.monitoring.model_monitor import ModelMonitor
from adserving.router.model_router import ModelRouter

from .response_model import (
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    ModelStatsResponse,
    ReadinessResponse,
    ServiceInfoResponse,
)

logger = logging.getLogger(__name__)

# Global variables for dependency injection
model_manager: Optional[ModelManager] = None
model_router: Optional[ModelRouter] = None
monitor: Optional[ModelMonitor] = None
data_handler: Optional[DataHandler] = None
service_start_time: datetime = datetime.now()

# Service readiness tracking (set by the main service)
service_readiness: Dict[str, Any] = {
    "ready": False,
    "models_loaded": 0,
    "models_failed": 0,
    "initialization_complete": False,
}

# Create API Router for all endpoints
api_router = APIRouter()


def create_error_response(
    error_code: int, message: str, detail: str = "", request_id: str = ""
) -> ErrorResponse:
    """Create standardized error response"""
    return ErrorResponse(
        error_code=error_code,
        message=message,
        detail=detail,
        request_id=request_id or str(uuid.uuid4()),
    )


# Dependency functions
def get_model_manager() -> ModelManager:
    """Get model manager dependency"""
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    return model_manager


def get_model_router() -> ModelRouter:
    """Get model router dependency"""
    if model_router is None:
        raise HTTPException(status_code=503, detail="Model router not initialized")
    return model_router


def get_monitor() -> ModelMonitor:
    """Get monitor dependency"""
    if monitor is None:
        raise HTTPException(status_code=503, detail="Monitor not initialized")
    return monitor


def get_input_handler() -> DataHandler:
    """Get input handler dependency"""
    if data_handler is None:
        raise HTTPException(status_code=503, detail="Input handler not initialized")
    return data_handler


# Application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global service_start_time
    service_start_time = datetime.now()

    logger.info("Anomaly Detection Serve API starting up...")
    yield
    logger.info("Anomaly Detection Serve API shutting down...")


# Create FastAPI app
def create_app(api_prefix: str = "") -> FastAPI:
    """Create and configure FastAPI application"""
    # Create base FastAPI config
    api_config = {
        "title": "Anomaly Detection API",
        "description": "Enhanced MLOps serving system for hundreds of models",
        "version": get_config().api_version,
        "lifespan": lifespan,
    }

    # Add root_path only if api_prefix is provided
    if api_prefix and api_prefix.strip():
        api_config["root_path"] = api_prefix

    app = FastAPI(**api_config)

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=create_error_response(exc.status_code, str(exc.detail)).dict(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        # Extract detailed error messages from Pydantic validation errors
        error_details = []
        for error in exc.errors():
            field_path = " -> ".join(str(loc) for loc in error["loc"])
            error_msg = error["msg"]
            error_type = error["type"]

            # Format the error message with field information
            if error_type == "value_error":
                # Custom validator error messages (our Vietnamese messages)
                formatted_error = f"Lỗi tại trường '{field_path}': {error_msg}"
            elif error_type == "type_error":
                # Pydantic type errors
                formatted_error = (
                    f"Lỗi kiểu dữ liệu tại trường '{field_path}': {error_msg}"
                )
            elif error_type == "missing":
                # Missing required fields
                formatted_error = f"Thiếu trường bắt buộc: '{field_path}'"
            else:
                # Other validation errors
                formatted_error = (
                    f"Lỗi validation tại trường '{field_path}': {error_msg}"
                )

            error_details.append(formatted_error)

        # Create comprehensive error message
        main_message = "Dữ liệu đầu vào không hợp lệ"
        detail_message = "; ".join(error_details)

        return JSONResponse(
            status_code=422,
            content=create_error_response(422, main_message, detail_message).dict(),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content=create_error_response(
                500, "Internal server error", str(exc)
            ).dict(),
        )

    # Include the API router
    app.include_router(api_router)

    return app


# Global app variable - will be initialized by initialize_app_with_config
app: Optional[FastAPI] = None


def initialize_app_with_config(api_prefix: str = "") -> FastAPI:
    """Initialize app with configuration"""
    global app
    app = create_app(api_prefix)
    return app


# API Routes
@api_router.get("/", response_model=ServiceInfoResponse)
async def service_info(
    manager: ModelManager = Depends(get_model_manager),
    router: ModelRouter = Depends(get_model_router),
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
                "service_info": "/",
                "health_check": "/health",
                "readiness_check": "/ready",
                "predict": "/predict",
                "model_stats": "/models/stats",
                "model_info": "/models/{model_name}/info",
                "routing_stats": "/routing/stats",
                "cache_stats": "/cache/stats",
            },
            timestamp=datetime.now().isoformat(),
            description="Anomaly Detection serving system for"
            " hundreds of models with tiered loading and intelligent routing",
            features=[
                "Tiered model loading (Hot/Warm/Cold)",
                "Single endpoint routing with intelligent selection",
                "Batch processing capabilities",
                "Advanced monitoring and metrics",
                "Resource sharing and GPU optimization",
                "Prometheus metrics export",
                "Auto-scaling and capacity planning",
            ],
        )
    except Exception as e:
        logger.error(f"Error getting service info: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get service info: {str(e)}"
        )


@api_router.get("/health", response_model=HealthResponse)
async def health_check(
    manager: ModelManager = Depends(get_model_manager),
    router: ModelRouter = Depends(get_model_router),
    monitor: ModelMonitor = Depends(get_monitor),
):
    """Comprehensive health check"""
    try:
        uptime = datetime.now() - service_start_time
        uptime_str = str(uptime).split(".")[0]

        # Get comprehensive stats
        cache_stats = manager.get_cache_stats()
        dashboard_data = monitor.get_dashboard_data()

        models_loaded = cache_stats.get("total_models", 0)

        return HealthResponse(
            status="healthy",
            version=get_config().api_version,
            timestamp=datetime.now().isoformat(),
            models_loaded=models_loaded,
            uptime=uptime_str,
            cache_stats=cache_stats,
            deployment_stats=dashboard_data.get("deployments", {}),
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@api_router.get("/ready", response_model=ReadinessResponse)
async def readiness_check():
    """
    Readiness check endpoint for cold start optimization.

    Returns service readiness status including model loading progress.
    This endpoint helps clients determine when the service is fully initialized
    and ready to handle requests without cold start delays.
    """
    try:
        global service_readiness

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
                    f"Service fully ready."
                    f" All {models_loaded} models loaded successfully."
                )
            else:
                message = (
                    f"Service ready with warnings."
                    f" {models_loaded}/{total_models} models loaded successfully."
                )
        elif initialization_complete:
            status = "not_ready"
            message = (
                f"Service initialization complete but not ready."
                f" {models_loaded}/{total_models} models loaded."
            )
        else:
            status = "initializing"
            message = (
                f"Service still initializing. {models_loaded} models loaded so far."
            )

        return ReadinessResponse(
            ready=ready,
            status=status,
            timestamp=datetime.now().isoformat(),
            models_loaded=models_loaded,
            models_failed=models_failed,
            total_models=total_models,
            initialization_complete=initialization_complete,
            message=message,
        )
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return ReadinessResponse(
            ready=False,
            status="error",
            timestamp=datetime.now().isoformat(),
            models_loaded=0,
            models_failed=0,
            total_models=0,
            initialization_complete=False,
            message=f"Readiness check failed: {str(e)}",
        )


@api_router.post("/predict", response_model=APIResponse)
async def predict(
    request: PredictionRequest,
    handler: DataHandler = Depends(get_input_handler),
    router: ModelRouter = Depends(get_model_router),
    monitor: ModelMonitor = Depends(get_monitor),
):
    """
    Prediction endpoint supporting only the specified input format:
    {
        "ma_don_vi": "UBND.0019",
        "ma_bao_cao": "10628953",
        "ky_du_lieu": "2025-07-10",
        "data": [
            {
                "ma_tieu_chi": "CT_TONGCONG",
                "FN01": 1.23,
                "FN02": 45.6,
                "FN03": 212,
                "FN04": 3.14
            }
        ]
    }

    Model names are generated using pattern:
    [ma_don_vi]_[ma_bao_cao]_[ma_tieu_chi]_[fld_code]
    where fld_code is derived from the FNxx
    fields in the data items (e.g., FN01, FN02, etc.).
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        # Process and validate input
        processed_request = await handler.process_request(request)

        # Route request to appropriate model
        result = await router.route_request(processed_request)

        # Record metrics
        inference_time = time.time() - start_time
        success = result.get("status") == "success"
        model_name = result.get("model_name", "unknown")

        if hasattr(monitor, "metrics_collector"):
            monitor.metrics_collector.record_model_request(
                model_name=model_name, response_time=inference_time, success=success
            )

        # Format response
        response = await handler.format_response(result, request_id, inference_time)

        return response

    except Exception as e:
        logger.error(f"Prediction error for request {request_id}: {e}")

        # Record error metrics
        if hasattr(monitor, "metrics_collector"):
            monitor.metrics_collector.record_model_request(
                model_name="unknown",
                response_time=time.time() - start_time,
                success=False,
            )

        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@api_router.get("/models/stats", response_model=ModelStatsResponse)
async def get_model_stats(manager: ModelManager = Depends(get_model_manager)):
    """Get detailed model statistics"""
    try:
        cache_stats = manager.get_cache_stats()

        # Calculate statistics
        hot_models = cache_stats.get("hot_cache", {}).get("size", 0)
        warm_models = cache_stats.get("warm_cache", {}).get("size", 0)
        cold_models = cache_stats.get("cold_cache", {}).get("size", 0)
        total_models = cache_stats.get("total_models", 0)

        # Calculate cache hit rate (simplified)
        cache_hit_rate = 0.85  # This would be calculated from actual metrics
        avg_load_time = 1.2  # This would be calculated from actual metrics

        return ModelStatsResponse(
            total_models=total_models,
            hot_models=hot_models,
            warm_models=warm_models,
            cold_models=cold_models,
            cache_hit_rate=cache_hit_rate,
            avg_load_time=avg_load_time,
            model_breakdown={
                "hot": hot_models,
                "warm": warm_models,
                "cold": cold_models,
            },
            tier_distribution={
                "hot": hot_models,
                "warm": warm_models,
                "cold": cold_models,
            },
            hot_models_list=cache_stats.get("hot_cache", {}).get("models", []),
            warm_models_list=cache_stats.get("warm_cache", {}).get("models", []),
            cold_models_list=cache_stats.get("cold_cache", {}).get("models", []),
        )
    except Exception as e:
        logger.error(f"Error getting model stats: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get model stats: {str(e)}"
        )


@api_router.get("/models/{model_name}/info", response_model=ModelInfoResponse)
async def get_model_info(
    model_name: str, manager: ModelManager = Depends(get_model_manager)
):
    """Get detailed information about a specific model"""
    try:
        model_info = manager.get_model_info(model_name)

        if not model_info:
            raise HTTPException(
                status_code=404, detail=f"Model '{model_name}' not found or not loaded"
            )

        # Calculate success rate
        total_requests = model_info.success_count + model_info.error_count
        success_rate = (
            (model_info.success_count / total_requests) if total_requests > 0 else 0.0
        )

        # Convert timestamps to ISO format
        loaded_at = datetime.fromtimestamp(model_info.loaded_at).isoformat()
        last_accessed = datetime.fromtimestamp(model_info.last_accessed).isoformat()

        return ModelInfoResponse(
            model_name=model_info.model_name,
            model_version=model_info.model_version,
            model_uri=model_info.model_uri,
            loaded_at=loaded_at,
            last_accessed=last_accessed,
            access_count=model_info.access_count,
            tier=model_info.tier.value,
            memory_usage=model_info.memory_usage,
            avg_inference_time=model_info.avg_inference_time,
            error_count=model_info.error_count,
            success_count=model_info.success_count,
            success_rate=success_rate,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info for {model_name}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get model info: {str(e)}"
        )


@api_router.get("/routing/stats")
async def get_routing_stats(router: ModelRouter = Depends(get_model_router)):
    """Get routing statistics"""
    try:
        return router.get_routing_stats()
    except Exception as e:
        logger.error(f"Error getting routing stats: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get routing stats: {str(e)}"
        )


@api_router.get("/cache/stats")
async def get_cache_stats(manager: ModelManager = Depends(get_model_manager)):
    """Get cache statistics"""
    try:
        return manager.get_cache_stats()
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get cache stats: {str(e)}"
        )


@api_router.get("/models/deployment-stats")
async def get_deployment_stats(manager: ModelManager = Depends(get_model_manager)):
    """Get zero-downtime deployment statistics"""
    try:
        return manager.get_deployment_stats()
    except Exception as e:
        logger.error(f"Error getting deployment stats: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get deployment stats: {str(e)}"
        )


@api_router.get("/dashboard")
async def get_dashboard_data(monitor: ModelMonitor = Depends(get_monitor)):
    """Get comprehensive dashboard data"""
    try:
        return monitor.get_dashboard_data()
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get dashboard data: {str(e)}"
        )


@api_router.get("/metrics")
async def get_prometheus_metrics(monitor: ModelMonitor = Depends(get_monitor)):
    """Get Prometheus metrics"""
    try:
        metrics = monitor.get_prometheus_metrics()
        return {"metrics": metrics}
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


# Initialization function
def initialize_dependencies(
    model_mgr: ModelManager,
    router: ModelRouter,
    mon: ModelMonitor,
    handler: DataHandler,
):
    """Initialize global dependencies"""
    global model_manager, model_router, monitor, data_handler
    model_manager = model_mgr
    model_router = router
    monitor = mon
    data_handler = handler

    logger.info("API dependencies initialized successfully")


def update_service_readiness(
    ready: bool, models_loaded: int, models_failed: int, initialization_complete: bool
):
    """Update service readiness state from main service"""
    global service_readiness
    service_readiness.update(
        {
            "ready": ready,
            "models_loaded": models_loaded,
            "models_failed": models_failed,
            "initialization_complete": initialization_complete,
        }
    )
