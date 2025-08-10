"""
Core Service Endpoints (Health, Readiness, Service Info)
"""

from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from ray import serve

from adserving.src.config.config_manager import get_config
from adserving.src.utils.logger import get_logger

from .api_dependencies import (
    service_start_time,
    service_readiness,  # THÊM: fallback trạng thái sẵn sàng
)

from .response_model import HealthResponse, ServiceInfoResponse

logger = get_logger()
router = APIRouter()

# (Phần còn lại giữ nguyên: service_info, health_check ...)


@router.get("/", response_model=ServiceInfoResponse)
async def service_info(
):
    """Get service information and status"""
    try:
        uptime = datetime.now() - service_start_time
        uptime_str = str(uptime).split(".")[0]  # Remove microseconds

        return ServiceInfoResponse(
            service="Anomaly Detection API",
            version=get_config().api_version,
            status="running",
            uptime=uptime_str,
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
):
    """
    Comprehensive health check phù hợp hệ thống:
    - Ưu tiên trạng thái Ray Serve app 'preloaded_model_server'.
    - Lấy số model đã preload trực tiếp từ Serve nếu có.
    - Rơi về service_readiness nếu Serve chưa sẵn sàng.
    - Trả HTTP 200 / 206 / 503 phù hợp.
    """
    try:
        cfg = get_config()
        uptime = datetime.now() - service_start_time
        uptime_str = str(uptime).split(".")[0]

        # 1) Mặc định các thống kê rỗng/an toàn
        cache_stats: Dict[str, Any] = {}
        deployment_stats: Dict[str, Any] = {}
        models_loaded = 0

        # 2) Kiểm tra Ray Serve app readiness
        serve_ready = False
        try:
            handle = serve.get_app_handle("preloaded_model_server")
            # Thử gọi nhanh (đồng bộ) một RPC nhẹ để đánh giá sẵn sàng
            # Ở đây gọi list_models (nếu raise sẽ vào except)
            names = await handle.list_models.remote()
            if names is not None:
                serve_ready = True
                models_loaded = len(names)
                deployment_stats["preloaded_model_server"] = {
                    "status": "running",
                    "models_loaded": models_loaded,
                }
        except Exception as e:
            # Serve chưa sẵn sàng hoặc có lỗi
            logger.debug(f"Serve app not ready: {e}")
            deployment_stats["preloaded_model_server"] = {
                "status": "not_ready",
                "error": str(e),
            }

        # 4) Nếu serve_ready = False, dùng service_readiness làm fallback
        if not serve_ready:
            models_loaded = int(service_readiness.get("models_loaded", 0))
            deployment_stats.setdefault("preloaded_model_server", {})
            deployment_stats["preloaded_model_server"].update(
                {
                    "status": "not_ready",
                    "initialization_complete": service_readiness.get(
                        "initialization_complete", False
                    ),
                    "models_failed": service_readiness.get("models_failed", 0),
                }
            )

        # 5) Xác định status chung
        total_models = (
            int(service_readiness.get("models_loaded", 0))
            + int(service_readiness.get("models_failed", 0))
        )
        has_failed = int(service_readiness.get("models_failed", 0)) > 0

        # Quy tắc trả mã:
        # - 503 nếu Serve chưa sẵn sàng.
        # - 206 nếu có failed_models > 0 hoặc có cảnh báo nhẹ (Serve sẵn sàng nhưng có lỗi nhỏ).
        # - 200 nếu hoàn toàn healthy.
        if not serve_ready:
            status_code = 503
            status = "unhealthy"
        elif has_failed and total_models > 0:
            status_code = 206
            status = "partially_healthy"
        else:
            status_code = 200
            status = "healthy"

        response = HealthResponse(
            status=status,
            version=cfg.api_version,
            timestamp=datetime.now().isoformat(),
            models_loaded=models_loaded,
            uptime=uptime_str,
            deployment_stats=deployment_stats,
        )

        if status_code == 200:
            return response
        else:
            # Trả JSONResponse để gán status code tuỳ biến (206 / 503)
            return JSONResponse(status_code=status_code, content=response.model_dump())

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

