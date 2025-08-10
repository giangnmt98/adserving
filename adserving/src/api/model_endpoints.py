from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ray import serve

from adserving.src.config.config_manager import get_config
from adserving.src.mlflow_utils.mlflow_parameter_updater import MLflowParameterUpdater
from adserving.src.mlflow_utils.mlflow_client import MLflowClient as WrappedMLflowClient
from adserving.src.api.response_model import ModelInfoResponse
from adserving.src.utils.logger import get_logger

logger = get_logger()
router = APIRouter()


def _get_serve_handle():
    try:
        return serve.get_app_handle("preloaded_model_server")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Serve not ready: {e}")


def _get_param_updater() -> MLflowParameterUpdater:
    # Tạo updater với client MLflow từ config (đúng kiểu đối số)
    try:
        cfg = get_config()
        client = WrappedMLflowClient(tracking_uri=cfg.mlflow.tracking_uri)  # TẠO CLIENT ĐÚNG
        return MLflowParameterUpdater(client)  # TRUYỀN CLIENT, KHÔNG TRUYỀN cfg.mlflow
    except Exception as e:
        logger.error(f"Cannot create MLflowParameterUpdater: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to initialize MLflow updater: {e}"
        )


# Request/Response models for parameter management
class ParameterUpdateRequest(BaseModel):
    parameters: Dict[str, float]
    comment: Optional[str] = None


class AnomalyThresholdUpdateRequest(BaseModel):
    threshold: float
    comment: Optional[str] = None


class RollbackRequest(BaseModel):
    model_name: str
    target_version: str


class ParameterValidationResponse(BaseModel):
    valid: bool
    errors: List[str] = []
    warnings: List[str] = []


class ParameterHistoryResponse(BaseModel):
    model_name: str
    updates: List[Dict]
    total_count: int


@router.get("/models/{model_name}/info", response_model=ModelInfoResponse)
async def get_model_info(model_name: str):
    """
    Lấy thông tin chi tiết model:
    - loaded status và version từ Serve
    - Thống kê cơ bản nếu monitor có cung cấp
    """
    try:
        handle = _get_serve_handle()
        details = await handle.get_model_details.remote(model_name)
        if not details or not isinstance(details, dict):
            raise HTTPException(status_code=404, detail="Model info unavailable")

        loaded = bool(details.get("loaded", False))
        model_version = details.get("model_version")
        # Nếu cần thêm thông tin (avg_inference_time, error/success counters) có thể lấy từ monitor
        avg_inference = 0.0
        error_count = 0
        success_count = 0

        return ModelInfoResponse(
            model_name=model_name,
            model_version=model_version or "",
            model_uri=f"models:/{model_name}/{model_version or 'Production'}",
            loaded_at=datetime.now().isoformat(),  # Không có thời điểm chính xác từ Serve → tạm thời now()
            last_accessed=datetime.now().isoformat(),
            access_count=0,
            memory_usage=0.0,
            avg_inference_time=avg_inference,
            error_count=error_count,
            success_count=success_count,
            success_rate=(
                (success_count / (error_count + success_count))
                if (error_count + success_count) > 0
                else 0.0
            ),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info for {model_name}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get model info: {str(e)}"
        )


@router.get("/models/production")
async def get_production_models():
    """
    Lấy danh sách các model Production hiện có từ Serve (đang preload) hoặc fallback MLflow.
    """
    try:
        handle = _get_serve_handle()
        names = await handle.list_models.remote()
        names = names or []

        return {
            "production_models": names,
            "count": len(names),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting production models: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get production models: {str(e)}"
        )


@router.get("/models/{model_name}/parameters")
async def get_model_parameters(model_name: str):
    """Lấy bộ tham số hiện tại của model (từ MLflow)."""
    try:
        updater = _get_param_updater()
        parameters = updater.get_current_parameters(model_name)
        if parameters is None:
            raise HTTPException(
                status_code=404, detail=f"Model '{model_name}' not found"
            )

        return {
            "model_name": model_name,
            "parameters": parameters,
            "timestamp": datetime.now().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting parameters for {model_name}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get model parameters: {str(e)}"
        )


@router.post("/models/{model_name}/parameters/validate")
async def validate_parameters(model_name: str, request: ParameterUpdateRequest):
    """Validate parameter updates trước khi apply."""
    try:
        updater = _get_param_updater()
        validation = updater.validate_parameter_update(model_name, request.parameters)
        return ParameterValidationResponse(
            valid=validation["valid"],
            errors=validation.get("errors", []),
            warnings=validation.get("warnings", []),
        )
    except Exception as e:
        logger.error(f"Error validating parameters for {model_name}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to validate parameters: {str(e)}"
        )


@router.put("/models/{model_name}/threshold")
async def update_model_threshold(model_name: str, request: AnomalyThresholdUpdateRequest):
    """Cập nhật threshold model trong MLflow."""
    try:
        if not 0.0 <= request.threshold <= 1.0:
            raise HTTPException(
                status_code=400, detail="Threshold must be between 0.0 and 1.0"
            )

        updater = _get_param_updater()
        success = updater.update_anomaly_threshold(
            model_name=model_name,
            new_threshold=request.threshold,
            comment=request.comment or f"API threshold update to {request.threshold}",
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to update threshold")

        return {
            "status": "success",
            "model_name": model_name,
            "new_threshold": request.threshold,
            "timestamp": datetime.now().isoformat(),
            "message": "Threshold updated successfully",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating threshold for {model_name}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to update threshold: {str(e)}"
        )


@router.put("/models/{model_name}/parameters")
async def update_model_parameters(model_name: str, request: ParameterUpdateRequest):
    """Cập nhật nhiều tham số cho model trong MLflow."""
    try:
        updater = _get_param_updater()
        validation = updater.validate_parameter_update(model_name, request.parameters)
        if not validation["valid"]:
            raise HTTPException(
                status_code=400,
                detail=f"Parameter validation failed: {validation['errors']}",
            )

        success = updater.create_parameter_version(
            model_name=model_name,
            parameter_updates=request.parameters,
            comment=request.comment or "API batch parameter update",
        )
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update parameters")

        return {
            "status": "success",
            "model_name": model_name,
            "updated_parameters": request.parameters,
            "timestamp": datetime.now().isoformat(),
            "message": "Parameters updated successfully",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating parameters for {model_name}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to update parameters: {str(e)}"
        )


@router.post("/models/rollback")
async def rollback_model_version(request: RollbackRequest):
    """Rollback model về version trước trong MLflow."""
    try:
        updater = _get_param_updater()
        model_name = request.model_name
        target_version = request.target_version
        success = updater.rollback_to_version(
            model_name= model_name,
            target_version= target_version,
        )
        if not success:
            raise HTTPException(status_code=500, detail="Failed to rollback model")

        return {
            "status": "success",
            "model_name": model_name,
            "target_version": target_version,
            "timestamp": datetime.now().isoformat(),
            "message": f"Model rolled back to version {request.target_version}",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rolling back {model_name}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to rollback model: {str(e)}"
        )


@router.get("/models/{model_name}/history", response_model=ParameterHistoryResponse)
async def get_parameter_history(model_name: str, limit: int = 10):
    """Xem lịch sử cập nhật tham số trên MLflow."""
    try:
        updater = _get_param_updater()
        history = updater.get_parameter_update_history(model_name, limit=limit)
        return ParameterHistoryResponse(
            model_name=model_name,
            updates=history or [],
            total_count=len(history) if history else 0,
        )
    except Exception as e:
        logger.error(f"Error getting history for {model_name}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get parameter history: {str(e)}"
        )


@router.post("/models/{model_name}/warm")
async def warm_model(model_name: str):
    """Warm (load) model vào cache của Serve nếu chưa có."""
    try:
        handle = _get_serve_handle()
        result = await handle.warm_model.remote(model_name)
        if not isinstance(result, dict):
            raise HTTPException(status_code=500, detail="Invalid warm result")
        status = result.get("status", "error")
        if status == "error":
            raise HTTPException(status_code=500, detail=result.get("message", "Warm failed"))
        return {
            "status": "success",
            "model_name": model_name,
            "model_version": result.get("model_version"),
            "timestamp": datetime.now().isoformat(),
            "message": result.get("message", "Model warmed up successfully"),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error warming up model {model_name}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to warm up model: {str(e)}"
        )


@router.delete("/models/{model_name}/cache")
async def evict_model_from_cache(model_name: str):
    """Evict model khỏi cache của Serve."""
    try:
        handle = _get_serve_handle()
        result = await handle.evict_model.remote(model_name)
        if not isinstance(result, dict):
            raise HTTPException(status_code=500, detail="Invalid evict result")
        status = result.get("status", "error")
        message = result.get("message", "Evict failed")
        return {
            "status": status,
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "message": message,
        }
    except Exception as e:
        logger.error(f"Error evicting model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to evict model: {str(e)}")
