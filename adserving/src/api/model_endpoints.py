"""
Model Management Endpoints
"""

from datetime import datetime
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from adserving.src.core.model_manager import ModelManager
from adserving.src.core.utils.mlflow_parameter_updater import MLflowParameterUpdater
from adserving.src.monitoring.model_monitor import ModelMonitor
from adserving.src.router.model_router import ModelRouter
from adserving.src.utils.logger import get_logger

from .api_dependencies import get_model_manager, get_model_router, get_monitor
from .response_model import ModelInfoResponse, ModelStatsResponse

logger = get_logger()

router = APIRouter()


# Request/Response models for parameter management
class ParameterUpdateRequest(BaseModel):
    """Request model for parameter updates"""

    parameters: Dict[str, float]
    comment: Optional[str] = None


class AnomalyThresholdUpdateRequest(BaseModel):
    """Request model for threshold updates"""

    threshold: float
    comment: Optional[str] = None


class RollbackRequest(BaseModel):
    """Request model for version rollback"""

    target_version: str
    comment: Optional[str] = None


class ParameterValidationResponse(BaseModel):
    """Response model for parameter validation"""

    valid: bool
    errors: List[str] = []
    warnings: List[str] = []


class ParameterHistoryResponse(BaseModel):
    """Response model for parameter update history"""

    model_name: str
    updates: List[Dict]
    total_count: int


@router.get("/models/{model_name}/info", response_model=ModelInfoResponse)
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


# New parameter management endpoints
@router.get("/models/production")
async def get_production_models(manager: ModelManager = Depends(get_model_manager)):
    """Get list of all production models"""
    try:
        models = manager.get_production_models()
        return {
            "production_models": models,
            "count": len(models),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting production models: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get production models: {str(e)}"
        )


@router.get("/models/{model_name}/parameters")
async def get_model_parameters(
    model_name: str, manager: ModelManager = Depends(get_model_manager)
):
    """Get current parameters for a specific model"""
    try:
        parameter_updater = MLflowParameterUpdater(manager.mlflow_client)
        parameters = parameter_updater.get_current_parameters(model_name)

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
async def validate_parameters(
    model_name: str,
    request: ParameterUpdateRequest,
    manager: ModelManager = Depends(get_model_manager),
):
    """Validate parameter updates before applying them"""
    try:
        parameter_updater = MLflowParameterUpdater(manager.mlflow_client)
        validation = parameter_updater.validate_parameter_update(
            model_name, request.parameters
        )

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
async def update_model_threshold(
    model_name: str,
    request: AnomalyThresholdUpdateRequest,
    manager: ModelManager = Depends(get_model_manager),
):
    """Update an anomaly threshold for a specific model"""
    try:
        parameter_updater = MLflowParameterUpdater(manager.mlflow_client)

        # Validate threshold value
        if not 0.0 <= request.threshold <= 1.0:
            raise HTTPException(
                status_code=400, detail="Threshold must be between 0.0 and 1.0"
            )

        success = parameter_updater.update_anomaly_threshold(
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
async def update_model_parameters(
    model_name: str,
    request: ParameterUpdateRequest,
    manager: ModelManager = Depends(get_model_manager),
):
    """Update multiple parameters for a specific model"""
    try:
        parameter_updater = MLflowParameterUpdater(manager.mlflow_client)

        # Validate parameters first
        validation = parameter_updater.validate_parameter_update(
            model_name, request.parameters
        )

        if not validation["valid"]:
            raise HTTPException(
                status_code=400,
                detail=f"Parameter validation failed: {validation['errors']}",
            )

        # Perform update
        success = parameter_updater.create_parameter_version(
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


@router.post("/models/{model_name}/rollback")
async def rollback_model_version(
    model_name: str,
    request: RollbackRequest,
    manager: ModelManager = Depends(get_model_manager),
):
    """Rollback model to a previous version"""
    try:
        parameter_updater = MLflowParameterUpdater(manager.mlflow_client)

        success = parameter_updater.rollback_to_version(
            model_name=model_name,
            target_version=request.target_version,
            comment=request.comment
            or f"API rollback to version {request.target_version}",
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to rollback model")

        return {
            "status": "success",
            "model_name": model_name,
            "target_version": request.target_version,
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
async def get_parameter_history(
    model_name: str, limit: int = 10, manager: ModelManager = Depends(get_model_manager)
):
    """Get parameter update history for a model"""
    try:
        parameter_updater = MLflowParameterUpdater(manager.mlflow_client)
        history = parameter_updater.get_parameter_update_history(
            model_name, limit=limit
        )

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


@router.post("/models/batch-threshold-update")
async def batch_threshold_update(
    updates: Dict[str, float],
    comment: Optional[str] = None,
    manager: ModelManager = Depends(get_model_manager),
):
    """Update thresholds for multiple models in batch"""
    try:
        parameter_updater = MLflowParameterUpdater(manager.mlflow_client)
        results = {}

        for model_name, threshold in updates.items():
            try:
                # Validate threshold
                if not 0.0 <= threshold <= 1.0:
                    results[model_name] = {
                        "status": "error",
                        "message": "Threshold must be between 0.0 and 1.0",
                    }
                    continue

                # Validate parameters
                validation = parameter_updater.validate_parameter_update(
                    model_name, {"anomaly_threshold": threshold}
                )

                if not validation["valid"]:
                    results[model_name] = {
                        "status": "error",
                        "message": f"Validation failed: {validation['errors']}",
                    }
                    continue

                # Update
                success = parameter_updater.update_anomaly_threshold(
                    model_name,
                    threshold,
                    comment=comment or f"Batch threshold update to {threshold}",
                )

                results[model_name] = {
                    "status": "success" if success else "error",
                    "new_threshold": threshold,
                    "message": "Updated successfully" if success else "Update failed",
                }

            except Exception as e:
                results[model_name] = {"status": "error", "message": str(e)}

        # Summary
        success_count = sum(1 for r in results.values() if r["status"] == "success")
        total_count = len(results)

        return {
            "status": "completed",
            "summary": {
                "total_models": total_count,
                "successful_updates": success_count,
                "failed_updates": total_count - success_count,
            },
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error in batch threshold update: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to perform batch update: {str(e)}"
        )


@router.post("/models/{model_name}/warm")
async def warm_model(
    model_name: str, manager: ModelManager = Depends(get_model_manager)
):
    """Warm up a model (preload into cache)"""
    try:
        model_info = await manager.load_model_async(model_name)

        if not model_info:
            raise HTTPException(
                status_code=404, detail=f"Failed to load model '{model_name}'"
            )

        return {
            "status": "success",
            "model_name": model_name,
            "model_version": model_info.model_version,
            "tier": model_info.tier.value,
            "timestamp": datetime.now().isoformat(),
            "message": "Model warmed up successfully",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error warming up model {model_name}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to warm up model: {str(e)}"
        )


@router.delete("/models/{model_name}/cache")
async def evict_model_from_cache(
    model_name: str, manager: ModelManager = Depends(get_model_manager)
):
    """Evict a model from cache"""
    try:
        removed = manager.cache.remove(model_name)

        return {
            "status": "success" if removed else "not_found",
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "message": f"Model "
            f"{'removed from cache' if removed else 'not found in cache'}",
        }
    except Exception as e:
        logger.error(f"Error evicting model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to evict model: {str(e)}")

@router.get("/metrics")
async def get_prometheus_metrics(monitor: ModelMonitor = Depends(get_monitor)):
    """Get Prometheus metrics"""
    try:
        metrics = monitor.get_prometheus_metrics()
        return {"metrics": metrics}
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")
