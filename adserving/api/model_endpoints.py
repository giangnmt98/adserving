"""
Model Management Endpoints
"""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException

from adserving.core.model_manager import ModelManager
from adserving.monitoring.model_monitor import ModelMonitor
from adserving.router.model_router import ModelRouter

from .api_dependencies import get_model_manager, get_model_router, get_monitor
from .response_model import ModelInfoResponse, ModelStatsResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/models/stats", response_model=ModelStatsResponse)
async def get_model_stats(manager: ModelManager = Depends(get_model_manager)):
    """Get detailed model statistics"""
    try:
        cache_stats = manager.get_cache_stats()

        # Calculate statistics
        hot_models = cache_stats.get("hot_cache", {}).get("size", 0)
        warm_models = cache_stats.get("warm_cache", {}).get("size", 0)
        cold_models = cache_stats.get("cold_cache", {}).get("size", 0)
        total_models = cache_stats.get("total_models", 0)

        # These would be calculated from actual metrics in production
        cache_hit_rate = 0.85
        avg_load_time = 1.2

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


@router.get("/routing/stats")
async def get_routing_stats(model_router: ModelRouter = Depends(get_model_router)):
    """Get routing statistics"""
    try:
        return model_router.get_routing_stats()
    except Exception as e:
        logger.error(f"Error getting routing stats: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get routing stats: {str(e)}"
        )


@router.get("/cache/stats")
async def get_cache_stats(manager: ModelManager = Depends(get_model_manager)):
    """Get cache statistics"""
    try:
        return manager.get_cache_stats()
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get cache stats: {str(e)}"
        )


@router.get("/models/deployment-stats")
async def get_deployment_stats(manager: ModelManager = Depends(get_model_manager)):
    """Get zero-downtime deployment statistics"""
    try:
        return manager.get_deployment_stats()
    except Exception as e:
        logger.error(f"Error getting deployment stats: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get deployment stats: {str(e)}"
        )


@router.get("/dashboard")
async def get_dashboard_data(monitor: ModelMonitor = Depends(get_monitor)):
    """Get comprehensive dashboard data"""
    try:
        return monitor.get_dashboard_data()
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get dashboard data: {str(e)}"
        )


@router.get("/metrics")
async def get_prometheus_metrics(monitor: ModelMonitor = Depends(get_monitor)):
    """Get Prometheus metrics"""
    try:
        metrics = monitor.get_prometheus_metrics()
        return {"metrics": metrics}
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")
