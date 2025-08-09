"""
API Dependencies and Global State Management
Enhanced with Unified Error Handling
"""

import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import HTTPException

from adserving.src.core.model_manager import ModelManager
from adserving.src.core.ray_model_deployment_manager import RayModelDeploymentManager
from adserving.src.datahandler.data_handler import DataHandler
from adserving.src.monitoring.model_monitor import ModelMonitor
from adserving.src.router.model_router import ModelRouter
from adserving.src.utils.logger import get_logger

from .response_model import ErrorResponse

logger = get_logger()

# Global variables for dependency injection
model_manager: Optional[ModelManager] = None
model_router: Optional[ModelRouter] = None
monitor: Optional[ModelMonitor] = None
data_handler: Optional[DataHandler] = None
ray_deployment_manager: Optional[RayModelDeploymentManager] = None
tier_orchestrator: Optional[Any] = None
use_tier_based_deployment: bool = False
ultra_scale_orchestrator: Optional[Any] = None
use_ultra_scale_deployment: bool = False
service_start_time: datetime = datetime.now()

# Enhanced error handling components (lazy loaded)
_unified_router_cache: Optional[Any] = None
_enhanced_error_handling_enabled: bool = False

# Service readiness tracking
service_readiness: Dict[str, Any] = {
    "ready": False,
    "models_loaded": 0,
    "models_failed": 0,
    "initialization_complete": False,
}


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


def get_model_manager() -> ModelManager:
    """Get model manager dependency with proper error handling"""
    if model_manager is None:
        raise HTTPException(
            status_code=503, detail="Service unavailable: Model manager not initialized"
        )
    return model_manager


def get_model_router() -> ModelRouter:
    """Get model router dependency with proper error handling"""
    if model_router is None:
        raise HTTPException(
            status_code=503, detail="Service unavailable: Model router not initialized"
        )
    return model_router


def get_monitor() -> ModelMonitor:
    """Get monitor dependency with proper error handling"""
    if monitor is None:
        raise HTTPException(
            status_code=503, detail="Service unavailable: Monitor not initialized"
        )
    return monitor


def get_input_handler() -> DataHandler:
    """Get input handler dependency with proper error handling"""
    if data_handler is None:
        raise HTTPException(
            status_code=503, detail="Service unavailable: Input handler not initialized"
        )
    return data_handler


def get_ray_deployment_manager() -> RayModelDeploymentManager:
    """Get Ray deployment manager dependency with proper error handling"""
    if ray_deployment_manager is None:
        raise HTTPException(
            status_code=503, detail="Service unavailable: Ray deployment manager not initialized"
        )
    return ray_deployment_manager


def get_tier_orchestrator():
    """Get tier orchestrator dependency - DISABLED (tier deployment removed)"""
    # Tier deployment mechanism has been completely removed
    # Return None instead of raising exception to prevent runtime errors
    return None


def get_ultra_scale_orchestrator():
    """Get ultra-scale deployment orchestrator dependency"""
    if ultra_scale_orchestrator is None:
        raise HTTPException(
            status_code=503, detail="Service unavailable: Ultra-scale orchestrator not initialized"
        )
    return ultra_scale_orchestrator


def is_tier_based_deployment_enabled() -> bool:
    """Check if tier-based deployment is enabled"""
    return use_tier_based_deployment


def is_ultra_scale_deployment_enabled() -> bool:
    """Check if ultra-scale deployment is enabled"""
    return use_ultra_scale_deployment


def is_enhanced_error_handling_enabled() -> bool:
    """Check if enhanced error handling is enabled"""
    return _enhanced_error_handling_enabled


def initialize_dependencies(
    model_mgr: ModelManager,
    router: ModelRouter,
    mon: ModelMonitor,
    handler: DataHandler,
    ray_mgr: Optional[RayModelDeploymentManager] = None,
    orchestrator: Optional[Any] = None,
    tier_based: bool = False,
    ultra_orchestrator: Optional[Any] = None,
    ultra_scale: bool = False,
) -> None:
    """Initialize global dependencies"""
    global model_manager, model_router, monitor, data_handler
    global ray_deployment_manager, tier_orchestrator, use_tier_based_deployment
    global ultra_scale_orchestrator, use_ultra_scale_deployment

    model_manager = model_mgr
    model_router = router
    monitor = mon
    data_handler = handler
    ray_deployment_manager = ray_mgr
    tier_orchestrator = orchestrator
    use_tier_based_deployment = tier_based
    ultra_scale_orchestrator = ultra_orchestrator
    use_ultra_scale_deployment = ultra_scale

    logger.info(
        f"API dependencies initialized successfully "
        f"(tier-based: {tier_based}, ray-deployment: {ray_mgr is not None}, "
        f"ultra-scale: {ultra_scale})"
    )


def enable_enhanced_error_handling(enabled: bool = True) -> None:
    """
    Enable or disable enhanced error handling features.

    Args:
        enabled: Whether to enable enhanced error handling
    """
    global _enhanced_error_handling_enabled, _unified_router_cache

    _enhanced_error_handling_enabled = enabled

    # Clear cache to force re-creation with new settings
    _unified_router_cache = None

    logger.info(f"Enhanced error handling: " f"{'enabled' if enabled else 'disabled'}")


def get_request_context_extractor():
    """Get request context extractor for enhanced error handling."""
    try:
        from adserving.utils.context_extractor import RequestContextExtractor

        return RequestContextExtractor
    except ImportError as e:
        logger.warning(f"Context extractor not available: {e}")
        return None


def update_service_readiness(
    ready: bool, models_loaded: int, models_failed: int, initialization_complete: bool
) -> None:
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


def get_service_info() -> Dict[str, Any]:
    """Get comprehensive service information"""
    info = {
        "service_readiness": service_readiness.copy(),
        "tier_based_deployment": use_tier_based_deployment,
        "ultra_scale_deployment": use_ultra_scale_deployment,
        "enhanced_error_handling": _enhanced_error_handling_enabled,
        "service_start_time": service_start_time.isoformat(),
        "components_initialized": {
            "model_manager": model_manager is not None,
            "model_router": model_router is not None,
            "monitor": monitor is not None,
            "data_handler": data_handler is not None,
            "ray_deployment_manager": ray_deployment_manager is not None,
            "tier_orchestrator": tier_orchestrator is not None,
            "ultra_scale_orchestrator": ultra_scale_orchestrator is not None,
            "unified_router": _unified_router_cache is not None,
        },
    }

    # Add routing stats if available
    try:
        if _unified_router_cache:
            info["routing_stats"] = _unified_router_cache.get_routing_stats()
        elif model_router:
            info["routing_stats"] = model_router.get_routing_stats()
    except Exception as e:
        logger.debug(f"Could not get routing stats: {e}")
        info["routing_stats"] = {"error": "stats unavailable"}

    return info


# Backward compatibility aliases
should_use_tier_based_deployment = is_tier_based_deployment_enabled
