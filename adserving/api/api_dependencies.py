"""
API Dependencies and Global State Management
Enhanced with Unified Error Handling
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import HTTPException

from adserving.core.model_manager import ModelManager
from adserving.datahandler.data_handler import DataHandler
from adserving.monitoring.model_monitor import ModelMonitor
from adserving.router.model_router import ModelRouter

from .response_model import ErrorResponse

logger = logging.getLogger(__name__)

# Global variables for dependency injection
model_manager: Optional[ModelManager] = None
model_router: Optional[ModelRouter] = None
monitor: Optional[ModelMonitor] = None
data_handler: Optional[DataHandler] = None
tier_orchestrator: Optional[Any] = None
use_tier_based_deployment: bool = False
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
            status_code=503,
            detail="Service unavailable: Model manager not initialized"
        )
    return model_manager


def get_model_router() -> ModelRouter:
    """Get model router dependency with proper error handling"""
    if model_router is None:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: Model router not initialized"
        )
    return model_router


def get_unified_model_router():
    """
    Get unified model router - enhanced wrapper around existing ModelRouter.

    Returns UnifiedModelRouter if enhanced error handling is enabled,
    otherwise returns regular ModelRouter for backward compatibility.
    """
    global _unified_router_cache

    if not _enhanced_error_handling_enabled:
        # Return regular router for backward compatibility
        return get_model_router()

    if _unified_router_cache is None:
        try:
            # Lazy import to avoid circular dependencies
            from adserving.router.unified_model_router import UnifiedModelRouter
            from adserving.router.unified_model_router_helpers import (
                UnifiedModelRouterHelpers
            )

            existing_router = get_model_router()
            logger.info("Creating unified router wrapper")

            _unified_router_cache = UnifiedModelRouter(existing_router, logger)

            # Inject helper methods
            _unified_router_cache._create_model_name_error = (
                UnifiedModelRouterHelpers.create_model_name_error
            )
            _unified_router_cache._create_model_not_found_error = (
                UnifiedModelRouterHelpers.create_model_not_found_error
            )
            _unified_router_cache._create_model_unavailable_error = (
                UnifiedModelRouterHelpers.create_model_unavailable_error
            )
            _unified_router_cache._create_model_load_error = (
                UnifiedModelRouterHelpers.create_model_load_error
            )
            _unified_router_cache._create_deployment_unavailable_error = (
                UnifiedModelRouterHelpers.create_deployment_unavailable_error
            )

        except ImportError as e:
            logger.warning(
                f"Enhanced error handling components not available: {e}"
            )
            logger.warning("Falling back to regular ModelRouter")
            return get_model_router()

    return _unified_router_cache


def get_monitor() -> ModelMonitor:
    """Get monitor dependency with proper error handling"""
    if monitor is None:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: Monitor not initialized"
        )
    return monitor


def get_input_handler() -> DataHandler:
    """Get input handler dependency with proper error handling"""
    if data_handler is None:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: Input handler not initialized"
        )
    return data_handler


def get_tier_orchestrator():
    """Get tier orchestrator dependency with proper error handling"""
    if tier_orchestrator is None:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: Tier orchestrator not initialized",
        )
    return tier_orchestrator


def is_tier_based_deployment_enabled() -> bool:
    """Check if tier-based deployment is enabled"""
    return use_tier_based_deployment


def is_enhanced_error_handling_enabled() -> bool:
    """Check if enhanced error handling is enabled"""
    return _enhanced_error_handling_enabled


def initialize_dependencies(
    model_mgr: ModelManager,
    router: ModelRouter,
    mon: ModelMonitor,
    handler: DataHandler,
    orchestrator: Optional[Any] = None,
    tier_based: bool = False,
) -> None:
    """Initialize global dependencies"""
    global model_manager, model_router, monitor, data_handler
    global tier_orchestrator, use_tier_based_deployment

    model_manager = model_mgr
    model_router = router
    monitor = mon
    data_handler = handler
    tier_orchestrator = orchestrator
    use_tier_based_deployment = tier_based

    logger.info(
        f"API dependencies initialized successfully "
        f"(tier-based: {tier_based})"
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

    logger.info(
        f"Enhanced error handling: "
        f"{'enabled' if enabled else 'disabled'}"
    )


def get_request_context_extractor():
    """Get request context extractor for enhanced error handling."""
    try:
        from adserving.utils.context_extractor import RequestContextExtractor
        return RequestContextExtractor
    except ImportError as e:
        logger.warning(f"Context extractor not available: {e}")
        return None


def update_service_readiness(
    ready: bool,
    models_loaded: int,
    models_failed: int,
    initialization_complete: bool
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
        "enhanced_error_handling": _enhanced_error_handling_enabled,
        "service_start_time": service_start_time.isoformat(),
        "components_initialized": {
            "model_manager": model_manager is not None,
            "model_router": model_router is not None,
            "monitor": monitor is not None,
            "data_handler": data_handler is not None,
            "tier_orchestrator": tier_orchestrator is not None,
            "unified_router": _unified_router_cache is not None,
        }
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