# Python
"""
API Dependencies and Global State Management
Enhanced with Unified Error Handling
"""

from datetime import datetime
from typing import Optional, Dict, Any

from fastapi import HTTPException

from adserving.src.datahandler.data_handler import DataHandler
from adserving.src.utils.logger import get_logger

logger = get_logger()

# Global variables for dependency injection
data_handler: Optional[DataHandler] = None
service_start_time: datetime = datetime.now()

# THÊM: Service readiness state (fallback khi Serve chưa sẵn sàng)
service_readiness: Dict[str, Any] = {
    "ready": False,
    "models_loaded": 0,
    "models_failed": 0,
    "initialization_complete": False,
    "last_update": service_start_time.isoformat(),
}


def update_service_readiness(
    *,
    ready: bool,
    models_loaded: int,
    models_failed: int,
    initialization_complete: bool,
) -> None:
    """Update service readiness state."""
    service_readiness.update(
        {
            "ready": bool(ready),
            "models_loaded": int(models_loaded),
            "models_failed": int(models_failed),
            "initialization_complete": bool(initialization_complete),
            "last_update": datetime.now().isoformat(),
        }
    )
    logger.info(
        f"Readiness updated: ready={ready}, loaded={models_loaded}, "
        f"failed={models_failed}, init_complete={initialization_complete}"
    )


def get_input_handler() -> DataHandler:
    """Get input handler dependency with proper error handling"""
    if data_handler is None:
        raise HTTPException(
            status_code=503, detail="Service unavailable: Input handler not initialized"
        )
    return data_handler


def initialize_dependencies(
    handler: DataHandler,
) -> None:
    """Initialize global dependencies"""
    global monitor, data_handler
    data_handler = handler

    # Reset readiness về trạng thái khởi tạo
    update_service_readiness(
        ready=False,
        models_loaded=0,
        models_failed=0,
        initialization_complete=False,
    )
