"""
API Module for Anomaly Detection Serve
Refactored for PEP8 compliance and better maintainability
"""

from .api_dependencies import (
    initialize_dependencies,
    update_service_readiness,
)
from .api_routes import initialize_app_with_config

__all__ = [
    "initialize_dependencies",
    "update_service_readiness",
    "initialize_app_with_config",
]
