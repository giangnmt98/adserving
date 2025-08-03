"""
Unified Model Exception System.

Hệ thống exception thống nhất cho tất cả model-related errors.
"""

import time
from typing import Dict, List, Optional, Any


class UnifiedModelError(Exception):
    """Base exception for all model-related errors."""

    def __init__(
            self,
            message: str,
            context: Dict = None,
            suggestions: List[str] = None
    ):
        super().__init__(message)
        self.context = context or {}
        self.suggestions = suggestions or []
        self.timestamp = time.time()
        self.error_type = self.__class__.__name__

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for API responses."""
        return {
            "error_type": self.error_type,
            "message": str(self),
            "context": self.context,
            "suggestions": self.suggestions,
            "timestamp": self.timestamp,
        }


class ModelNotAvailableError(UnifiedModelError):
    """Model không có sẵn - cần load hoặc deploy."""

    def __init__(
            self,
            model_name: str,
            deployment_type: str = None,
            **kwargs
    ):
        self.model_name = model_name
        self.deployment_type = deployment_type

        message = f"Model '{model_name}' is not available"
        if deployment_type:
            message += f" in {deployment_type} deployment"

        super().__init__(message, **kwargs)


class ModelLoadFailedError(UnifiedModelError):
    """Model load thất bại."""

    def __init__(self, model_name: str, load_error: str = None, **kwargs):
        self.model_name = model_name
        self.load_error = load_error

        message = f"Failed to load model '{model_name}'"
        if load_error:
            message += f": {load_error}"

        super().__init__(message, **kwargs)


class ModelNotFoundError(UnifiedModelError):
    """Model không tồn tại trong registry."""

    def __init__(
            self,
            model_name: str = None,
            registry_type: str = None,
            **kwargs
    ):
        self.model_name = model_name
        self.registry_type = registry_type

        if model_name:
            message = f"Model '{model_name}' not found"
            if registry_type:
                message += f" in {registry_type} registry"
        else:
            message = "Could not determine model name from request"

        super().__init__(message, **kwargs)


class SingleElementModelError(ModelNotFoundError):
    """Lỗi cụ thể cho single element request."""

    def __init__(
            self,
            data_element: Dict,
            attempted_model: str = None,
            **kwargs
    ):
        self.data_element = data_element
        self.attempted_model = attempted_model

        ma_tieu_chi = data_element.get("ma_tieu_chi", "UNKNOWN")
        fn_fields = [k for k in data_element.keys() if k.startswith("FN")]

        message = (
            f"Không tìm thấy model cho tiêu chí '{ma_tieu_chi}' "
            f"với các trường: {fn_fields}"
        )

        if attempted_model:
            message += f" (Model được tạo: {attempted_model})"

        context = kwargs.get('context', {})
        context.update({
            "ma_tieu_chi": ma_tieu_chi,
            "fn_fields": fn_fields,
            "attempted_model": attempted_model,
            "data_element": data_element,
            "is_single_element": True,
        })

        super().__init__(attempted_model, context=context, **kwargs)


class DeploymentUnavailableError(UnifiedModelError):
    """Deployment service không khả dụng."""

    def __init__(
            self,
            deployment_name: str = None,
            service_type: str = None,
            **kwargs
    ):
        self.deployment_name = deployment_name
        self.service_type = service_type

        message = "Deployment service unavailable"
        if deployment_name:
            message = f"Deployment '{deployment_name}' unavailable"
        if service_type:
            message += f" ({service_type})"

        super().__init__(message, **kwargs)


class TierRoutingError(UnifiedModelError):
    """Lỗi trong tier-based routing."""

    def __init__(self, model_name: str, tier: str = None, **kwargs):
        self.model_name = model_name
        self.tier = tier

        message = f"Failed to route model '{model_name}'"
        if tier:
            message += f" to tier '{tier}'"

        super().__init__(message, **kwargs)


class PoolRoutingError(UnifiedModelError):
    """Lỗi trong pool-based routing."""

    def __init__(self, model_name: str, pool_name: str = None, **kwargs):
        self.model_name = model_name
        self.pool_name = pool_name

        message = f"Failed to route model '{model_name}'"
        if pool_name:
            message += f" to pool '{pool_name}'"

        super().__init__(message, **kwargs)