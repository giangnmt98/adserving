"""
Unified Model Router - Enhanced từ ModelRouter hiện tại.

Sử dụng lại logic có sẵn và thêm unified error handling.
"""

import logging
import time
from typing import Any, Dict, Optional

from adserving.exceptions.unified_model_exceptions import (
    SingleElementModelError,
    ModelNotAvailableError,
    ModelNotFoundError,
    DeploymentUnavailableError,
    UnifiedModelError,
)
from adserving.router.model_router import ModelRouter
from adserving.utils.context_extractor import RequestContextExtractor


class UnifiedModelRouter:
    """
    Wrapper cho ModelRouter hiện có với unified error handling.

    Tái sử dụng toàn bộ logic routing có sẵn.
    """

    def __init__(
            self,
            model_router: ModelRouter,
            logger: Optional[logging.Logger] = None
    ):
        self.model_router = model_router
        self.logger = logger or logging.getLogger(__name__)

    async def route_request_with_context(
            self,
            request: Dict[str, Any],
            request_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Route request với context-aware error handling.

        Sử dụng ModelRouter.route_request() có sẵn và enhance error handling.
        """
        start_time = time.time()

        try:
            # Sử dụng ModelRouter hiện có
            result = await self.model_router.route_request(request)

            # Check for errors trong result
            if isinstance(result, dict) and result.get("status") == "routing_error":
                return self._handle_routing_error(
                    result, request_context, start_time
                )

            # Success case
            return result

        except Exception as e:
            return self._handle_routing_exception(e, request_context, start_time)

    def _handle_routing_error(
            self,
            result: Dict[str, Any],
            request_context: Dict[str, Any],
            start_time: float
    ) -> Dict[str, Any]:
        """
        Handle routing errors từ ModelRouter.route_request().

        Transform error thành context-aware format.
        """
        error_message = result.get("error", "Unknown routing error")
        routing_time = result.get("routing_time", time.time() - start_time)

        # Parse error message để determine error type
        if "Could not determine model name" in error_message:
            unified_error = self._create_model_name_error(request_context)
        elif "does not exist in MLflow" in error_message:
            unified_error = self._create_model_not_found_error(
                error_message, request_context
            )
        elif "no Production version" in error_message:
            unified_error = self._create_model_unavailable_error(
                error_message, request_context
            )
        elif "failed to load" in error_message:
            unified_error = self._create_model_load_error(
                error_message, request_context
            )
        elif "No deployments are currently available" in error_message:
            unified_error = self._create_deployment_unavailable_error(
                error_message, request_context
            )
        else:
            unified_error = UnifiedModelError(
                error_message,
                context=request_context,
                suggestions=["Check service logs", "Verify system health"]
            )

        # Return enhanced error response
        return {
            **result,
            "unified_error": unified_error.to_dict(),
            "error_category": unified_error.error_type,
            "troubleshooting": {
                "suggestions": unified_error.suggestions,
                "context": unified_error.context,
            }
        }

    def _handle_routing_exception(
            self,
            exception: Exception,
            request_context: Dict[str, Any],
            start_time: float
    ) -> Dict[str, Any]:
        """Handle unexpected exceptions."""
        routing_time = time.time() - start_time

        unified_error = UnifiedModelError(
            f"Unexpected routing error: {str(exception)}",
            context={
                **request_context,
                "exception_type": type(exception).__name__,
                "routing_time": routing_time,
            },
            suggestions=[
                "Check service health",
                "Verify Ray Serve status",
                "Check system resources",
            ]
        )

        return {
            "status": "routing_error",
            "error": str(exception),
            "routing_time": routing_time,
            "unified_error": unified_error.to_dict(),
            "error_category": "unexpected_error",
        }