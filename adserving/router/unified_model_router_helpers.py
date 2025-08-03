"""
Helper methods for UnifiedModelRouter.

Tách riêng để giữ file chính không quá 300 dòng.
"""

import re
from typing import Optional, Dict, Any

from adserving.exceptions.unified_model_exceptions import (
    SingleElementModelError,
    ModelNotAvailableError,
    ModelNotFoundError,
    DeploymentUnavailableError,
    UnifiedModelError,
)
from adserving.utils.context_extractor import RequestContextExtractor


class UnifiedModelRouterHelpers:
    """Helper methods for UnifiedModelRouter."""

    @staticmethod
    def create_model_name_error(request_context: Dict[str, Any]) -> UnifiedModelError:
        """Create context-aware model name extraction error."""
        is_single_element = request_context.get("is_single_element", False)
        data_elements = request_context.get("data_elements", [])

        if is_single_element and data_elements:
            return SingleElementModelError(
                data_elements[0],
                suggestions=RequestContextExtractor.generate_error_patterns(
                    request_context
                )
            )
        else:
            return ModelNotFoundError(
                context=request_context,
                suggestions=[
                    "Check request format",
                    "Verify ma_tieu_chi field",
                    "Ensure data fields are properly formatted"
                ]
            )

    @staticmethod
    def create_model_not_found_error(
            error_message: str,
            request_context: Dict[str, Any]
    ) -> UnifiedModelError:
        """Create model not found error with suggestions."""
        # Extract model name from error message
        model_name = UnifiedModelRouterHelpers.extract_model_name_from_error(
            error_message
        )

        is_single_element = request_context.get("is_single_element", False)
        data_elements = request_context.get("data_elements", [])

        if is_single_element and data_elements and model_name:
            return SingleElementModelError(
                data_elements[0],
                attempted_model=model_name,
                suggestions=RequestContextExtractor.generate_error_patterns(
                    request_context
                )
            )
        else:
            return ModelNotFoundError(
                model_name=model_name,
                registry_type="MLflow",
                context=request_context,
                suggestions=[
                    "Check model registry",
                    "Verify model name format",
                    "Ensure model is registered in MLflow"
                ]
            )

    @staticmethod
    def create_model_unavailable_error(
            error_message: str,
            request_context: Dict[str, Any]
    ) -> ModelNotAvailableError:
        """Create model unavailable error."""
        model_name = UnifiedModelRouterHelpers.extract_model_name_from_error(
            error_message
        )

        return ModelNotAvailableError(
            model_name or "unknown",
            deployment_type="mlflow",
            context=request_context,
            suggestions=[
                "Promote model version to Production in MLflow",
                "Check model staging status",
                "Verify MLflow connectivity"
            ]
        )

    @staticmethod
    def create_model_load_error(
            error_message: str,
            request_context: Dict[str, Any]
    ) -> UnifiedModelError:
        """Create model load error with specific suggestions."""
        model_name = UnifiedModelRouterHelpers.extract_model_name_from_error(
            error_message
        )

        # Determine specific load error type từ existing logic
        if "cloudpickle" in error_message:
            suggestions = [
                "Update cloudpickle version",
                "Check model serialization compatibility",
                "Verify Python environment"
            ]
        elif "mlflow" in error_message.lower() and "version" in error_message.lower():
            suggestions = [
                "Update MLflow version",
                "Check model's MLflow requirements",
                "Verify environment compatibility"
            ]
        elif "numpy" in error_message.lower():
            suggestions = [
                "Update NumPy version",
                "Check model's NumPy requirements",
                "Verify dependencies"
            ]
        else:
            suggestions = [
                "Check model dependencies",
                "Verify environment setup",
                "Check service logs for details"
            ]

        return UnifiedModelError(
            f"Model load failed: {error_message}",
            context={
                **request_context,
                "model_name": model_name,
                "load_error": error_message,
            },
            suggestions=suggestions,
        )

    @staticmethod
    def create_deployment_unavailable_error(
            error_message: str,
            request_context: Dict[str, Any]
    ) -> DeploymentUnavailableError:
        """Create deployment unavailable error."""
        return DeploymentUnavailableError(
            service_type="pooled_deployment",
            context=request_context,
            suggestions=[
                "Check Ray Serve status",
                "Verify deployment pool health",
                "Ensure service is properly initialized"
            ]
        )

    @staticmethod
    def extract_model_name_from_error(error_message: str) -> Optional[str]:
        """Extract model name from error message."""
        # Look for patterns like "Model 'model_name'" or "'model_name'"
        patterns = [
            r"Model '([^']+)'",
            r"'([^']+)' does not exist",
            r"'([^']+)' exists in MLflow",
            r"'([^']+)' failed to load",
        ]

        for pattern in patterns:
            match = re.search(pattern, error_message)
            if match:
                return match.group(1)

        return None