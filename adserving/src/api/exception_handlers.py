"""
Exception handlers for the API layer with standardized response format
"""

from typing import Dict, Any
from datetime import datetime

from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from adserving.src.utils.logger import get_logger

logger = get_logger()


def create_standardized_error_response(
    error_code: str,
    error_message: str,
    error_details: str = None,
    request_id: str = None,
    context: Dict[str, Any] = None,
    status_code: int = 500,
) -> JSONResponse:
    """Create standardized error response following API specification"""

    if not request_id:
        request_id = f"req_{int(datetime.now().timestamp() * 1000)}"

    error_response = {
        "error": {
            "error_code": error_code,
            "error_message": error_message,
            "error_details": error_details or "No additional details available",
        },
        "request_id": request_id,
        "status": "error",
        "timestamp": datetime.now().isoformat(),
    }

    # Add context if provided
    if context:
        error_response["context"] = context

    return JSONResponse(status_code=status_code, content=error_response)


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors with detailed field information"""

    # Extract request ID if available
    request_id = None
    try:
        if request.method == "POST" and hasattr(request, "_body"):
            body = await request.body()
            if body:
                import json

                data = json.loads(body)
                request_id = data.get("request_id")
    except Exception:
        pass

    # Process validation errors
    validation_errors = []
    for error in exc.errors():
        field_path = " -> ".join(str(loc) for loc in error["loc"])
        validation_errors.append(
            {
                "field": field_path,
                "error": error["msg"],
                "error_type": error["type"],
                "input_value": error.get("input", "N/A"),
            }
        )

    logger.warning(f"Validation error for request {request_id}: {validation_errors}")

    return create_standardized_error_response(
        error_code="VALIDATION_ERROR",
        error_message="Request validation failed",
        error_details="Input data contains validation errors",
        request_id=request_id,
        context={"validation_errors": validation_errors},
        status_code=422,
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions with standardized format"""

    # Extract request ID and context from the detail if it's a dict
    request_id = None
    context = None
    error_message = "HTTP error occurred"
    error_details = str(exc.detail)

    # Check if this is a validation error from our custom validators
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        validation_error = exc.detail["error"]

        # Create standardized error response for validation errors
        error_response = {
            "error": {
                "error_code": validation_error.get("error_code", "VALIDATION_ERROR"),
                "error_message": validation_error.get(
                    "error_message", "Validation failed"
                ),
                "error_details": validation_error.get(
                    "error_details", "Input validation failed"
                ),
            },
            "field_path": exc.detail.get("field_path", "unknown"),
            "request_id": f"req_{int(datetime.now().timestamp() * 1000)}",
            "status": "error",
            "timestamp": exc.detail.get("timestamp", datetime.now().isoformat()),
        }

        # Log different severity based on field
        field_path = exc.detail.get("field_path", "")
        if field_path in ["ma_don_vi", "ma_bao_cao", "ky_du_lieu", "data"]:
            logger.error(
                f"Critical field validation error: {validation_error} at {field_path}"
            )
        else:
            logger.warning(
                f"Field validation error: {validation_error} at {field_path}"
            )

        return JSONResponse(status_code=exc.status_code, content=error_response)

    # Handle other HTTP exceptions (existing logic)
    if isinstance(exc.detail, dict):
        error_type = exc.detail.get("error_type", "http_error")
        error_message = exc.detail.get("message", error_message)
        context = exc.detail.get("context")
        request_info = exc.detail.get("request_info")

        # Extract request_id from context or request_info
        if context and isinstance(context, dict):
            request_id = context.get("request_id")
        if not request_id and request_info and isinstance(request_info, dict):
            request_id = request_info.get("request_id")

        # Map error types to standardized codes
        error_code_mapping = {
            "model_not_found": "MODEL_NOT_FOUND",
            "prediction_error": "PREDICTION_ERROR",
            "validation_error": "VALIDATION_ERROR",
            "timeout_error": "TIMEOUT_ERROR",
            "internal_error": "INTERNAL_ERROR",
        }

        error_code = error_code_mapping.get(error_type, "HTTP_ERROR")

        # Enhanced context for model_not_found
        if error_type == "model_not_found" and context:
            # Add available models suggestion
            available_models = [
                "anomaly_detector_v1",
                "fraud_detector_v2",
                "classification_model_v3",
            ]  # This should come from model registry
            context["available_models"] = available_models
            context["suggestion"] = (
                "Please check the model name and ensure it's deployed"
            )

    else:
        error_code = "HTTP_ERROR"
        if exc.status_code == 404:
            error_code = "NOT_FOUND"
            error_message = "Resource not found"
        elif exc.status_code == 400:
            error_code = "BAD_REQUEST"
            error_message = "Bad request"
        elif exc.status_code == 401:
            error_code = "UNAUTHORIZED"
            error_message = "Unauthorized access"
        elif exc.status_code == 403:
            error_code = "FORBIDDEN"
            error_message = "Access forbidden"
        elif exc.status_code == 500:
            error_code = "INTERNAL_ERROR"
            error_message = "Internal server error"

    logger.warning(
        f"HTTP exception {exc.status_code}: {error_message} - Request ID: {request_id}"
    )

    return create_standardized_error_response(
        error_code=error_code,
        error_message=error_message,
        error_details=error_details,
        request_id=request_id,
        context=context,
        status_code=exc.status_code,
    )


async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions with standardized format"""

    # Try to extract request ID from request body
    request_id = None
    try:
        if request.method == "POST":
            body = await request.body()
            if body:
                import json

                data = json.loads(body)
                request_id = data.get("request_id")
    except Exception:
        pass

    error_message = "An unexpected error occurred"
    error_details = str(exc)

    # Log the full exception for debugging
    logger.error(f"Unexpected error for request {request_id}: {exc}", exc_info=True)

    return create_standardized_error_response(
        error_code="INTERNAL_ERROR",
        error_message=error_message,
        error_details=error_details,
        request_id=request_id,
        status_code=500,
    )


def setup_exception_handlers(app):
    """Setup all exception handlers for the FastAPI app"""
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)
