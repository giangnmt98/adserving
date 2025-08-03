"""
Exception Handlers for FastAPI Application
"""

import logging

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .api_dependencies import create_error_response

logger = logging.getLogger(__name__)


async def http_exception_handler(request: Request, exc) -> JSONResponse:
    """Handle HTTP exceptions with standardized error response"""
    return JSONResponse(
        status_code=exc.status_code,
        content=create_error_response(exc.status_code, str(exc.detail)).dict(),
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle validation errors with detailed Vietnamese messages"""
    error_details = []

    for error in exc.errors():
        field_path = " -> ".join(str(loc) for loc in error["loc"])
        error_msg = error["msg"]
        error_type = error["type"]

        # Format error message with field information
        if error_type == "value_error":
            formatted_error = f"Lỗi tại trường '{field_path}': {error_msg}"
        elif error_type == "type_error":
            formatted_error = f"Lỗi kiểu dữ liệu tại trường '{field_path}': {error_msg}"
        elif error_type == "missing":
            formatted_error = f"Thiếu trường bắt buộc: '{field_path}'"
        else:
            formatted_error = f"Lỗi validation tại trường '{field_path}': {error_msg}"

        error_details.append(formatted_error)

    # Create comprehensive error message
    main_message = "Dữ liệu đầu vào không hợp lệ"
    detail_message = "; ".join(error_details)

    return JSONResponse(
        status_code=422,
        content=create_error_response(422, main_message, detail_message).dict(),
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle general exceptions with logging"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content=create_error_response(500, "Internal server error", str(exc)).dict(),
    )
