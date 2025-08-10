# Python
from datetime import datetime
from typing import Any, List, Optional

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from adserving.src.utils.logger import get_logger

logger = get_logger()


def _now_iso() -> str:
    return datetime.now().isoformat()


def _ensure_request_id(request_id: Optional[str]) -> str:
    return request_id or f"req_{int(datetime.now().timestamp() * 1000)}"


def _unified_error_payload(
    *,
    error_code: str,
    error_message: str,
    error_details: str,
    field_path: str,
    request_id: Optional[str] = None,
    status_code: int = 400,
) -> JSONResponse:
    """Trả về payload lỗi theo format duy nhất."""
    rid = _ensure_request_id(request_id)
    payload = {
        "error": {
            "error_code": error_code,
            "error_message": error_message,
            "error_details": error_details or "Input validation failed",
        },
        "field_path": field_path or "unknown",
        "request_id": rid,
        "status": "error",
        "timestamp": _now_iso(),
    }
    return JSONResponse(status_code=status_code, content=payload)


def _pydantic_error_code(err_type: str) -> str:
    # Map cơ bản cho Pydantic error types
    if err_type == "missing":
        return "MISSING_REQUIRED_FIELD"
    if err_type.startswith("type_error"):
        return "INVALID_DATA_TYPE"
    if err_type.startswith("value_error.date") or err_type.startswith(
        "value_error.time"
    ):
        return "INVALID_DATE_FORMAT"
    if err_type.startswith("value_error"):
        return "VALIDATION_ERROR"
    return "VALIDATION_ERROR"


def _format_field_path_from_loc(loc: List[Any]) -> str:
    # loc có dạng ("body","data", 0, "ma_tieu_chi") → "data[0].ma_tieu_chi"
    if not loc:
        return "unknown"
    # Bỏ "body" nếu là phần tử đầu
    parts = list(loc)
    if parts and str(parts[0]) == "body":
        parts = parts[1:]

    out: List[str] = []
    for p in parts:
        if isinstance(p, int):
            # áp dụng vào phần trước đó nếu là mảng
            if out:
                out[-1] = f"{out[-1]}[{p}]"
            else:
                out.append(f"[{p}]")
        else:
            # tên trường
            out.append(str(p))
    # Ghép dấu chấm giữa các trường
    # “data[0].ma_tieu_chi” nếu có mảng
    return ".".join(out) if out else "unknown"


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Chuẩn hóa Pydantic/ValidationError về format duy nhất."""
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

    errors = exc.errors() or []
    if not errors:
        return _unified_error_payload(
            error_code="VALIDATION_ERROR",
            error_message="Request validation failed",
            error_details="Input validation failed",
            field_path="unknown",
            request_id=request_id,
            status_code=422,
        )

    # Lấy lỗi đầu tiên để phản hồi (giữ đơn giản và nhất quán)
    e = errors[0]
    err_type = e.get("type", "validation_error")
    error_code = _pydantic_error_code(err_type)
    msg = e.get("msg") or "Validation failed"
    loc = e.get("loc") or []
    field_path = _format_field_path_from_loc(list(loc))

    # Tối ưu thông điệp khi thiếu trường cụ thể trong data[i]
    # Ví dụ: thiếu "ma_tieu_chi" → error_message rõ ràng
    # Pydantic “Field required” cho thiếu trường → giữ msg, field_path cung cấp vị trí
    error_message = msg
    error_details = "Input validation failed"

    # Nếu là MISSING_REQUIRED_FIELD và có tên trường cuối trong field_path
    if error_code == "MISSING_REQUIRED_FIELD":
        last_segment = str(field_path.split(".")[-1]) if field_path else ""
        # nếu có dạng data[<i>].<field>
        if field_path.startswith("data[") and "." in field_path:
            idx_part, field_part = field_path.split(".", 1)
            try:
                idx = int(idx_part[idx_part.find("[") + 1 : idx_part.find("]")])
            except Exception:
                idx = None
            if idx is not None and last_segment:
                error_message = (
                    f"Data item tại vị trí "
                    f"{idx + 1} thiếu trường"
                    f" bắt buộc '{last_segment}'"
                )

    logger.warning(
        f"Validation error " f"({error_code}) at {field_path}: {error_message}"
    )
    status_code = 400 if error_code == "MISSING_REQUIRED_FIELD" else 422

    return _unified_error_payload(
        error_code=error_code,
        error_message=error_message,
        error_details=error_details,
        field_path=field_path,
        request_id=request_id,
        status_code=status_code,
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Chuẩn hóa mọi HTTPException về format duy nhất."""
    # Nếu detail đã theo format mục tiêu thì passthrough
    if (
        isinstance(exc.detail, dict)
        and "error" in exc.detail
        and "status" in exc.detail
    ):
        # Đảm bảo status code của response là exc.status_code
        return JSONResponse(status_code=exc.status_code, content=exc.detail)

    # Còn lại: dựng format thống nhất từ HTTPException
    # Ưu tiên field_path, error_code, error_message trong detail nếu có
    detail = exc.detail
    error_code = "HTTP_ERROR"
    error_message = "HTTP error occurred"
    error_details = str(detail)
    field_path = "unknown"

    if isinstance(detail, dict):
        # Map nhẹ nếu có
        error_message = (
            detail.get("message") or detail.get("error_message") or error_message
        )
        error_details = detail.get("error_details") or error_details
        field_path = detail.get("field_path") or field_path

        # Thử lấy error_code
        if "error" in detail and isinstance(detail["error"], dict):
            error_code = detail["error"].get("error_code") or error_code
        else:
            error_code = detail.get("error_code") or error_code

    # Nếu Not Found/Bad Request/... gán mã tương ứng
    if exc.status_code == 404:
        error_code = "NOT_FOUND" if error_code == "HTTP_ERROR" else error_code
        error_message = error_message or "Resource not found"
    elif exc.status_code == 400:
        if error_code == "HTTP_ERROR":
            error_code = "BAD_REQUEST"
        error_message = error_message or "Bad request"
    elif exc.status_code == 401:
        error_code = "UNAUTHORIZED" if error_code == "HTTP_ERROR" else error_code
    elif exc.status_code == 403:
        error_code = "FORBIDDEN" if error_code == "HTTP_ERROR" else error_code
    elif exc.status_code == 500:
        error_code = "INTERNAL_ERROR" if error_code == "HTTP_ERROR" else error_code
        error_message = error_message or "Internal server error"

    # request_id nếu có trong detail/context
    request_id = None
    if isinstance(detail, dict):
        request_id = detail.get("request_id")
        ctx = detail.get("context")
        if not request_id and isinstance(ctx, dict):
            request_id = ctx.get("request_id")

    return _unified_error_payload(
        error_code=error_code,
        error_message=error_message or "Error",
        error_details=error_details or "",
        field_path=field_path,
        request_id=request_id,
        status_code=exc.status_code,
    )


async def general_exception_handler(request: Request, exc: Exception):
    """Fallback thống nhất cho lỗi không bắt được."""
    logger.error(f"Unexpected error: {exc}")
    return _unified_error_payload(
        error_code="INTERNAL_ERROR",
        error_message="An unexpected error occurred",
        error_details=str(exc),
        field_path="unknown",
        request_id=None,
        status_code=500,
    )


def setup_exception_handlers(app):
    """Đăng ký tất cả exception handlers thống nhất format."""
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)
