"""Data models and validation for prediction requests and responses."""

# Python
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from fastapi import HTTPException
from pydantic import BaseModel, Field, validator

from adserving.src.datahandler.validators import (
    collect_and_validate_fn_fields,
    ensure_list_non_empty,
    extract_ma_tieu_chi_with_errors,
    finalize_validated_list_with_errors,
    validate_item_is_dict,
)


class PredictionRequest(BaseModel):
    """Prediction request for the specified format only"""

    ma_don_vi: str = Field(..., description="Mã đơn vị")
    ma_bao_cao: str = Field(..., description="Mã báo cáo")
    ky_du_lieu: str = Field(..., description="Kỳ dữ liệu (YYYY-MM-DD)")
    data: List[Dict[str, Any]] = Field(
        ...,
        description=("Danh sách dữ liệu với ma_tieu_chi và các trường FNxx"),
    )

    @validator("ma_don_vi")
    def validate_ma_don_vi(cls, v):
        """Validate ma_don_vi - must return error immediately if invalid"""
        if not isinstance(v, str):
            err_msg = (
                "Trường ma_don_vi phải là kiểu chuỗi (string), nhận được kiểu "
                f"{type(v).__name__}"
            )
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "error_code": "INVALID_DATA_TYPE",
                        "error_message": err_msg,
                    },
                    "field_path": "ma_don_vi",
                    "timestamp": datetime.now().isoformat(),
                },
            )

        if not v or not v.strip():
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "error_code": "EMPTY_REQUIRED_FIELD",
                        "error_message": "Trường ma_don_vi không được rỗng",
                    },
                    "field_path": "ma_don_vi",
                    "timestamp": datetime.now().isoformat(),
                },
            )

        return v.strip()

    @validator("ma_bao_cao")
    def validate_ma_bao_cao(cls, v):
        """Validate ma_bao_cao - must return an error immediately if invalid"""
        if not isinstance(v, str):
            err_msg = (
                "Trường ma_bao_cao phải là kiểu chuỗi (string), nhận được kiểu "
                f"{type(v).__name__}"
            )
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "error_code": "INVALID_DATA_TYPE",
                        "error_message": err_msg,
                    },
                    "field_path": "ma_bao_cao",
                    "timestamp": datetime.now().isoformat(),
                },
            )

        if not v or not v.strip():
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "error_code": "EMPTY_REQUIRED_FIELD",
                        "error_message": "Trường bắt buộc không được rỗng",
                    },
                    "field_path": "ma_bao_cao",
                    "timestamp": datetime.now().isoformat(),
                },
            )

        return v.strip()

    @validator("ky_du_lieu")
    def validate_ky_du_lieu(cls, v):
        """Validate ky_du_lieu - must return error immediately if invalid"""
        if not isinstance(v, str):
            err_details = (
                "Trường ky_du_lieu phải là kiểu chuỗi (string), nhận được kiểu "
                f"{type(v).__name__}"
            )
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "error_code": "INVALID_DATA_TYPE",
                        "error_message": "Kiểu dữ liệu không hợp lệ",
                        "error_details": err_details,
                    },
                    "field_path": "ky_du_lieu",
                    "timestamp": datetime.now().isoformat(),
                },
            )

        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            err_details = (
                "Trường ky_du_lieu phải có format YYYY-MM-DD "
                f"(ví dụ: 2024-01-01), nhận được: {v}"
            )
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "error_code": "INVALID_DATE_FORMAT",
                        "error_message": "Định dạng ngày không hợp lệ",
                        "error_details": err_details,
                    },
                    "field_path": "ky_du_lieu",
                    "timestamp": datetime.now().isoformat(),
                },
            )

    @validator("data")
    def validate_data(cls, v):
        """
        Validate data - return 400 immediately if empty,
        collect child validation errors for later processing
        """
        # 1) Kiểm tra kiểu & rỗng (giữ nguyên logic ném 400 ngay)
        v = ensure_list_non_empty(v, field_path="data")

        # 2) Thu thập lỗi phần tử con
        validation_errors: List[Dict[str, Any]] = []

        for i, item in enumerate(v):
            # 2.1) Bắt buộc là dict
            if not validate_item_is_dict(i, item, validation_errors):
                # Item không đúng kiểu -> đã đẩy lỗi, bỏ qua các check tiếp theo
                continue

            # 2.2) Trích ma_tieu_chi + đính kèm các lỗi liên quan
            ma_tieu_chi = extract_ma_tieu_chi_with_errors(
                i=i,
                item=item,
                errors=validation_errors,
            )

            # 2.3) Kiểm tra tối thiểu một trường FN và validate từng FN
            collect_and_validate_fn_fields(
                i=i,
                item=item,
                ma_tieu_chi=ma_tieu_chi,
                errors=validation_errors,
            )

        # 3) Trả về danh sách đã gắn lỗi con (nếu có), hoặc v gốc
        if validation_errors:
            return finalize_validated_list_with_errors(v, validation_errors)

        return v


class PredictionResult(BaseModel):
    """Single prediction result"""

    model_name: str
    prediction: Union[float, List[float]]
    confidence: Optional[float] = None
    tier: Optional[str] = None
    inference_time: float
    metadata: Optional[Dict[str, Any]] = None


class TieuChiResult(BaseModel):
    """Result for a single criterion"""

    ma_tieu_chi: str
    fld_code: str
    gia_tri: float
    is_anomaly: bool
    anomaly_score: Optional[float] = None
    prediction: Optional[float] = None
    model_name: Optional[str] = None
    tier: Optional[str] = None


class ModelUsageInfo(BaseModel):
    """Information about a model used in the request"""

    model_name: str
    model_version: str
    inference_time: float


class RequestInfo(BaseModel):
    """Request information section"""

    ma_don_vi: str
    ma_bao_cao: str
    ky_du_lieu: str


class Metadata(BaseModel):
    """Metadata section with system information"""

    status: str
    timestamp: float
    request_id: str
    api_version: str
    total_time: float


class PredictionError(BaseModel):
    """Individual prediction error information with standardized fields"""

    ma_tieu_chi: Optional[str] = Field(
        default=None,
        description="Mã tiêu chí gây lỗi",
    )
    fn_field: Optional[str] = Field(
        default=None,
        description="Trường FN gây lỗi",
    )
    field_path: str = Field(..., description="Đường dẫn đến trường lỗi")
    error_code: str = Field(..., description="Mã lỗi chuẩn hóa")
    error_message: str = Field(..., description="Thông báo lỗi")

    @classmethod
    def from_validation_error(
        cls,
        validation_error: Dict[str, Any],
    ) -> "PredictionError":
        """Create PredictionError from validation error dict"""
        return cls(
            ma_tieu_chi=validation_error.get("ma_tieu_chi"),
            fn_field=validation_error.get("fn_field"),
            field_path=validation_error["field_path"],
            error_code=validation_error["error_code"],
            error_message=validation_error["error_message"],
        )

    @classmethod
    def from_model_error(
        cls,
        ma_tieu_chi: str,
        error_message: str,
    ) -> "PredictionError":
        """Create PredictionError from a model processing error"""
        return cls(
            ma_tieu_chi=ma_tieu_chi,
            fn_field=None,
            field_path=f"model_processing.{ma_tieu_chi}",
            error_code="MODEL_PROCESSING_ERROR",
            error_message=error_message,
        )

    @classmethod
    def from_prediction_failure(
        cls,
        ma_tieu_chi,
        fn_field,
        error_message,
    ) -> "PredictionError":
        """Create PredictionError from prediction failure"""
        field_suffix = f".{fn_field}" if fn_field else ""
        return cls(
            ma_tieu_chi=ma_tieu_chi,
            fn_field=fn_field,
            field_path=f"prediction.{ma_tieu_chi}{field_suffix}",
            error_code="PREDICTION_FAILED",
            error_message=error_message or "Prediction failed",
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "ma_tieu_chi": self.ma_tieu_chi,
            "fn_field": self.fn_field,
            "field_path": self.field_path,
            "error_code": self.error_code,
            "error_message": self.error_message,
        }


# Thêm class mới DetailedPredictionResult
@dataclass
class DetailedPredictionResult:
    """Detailed prediction result with comprehensive tracking fields"""

    model_name: str
    ma_tieu_chi: str
    fld_code: str
    is_anomaly: bool
    anomaly_score: Optional[float]
    anomaly_threshold: Optional[float]
    processing_time: float
    model_version: Optional[str] = None
    error_message: Optional[str] = None
    status: str = "success"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "model_name": self.model_name,
            "ma_tieu_chi": self.ma_tieu_chi,
            "fld_code": self.fld_code,
            "is_anomaly": self.is_anomaly,
            "anomaly_score": self.anomaly_score,
            "anomaly_threshold": self.anomaly_threshold,
            "processing_time": self.processing_time,
            "model_version": self.model_version,
            "error_message": self.error_message,
            "status": self.status,
        }


# Cập nhật APIResponse để hỗ trợ detailed results
@dataclass
class APIResponse:
    """API response model with optional detailed results support"""

    metadata: Metadata
    request_info: RequestInfo
    results: Dict[str, List[Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with conditional detailed_results"""
        meta = (
            self.metadata.to_dict()
            if hasattr(self.metadata, "to_dict")
            else self.metadata.__dict__
        )
        req_info = (
            self.request_info.to_dict()
            if hasattr(self.request_info, "to_dict")
            else self.request_info.__dict__
        )
        # Giữ nguyên cấu trúc results (dict), chỉ chuyển từng phần tử nếu cần
        results_obj: Dict[str, Any] = {}
        for k, v in (self.results or {}).items():
            results_obj[k] = [
                (item.to_dict() if hasattr(item, "to_dict") else item)
                for item in (v or [])
            ]

        return {
            "metadata": meta,
            "request_info": req_info,
            "results": results_obj,
        }
