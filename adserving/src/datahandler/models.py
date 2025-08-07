from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from pydantic import BaseModel, Field, validator
from fastapi import HTTPException

from adserving.src.config.config_manager import get_config


class PredictionRequest(BaseModel):
    """Prediction request for the specified format only"""

    ma_don_vi: str = Field(..., description="Mã đơn vị")
    ma_bao_cao: str = Field(..., description="Mã báo cáo")
    ky_du_lieu: str = Field(..., description="Kỳ dữ liệu (YYYY-MM-DD)")
    data: List[Dict[str, Any]] = Field(
        ..., description="Danh sách dữ liệu với ma_tieu_chi và các trường FNxx"
    )

    @validator("ma_don_vi")
    def validate_ma_don_vi(cls, v):
        """Validate ma_don_vi - must return error immediately if invalid"""
        if not isinstance(v, str):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "error_code": "INVALID_DATA_TYPE",
                        "error_message": f"Trường ma_don_vi phải là kiểu chuỗi (string), nhận được kiểu {type(v).__name__}",
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
        """Validate ma_bao_cao - must return error immediately if invalid"""
        if not isinstance(v, str):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "error_code": "INVALID_DATA_TYPE",
                        "error_message": f"Trường ma_bao_cao phải là kiểu chuỗi (string), nhận được kiểu {type(v).__name__}",
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
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "error_code": "INVALID_DATA_TYPE",
                        "error_message": "Kiểu dữ liệu không hợp lệ",
                        "error_details": f"Trường ky_du_lieu phải là kiểu chuỗi (string), nhận được kiểu {type(v).__name__}",
                    },
                    "field_path": "ky_du_lieu",
                    "timestamp": datetime.now().isoformat(),
                },
            )

        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "error_code": "INVALID_DATE_FORMAT",
                        "error_message": "Định dạng ngày không hợp lệ",
                        "error_details": f"Trường ky_du_lieu phải có format YYYY-MM-DD (ví dụ: 2024-01-01), nhận được: {v}",
                    },
                    "field_path": "ky_du_lieu",
                    "timestamp": datetime.now().isoformat(),
                },
            )

    @validator("data")
    def validate_data(cls, v):
        """Validate data - return 400 immediately if empty, collect child validation errors for later processing"""

        # Check data type first - return 400 immediately
        if not isinstance(v, list):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "error_code": "INVALID_DATA_TYPE",
                        "error_message": f"Trường data phải là kiểu mảng (array), nhận được kiểu {type(v).__name__}",
                    },
                    "field_path": "data",
                    "timestamp": datetime.now().isoformat(),
                },
            )

        # Check if data is empty - return 400 immediately
        if not v:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "error_code": "EMPTY_REQUIRED_FIELD",
                        "error_message": "Trường data không được rỗng, phải chứa ít nhất một phần tử",
                    },
                    "field_path": "data",
                    "timestamp": datetime.now().isoformat(),
                },
            )

        # Now collect validation errors for child elements only
        validation_errors = []

        # Validate each data element
        for i, item in enumerate(v):
            element_path = f"data[{i}]"

            # Check if item is dict
            if not isinstance(item, dict):
                validation_errors.append(
                    {
                        "error_message": f"Phần tử thứ {i + 1} trong data phải là kiểu đối tượng (object), "
                        f"nhận được kiểu {type(item).__name__}",
                    }
                )
                continue

            # Validate ma_tieu_chi field
            ma_tieu_chi = item.get("ma_tieu_chi")

            if "ma_tieu_chi" not in item:
                validation_errors.append(
                    {
                        "error_message": f"Phần tử thứ {i + 1} "
                        f"trong data thiếu trường bắt buộc ma_tieu_chi",
                    }
                )
                ma_tieu_chi = f"element_{i + 1}"  # Default for further validation
            else:
                if not isinstance(item["ma_tieu_chi"], str):
                    validation_errors.append(
                        {
                            "error_message": f"Trường ma_tieu_chi trong phần tử thứ {i + 1}"
                            f" phải là kiểu chuỗi (string),"
                            f" nhận được kiểu {type(item['ma_tieu_chi']).__name__}",
                        }
                    )
                elif not item["ma_tieu_chi"].strip():
                    validation_errors.append(
                        {
                            "error_message": f"Trường ma_tieu_chi trong phần tử "
                            f"thứ {i + 1} không được rỗng",
                        }
                    )

            # Check for at least one FN field
            fn_fields = [k for k in item.keys() if k.startswith("FN")]
            if not fn_fields:
                validation_errors.append(
                    {
                        "error_message": f"Phần tử thứ {i + 1} trong data (ma_tieu_chi: {ma_tieu_chi}) "
                        f"phải có ít nhất một trường FN (ví dụ: FN01, FN02, ...)",
                    }
                )
            else:
                # Validate each FN field
                for fn_field in fn_fields:
                    fn_value = item[fn_field]

                    # Check data type
                    if not isinstance(fn_value, (int, float)):
                        validation_errors.append(
                            {
                                "error_message": f"Trường {fn_field} phải là kiểu số (number),"
                                f" nhận được kiểu {type(fn_value).__name__}",
                                "ma_tieu_chi": ma_tieu_chi,
                            }
                        )
                        continue

        # Only store validation errors for child elements, not for the data field itself
        if validation_errors:
            # Create a custom list class that can hold validation errors
            class ValidatedList(list):
                pass

            validated_data = ValidatedList(v)
            validated_data._validation_errors = validation_errors

            # Distribute errors to individual items for backward compatibility
            for item in validated_data:
                if isinstance(item, dict):
                    if "_validation_errors" not in item:
                        item["_validation_errors"] = []

            # Group errors by ma_tieu_chi and add to respective items
            for error in validation_errors:
                if error["ma_tieu_chi"]:
                    for item in validated_data:
                        if (
                            isinstance(item, dict)
                            and item.get("ma_tieu_chi") == error["ma_tieu_chi"]
                        ):
                            if "_validation_errors" not in item:
                                item["_validation_errors"] = []
                            item["_validation_errors"].append(error)

            return validated_data

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


class ModelInfo(BaseModel):
    """Model deployment and routing information"""

    deployment_strategy: str
    routing_strategy: str
    models_used: List[ModelUsageInfo]


class RequestInfo(BaseModel):
    """Request information section"""

    ma_don_vi: str
    ma_bao_cao: str
    ky_du_lieu: str


class Metadata(BaseModel):
    """Metadata section with system information"""

    status: str
    timestamp: str
    request_id: str
    api_version: str
    total_time: float


class PredictionError(BaseModel):
    """Individual prediction error information with standardized fields"""

    ma_tieu_chi: Optional[str] = Field(default=None, description="Mã tiêu chí gây lỗi")
    fn_field: Optional[str] = Field(default=None, description="Trường FN gây lỗi")
    field_path: str = Field(..., description="Đường dẫn đến trường lỗi")
    error_code: str = Field(..., description="Mã lỗi chuẩn hóa")
    error_message: str = Field(..., description="Thông báo lỗi")

    @classmethod
    def from_validation_error(
        cls, validation_error: Dict[str, Any]
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
        cls, ma_tieu_chi: str, error_message: str, detail: str = None
    ) -> "PredictionError":
        """Create PredictionError from model processing error"""
        return cls(
            ma_tieu_chi=ma_tieu_chi,
            fn_field=None,
            field_path=f"model_processing.{ma_tieu_chi}",
            error_code="MODEL_PROCESSING_ERROR",
            error_message=error_message,
            detail=detail or error_message,
        )

    @classmethod
    def from_prediction_failure(
        cls,
        ma_tieu_chi: str,
        fn_field: str = None,
        error_message: str = None,
        detail: str = None,
    ) -> "PredictionError":
        """Create PredictionError from prediction failure"""
        field_suffix = f".{fn_field}" if fn_field else ""
        return cls(
            ma_tieu_chi=ma_tieu_chi,
            fn_field=fn_field,
            field_path=f"prediction.{ma_tieu_chi}{field_suffix}",
            error_code="PREDICTION_FAILED",
            error_message=error_message or "Prediction failed",
            detail=detail or f"Failed to predict for {ma_tieu_chi}",
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
    results: List[DetailedPredictionResult]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with conditional detailed_results"""
        response_dict = {
            "metadata": (
                self.metadata.to_dict()
                if hasattr(self.metadata, "to_dict")
                else self.metadata.__dict__
            ),
            "request_info": (
                self.request_info.to_dict()
                if hasattr(self.request_info, "to_dict")
                else self.request_info.__dict__
            ),
            "results": [
                result.to_dict() if hasattr(result, "to_dict") else result.__dict__
                for result in self.results
            ],
        }
        return response_dict

    @classmethod
    def create_success_response(
        cls,
        request_id: str,
        request_info: RequestInfo,
        results: List[DetailedPredictionResult],
        total_time: float,
        warnings: Optional[List] = None,
    ):
        """Create a success response with optional detailed results"""
        metadata = Metadata(
            status="success",
            timestamp=datetime.now().isoformat(),
            request_id=request_id,
            api_version=get_config().api_version,
            total_time=total_time,
        )

        return cls(
            metadata=metadata,
            request_info=request_info,
            results=results,
        )
