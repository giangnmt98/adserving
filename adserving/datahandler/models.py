from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator
from fastapi import HTTPException


class ValidationError(Exception):
    """Custom validation error with standardized format"""

    def __init__(
        self, error_code: str, error_message: str, detail: str, field_path: str = None
    ):
        self.error_code = error_code
        self.error_message = error_message
        self.detail = detail
        self.field_path = field_path
        super().__init__(detail)


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
                        "error_message": "Kiểu dữ liệu không hợp lệ",
                        "error_details": f"Trường ma_don_vi phải là kiểu chuỗi (string), nhận được kiểu {type(v).__name__}",
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
                        "error_message": "Trường bắt buộc không được rỗng",
                        "error_details": "Trường ma_don_vi không được rỗng",
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
                        "error_message": "Kiểu dữ liệu không hợp lệ",
                        "error_details": f"Trường ma_bao_cao phải là kiểu chuỗi (string), nhận được kiểu {type(v).__name__}",
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
                        "error_details": "Trường ma_bao_cao không được rỗng",
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
                        "error_message": "Kiểu dữ liệu không hợp lệ",
                        "error_details": f"Trường data phải là kiểu mảng (array), nhận được kiểu {type(v).__name__}",
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
                        "error_message": "Trường bắt buộc không được rỗng",
                        "error_details": "Trường data không được rỗng, phải chứa ít nhất một phần tử",
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
                        "field_path": element_path,
                        "error_code": "INVALID_ELEMENT_TYPE",
                        "error_message": "Kiểu phần tử không hợp lệ",
                        "detail": f"Phần tử thứ {i + 1} trong data phải là kiểu đối tượng (object), nhận được kiểu {type(item).__name__}",
                        "ma_tieu_chi": None,
                        "fn_field": None,
                    }
                )
                continue

            # Validate ma_tieu_chi field
            ma_tieu_chi = item.get("ma_tieu_chi")

            if "ma_tieu_chi" not in item:
                validation_errors.append(
                    {
                        "field_path": f"{element_path}.ma_tieu_chi",
                        "error_code": "MISSING_REQUIRED_FIELD",
                        "error_message": "Thiếu trường bắt buộc",
                        "detail": f"Phần tử thứ {i + 1} trong data thiếu trường bắt buộc ma_tieu_chi",
                        "ma_tieu_chi": None,
                        "fn_field": None,
                    }
                )
                ma_tieu_chi = f"element_{i + 1}"  # Default for further validation
            else:
                if not isinstance(item["ma_tieu_chi"], str):
                    validation_errors.append(
                        {
                            "field_path": f"{element_path}.ma_tieu_chi",
                            "error_code": "INVALID_DATA_TYPE",
                            "error_message": "Kiểu dữ liệu không hợp lệ",
                            "detail": f"Trường ma_tieu_chi trong phần tử thứ {i + 1} phải là kiểu chuỗi (string), nhận được kiểu {type(item['ma_tieu_chi']).__name__}",
                            "ma_tieu_chi": (
                                str(item["ma_tieu_chi"])
                                if item["ma_tieu_chi"] is not None
                                else None
                            ),
                            "fn_field": None,
                        }
                    )
                elif not item["ma_tieu_chi"].strip():
                    validation_errors.append(
                        {
                            "field_path": f"{element_path}.ma_tieu_chi",
                            "error_code": "EMPTY_REQUIRED_FIELD",
                            "error_message": "Trường bắt buộc không được rỗng",
                            "detail": f"Trường ma_tieu_chi trong phần tử thứ {i + 1} không được rỗng",
                            "ma_tieu_chi": item["ma_tieu_chi"],
                            "fn_field": None,
                        }
                    )

            # Check for at least one FN field
            fn_fields = [k for k in item.keys() if k.startswith("FN")]
            if not fn_fields:
                validation_errors.append(
                    {
                        "field_path": f"{element_path}.FN_fields",
                        "error_code": "MISSING_FN_FIELDS",
                        "error_message": "Thiếu trường FN",
                        "detail": f"Phần tử thứ {i + 1} trong data (ma_tieu_chi: {ma_tieu_chi}) phải có ít nhất một trường FN (ví dụ: FN01, FN02, ...)",
                        "ma_tieu_chi": ma_tieu_chi,
                        "fn_field": None,
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
                                "field_path": f"{element_path}.{fn_field}",
                                "error_code": "INVALID_DATA_TYPE",
                                "error_message": "Kiểu dữ liệu không hợp lệ",
                                "detail": f"Trường {fn_field} phải là kiểu số (number), nhận được kiểu {type(fn_value).__name__}",
                                "ma_tieu_chi": ma_tieu_chi,
                                "fn_field": fn_field,
                            }
                        )
                        continue

                    # Range validation for FN fields
                    if isinstance(fn_value, (int, float)):
                        if fn_value < 0:
                            validation_errors.append(
                                {
                                    "field_path": f"{element_path}.{fn_field}",
                                    "error_code": "INVALID_VALUE_RANGE",
                                    "error_message": "Giá trị ngoài phạm vi cho phép",
                                    "detail": f"Trường {fn_field} không được âm, nhận được giá trị {fn_value}",
                                    "ma_tieu_chi": ma_tieu_chi,
                                    "fn_field": fn_field,
                                }
                            )

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


class CriterionAnomalyResult(BaseModel):
    """Anomaly detection result for a single criterion"""

    ma_tieu_chi: str
    is_anomaly: bool
    list_fn_abnormal: List[str]


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


class CriterionResult(BaseModel):
    """Result for a single criterion in the new format"""

    ma_tieu_chi: str
    anomaly_FN: List[str]


class PredictionError(BaseModel):
    """Individual prediction error information with standardized fields"""

    ma_tieu_chi: Optional[str] = Field(default=None, description="Mã tiêu chí gây lỗi")
    fn_field: Optional[str] = Field(default=None, description="Trường FN gây lỗi")
    field_path: str = Field(..., description="Đường dẫn đến trường lỗi")
    error_code: str = Field(..., description="Mã lỗi chuẩn hóa")
    error_message: str = Field(..., description="Thông báo lỗi")
    detail: str = Field(..., description="Chi tiết lỗi cụ thể")

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
            detail=validation_error["detail"],
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
            "detail": self.detail,
        }


class APIResponse(BaseModel):
    """API response format with comprehensive information"""

    metadata: Metadata
    request_info: RequestInfo
    results: List[CriterionResult]
    warnings: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="List of validation and processing warnings"
    )

    @classmethod
    def create_success_response(
        cls,
        request_id: str,
        request_info: RequestInfo,
        results: List[CriterionResult],
        total_time: float,
        warnings: List[PredictionError] = None,
    ) -> "APIResponse":
        """Create successful API response"""

        metadata = Metadata(
            status="success",
            timestamp=datetime.now().isoformat(),
            request_id=request_id,
            api_version="1.0.0",
            total_time=total_time,
        )

        warning_dicts = None
        if warnings:
            warning_dicts = [warning.to_dict() for warning in warnings]

        return cls(
            metadata=metadata,
            request_info=request_info,
            results=results,
            warnings=warning_dicts,
        )

    @classmethod
    def create_validation_error_response(
        cls, request_id: str, validation_errors: List[PredictionError]
    ) -> "APIResponse":
        """Create validation error response"""

        metadata = Metadata(
            status="validation_error",
            timestamp=datetime.now().isoformat(),
            request_id=request_id,
            api_version="1.0.0",
            total_time=0.0,
        )

        request_info = RequestInfo(
            ma_don_vi="VALIDATION_FAILED",
            ma_bao_cao="VALIDATION_FAILED",
            ky_du_lieu="VALIDATION_FAILED",
        )

        # Group errors by criteria
        errors_by_criteria = {}
        general_errors = []

        for error in validation_errors:
            if error.ma_tieu_chi:
                if error.ma_tieu_chi not in errors_by_criteria:
                    errors_by_criteria[error.ma_tieu_chi] = []
                errors_by_criteria[error.ma_tieu_chi].append(error)
            else:
                general_errors.append(error)

        warnings = [error.to_dict() for error in validation_errors]

        return cls(
            metadata=metadata,
            request_info=request_info,
            results=[],
            warnings=warnings,
        )


class AnomalyDetectionResponse(BaseModel):
    """New anomaly detection response format with monitoring support"""

    # Monitoring fields
    request_id: str
    status: str
    timestamp: str
    total_time: float

    # Business data fields
    ma_don_vi: str
    ma_bao_cao: str
    ky_du_lieu: str
    result: List[CriterionAnomalyResult]


class PredictionResponse(BaseModel):
    """Unified prediction response"""

    request_id: str
    timestamp: str
    status: str
    total_time: float

    results: Optional[List[PredictionResult]] = None

    # Anomaly format response
    ma_don_vi: Optional[str] = None
    ma_bao_cao: Optional[str] = None
    ky_du_lieu: Optional[str] = None
    criteria_results: Optional[List[TieuChiResult]] = None

    # Direct prediction response
    prediction: Optional[Union[float, List[float]]] = None
    model_name: Optional[str] = None

    # Common fields
    errors: List[str] = []
    warnings: List[str] = []
    summary: Optional[Dict[str, Any]] = None
