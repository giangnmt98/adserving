from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


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
        if not isinstance(v, str):
            raise ValueError(
                f"Trường ma_don_vi phải là kiểu chuỗi (string), "
                f"nhận được kiểu {type(v).__name__}"
            )
        if not v or not v.strip():
            raise ValueError("Trường ma_don_vi không được rỗng")
        return v.strip()

    @validator("ma_bao_cao")
    def validate_ma_bao_cao(cls, v):
        if not isinstance(v, str):
            raise ValueError(
                f"Trường ma_bao_cao phải là kiểu chuỗi (string), "
                f"nhận được kiểu {type(v).__name__}"
            )
        if not v or not v.strip():
            raise ValueError("Trường ma_bao_cao không được rỗng")
        return v.strip()

    @validator("ky_du_lieu")
    def validate_ky_du_lieu(cls, v):
        if not isinstance(v, str):
            raise ValueError(
                f"Trường ky_du_lieu phải là kiểu chuỗi (string), "
                f"nhận được kiểu {type(v).__name__}"
            )
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError(
                "Trường ky_du_lieu phải có format YYYY-MM-DD " "(ví dụ: 2024-01-01)"
            )

    @validator("data")
    def validate_data(cls, v):
        if not isinstance(v, list):
            raise ValueError(
                f"Trường data phải là kiểu mảng (array), "
                f"nhận được kiểu {type(v).__name__}"
            )

        if not v:
            raise ValueError("Trường data không được rỗng")

        # Store validation errors for FNxx fields
        validation_errors = []

        for i, item in enumerate(v):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Phần tử thứ {i+1} trong data phải là kiểu đối tượng "
                    f"(object), nhận được kiểu {type(item).__name__}"
                )

            # Validate ma_tieu_chi field
            if "ma_tieu_chi" not in item:
                raise ValueError(
                    f"Phần tử thứ {i+1} trong data thiếu trường bắt buộc "
                    f"ma_tieu_chi"
                )

            if not isinstance(item["ma_tieu_chi"], str):
                raise ValueError(
                    f"Trường ma_tieu_chi trong phần tử thứ {i+1} phải là "
                    f"kiểu chuỗi (string), nhận được kiểu "
                    f'{type(item["ma_tieu_chi"]).__name__}'
                )

            if not item["ma_tieu_chi"].strip():
                raise ValueError(
                    f"Trường ma_tieu_chi trong phần tử thứ {i+1} " f"không được rỗng"
                )

            # Check for at least one FN field
            fn_fields = [k for k in item.keys() if k.startswith("FN")]
            if not fn_fields:
                raise ValueError(
                    f"Phần tử thứ {i+1} trong data "
                    f'(ma_tieu_chi: {item.get("ma_tieu_chi")}) '
                    f"phải có ít nhất một trường FN (ví dụ: FN01, FN02, ...)"
                )

            # Validate FN field types and values - collect errors instead of raising
            ma_tieu_chi = item.get("ma_tieu_chi", f"element_{i+1}")
            for fn_field in fn_fields:
                fn_value = item[fn_field]
                
                # Check type
                if not isinstance(fn_value, (int, float)):
                    validation_errors.append({
                        "ma_tieu_chi": ma_tieu_chi,
                        "fld_code": fn_field,
                        "error_message": "data không hợp lệ",
                        "detail": f"Trường {fn_field} phải là kiểu số (number), nhận được kiểu {type(fn_value).__name__}"
                    })
                    continue

                # Check for reasonable numeric values
                if isinstance(fn_value, (int, float)) and (
                    fn_value < 0 or fn_value > 1e15
                ):
                    validation_errors.append({
                        "ma_tieu_chi": ma_tieu_chi,
                        "fld_code": fn_field,
                        "error_message": "data không hợp lệ",
                        "detail": f"Trường {fn_field} có giá trị không hợp lệ: {fn_value}. Giá trị phải từ 0 đến 1,000,000,000,000,000"
                    })

        # Store validation errors in the data for later processing
        if validation_errors:
            # Add validation errors to the data structure for later processing
            for item in v:
                if not hasattr(item, '_validation_errors'):
                    item['_validation_errors'] = []
            
            # Distribute errors to their respective items
            for error in validation_errors:
                for item in v:
                    if item.get("ma_tieu_chi") == error["ma_tieu_chi"]:
                        if '_validation_errors' not in item:
                            item['_validation_errors'] = []
                        item['_validation_errors'].append(error)

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
    """Individual prediction error information"""

    ma_tieu_chi: str
    fld_code: str
    error_message: str


class APIResponse(BaseModel):
    """API response format with comprehensive information"""

    metadata: Metadata
    request_info: RequestInfo
    results: List[CriterionResult]
    notes: Optional[List[PredictionError]] = None


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
