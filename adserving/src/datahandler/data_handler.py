"""Data processing and validation module for ad serving requests."""

import re
import time
from typing import Any, Dict, List, Optional

from adserving.src.config.config import get_config
from adserving.src.datahandler.models import (
    APIResponse,
    Metadata,
    PredictionRequest,
    RequestInfo,
)
from adserving.src.utils.logger import get_logger

logger = get_logger()


class DataHandler:
    """Handles input processing, validation, and transformation"""

    def __init__(self):
        """Initialize DataHandler with logger and configuration."""
        self.logger = get_logger()
        self.config = get_config()

    async def process_request(self, request: PredictionRequest) -> Dict[str, Any]:
        """Process and validate input request"""
        try:
            return await self._process_new_data_request(request)
        except Exception as e:
            self.logger.error(f"Error processing request: {e}")
            raise

    async def _process_new_data_request(
        self, request: PredictionRequest
    ) -> Dict[str, Any]:
        """Process the specified data format request"""
        validation_errors = []
        for item in request.data:
            if isinstance(item, dict) and "_validation_errors" in item:
                validation_errors.extend(item["_validation_errors"])
                del item["_validation_errors"]

        result = {
            "ma_don_vi": request.ma_don_vi,
            "ma_bao_cao": request.ma_bao_cao,
            "ky_du_lieu": request.ky_du_lieu,
            "data": request.data,
        }

        if validation_errors:
            result["_validation_errors"] = validation_errors

        return result

    async def format_response(
        self,
        request_id: str,
        timestamp,
        total_time: float,
        ma_don_vi,
        ma_bao_cao,
        ky_du_lieu,
        detailed_results: Dict[str, Any],
        validation_errors: Optional[List] = None,
    ) -> APIResponse:
        """Format API response with metadata and results."""
        try:
            status = detailed_results.get("status", "success")
            anomalies = detailed_results.get("anomalies", [])
            details = detailed_results.get("details", [])
            failed: List[Dict[str, str]] = detailed_results.get("failed", []) or []

            if validation_errors:
                failed.extend(self._process_validation_errors(validation_errors))
                # Có lỗi validate → ít nhất partial_success
                status = "partial_success" if status == "success" else status

            # Nếu không có details mà có failed → error
            if not details and failed and status == "success":
                status = "error"

            metadata = self._create_metadata(
                request_id=request_id,
                timestamp=timestamp,
                status=status,
                total_time=total_time,
            )
            request_info = self._create_request_info(ma_don_vi, ma_bao_cao, ky_du_lieu)
            response_body = self._create_response_body(anomalies, failed, details)

            return APIResponse(
                metadata=metadata,
                request_info=request_info,
                results=response_body,
            )

        except Exception as e:
            self.logger.error(f"Error formatting detailed response: {e}")
            return self._create_detailed_error_response(
                request_id, total_time, detailed_results
            )

    def _infer_column_from_message(self, msg: str) -> str:
        """Infer column name from error message by searching for FN patterns."""
        if not isinstance(msg, str):
            return "UNKNOWN"
        m = re.search(r"\bFN\d{1,2}\b", msg, flags=re.IGNORECASE)
        return m.group(0).upper() if m else "UNKNOWN"

    def _process_validation_error_item(
        self, err, is_dict: bool = True
    ) -> Dict[str, str]:
        """Process individual validation error into standardized format."""
        if is_dict:
            e = err.to_dict() if hasattr(err, "to_dict") else err
            mtc = (e.get("ma_tieu_chi") or "").strip() or "UNKNOWN"
            msg = (
                e.get("error_message")
                or e.get("message")
                or e.get("error")
                or "validation_error"
            )
        else:
            mtc = "UNKNOWN"
            msg = str(err)

        col = (
            e.get("fn_field")
            if is_dict
            else (
                None or e.get("fld_code")
                if is_dict
                else None or self._infer_column_from_message(str(msg))
            )
        )

        if not col or col == "UNKNOWN":
            if "ma_tieu_chi" in str(msg).lower():
                col = "ma_tieu_chi"

        return {"ma_tieu_chi": mtc, "column": col, "error_message": str(msg)}

    def _process_validation_errors(
        self, validation_errors: List
    ) -> List[Dict[str, str]]:
        """Process list of validation errors into standardized format."""
        failed = []
        for err in validation_errors:
            if hasattr(err, "to_dict") or isinstance(err, dict):
                failed.append(self._process_validation_error_item(err, True))
            else:
                failed.append(self._process_validation_error_item(err, False))
        return failed

    def _create_metadata(
        self, request_id: str, timestamp: float, status: str, total_time: float
    ) -> Metadata:
        """Create metadata object for API response."""
        return Metadata(
            status=status,
            timestamp=timestamp,
            request_id=request_id,
            api_version=self.config.api_version,
            total_time=total_time,
        )

    def _create_request_info(
        self, ma_don_vi: str, ma_bao_cao: str, ky_du_lieu: str
    ) -> RequestInfo:
        """Create request info object with organizational and reporting details."""
        return RequestInfo(
            ma_don_vi=ma_don_vi, ma_bao_cao=ma_bao_cao, ky_du_lieu=ky_du_lieu
        )

    def _create_response_body(
        self, anomalies: List, failed: List, details: List
    ) -> Dict[str, List]:
        """Create response body with anomalies, failed items, and details."""
        return {
            "anomalies": anomalies,  # [{ma_tieu_chi, list_anomaly}]
            "failed": failed,  # [{ma_tieu_chi, column, error_message}]
            "details": details,  # có thể bỏ nếu muốn tối ưu latency
        }

    def _create_detailed_error_response(
        self,
        request_id: str,
        total_time: float,
        detailed_results: Dict[str, Any],
    ) -> APIResponse:
        """Create error response when detailed response formatting fails."""
        metadata = Metadata(
            status="error",
            timestamp=time.time() * 1000,
            request_id=request_id,
            api_version=self.config.api_version,
            total_time=total_time,
        )

        request_info = RequestInfo(
            ma_don_vi=detailed_results.get("ma_don_vi", "UNKNOWN"),
            ma_bao_cao=detailed_results.get("ma_bao_cao", "UNKNOWN"),
            ky_du_lieu=detailed_results.get("ky_du_lieu", "UNKNOWN"),
        )

        return APIResponse(
            metadata=metadata,
            request_info=request_info,
            results={"anomalies": [], "failed_elements": [], "details": []},
        )
