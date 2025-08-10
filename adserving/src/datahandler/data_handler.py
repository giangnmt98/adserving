import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from adserving.src.config.config_manager import get_config
from adserving.src.datahandler.models import (APIResponse, Metadata,
                                              PredictionError,
                                              PredictionRequest, RequestInfo)
from adserving.src.utils.logger import get_logger

logger = get_logger()


class DataHandler:
    """Handles input processing, validation, and transformation"""

    def __init__(self):
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
        total_time: float,
        ma_don_vi: str,
        ma_bao_cao: str,
        ky_du_lieu: str,
        detailed_results: Dict[str, Any],
        validation_errors: Optional[List] = None,
    ) -> APIResponse:
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

            metadata = self._create_metadata(request_id, status, total_time)
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
        if not isinstance(msg, str):
            return "UNKNOWN"
        m = re.search(r"\bFN\d{1,2}\b", msg, flags=re.IGNORECASE)
        return m.group(0).upper() if m else "UNKNOWN"

    def _process_validation_error_item(
        self, err, is_dict: bool = True
    ) -> Dict[str, str]:
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
        failed = []
        for err in validation_errors:
            if hasattr(err, "to_dict") or isinstance(err, dict):
                failed.append(self._process_validation_error_item(err, True))
            else:
                failed.append(self._process_validation_error_item(err, False))
        return failed

    def _create_metadata(
        self, request_id: str, status: str, total_time: float
    ) -> Metadata:
        return Metadata(
            status=status,
            timestamp=datetime.now().isoformat(),
            request_id=request_id,
            api_version=self.config.api_version,
            total_time=total_time,
        )

    def _create_request_info(
        self, ma_don_vi: str, ma_bao_cao: str, ky_du_lieu: str
    ) -> RequestInfo:
        return RequestInfo(
            ma_don_vi=ma_don_vi, ma_bao_cao=ma_bao_cao, ky_du_lieu=ky_du_lieu
        )

    def _create_response_body(
        self, anomalies: List, failed: List, details: List
    ) -> Dict[str, List]:
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
        metadata = Metadata(
            status="error",
            timestamp=datetime.now().isoformat(),
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

    def _create_metadata_and_request_info(
        self, first_result: Dict[str, Any], request_id: str, total_time: float
    ) -> Tuple[Metadata, RequestInfo]:
        """Create metadata and request info from first result"""
        ma_don_vi = first_result.get("ma_don_vi", "")
        ma_bao_cao = first_result.get("ma_bao_cao", "")
        ky_du_lieu = first_result.get("ky_du_lieu", "")

        metadata = Metadata(
            status="success",
            timestamp=datetime.now().isoformat(),
            request_id=request_id,
            api_version=self.config.api_version,
            total_time=total_time,
        )

        request_info = RequestInfo(
            ma_don_vi=ma_don_vi, ma_bao_cao=ma_bao_cao, ky_du_lieu=ky_du_lieu
        )

        return metadata, request_info

    def _process_results(
        self, results_list: List[Dict[str, Any]], request_info: RequestInfo
    ) -> Tuple[Dict[str, List[str]], List[PredictionError]]:
        """Process results list with partial success support"""
        criteria_groups = {}
        prediction_errors = []

        for task_result in results_list:
            if task_result.get("status") != "success":
                error = self._handle_failed_result(task_result)
                if error:
                    prediction_errors.append(error)
                continue
            # Process successful results
            ma_tieu_chi = task_result.get("ma_tieu_chi")
            fld_code = task_result.get("fld_code")
            is_anomaly = task_result.get("is_anomaly")

            # Skip if ma_tieu_chi is empty or None (this should be caught in validation)
            if not ma_tieu_chi or not ma_tieu_chi.strip():
                self.logger.warning(
                    f"Skipping result with empty ma_tieu_chi: {task_result}"
                )
                prediction_errors.append(
                    PredictionError.from_validation_error(
                        {
                            "error_message": "ma_tieu_chi "
                            "field is required and cannot be empty",
                            "ma_tieu_chi": ma_tieu_chi,
                        }
                    )
                )
                continue

            # Skip if fld_code is missing (field validation error)
            if not fld_code:
                self.logger.warning(
                    f"Skipping result with missing fld_code for {ma_tieu_chi}"
                )
                prediction_errors.append(
                    PredictionError.from_validation_error(
                        {
                            "error_message": f"Field code "
                            f"is missing for ma_tieu_chi:"
                            f" {ma_tieu_chi}",
                            "ma_tieu_chi": ma_tieu_chi,
                        }
                    )
                )
                continue

            # Process valid field
            if ma_tieu_chi not in criteria_groups:
                criteria_groups[ma_tieu_chi] = []

            if is_anomaly:
                criteria_groups[ma_tieu_chi].append(fld_code)
                self.logger.debug(f"Added anomaly field {fld_code} to {ma_tieu_chi}")

        return criteria_groups, prediction_errors

    def _handle_failed_result(self, task_result: Dict[str, Any]) -> PredictionError:
        """Handle failed task result and create PredictionError"""
        ma_tieu_chi = task_result.get("ma_tieu_chi", "UNKNOWN")
        fn_field = task_result.get("fn_field")
        error_message = task_result.get("error", "Unknown error occurred")

        # Determine error type and create appropriate PredictionError
        if "model_not_found" in error_message.lower():
            return PredictionError.from_model_error(
                ma_tieu_chi=ma_tieu_chi,
                error_message="Model not found",
                detail=f"No model available for criterion {ma_tieu_chi}",
            )
        elif "timeout" in error_message.lower():
            return PredictionError.from_prediction_failure(
                ma_tieu_chi=ma_tieu_chi,
                fn_field=fn_field,
                error_message="Prediction timeout",
                detail=f"Prediction timed out for criterion {ma_tieu_chi}",
            )
        elif "invalid_input" in error_message.lower():
            return PredictionError.from_prediction_failure(
                ma_tieu_chi=ma_tieu_chi,
                fn_field=fn_field,
                error_message="Invalid input data",
                detail=f"Input data validation failed for criterion {ma_tieu_chi}",
            )
        else:
            return PredictionError.from_prediction_failure(
                ma_tieu_chi=ma_tieu_chi,
                fn_field=fn_field,
                error_message="Prediction failed",
                detail=error_message,
            )

    def _create_error_response_with_validation(
        self,
        request_id: str,
        total_time: float,
        result: Dict[str, Any],
        validation_errors: List[PredictionError],
    ) -> APIResponse:
        """Create an error response with validation errors included"""

        # Extract request info from result if available
        ma_don_vi = result.get("ma_don_vi", "UNKNOWN")
        ma_bao_cao = result.get("ma_bao_cao", "UNKNOWN")
        ky_du_lieu = result.get("ky_du_lieu", "UNKNOWN")

        # Create basic error from result
        result_errors = []
        if result.get("error"):
            result_errors.append(
                PredictionError.from_model_error(
                    ma_tieu_chi="SYSTEM",
                    error_message="System error",
                    detail=result.get("error", "Unknown system error"),
                )
            )

        metadata = Metadata(
            status="error",
            timestamp=datetime.now().isoformat(),
            request_id=request_id,
            api_version=self.config.api_version,
            total_time=total_time,
        )

        request_info = RequestInfo(
            ma_don_vi=ma_don_vi, ma_bao_cao=ma_bao_cao, ky_du_lieu=ky_du_lieu
        )

        return APIResponse(
            metadata=metadata,
            request_info=request_info,
            results=[],
        )
