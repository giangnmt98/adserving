from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

from adserving.src.config.config_manager import get_config
from adserving.src.datahandler.models import (
    APIResponse,
    Metadata,
    PredictionError,
    PredictionRequest,
    RequestInfo,
)
from adserving.src.utils.logger import get_logger

logger = get_logger()


class DataHandler:
    """Handles input processing, validation, and transformation"""

    def __init__(self):
        self.logger = get_logger()
        self.config = get_config()

    async def process_request(self, request: PredictionRequest) -> Dict[str, Any]:
        """Process and validate input request"""
        print(request)
        try:
            # Process the request in the specified format
            return await self._process_new_data_request(request)

        except Exception as e:
            self.logger.error(f"Error processing request: {e}")
            raise

    async def _process_new_data_request(
        self, request: PredictionRequest
    ) -> Dict[str, Any]:
        """Process the specified data format request"""
        # Extract validation errors from request data
        validation_errors = []
        for item in request.data:
            if isinstance(item, dict) and "_validation_errors" in item:
                validation_errors.extend(item["_validation_errors"])
                # Remove validation errors from the data before processing
                del item["_validation_errors"]

        # Return the request data in the format expected by pooled deployment
        result = {
            "ma_don_vi": request.ma_don_vi,
            "ma_bao_cao": request.ma_bao_cao,
            "ky_du_lieu": request.ky_du_lieu,
            "data": request.data,
        }

        # Add validation errors if any exist
        if validation_errors:
            result["_validation_errors"] = validation_errors

        return result

    # Thêm method mới cho detailed response formatting
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
            warnings = None
            if validation_errors:
                warnings = [
                    (
                        {**val_error.to_dict(), "status": "error"}
                        if hasattr(val_error, "to_dict")
                        else (
                            {**val_error, "status": "error"}
                            if isinstance(val_error, dict)
                            else {"error": str(val_error), "status": "error"}
                        )
                    )
                    for val_error in validation_errors
                ]

            results_list = detailed_results["results"]
            metadata, request_info = self._create_metadata_and_request_info(
                results_list[0], request_id, total_time
            )

            # Group anomalies by criteria
            criteria_groups = {}
            for result in results_list:
                if result["status"] == "success" and result["is_anomaly"]:
                    criteria_groups.setdefault(result["ma_tieu_chi"], []).append(
                        result["fld_code"]
                    )

            # Calculate status based on prediction results
            failed_count = sum(1 for r in results_list if r["status"] == "error")

            if failed_count == 0:
                metadata.status = "success"
            if failed_count == len(results_list):
                metadata.status = "error"
            if warnings:
                metadata.status = "partial_success"

            if warnings:
                results_list.extend(warnings)

            request_info = RequestInfo(
                ma_don_vi=ma_don_vi, ma_bao_cao=ma_bao_cao, ky_du_lieu=ky_du_lieu
            )

            return APIResponse(
                metadata=metadata,
                request_info=request_info,
                results=results_list,
            )

        except Exception as e:
            self.logger.error(f"Error formatting detailed response: {e}")
            return self._create_detailed_error_response(
                request_id, total_time, detailed_results
            )

    # Thêm method mới cho detailed error response
    def _create_detailed_error_response(
        self,
        request_id: str,
        total_time: float,
        detailed_results: Dict[str, Any],
    ) -> APIResponse:
        """Create detailed error response"""

        metadata = Metadata(
            status="error",
            timestamp=datetime.now().isoformat(),
            request_id=request_id,
            api_version=self.config.api_version,
            total_time=total_time,
        )

        request_info = RequestInfo(
            ma_don_vi=detailed_results["results"][0]["ma_don_vi"],
            ma_bao_cao=detailed_results["results"][0]["ma_bao_cao"],
            ky_du_lieu=detailed_results["results"][0]["ky_du_lieu"],
        )

        return APIResponse(
            metadata=metadata,
            request_info=request_info,
            results=[],
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
                            "error_message": "ma_tieu_chi field is required and cannot be empty",
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
                            "error_message": f"Field code is missing for ma_tieu_chi: {ma_tieu_chi}",
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

        # Combine all errors
        all_errors = validation_errors + result_errors

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
