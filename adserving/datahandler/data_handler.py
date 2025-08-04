import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple

from adserving.config.config import get_config
from adserving.datahandler.models import (
    APIResponse,
    CriterionResult,
    Metadata,
    PredictionError,
    PredictionRequest,
    RequestInfo,
)

logger = logging.getLogger(__name__)


class DataHandler:
    """Handles input processing, validation, and transformation"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = get_config()

    async def process_request(self, request: PredictionRequest) -> Dict[str, Any]:
        """Process and validate input request"""
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

    def _get_model_threshold(
        self, model_name: str, model_metadata: Dict[str, Any] = None
    ) -> float:
        """
        Determine anomaly threshold for a model based on metadata and configuration.
        """
        anomaly_config = self.config.anomaly_detection

        # 1. Try to get threshold from model metadata (if enabled)
        if anomaly_config.enable_metadata_threshold and model_metadata:
            # Check if metadata contains threshold information
            if "anomaly_threshold" in model_metadata:
                threshold = model_metadata["anomaly_threshold"]
                self.logger.debug(
                    f"Using metadata threshold {threshold} for model {model_name}"
                )
                return float(threshold)

            # Check if metadata contains output type information
            if "output_type" in model_metadata:
                output_type = model_metadata["output_type"].lower()
                if output_type == "probability":
                    threshold = model_metadata.get(
                        "probability_threshold",
                        anomaly_config.default_probability_threshold,
                    )
                    self.logger.debug(
                        f"Using metadata probability threshold {threshold} for model {model_name}"
                    )
                    return float(threshold)
                elif output_type == "label":
                    threshold = model_metadata.get(
                        "label_threshold", anomaly_config.default_label_threshold
                    )
                    self.logger.debug(
                        f"Using metadata label threshold {threshold} for model {model_name}"
                    )
                    return float(threshold)

        # 2. Try model-specific configuration
        if model_name in anomaly_config.model_specific_thresholds:
            threshold = anomaly_config.model_specific_thresholds[model_name]
            self.logger.debug(
                f"Using configured threshold {threshold} for model {model_name}"
            )
            return threshold

        # 3. Use default threshold based on configured output type
        if model_name in anomaly_config.model_output_types:
            output_type = anomaly_config.model_output_types[model_name].lower()
            if output_type == "probability":
                threshold = anomaly_config.default_probability_threshold
                self.logger.debug(
                    f"Using default probability threshold {threshold} for model {model_name}"
                )
                return threshold
            elif output_type == "label":
                threshold = anomaly_config.default_label_threshold
                self.logger.debug(
                    f"Using default label threshold {threshold} for model {model_name}"
                )
                return threshold

        # 4. Fallback to default probability threshold
        threshold = anomaly_config.default_probability_threshold
        self.logger.debug(
            f"Using fallback threshold {threshold} for model {model_name}"
        )
        return threshold

    async def format_response(
        self,
        result: Dict[str, Any],
        request_id: str,
        total_time: float,
        validation_errors=None,
    ) -> APIResponse:
        """Response formatting with a PredictionError structure"""

        # Convert validation errors to PredictionError objects first
        validation_prediction_errors = []
        if validation_errors:
            for val_error in validation_errors:
                validation_prediction_errors.append(
                    PredictionError.from_validation_error(val_error)
                )

        # Handle early failures with validation errors
        if result.get("status") != "success" or "results" not in result:
            return self._create_error_response_with_validation(
                request_id, total_time, result, validation_prediction_errors
            )

        results_list = result["results"]
        self.logger.info(
            f"Processing results_list with {len(results_list)} items: {results_list}"
        )

        if not results_list:
            return self._create_empty_response_with_validation(
                request_id, total_time, result, validation_prediction_errors
            )

        try:
            # Create metadata and request info
            metadata, request_info = self._create_metadata_and_request_info(
                results_list[0], request_id, total_time
            )

            # Process prediction results and collect errors
            criteria_groups, prediction_errors = self._process_results(
                results_list, request_info
            )

            # Combine validation errors with prediction errors
            all_errors = validation_prediction_errors + prediction_errors

            # Create final results
            results = [
                CriterionResult(ma_tieu_chi=ma_tieu_chi, anomaly_FN=anomaly_fns)
                for ma_tieu_chi, anomaly_fns in criteria_groups.items()
            ]

            # Determine final status
            if all_errors:
                metadata.status = "partial_success" if results else "error"
            else:
                metadata.status = "success"

            # Create success response with warnings if needed
            return APIResponse.create_success_response(
                request_id=request_id,
                request_info=request_info,
                results=results,
                total_time=total_time,
                warnings=all_errors if all_errors else None,
            )

        except Exception as e:
            self.logger.error(f"Error formatting response: {e}")
            return self._create_error_response_with_validation(
                request_id, total_time, result, validation_prediction_errors
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
            prediction = task_result.get("prediction")

            # Skip if ma_tieu_chi is empty or None (this should be caught in validation)
            if not ma_tieu_chi or not ma_tieu_chi.strip():
                self.logger.warning(
                    f"Skipping result with empty ma_tieu_chi: {task_result}"
                )
                prediction_errors.append(
                    PredictionError.from_validation_error(
                        {
                            "field_path": "ma_tieu_chi",
                            "error_message": "ma_tieu_chi cannot be empty",
                            "error_code": "EMPTY_REQUIRED_FIELD",
                            "detail": "ma_tieu_chi field is required and cannot be empty",
                            "ma_tieu_chi": ma_tieu_chi,
                            "fn_field": None,
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
                            "field_path": f"{ma_tieu_chi}.fld_code",
                            "error_message": "Missing field code",
                            "error_code": "MISSING_FIELD_CODE",
                            "detail": f"Field code is missing for ma_tieu_chi: {ma_tieu_chi}",
                            "ma_tieu_chi": ma_tieu_chi,
                            "fn_field": None,
                        }
                    )
                )
                continue

            # Skip if prediction is None/null (field value validation error)
            if prediction is None:
                self.logger.warning(
                    f"Skipping result with null prediction for {ma_tieu_chi}.{fld_code}"
                )
                prediction_errors.append(
                    PredictionError.from_validation_error(
                        {
                            "field_path": f"{ma_tieu_chi}.{fld_code}",
                            "error_message": "Null field value",
                            "error_code": "NULL_FIELD_VALUE",
                            "detail": f"Field {fld_code} has null value and cannot be processed",
                            "ma_tieu_chi": ma_tieu_chi,
                            "fn_field": fld_code,
                        }
                    )
                )
                continue

            # Process valid field
            if ma_tieu_chi not in criteria_groups:
                criteria_groups[ma_tieu_chi] = []

            # Check if prediction indicates anomaly
            is_anomaly = False
            if hasattr(prediction, "__iter__") and not isinstance(prediction, str):
                is_anomaly = bool(prediction[0]) if len(prediction) > 0 else False
            else:
                is_anomaly = bool(prediction)

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
        """Create error response with validation errors included"""

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
            warnings=[error.to_dict() for error in all_errors] if all_errors else None,
        )

    def _create_empty_response_with_validation(
        self,
        request_id: str,
        total_time: float,
        result: Dict[str, Any],
        validation_errors: List[PredictionError],
    ) -> APIResponse:
        """Create empty response with validation errors when no results"""

        # If we have validation errors, this is an error response
        status = "error" if validation_errors else "success"

        # Extract request info from result
        ma_don_vi = result.get("ma_don_vi", "UNKNOWN")
        ma_bao_cao = result.get("ma_bao_cao", "UNKNOWN")
        ky_du_lieu = result.get("ky_du_lieu", "UNKNOWN")

        metadata = Metadata(
            status=status,
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
            warnings=(
                [error.to_dict() for error in validation_errors]
                if validation_errors
                else None
            ),
        )

    # Legacy methods for backward compatibility
    def _create_error_response(self, request_id: str, total_time: float) -> APIResponse:
        """Create error response efficiently (legacy method)"""
        return APIResponse(
            metadata=Metadata(
                status="error",
                timestamp=datetime.now().isoformat(),
                request_id=request_id,
                api_version=self.config.api_version,
                total_time=total_time,
            ),
            request_info=RequestInfo(ma_don_vi="", ma_bao_cao="", ky_du_lieu=""),
            results=[],
            warnings=None,
        )

    def _create_empty_response(self, request_id: str, total_time: float) -> APIResponse:
        """Create empty response with error status when no results (legacy method)"""
        return APIResponse(
            metadata=Metadata(
                status="error",
                timestamp=datetime.now().isoformat(),
                request_id=request_id,
                api_version=self.config.api_version,
                total_time=total_time,
            ),
            request_info=RequestInfo(ma_don_vi="", ma_bao_cao="", ky_du_lieu=""),
            results=[],
            warnings=None,
        )
