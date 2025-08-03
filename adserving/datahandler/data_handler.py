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


# Input Models


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
            if isinstance(item, dict) and '_validation_errors' in item:
                validation_errors.extend(item['_validation_errors'])
                # Remove validation errors from the data before processing
                del item['_validation_errors']
        
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

    # def _normalize_field_code(self, field_name: str) -> str:
    #     """Normalize field code similar to test_simple_api"""
    #     if not field_name.upper().startswith("FN"):
    #         return field_name.upper()
    #
    #     # Extract number part
    #     number_part = field_name[2:]
    #     if number_part.isdigit():
    #         # Pad with zero if single digit
    #         if len(number_part) == 1:
    #             return f"FN0{number_part}"
    #         else:
    #             return f"FN{number_part}"
    #
    #     return field_name.upper()

    def _get_model_threshold(
        self, model_name: str, model_metadata: Dict[str, Any] = None
    ) -> float:
        """
        Determine anomaly threshold for a model based on metadata and configuration.

        Priority order:
        1. Model metadata (if available and enabled)
        2. Model-specific configuration
        3. Default threshold based on output type
        4. Fallback to a default probability threshold

        Args:
            model_name: Name of the model
            model_metadata: Optional model metadata from MLflow

        Returns:
            float: Threshold value to use for anomaly detection
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
                        f"Using metadata probability "
                        f"threshold {threshold} for model {model_name}"
                    )
                    return float(threshold)
                elif output_type == "label":
                    threshold = model_metadata.get(
                        "label_threshold", anomaly_config.default_label_threshold
                    )
                    self.logger.debug(
                        f"Using metadata label "
                        f"threshold {threshold} for model {model_name}"
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
                    f"Using default probability threshold "
                    f"{threshold} for model {model_name}"
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
        self, result: Dict[str, Any], request_id: str, total_time: float
    ) -> APIResponse:
        """Optimized response formatting processing only essential fields"""

        # Check for validation errors first
        validation_errors = result.get("_validation_errors", [])
        
        # Fast path: early validation with minimal processing
        if result.get("status") != "success" or "results" not in result:
            # If we have validation errors, create a response with them
            if validation_errors:
                return self._create_validation_error_response(
                    validation_errors, result, request_id, total_time
                )
            return self._create_error_response(request_id, total_time)

        results_list = result["results"]
        if not results_list:
            # If we have validation errors, create a response with them
            if validation_errors:
                return self._create_validation_error_response(
                    validation_errors, result, request_id, total_time
                )
            return self._create_empty_response(request_id, total_time)

        try:
            metadata, request_info = self._create_metadata_and_request_info(
                results_list[0], request_id, total_time
            )

            criteria_groups, prediction_errors = self._process_results(
                results_list, request_info
            )

            # Add validation errors to prediction errors
            if validation_errors:
                for val_error in validation_errors:
                    prediction_errors.append(
                        PredictionError(
                            ma_tieu_chi=val_error["ma_tieu_chi"],
                            fld_code=val_error["fld_code"],
                            error_message=val_error["error_message"]
                        )
                    )
                # Set status to error if we have validation errors
                metadata.status = "error"

            results = [
                CriterionResult(ma_tieu_chi=ma_tieu_chi, anomaly_FN=anomaly_fns)
                for ma_tieu_chi, anomaly_fns in criteria_groups.items()
            ]

            if not results and not validation_errors:
                metadata.status = "error"

            return APIResponse(
                metadata=metadata,
                request_info=request_info,
                results=results,
                notes=prediction_errors if prediction_errors else None,
            )

        except Exception as e:
            self.logger.error(f"Error formatting response: {e}")
            return self._create_error_response(request_id, total_time)

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
        """Process results list and extract criteria groups and errors"""
        criteria_groups = {}
        prediction_errors = []

        for task_result in results_list:
            if task_result.get("status") != "success":
                self._handle_failed_result(task_result, prediction_errors)
                continue

            self._process_successful_result(task_result, request_info, criteria_groups)

        return criteria_groups, prediction_errors

    def _handle_failed_result(
        self, task_result: Dict[str, Any], prediction_errors: List[PredictionError]
    ) -> None:
        """Handle failed task result and collect error info"""
        ma_tieu_chi = task_result.get("ma_tieu_chi")
        fld_code = task_result.get("fld_code")
        error_message = task_result.get("error", "Unknown error")

        if ma_tieu_chi and fld_code:
            prediction_errors.append(
                PredictionError(
                    ma_tieu_chi=ma_tieu_chi,
                    fld_code=fld_code,
                    error_message=error_message,
                )
            )

    def _process_successful_result(
        self,
        task_result: Dict[str, Any],
        request_info: RequestInfo,
        criteria_groups: Dict[str, List[str]],
    ) -> None:
        """Process successful task result and update criteria groups"""
        ma_tieu_chi = task_result.get("ma_tieu_chi")
        if not ma_tieu_chi:
            return

        prediction = task_result.get("prediction", 0)
        fld_code = task_result.get("fld_code")
        if not fld_code:
            return

        model_name = self._get_model_name(
            task_result, request_info, ma_tieu_chi, fld_code
        )
        model_metadata = task_result.get("model_metadata")
        anomaly_threshold = self._get_model_threshold(model_name, model_metadata)

        if prediction > anomaly_threshold:
            if ma_tieu_chi not in criteria_groups:
                criteria_groups[ma_tieu_chi] = []
            criteria_groups[ma_tieu_chi].append(fld_code)

    def _get_model_name(
        self,
        task_result: Dict[str, Any],
        request_info: RequestInfo,
        ma_tieu_chi: str,
        fld_code: str,
    ) -> str:
        """Get or construct model name"""
        model_name = task_result.get("model_name")
        if not model_name:
            model_name = (
                f"{request_info.ma_don_vi}"
                f"_{request_info.ma_bao_cao}"
                f"_{ma_tieu_chi}_{fld_code}"
            )
        return model_name

    def _create_error_response(self, request_id: str, total_time: float) -> APIResponse:
        """Create error response efficiently"""
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
            notes=None,
        )

    def _create_empty_response(self, request_id: str, total_time: float) -> APIResponse:
        """Create empty response with error status when no results"""
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
            notes=None,
        )

    def _create_validation_error_response(
        self, validation_errors: List[Dict[str, Any]], result: Dict[str, Any], 
        request_id: str, total_time: float
    ) -> APIResponse:
        """Create response with validation errors"""
        # Extract request info from result
        ma_don_vi = result.get("ma_don_vi", "")
        ma_bao_cao = result.get("ma_bao_cao", "")
        ky_du_lieu = result.get("ky_du_lieu", "")
        
        # Convert validation errors to PredictionError objects
        prediction_errors = []
        for val_error in validation_errors:
            prediction_errors.append(
                PredictionError(
                    ma_tieu_chi=val_error["ma_tieu_chi"],
                    fld_code=val_error["fld_code"],
                    error_message=val_error["error_message"]
                )
            )
        
        return APIResponse(
            metadata=Metadata(
                status="error",
                timestamp=datetime.now().isoformat(),
                request_id=request_id,
                api_version=self.config.api_version,
                total_time=total_time,
            ),
            request_info=RequestInfo(
                ma_don_vi=ma_don_vi, 
                ma_bao_cao=ma_bao_cao, 
                ky_du_lieu=ky_du_lieu
            ),
            results=[],
            notes=prediction_errors,
        )
