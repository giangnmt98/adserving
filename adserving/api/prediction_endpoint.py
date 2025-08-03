"""
Enhanced Prediction Endpoint with Single Element Error Handling
"""

import logging
import time
import uuid
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from ray import serve

from adserving.datahandler.data_handler import DataHandler
from adserving.datahandler.models import APIResponse, PredictionRequest
from adserving.monitoring.model_monitor import ModelMonitor
from adserving.router.model_router import ModelRouter

from .api_dependencies import (
    get_input_handler,
    get_model_router,
    get_monitor,
    get_tier_orchestrator,
    is_tier_based_deployment_enabled,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class ModelNotFoundError(Exception):
    """Custom exception for model not found"""

    def __init__(self, message: str, model_context: Optional[Dict] = None):
        super().__init__(message)
        self.model_context = model_context or {}


class ServiceUnavailableError(Exception):
    """Custom exception for service unavailable"""

    pass


class SingleElementModelNotFoundError(ModelNotFoundError):
    """Specific exception for single element model not found"""

    def __init__(self, data_element: Dict, attempted_model_name: str = None):
        self.data_element = data_element
        self.attempted_model_name = attempted_model_name

        # Create detailed error message
        ma_tieu_chi = data_element.get("ma_tieu_chi", "UNKNOWN")
        fn_fields = [k for k in data_element.keys() if k.startswith("FN")]

        message = (
            f"Không tìm thấy model cho tiêu chí '{ma_tieu_chi}' "
            f"với các trường dữ liệu: {fn_fields}"
        )

        if attempted_model_name:
            message += f" (Tên model được tạo: {attempted_model_name})"

        super().__init__(
            message,
            {
                "ma_tieu_chi": ma_tieu_chi,
                "fn_fields": fn_fields,
                "attempted_model_name": attempted_model_name,
                "data_element": data_element,
            },
        )


def _extract_detailed_model_info(request_data: Dict) -> Dict:
    """Extract detailed information for model identification"""
    info = {
        "ma_don_vi": request_data.get("ma_don_vi", "UNKNOWN"),
        "ma_bao_cao": request_data.get("ma_bao_cao", "UNKNOWN"),
        "ky_du_lieu": request_data.get("ky_du_lieu", "UNKNOWN"),
        "data_elements": [],
    }

    data_list = request_data.get("data", [])
    for element in data_list:
        element_info = {
            "ma_tieu_chi": element.get("ma_tieu_chi", "UNKNOWN"),
            "fn_fields": [k for k in element.keys() if k.startswith("FN")],
            "fn_count": len([k for k in element.keys() if k.startswith("FN")]),
        }
        info["data_elements"].append(element_info)

    info["total_elements"] = len(data_list)
    info["is_single_element"] = len(data_list) == 1

    return info


def _generate_expected_model_patterns(info: Dict) -> List[str]:
    """Generate possible model name patterns for debugging"""
    patterns = []

    ma_don_vi = info["ma_don_vi"]
    ma_bao_cao = info["ma_bao_cao"]

    for element in info["data_elements"]:
        ma_tieu_chi = element["ma_tieu_chi"]
        fn_count = element["fn_count"]

        # Common pattern variations
        base_pattern = f"{ma_don_vi}_{ma_bao_cao}_{ma_tieu_chi}"
        patterns.extend(
            [
                f"{base_pattern}_FN{fn_count:02d}",  # FN04
                f"{base_pattern}_FN{fn_count}",  # FN4
                f"{base_pattern}_{fn_count}F",  # 4F
                f"{base_pattern}_fields_{fn_count}",  # fields_4
            ]
        )

    return patterns[:5]  # Limit to 5 most common patterns


@router.post("/predict", response_model=APIResponse)
async def predict(
    request: PredictionRequest,
    handler: DataHandler = Depends(get_input_handler),
    model_router: ModelRouter = Depends(get_model_router),
    monitor: ModelMonitor = Depends(get_monitor),
):
    """
    Enhanced prediction endpoint with single element error handling.

    Properly handles cases where:
    - Single data element doesn't match any model
    - Multiple elements with some missing models
    - Model name extraction failures

    Example request:
    {
        "ma_don_vi": "UBND.0019",
        "ma_bao_cao": "10628953",
        "ky_du_lieu": "2025-07-10",
        "data": [
            {
                "ma_tieu_chi": "CT_TONGCONG",
                "FN01": 1.23,
                "FN02": 45.6,
                "FN03": 22,
                "FN4": 63.14
            }
        ]
    }
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    model_name = "unknown"

    # Extract request info for detailed error handling
    request_dict = request.dict() if hasattr(request, "dict") else request
    model_info = _extract_detailed_model_info(request_dict)

    logger.info(
        f"Processing prediction request {request_id}: "
        f"{model_info['total_elements']} elements, "
        f"single_element={model_info['is_single_element']}"
    )

    try:
        # Process and validate input
        processed_request = await handler.process_request(request)

        # Handle tier-based deployment
        if is_tier_based_deployment_enabled():
            result = await _handle_tier_based_prediction(
                processed_request, request_id, start_time, model_info
            )
            model_name = result.get("model_name", "unknown")
        else:
            # Traditional routing with enhanced error handling
            result = await _handle_traditional_routing(
                model_router, processed_request, request_id, model_info
            )
            model_name = result.get("model_name", "unknown")

        # Check for model-specific errors
        if result.get("error") == "model_not_found":
            raise _create_model_not_found_error(result, model_info, request_id)

        # Record success metrics
        inference_time = time.time() - start_time
        success = result.get("status") == "success"

        if hasattr(monitor, "metrics_collector"):
            monitor.metrics_collector.record_model_request(
                model_name=model_name, response_time=inference_time, success=success
            )

        # Format and return response
        response = await handler.format_response(result, request_id, inference_time)
        return response

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except SingleElementModelNotFoundError as e:
        logger.warning(
            f"Single element model not found for request {request_id}: "
            f"ma_tieu_chi={e.model_context.get('ma_tieu_chi')}, "
            f"attempted_model={e.model_context.get('attempted_model_name')}"
        )

        # Generate helpful suggestions
        expected_patterns = _generate_expected_model_patterns(model_info)

        error_detail = {
            "error_type": "single_element_model_not_found",
            "message": str(e),
            "context": e.model_context,
            "suggestions": {
                "expected_model_patterns": expected_patterns,
                "check_model_registry": True,
                "verify_data_fields": e.model_context.get("fn_fields", []),
            },
        }

        raise HTTPException(status_code=404, detail=error_detail)

    except ModelNotFoundError as e:
        logger.warning(f"Model not found for request {request_id}: {e}")

        error_detail = {
            "error_type": "model_not_found",
            "message": str(e),
            "context": getattr(e, "model_context", {}),
            "request_info": model_info,
        }

        raise HTTPException(status_code=404, detail=error_detail)

    except ServiceUnavailableError as e:
        logger.error(f"Service unavailable for request {request_id}: {e}")
        raise HTTPException(
            status_code=503, detail=f"Service temporarily unavailable: {str(e)}"
        )

    except ValueError as e:
        logger.error(f"Validation error for request {request_id}: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")

    except Exception as e:
        logger.error(f"Prediction error for request {request_id}: {e}")

        # Record error metrics
        if hasattr(monitor, "metrics_collector"):
            monitor.metrics_collector.record_model_request(
                model_name=model_name,
                response_time=time.time() - start_time,
                success=False,
            )

        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


def _create_model_not_found_error(
    result: Dict, model_info: Dict, request_id: str
) -> Exception:
    """Create appropriate model not found error based on context"""

    # Check if this is a single element case
    if model_info["is_single_element"] and model_info["data_elements"]:
        data_element = model_info["data_elements"][0]
        attempted_model = result.get("attempted_model_name")

        # Create the actual data element dict for the error
        # This would need to be passed from the result or reconstructed
        element_dict = {
            "ma_tieu_chi": data_element["ma_tieu_chi"],
            **{f"FN{i+1}": f"field_{i+1}" for i in range(data_element["fn_count"])},
        }

        return SingleElementModelNotFoundError(element_dict, attempted_model)
    else:
        return ModelNotFoundError(
            f"Model not found for request {request_id}", model_info
        )


async def _handle_traditional_routing(
    model_router: ModelRouter,
    processed_request: Dict,
    request_id: str,
    model_info: Dict,
) -> Dict:
    """Handle traditional routing with enhanced error context"""
    try:
        result = await model_router.route_request(processed_request)

        # Enhance result with context for better error handling
        if result.get("error") == "model_not_found":
            result["model_info"] = model_info
            result["request_id"] = request_id

        return result

    except Exception as e:
        logger.error(f"Traditional routing failed for {request_id}: {e}")
        return {
            "error": "routing_failed",
            "message": str(e),
            "model_info": model_info,
            "request_id": request_id,
        }


async def _handle_tier_based_prediction(
    processed_request: Dict, request_id: str, start_time: float, model_info: Dict
) -> Dict:
    """Handle tier-based prediction with enhanced error context"""
    try:
        orchestrator = get_tier_orchestrator()

        # Extract model name with context
        from ..router.model_name_extractor import ModelNameExtractor

        extractor = ModelNameExtractor()
        model_name = extractor.extract_model_name(processed_request)

        if not model_name:
            # Enhanced error for single element case
            if model_info["is_single_element"]:
                data_element = model_info["data_elements"][0]
                element_dict = {
                    "ma_tieu_chi": data_element["ma_tieu_chi"],
                    **{
                        f"FN{i+1}": f"field_{i+1}"
                        for i in range(data_element["fn_count"])
                    },
                }
                raise SingleElementModelNotFoundError(element_dict)
            else:
                raise ModelNotFoundError(
                    "Could not determine model name from request", model_info
                )

        # Route request with lazy loading support
        deployment_name = await orchestrator.route_request(
            model_name, processed_request
        )

        if not deployment_name:
            if model_info["is_single_element"]:
                data_element = model_info["data_elements"][0]
                element_dict = {
                    "ma_tieu_chi": data_element["ma_tieu_chi"],
                    **{
                        f"FN{i+1}": f"field_{i+1}"
                        for i in range(data_element["fn_count"])
                    },
                }
                raise SingleElementModelNotFoundError(element_dict, model_name)
            else:
                raise ModelNotFoundError(
                    f"No deployment available for model {model_name}",
                    {**model_info, "attempted_model_name": model_name},
                )

        # Send request to deployment

        try:
            deployment_handle = serve.get_app_handle(deployment_name)
            result = await deployment_handle.remote(processed_request)
        except Exception as e:
            raise ServiceUnavailableError(
                f"Failed to communicate with model deployment: {str(e)}"
            )

        # Record metrics for tier management
        inference_time = time.time() - start_time
        success = result.get("status") == "success"
        orchestrator.record_request_metrics(
            model_name, deployment_name, inference_time, not success
        )

        return result

    except (
        SingleElementModelNotFoundError,
        ModelNotFoundError,
        ServiceUnavailableError,
    ):
        # Re-raise custom exceptions
        raise
    except Exception as e:
        logger.error(f"Tier-based prediction failed: {e}")
        raise ServiceUnavailableError(f"Tier-based deployment error: {str(e)}")
