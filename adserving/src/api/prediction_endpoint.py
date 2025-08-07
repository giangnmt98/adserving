"""
Enhanced Prediction Endpoint with Single Element Error Handling
"""

import time
import uuid
import warnings
from typing import Dict

from fastapi import APIRouter, Depends, HTTPException
from ray import serve

from adserving.src.datahandler.data_handler import DataHandler
from adserving.src.datahandler.models import APIResponse, PredictionRequest
from adserving.src.monitoring.model_monitor import ModelMonitor
from adserving.src.router.model_name_extractor import ModelNameExtractor
from adserving.src.utils.logger import get_logger
from .api_dependencies import (
    get_input_handler,
    get_monitor,
    get_tier_orchestrator,
)

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.type_adapter")

logger = get_logger()

router = APIRouter()


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


@router.post("/predict", response_model=APIResponse)
async def predict(
    request: PredictionRequest,
    handler: DataHandler = Depends(get_input_handler),
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

    # Extract request info for detailed error handling
    request_dict = request.model_dump() if hasattr(request, "dict") else request
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
        result = await _handle_tier_based_prediction(
            processed_request, request_id, start_time, model_info
        )
        model_name = result.get("model_name", "unknown")

        # Record success metrics
        inference_time = time.time() - start_time
        success = result.get("status") == "success"

        if hasattr(monitor, "metrics_collector"):
            monitor.metrics_collector.record_model_request(
                model_name=model_name, response_time=inference_time, success=success
            )

        # Format and return response
        if (
            "_validation_errors" not in processed_request
            or processed_request["_validation_errors"] is None
        ):
            response = await handler.format_response(
                request_id=request_id,
                total_time=inference_time,
                ma_don_vi=request_dict["ma_don_vi"],
                ma_bao_cao=request_dict["ma_bao_cao"],
                ky_du_lieu=request_dict["ky_du_lieu"],
                detailed_results=result,
                validation_errors=[],
            )
        else:
            response = await handler.format_response(
                request_id=request_id,
                total_time=inference_time,
                ma_don_vi=request_dict["ma_don_vi"],
                ma_bao_cao=request_dict["ma_bao_cao"],
                ky_du_lieu=request_dict["ky_du_lieu"],
                detailed_results=result,
                validation_errors=processed_request["_validation_errors"],
            )
        return response

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise


async def _handle_tier_based_prediction(
    processed_request: Dict, request_id: str, start_time: float, model_info: Dict
) -> Dict:
    """Handle tier-based prediction with enhanced error context"""
    orchestrator = get_tier_orchestrator()

    # Extract model name with context
    extractor = ModelNameExtractor()
    model_name = extractor.extract_model_name(processed_request)

    # Route request with lazy loading support
    deployment_name = await orchestrator.route_request(model_name, processed_request)

    deployment_handle = serve.get_app_handle(deployment_name)
    processed_request["request_id"] = request_id
    result = await deployment_handle.remote(processed_request)

    # Record metrics for tier management
    inference_time = time.time() - start_time
    success = result.get("status") == "success"
    orchestrator.record_request_metrics(
        model_name, deployment_name, inference_time, not success
    )

    return result
