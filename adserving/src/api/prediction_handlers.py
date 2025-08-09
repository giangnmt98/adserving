"""
Prediction handler functions for processing prediction requests
"""

import asyncio
import time
from typing import Dict
import pandas as pd

from adserving.src.router.model_name_extractor import ModelNameExtractor
from adserving.src.core.ray_model_deployment_manager import RayModelDeploymentManager
from adserving.src.utils.logger import get_logger

logger = get_logger()


def _normalize_field_code(field_name: str) -> str:
    """Normalize field code similar to RequestProcessor"""
    if not field_name.upper().startswith("FN"):
        return field_name.upper()

    # Extract number part
    number_part = field_name[2:]
    if number_part.isdigit():
        # Pad with zero if single digit
        if len(number_part) == 1:
            return f"FN0{number_part}"
        else:
            return f"FN{number_part}"

    return field_name.upper()


async def handle_direct_model_prediction(
    processed_request: Dict, request_id: str, start_time: float, model_info: Dict, model_manager, ray_deployment_manager
) -> Dict:
    """Handle model prediction using Ray Model Deployment Manager with dedicated routers"""
    try:
        logger.info("Processing request using multi-model approach (one model per FN field)")

        # Get data elements from the request
        data_elements = processed_request.get("data", [])
        if not data_elements:
            logger.warning("No data elements found in request")
            return {
                "status": "error",
                "error_code": "NO_DATA",
                "error_message": "No data elements provided for prediction",
                "model_name": "N/A",
                "results": []
            }

        # Step 1: Create prediction tasks with individual model names for each FN field
        prediction_tasks = []
        request_metadata = {
            "ma_don_vi": processed_request.get("ma_don_vi", "UNKNOWN"),
            "ma_bao_cao": processed_request.get("ma_bao_cao", "UNKNOWN"),
            "ky_du_lieu": processed_request.get("ky_du_lieu", "UNKNOWN"),
        }

        for element in data_elements:
            ma_tieu_chi = element.get("ma_tieu_chi", "UNKNOWN")

            # Create tasks for each FN field with its specific model name
            for field_name, field_value in element.items():
                if field_name.startswith("FN") and field_value is not None:
                    # Extract model name for this specific FN field
                    normalized_fn = _normalize_field_code(field_name)
                    model_name = f"{request_metadata['ma_don_vi']}_{request_metadata['ma_bao_cao']}_{ma_tieu_chi}_{normalized_fn}"
                    
                    prediction_input = {
                        "ma_tieu_chi": ma_tieu_chi,
                        "fld_code": field_name,
                        "gia_tri": field_value,
                        "model_name": model_name,  # Each task has its own model
                        **request_metadata
                    }
                    prediction_tasks.append(prediction_input)

        if not prediction_tasks:
            logger.warning("No valid FN fields found for prediction")
            return {
                "status": "error",
                "error_code": "NO_VALID_FIELDS",
                "error_message": "No valid FN fields found for prediction",
                "model_name": "N/A",
                "results": []
            }

        logger.info(f"Step 2: Processing {len(prediction_tasks)} predictions with individual models")

        # Step 2: Route each prediction task to its specific model
        async def route_single_prediction_to_specific_model(prediction_input):
            """Route prediction to its specific model"""
            task_model_name = prediction_input.get("model_name")
            try:
                # Check if the specific model is deployed to Ray with enhanced verification
                if not ray_deployment_manager:
                    logger.error("Ray deployment manager not available")
                    return {
                        "status": "error",
                        "error_message": f"Ray deployment manager not available. Model {task_model_name} can only be processed through Ray model router.",
                        "model_name": task_model_name,
                        "ma_tieu_chi": prediction_input.get("ma_tieu_chi"),
                        "fld_code": prediction_input.get("fld_code"),
                        "is_anomaly": False,
                        "prediction": None
                    }
                
                # Enhanced model availability check - check multiple deployment sources
                is_model_available = False
                deployment_info = None
                
                # Method 1: Check ray_deployment_manager.deployed_models (primary)
                if task_model_name in ray_deployment_manager.deployed_models:
                    deployment_info = ray_deployment_manager.deployed_models[task_model_name]
                    is_model_available = True
                    logger.debug(f"Model {task_model_name} found in ray_deployment_manager.deployed_models")
                
                # Method 2: Check if model is available via model_manager (fallback)
                elif hasattr(ray_deployment_manager, 'model_manager') and ray_deployment_manager.model_manager:
                    try:
                        # Check if model is loaded in model manager (indicates it was deployed by other systems)
                        if hasattr(ray_deployment_manager.model_manager, 'is_model_loaded'):
                            if ray_deployment_manager.model_manager.is_model_loaded(task_model_name):
                                logger.info(f"Model {task_model_name} found in model_manager cache - attempting to create deployment")
                                # Try to load and deploy the model on-demand
                                deployment_result = await ray_deployment_manager.load_deploy_and_route_model(task_model_name)
                                if deployment_result.get("status") == "success":
                                    deployment_info = deployment_result.get("deployment_info")
                                    is_model_available = True
                                    logger.info(f"Model {task_model_name} deployed on-demand successfully")
                    except Exception as e:
                        logger.warning(f"Failed to check model availability via model_manager for {task_model_name}: {e}")
                
                # Method 3: Try to deploy model on-demand if it's not found (last resort)
                if not is_model_available:
                    try:
                        logger.info(f"Model {task_model_name} not found in deployment tracking - attempting on-demand deployment")
                        deployment_result = await ray_deployment_manager.load_deploy_and_route_model(task_model_name)
                        if deployment_result.get("status") in ["success", "already_deployed"]:
                            deployment_info = deployment_result.get("deployment_info")
                            is_model_available = True
                            logger.info(f"Model {task_model_name} deployed on-demand successfully")
                        else:
                            logger.error(f"On-demand deployment failed for {task_model_name}: {deployment_result.get('error_message', 'Unknown error')}")
                    except Exception as e:
                        logger.error(f"On-demand deployment failed for {task_model_name}: {e}")
                
                # If model is still not available, return error
                if not is_model_available:
                    logger.error(f"Model {task_model_name} not available after all deployment attempts")
                    return {
                        "status": "error",
                        "error_message": f"Model {task_model_name} is not available for prediction. Multiple deployment verification methods failed.",
                        "model_name": task_model_name,
                        "ma_tieu_chi": prediction_input.get("ma_tieu_chi"),
                        "fld_code": prediction_input.get("fld_code"),
                        "is_anomaly": False,
                        "prediction": None
                    }

                # Get the specific model's deployment info and router
                deployment_info = ray_deployment_manager.deployed_models[task_model_name]
                if not deployment_info or not hasattr(deployment_info, 'router_handle'):
                    return {
                        "status": "error",
                        "error_message": f"Invalid deployment info for {task_model_name}",
                        "model_name": task_model_name,
                        "ma_tieu_chi": prediction_input.get("ma_tieu_chi"),
                        "fld_code": prediction_input.get("fld_code"),
                        "is_anomaly": False,
                        "prediction": None
                    }

                router_handle = deployment_info.router_handle
                if not router_handle:
                    return {
                        "status": "error",
                        "error_message": f"Router handle not available for {task_model_name}",
                        "model_name": task_model_name,
                        "ma_tieu_chi": prediction_input.get("ma_tieu_chi"),
                        "fld_code": prediction_input.get("fld_code"),
                        "is_anomaly": False,
                        "prediction": None
                    }

                # Route to the specific model's router
                result = await router_handle.route_request.remote(prediction_input)
                return result

            except Exception as e:
                logger.error(f"Error routing prediction to {task_model_name}: {e}")
                return {
                    "status": "error",
                    "error_message": f"Routing error for {task_model_name}: {str(e)}",
                    "model_name": task_model_name,
                    "ma_tieu_chi": prediction_input.get("ma_tieu_chi", "UNKNOWN"),
                    "fld_code": prediction_input.get("fld_code", "UNKNOWN"),
                    "is_anomaly": False,
                    "prediction": None
                }

        # Execute all predictions through parallel processing (each with its own model)
        try:
            logger.info(f"Executing {len(prediction_tasks)} predictions in parallel with individual model routing")
            parallel_results = await asyncio.gather(
                *[route_single_prediction_to_specific_model(task) for task in prediction_tasks],
                return_exceptions=True
            )
        except Exception as e:
            logger.error(f"Multi-model parallel processing failed: {e}")
            return {
                "status": "error",
                "error_code": "MULTI_MODEL_PARALLEL_FAILED",
                "error_message": f"Multi-model parallel processing failed: {str(e)}",
                "model_name": "multi-model",
                "results": []
            }

        # Step 4: Process and aggregate results
        results = []
        successful_predictions = 0
        failed_predictions = 0

        for i, result in enumerate(parallel_results):
            if isinstance(result, Exception):
                # Handle task execution exceptions
                task_data = prediction_tasks[i]
                error_result = {
                    "status": "error",
                    "ma_don_vi": request_metadata["ma_don_vi"],
                    "ma_bao_cao": request_metadata["ma_bao_cao"],
                    "ky_du_lieu": request_metadata["ky_du_lieu"],
                    "ma_tieu_chi": task_data["ma_tieu_chi"],
                    "fld_code": task_data["fld_code"],
                    "model_name": task_data["model_name"],  # Use task-specific model name
                    "error_message": f"Prediction task error: {str(result)}",
                    "is_anomaly": False,
                    "prediction": None,
                }
                results.append(error_result)
                failed_predictions += 1
                logger.error(f"Prediction task {i} failed: {result}")
            else:
                # Add request metadata to successful results
                if result.get("status") == "success":
                    result.update(request_metadata)
                    successful_predictions += 1
                else:
                    failed_predictions += 1
                results.append(result)

        total_processing_time = time.time() - start_time

        # Collect unique model names used
        used_models = list(set([task.get("model_name", "unknown") for task in prediction_tasks]))
        models_summary = f"{len(used_models)} models: {', '.join(used_models[:3])}{'...' if len(used_models) > 3 else ''}"

        logger.info(
            f"Multi-model processing completed in {total_processing_time:.2f}s: "
            f"{successful_predictions} successful, {failed_predictions} failed "
            f"across {models_summary}"
        )

        # Return results with enhanced metadata
        overall_status = "success" if any(r.get("status") == "success" for r in results) else "error"

        return {
            "status": overall_status,
            "model_name": "multi-model",  # Indicates multiple models were used
            "models_used": used_models,    # List of all models used
            "results": results,
            "ray_model_router": True,
            "deployment_method": "multi-model",  # New deployment method indicator
            "total_tasks": len(prediction_tasks),
            "successful_tasks": successful_predictions,
            "failed_tasks": failed_predictions,
            "total_processing_time": total_processing_time,
        }

    except Exception as e:
        logger.error(f"Error in multi-model prediction processing: {e}")
        return {
            "status": "error",
            "error_code": "MULTI_MODEL_ERROR",
            "error_message": "Multi-model prediction processing failed",
            "error_details": str(e),
            "model_name": "multi-model",
            "results": []
        }

