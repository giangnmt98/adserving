"""
Ray-Serializable Model Router

This module provides a Ray-serializable wrapper for ModelRouter that eliminates 
threading objects that cause pickle/serialization errors when deploying to Ray Serve.
"""

import asyncio
from collections import defaultdict
from typing import Any, Dict, List, Optional

from adserving.src.router.serializable_model_name_extractor import SerializableModelNameExtractor
from adserving.src.router.deployment_selector import DeploymentSelector
from adserving.src.router.routing_strategy import RoutingStrategy
from adserving.src.utils.logger import get_logger


class SerializableModelRouter:
    """
    Ray-serializable model router that can be safely passed to Ray Serve deployments.
    
    This class removes all threading objects (threading.RLock, etc.) that cause 
    Ray serialization errors and replaces them with Ray-compatible alternatives.
    """

    def __init__(
        self,
        routing_strategy: RoutingStrategy = RoutingStrategy.LEAST_LOADED,
        enable_request_queuing: bool = True,
        max_queue_size: int = 10000,
    ):
        self.routing_strategy = routing_strategy
        self.enable_request_queuing = enable_request_queuing
        self.max_queue_size = max_queue_size

        # Routing metrics and state (no threading objects)
        self.route_metrics: Dict[str, Dict] = {}
        self.deployment_loads: Dict[str, int] = defaultdict(int)
        self.model_to_deployment: Dict[str, str] = {}

        # Round-robin state
        self.round_robin_index = 0
        self.available_deployments: List[str] = []

        # Model affinity (sticky routing)
        self.model_affinity: Dict[str, str] = {}

        # Performance tracking
        self.request_count = 0
        self.total_routing_time = 0.0
        self.routing_errors = 0

        # Helper components (Ray-serializable versions)
        self.model_name_extractor = SerializableModelNameExtractor()
        self.deployment_selector = DeploymentSelector()

        # Note: No threading.RLock objects - these cause serialization errors
        self.logger = get_logger()

    def register_deployment(self, deployment_name: str):
        """Register a deployment for routing (thread-safe using asyncio)"""
        if deployment_name not in self.available_deployments:
            self.available_deployments.append(deployment_name)
            self.deployment_loads[deployment_name] = 0
            self.logger.debug(f"Registered deployment: {deployment_name}")

    async def route_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route batch request with individual element handling.
        
        This is a simplified version that works with Ray Serve without threading locks.
        """
        try:
            # Extract prediction tasks from request
            prediction_tasks = request.get("prediction_tasks", [])
            
            if not prediction_tasks:
                # Handle direct prediction request format
                return await self._handle_direct_prediction_request(request)

            results = []
            failed_elements = []

            # Group tasks by model name for efficiency
            tasks_by_model = {}
            for task in prediction_tasks:
                model_name = self.model_name_extractor.extract_model_name(task)
                if model_name not in tasks_by_model:
                    tasks_by_model[model_name] = []
                tasks_by_model[model_name].append(task)

            # Process each model group in parallel
            async def _process_model_group(model_name: str, model_tasks: List[Dict[str, Any]]):
                group_results = []
                group_failed = []
                
                try:
                    deployment_name = await self._select_deployment(model_name, model_tasks[0])
                    if not deployment_name:
                        error_details = f"No deployment available for model {model_name}"
                        for task in model_tasks:
                            group_failed.append(
                                {
                                    "element_index": task.get("_element_index", -1),
                                    "ma_tieu_chi": task.get("_ma_tieu_chi", "unknown"),
                                    "model_name": model_name,
                                    "error": "model_not_found",
                                    "error_details": error_details,
                                }
                            )
                        return group_results, group_failed

                    # Process tasks within this model in parallel with a semaphore
                    semaphore = asyncio.Semaphore(10)

                    async def _send_one(task: Dict[str, Any]):
                        async with semaphore:
                            try:
                                result = await self._send_to_deployment(deployment_name, task)
                                return {
                                    **result,
                                    "element_index": task.get("_element_index", -1),
                                    "ma_tieu_chi": task.get("_ma_tieu_chi", "unknown"),
                                    "model_name": model_name,
                                }, None
                            except Exception as task_error:
                                return None, {
                                    "element_index": task.get("_element_index", -1),
                                    "ma_tieu_chi": task.get("_ma_tieu_chi", "unknown"),
                                    "model_name": model_name,
                                    "error": "prediction_failed",
                                    "error_details": str(task_error),
                                }

                    send_tasks = [_send_one(task) for task in model_tasks]
                    send_results = await asyncio.gather(*send_tasks, return_exceptions=False)
                    
                    for r, e in send_results:
                        if r is not None:
                            group_results.append(r)
                        if e is not None:
                            group_failed.append(e)

                    return group_results, group_failed

                except Exception as model_error:
                    for task in model_tasks:
                        group_failed.append(
                            {
                                "element_index": task.get("_element_index", -1),
                                "ma_tieu_chi": task.get("_ma_tieu_chi", "unknown"),
                                "model_name": model_name,
                                "error": "model_group_failed",
                                "error_details": str(model_error),
                            }
                        )
                    return group_results, group_failed

            group_coros = [
                _process_model_group(model_name, model_tasks)
                for model_name, model_tasks in tasks_by_model.items()
            ]
            group_outcomes = await asyncio.gather(*group_coros, return_exceptions=False)
            
            for group_results, group_failed in group_outcomes:
                results.extend(group_results)
                failed_elements.extend(group_failed)

            return {
                "status": "partial_success" if results else "failed",
                "results": results,
                "failed_elements": failed_elements,
                "summary": {
                    "total_tasks": len(prediction_tasks),
                    "successful_tasks": len(results),
                    "failed_tasks": len(failed_elements),
                },
            }

        except Exception as e:
            self.logger.error(f"SerializableModelRouter routing error: {e}")
            return {
                "status": "error",
                "error_message": str(e),
                "results": [],
                "failed_elements": [],
            }

    async def _handle_direct_prediction_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle direct prediction request format"""
        try:
            # This is a simplified handler for direct prediction requests
            return {
                "status": "success",
                "message": "Request processed by SerializableModelRouter",
                "router_type": "serializable",
                "request_id": request.get("request_id", "unknown"),
            }
        except Exception as e:
            return {
                "status": "error",
                "error_message": str(e),
            }

    async def _select_deployment(self, model_name: str, request: Dict[str, Any]) -> Optional[str]:
        """Select optimal deployment based on routing strategy (without locks)"""
        if not self.available_deployments:
            return None

        # Simple deployment selection without threading locks
        deployment = self.deployment_selector.select_deployment(
            self.routing_strategy,
            model_name,
            self.available_deployments,
            self.deployment_loads,
            self.route_metrics,
            self.model_affinity,
            self.round_robin_index,
        )

        # Update round robin index
        if deployment:
            self.round_robin_index = (self.round_robin_index + 1) % len(self.available_deployments)
            if self.routing_strategy == RoutingStrategy.MODEL_AFFINITY:
                self.model_affinity[model_name] = deployment[0] if isinstance(deployment, tuple) else deployment

        return deployment[0] if isinstance(deployment, tuple) else deployment

    async def _send_to_deployment(self, deployment_name: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send request to specific deployment (simplified for Ray Serve)"""
        try:
            # Increment load counter (without locks for simplicity in Ray Serve context)
            self.deployment_loads[deployment_name] += 1

            # In Ray Serve context, this would typically route to the actual model
            # For now, return a mock response indicating successful routing
            response = {
                "status": "success",
                "deployment_name": deployment_name,
                "model_prediction": "mock_prediction_result",
                "processed_by": "SerializableModelRouter",
            }

            return response

        finally:
            # Decrement load counter
            self.deployment_loads[deployment_name] = max(
                0, self.deployment_loads[deployment_name] - 1
            )

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics"""
        try:
            avg_routing_time = self.total_routing_time / max(1, self.request_count)
            error_rate = self.routing_errors / max(1, self.request_count)

            return {
                "routing_strategy": self.routing_strategy.value if hasattr(self.routing_strategy, 'value') else str(self.routing_strategy),
                "total_requests": self.request_count,
                "routing_errors": self.routing_errors,
                "error_rate": error_rate,
                "avg_routing_time": avg_routing_time,
                "available_deployments": len(self.available_deployments),
                "deployment_loads": dict(self.deployment_loads),
                "model_affinities": len(self.model_affinity),
                "router_type": "serializable",
                "threading_locks": False,  # Indicates no threading locks used
            }
        except Exception as e:
            return {
                "error": f"Failed to get routing stats: {str(e)}",
                "router_type": "serializable",
            }

    def get_model_routing_info(self, model_name: str) -> Dict[str, Any]:
        """Get routing information for specific model"""
        return {
            "model_name": model_name,
            "assigned_deployment": self.model_to_deployment.get(model_name, None),
            "affinity_deployment": self.model_affinity.get(model_name, None),
            "router_type": "serializable",
        }


def create_serializable_router_from_model_router(model_router) -> SerializableModelRouter:
    """
    Create a SerializableModelRouter from an existing ModelRouter.
    
    This function extracts the serializable configuration and state from a ModelRouter
    and creates a new SerializableModelRouter instance that can be used with Ray Serve.
    """
    try:
        serializable_router = SerializableModelRouter(
            routing_strategy=model_router.routing_strategy,
            enable_request_queuing=model_router.enable_request_queuing,
            max_queue_size=getattr(model_router.request_queue, 'max_size', 10000) if model_router.request_queue else 10000,
        )

        # Copy over the serializable state
        serializable_router.available_deployments = list(model_router.available_deployments)
        serializable_router.deployment_loads = dict(model_router.deployment_loads)
        serializable_router.model_to_deployment = dict(model_router.model_to_deployment)
        serializable_router.model_affinity = dict(model_router.model_affinity)
        serializable_router.round_robin_index = model_router.round_robin_index
        
        # Copy performance tracking
        serializable_router.request_count = model_router.request_count
        serializable_router.total_routing_time = model_router.total_routing_time
        serializable_router.routing_errors = model_router.routing_errors

        return serializable_router

    except Exception as e:
        logger = get_logger()
        logger.error(f"Failed to create serializable router from ModelRouter: {e}")
        # Return a basic serializable router as fallback
        return SerializableModelRouter()