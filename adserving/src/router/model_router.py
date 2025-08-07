"""
Model Router with Single Endpoint Strategy

This module implements intelligent request routing through a single /predict
endpoint instead of individual model endpoints, providing better scalability
for hundreds of models.

Key improvements:
- Single /predict endpoint for all models
- Intelligent model selection based on request data
- Request queuing and load balancing
- Dynamic routing based on model availability and load
"""

import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from ray import serve

from adserving.src.core.model_manager import ModelManager

from adserving.src.deployment.pooled_deployment import PooledModelDeployment
from adserving.src.router.deployment_selector import DeploymentSelector
from adserving.src.router.model_name_extractor import ModelNameExtractor
from adserving.src.router.request_queue import RequestQueue
from adserving.src.router.route_metrics import RouteMetrics
from adserving.src.router.routing_strategy import RoutingStrategy
from adserving.src.utils.logger import get_logger


class ModelRouter:
    """Intelligent model router with a single endpoint strategy"""

    def __init__(
        self,
        model_manager: ModelManager,
        pooled_deployment: PooledModelDeployment,
        routing_strategy: RoutingStrategy = RoutingStrategy.LEAST_LOADED,
        enable_request_queuing: bool = True,
        max_queue_size: int = 10000,
    ):

        self.model_manager = model_manager
        self.pooled_deployment = pooled_deployment
        self.routing_strategy = routing_strategy
        self.enable_request_queuing = enable_request_queuing

        # Request queue
        self.request_queue = (
            RequestQueue(max_queue_size) if enable_request_queuing else None
        )

        # Routing metrics and state
        self.route_metrics: Dict[str, RouteMetrics] = {}
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

        # Helper components
        self.model_name_extractor = ModelNameExtractor()
        self.deployment_selector = DeploymentSelector()

        self._lock = threading.RLock()
        self.logger = get_logger()

    def register_deployment(self, deployment_name: str):
        """Register a deployment for routing"""
        with self._lock:
            if deployment_name not in self.available_deployments:
                self.available_deployments.append(deployment_name)
                self.deployment_loads[deployment_name] = 0
                self.logger.debug(f"Registered deployment: {deployment_name}")

    async def route_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Route batch request with individual element handling"""

        # Extract all prediction tasks
        prediction_tasks = request.get("prediction_tasks", [])

        results = []
        failed_elements = []

        # Group tasks by model name for efficiency
        tasks_by_model = {}
        for task in prediction_tasks:
            model_name = self.model_name_extractor.extract_model_name(task)
            if model_name not in tasks_by_model:
                tasks_by_model[model_name] = []
            tasks_by_model[model_name].append(task)

        # Process each model group
        for model_name, model_tasks in tasks_by_model.items():
            try:
                # Check if model/deployment exists
                deployment_name = await self._select_deployment(
                    model_name, model_tasks[0]
                )

                if not deployment_name:
                    # Model doesn't exist - add all tasks to failed_elements
                    error_details = await self._diagnose_deployment_unavailability(
                        model_name, model_tasks[0]
                    )

                    for task in model_tasks:
                        failed_elements.append(
                            {
                                "element_index": task.get("_element_index", -1),
                                "ma_tieu_chi": task.get("_ma_tieu_chi", "unknown"),
                                "model_name": model_name,
                                "error": "model_not_found",
                                "error_details": error_details,
                            }
                        )
                    continue

                # Model exists - process tasks
                for task in model_tasks:
                    try:
                        result = await self._send_to_deployment(deployment_name, task)
                        results.append(
                            {
                                **result,
                                "element_index": task.get("_element_index", -1),
                                "ma_tieu_chi": task.get("_ma_tieu_chi", "unknown"),
                                "model_name": model_name,
                            }
                        )
                    except Exception as task_error:
                        failed_elements.append(
                            {
                                "element_index": task.get("_element_index", -1),
                                "ma_tieu_chi": task.get("_ma_tieu_chi", "unknown"),
                                "model_name": model_name,
                                "error": "prediction_failed",
                                "error_details": str(task_error),
                            }
                        )

            except Exception as model_error:
                # Model group failed - add all tasks to failed_elements
                for task in model_tasks:
                    failed_elements.append(
                        {
                            "element_index": task.get("_element_index", -1),
                            "ma_tieu_chi": task.get("_ma_tieu_chi", "unknown"),
                            "model_name": model_name,
                            "error": "model_group_failed",
                            "error_details": str(model_error),
                        }
                    )

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

    async def _diagnose_deployment_unavailability(
        self, model_name: str, request: Dict[str, Any]
    ) -> str:
        """Diagnose why no deployment is available WITHOUT triggering model loading"""
        try:
            if not self.available_deployments:
                return self._get_no_deployments_message()

            self.logger.warning(
                f"Model {model_name} - no deployment available, checking MLflow status"
            )

            # ### CRITICAL FIX: Check MLflow status WITHOUT loading model ###
            try:
                model_exists = await self._check_mlflow_model_exists_only(model_name)

                if not model_exists:
                    return self._get_mlflow_model_not_found_message(model_name)
                else:
                    # Model exists but no deployment available
                    return self._get_no_deployment_pools_message(model_name)

            except Exception as check_error:
                return (
                    f"Could not verify model {model_name} in MLflow: {str(check_error)}"
                )

        except Exception as diag_error:
            return (
                f"Could not diagnose deployment unavailability for model "
                f"'{model_name}': {str(diag_error)}"
            )

    async def _check_mlflow_model_exists_only(self, model_name: str) -> bool:
        """Check if model exists in MLflow WITHOUT attempting to load"""
        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient(tracking_uri=self.model_manager.mlflow_tracking_uri)

            try:
                # Try new alias-based approach first
                latest_version = client.get_model_version_by_alias(
                    model_name, "production"
                )
                self.logger.debug(f"Model {model_name} exists with production alias")
                return True
            except Exception:
                # Fallback to search_model_versions
                try:
                    versions = client.search_model_versions(
                        filter_string=f"name='{model_name}'",
                        order_by=["version_number DESC"],
                        max_results=1,
                    )
                    if versions and len(versions) > 0:
                        self.logger.debug(f"Model {model_name} exists in MLflow")
                        return True
                    else:
                        self.logger.debug(f"Model {model_name} has no versions")
                        return False
                except Exception as search_error:
                    if "RESOURCE_DOES_NOT_EXIST" in str(search_error):
                        self.logger.debug(f"Model {model_name} does not exist")
                        return False
                    else:
                        raise search_error

        except Exception as client_error:
            self.logger.error(f"Error checking model {model_name}: {client_error}")
            raise client_error

    def _get_no_deployments_message(self) -> str:
        return (
            "No deployments are currently available. This usually means "
            "the service hasn't been properly initialized with deployment "
            "pools. Please ensure the service is started correctly and "
            "deployment pools are created."
        )

    def _get_no_deployment_pools_message(self, model_name: str) -> str:
        return (
            f"Model '{model_name}' loaded successfully but no "
            f"deployment pools are available to serve it. This "
            f"indicates a deployment configuration issue."
        )

    def _get_mlflow_model_not_found_message(self, model_name: str) -> str:
        return (
            f"Model '{model_name}' does not exist in MLflow. Please verify "
            f"the model name is correct and the model has been registered "
            f"in MLflow."
        )

    def _get_mlflow_no_production_version_message(self, model_name: str) -> str:
        return (
            f"Model '{model_name}' exists in MLflow but has no Production "
            f"version. Please promote a model version to Production stage "
            f"in MLflow."
        )

    def _get_mlflow_load_failure_message(self, model_name: str) -> str:
        return (
            f"Model '{model_name}' exists in MLflow with Production version "
            f"but failed to load. This is likely due to dependency "
            f"mismatches (MLflow, Python, or package versions). Check the "
            f"service logs for specific dependency errors. You may need to "
            f"update your environment to match the model's requirements."
        )

    async def _check_mlflow_model_status(self, model_name: str) -> str:
        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient(tracking_uri=self.model_manager.mlflow_tracking_uri)

            try:
                latest_versions = await self._get_model_versions(client, model_name)

                if not latest_versions:
                    return self._get_mlflow_no_production_version_message(model_name)
                else:
                    return self._get_mlflow_load_failure_message(model_name)

            except Exception as mlflow_error:
                if "RESOURCE_DOES_NOT_EXIST" in str(mlflow_error):
                    return self._get_mlflow_model_not_found_message(model_name)
                else:
                    return (
                        f"Error checking model "
                        f"'{model_name}' in MLflow: {str(mlflow_error)}"
                    )

        except Exception as client_error:
            return (
                f"Could not connect to MLflow"
                f" to check model '{model_name}': {str(client_error)}"
            )

    async def _get_model_versions(self, client, model_name: str):
        try:
            latest_version = client.get_model_version_by_alias(model_name, "production")
            return [latest_version]
        except Exception:
            return client.search_model_versions(
                filter_string=f"name='{model_name}'",
                order_by=["version_number DESC"],
                max_results=1,
            )

    def _get_model_load_error_message(
        self, model_name: str, load_error: Exception
    ) -> str:
        error_msg = str(load_error)

        if "Can't get attribute '_class_setstate'" in error_msg:
            return (
                f"Model '{model_name}' failed to load due to cloudpickle "
                f"version mismatch. The model was saved with a different "
                f"version of cloudpickle than currently installed. "
                f"Error: {error_msg}"
            )
        elif "mlflow" in error_msg.lower() and "version" in error_msg.lower():
            return (
                f"Model '{model_name}' failed to load due to MLflow "
                f"version mismatch. Error: {error_msg}"
            )
        elif "numpy" in error_msg.lower() and "version" in error_msg.lower():
            return (
                f"Model '{model_name}' failed to load due to NumPy "
                f"version mismatch. Error: {error_msg}"
            )
        elif "python" in error_msg.lower() and "version" in error_msg.lower():
            return (
                f"Model '{model_name}' failed to load due to Python "
                f"version incompatibility. Error: {error_msg}"
            )
        else:
            return f"Model '{model_name}' failed to load with error: {error_msg}"

    async def _select_deployment(
        self, model_name: str, request: Dict[str, Any]
    ) -> Optional[str]:
        """Select optimal deployment based on routing strategy"""
        if not self.available_deployments:
            return None

        with self._lock:
            deployment, new_round_robin_index = (
                self.deployment_selector.select_deployment(
                    self.routing_strategy,
                    model_name,
                    self.available_deployments,
                    self.deployment_loads,
                    self.route_metrics,
                    self.model_affinity,
                    self.round_robin_index,
                )
            )

            # Update round robin index
            self.round_robin_index = new_round_robin_index

            # Update model affinity for MODEL_AFFINITY strategy
            if self.routing_strategy == RoutingStrategy.MODEL_AFFINITY and deployment:
                self.model_affinity[model_name] = deployment

            return deployment

    async def _send_to_deployment(
        self, deployment_name: str, request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send request to specific deployment"""
        try:
            # Increment load counter
            with self._lock:
                self.deployment_loads[deployment_name] += 1

            # Get deployment handle and send request
            handle = serve.get_app_handle(deployment_name).get_handle()
            result = await handle.remote(request)

            return result

        finally:
            # Decrement load counter
            with self._lock:
                self.deployment_loads[deployment_name] = max(
                    0, self.deployment_loads[deployment_name] - 1
                )

    def _update_routing_metrics(self, model_name: str, deployment_name: str):
        """Update routing decision metrics"""
        metric_key = f"{model_name}_{deployment_name}"

        if metric_key not in self.route_metrics:
            self.route_metrics[metric_key] = RouteMetrics(
                model_name=model_name,
                deployment_name=deployment_name,
                avg_response_time=0.0,
                current_load=0,
                success_rate=1.0,
                last_updated=time.time(),
            )

        # Update model to deployment mapping
        self.model_to_deployment[model_name] = deployment_name

    def _update_route_performance(
        self,
        model_name: str,
        deployment_name: str,
        response_time: float,
        success: bool,
    ):
        """Update route performance metrics"""
        metric_key = f"{model_name}_{deployment_name}"

        if metric_key in self.route_metrics:
            metric = self.route_metrics[metric_key]

            # Update response time (exponential moving average)
            alpha = 0.1  # Smoothing factor
            metric.avg_response_time = (
                alpha * response_time + (1 - alpha) * metric.avg_response_time
            )

            # Update success rate
            if success:
                metric.success_rate = min(1.0, metric.success_rate + 0.01)
            else:
                metric.success_rate = max(0.0, metric.success_rate - 0.05)

            metric.last_updated = time.time()

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics"""
        with self._lock:
            avg_routing_time = self.total_routing_time / max(1, self.request_count)
            error_rate = self.routing_errors / max(1, self.request_count)

            # Top models by request count
            model_counts = defaultdict(int)
            for metric_key, metric in self.route_metrics.items():
                model_counts[metric.model_name] += 1

            top_models = dict(
                sorted(model_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            )

            return {
                "routing_strategy": self.routing_strategy.value,
                "total_requests": self.request_count,
                "routing_errors": self.routing_errors,
                "error_rate": error_rate,
                "avg_routing_time": avg_routing_time,
                "available_deployments": len(self.available_deployments),
                "deployment_loads": dict(self.deployment_loads),
                "top_models": top_models,
                "model_affinities": len(self.model_affinity),
                "queue_stats": (
                    self.request_queue.get_queue_stats() if self.request_queue else None
                ),
            }

    def get_model_routing_info(self, model_name: str) -> Dict[str, Any]:
        """Get routing information for specific model"""
        with self._lock:
            # Find all metrics for this model
            model_metrics = {
                k: v
                for k, v in self.route_metrics.items()
                if v.model_name == model_name
            }

            current_deployment = self.model_to_deployment.get(model_name)
            affinity_deployment = self.model_affinity.get(model_name)

            return {
                "model_name": model_name,
                "current_deployment": current_deployment,
                "affinity_deployment": affinity_deployment,
                "metrics": {
                    k: {
                        "deployment": v.deployment_name,
                        "avg_response_time": v.avg_response_time,
                        "success_rate": v.success_rate,
                        "last_updated": v.last_updated,
                    }
                    for k, v in model_metrics.items()
                },
            }

    def update_routing_strategy(self, strategy: RoutingStrategy):
        """Update routing strategy dynamically"""
        with self._lock:
            old_strategy = self.routing_strategy
            self.routing_strategy = strategy
            self.logger.info(
                f"Updated routing strategy from {old_strategy.value} to "
                f"{strategy.value}"
            )

    def cleanup(self):
        """Cleanup router resources"""
        with self._lock:
            self.available_deployments.clear()
            self.deployment_loads.clear()
            self.route_metrics.clear()
            self.model_to_deployment.clear()
            self.model_affinity.clear()
