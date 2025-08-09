"""
Ray remote actor for model routing and parallel task management
"""

import asyncio
import time
import ray
from typing import Dict, Any, List

from adserving.src.utils.logger import get_logger


@ray.remote(num_cpus=0.5)
class RayModelRouter:
    """Ray remote actor for routing requests to model deployment actors and managing parallel tasks"""
    
    def __init__(self, model_name: str, deployment_actor):
        self.model_name = model_name
        self.deployment_actor = deployment_actor
        self.request_count = 0
        self.total_routing_time = 0.0
        self.failed_requests = 0
        self.logger = get_logger()
        
        # Load Ray optimization configuration for dynamic concurrent task limit
        try:
            from adserving.src.config.config_manager import get_config
            config = get_config()
            ray_config = getattr(config, 'ray', None)
            ray_optimization = getattr(ray_config, 'optimization', {}) if ray_config else {}
            
            # Use configurable max_concurrent_tasks instead of hardcoded value
            self.max_concurrent_tasks = ray_optimization.get('max_concurrent_tasks_per_router', 100)  # Default to ultra-scale 100
            self.logger.info(f"Ray model router for {model_name} configured with {self.max_concurrent_tasks} max concurrent tasks")
            
        except Exception as e:
            # Fallback to ultra-scale default if config loading fails
            self.max_concurrent_tasks = 100  # Ultra-scale fallback (increased from 20)
            self.logger.warning(f"Failed to load config for {model_name}, using ultra-scale fallback: {self.max_concurrent_tasks} concurrent tasks")
        
        # Task pool for parallel processing with configurable limit
        self.task_semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        
    async def route_request(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Route a single prediction request to the deployment actor"""
        routing_start = time.time()
        
        try:
            self.request_count += 1
            self.logger.debug(f"Ray router processing request for {self.model_name}: {input_data}")
            
            # Use semaphore to limit concurrent tasks and prevent resource exhaustion
            async with self.task_semaphore:
                # Route to the deployment actor for prediction
                result = await self.deployment_actor.predict.remote(input_data)
                
            routing_time = time.time() - routing_start
            self.total_routing_time += routing_time
            
            # Add routing metadata
            if isinstance(result, dict):
                result["routing_time"] = routing_time
                result["router_actor_id"] = ray.get_runtime_context().get_actor_id()
            
            self.logger.debug(f"Ray router completed request for {self.model_name} in {routing_time:.2f}s")
            return result
            
        except Exception as e:
            routing_time = time.time() - routing_start
            self.failed_requests += 1
            self.total_routing_time += routing_time
            
            self.logger.error(f"Ray router failed for {self.model_name}: {e}")
            
            return {
                "status": "error",
                "ma_don_vi": input_data.get("ma_don_vi", "UNKNOWN"),
                "ma_bao_cao": input_data.get("ma_bao_cao", "UNKNOWN"),
                "ky_du_lieu": input_data.get("ky_du_lieu", "UNKNOWN"),
                "ma_tieu_chi": input_data.get("ma_tieu_chi", "UNKNOWN"),
                "fld_code": input_data.get("fld_code", "UNKNOWN"),
                "model_name": self.model_name,
                "error_message": f"Ray router error: {str(e)}",
                "is_anomaly": False,
                "prediction": None,
                "routing_time": routing_time,
                "router_actor_id": ray.get_runtime_context().get_actor_id()
            }
    
    async def route_batch_requests(self, batch_requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Route multiple prediction requests in parallel using Ray tasks"""
        batch_start = time.time()
        
        try:
            self.logger.info(f"Ray router processing batch of {len(batch_requests)} requests for {self.model_name}")
            
            # Create Ray remote tasks for each request in the batch
            async def route_single_with_semaphore(request_data):
                async with self.task_semaphore:
                    return await self.deployment_actor.predict.remote(request_data)
            
            # Execute all requests in parallel using Ray remote calls
            batch_tasks = [route_single_with_semaphore(request) for request in batch_requests]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process results and handle exceptions
            processed_results = []
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    # Handle individual task failures
                    request_data = batch_requests[i]
                    error_result = {
                        "status": "error",
                        "ma_don_vi": request_data.get("ma_don_vi", "UNKNOWN"),
                        "ma_bao_cao": request_data.get("ma_bao_cao", "UNKNOWN"),
                        "ky_du_lieu": request_data.get("ky_du_lieu", "UNKNOWN"),
                        "ma_tieu_chi": request_data.get("ma_tieu_chi", "UNKNOWN"),
                        "fld_code": request_data.get("fld_code", "UNKNOWN"),
                        "model_name": self.model_name,
                        "error_message": f"Ray batch task error: {str(result)}",
                        "is_anomaly": False,
                        "prediction": None,
                        "router_actor_id": ray.get_runtime_context().get_actor_id()
                    }
                    processed_results.append(error_result)
                    self.failed_requests += 1
                else:
                    # Add router metadata to successful results
                    if isinstance(result, dict):
                        result["router_actor_id"] = ray.get_runtime_context().get_actor_id()
                    processed_results.append(result)
            
            batch_time = time.time() - batch_start
            self.request_count += len(batch_requests)
            self.total_routing_time += batch_time
            
            successful_count = len([r for r in processed_results if r.get("status") == "success"])
            failed_count = len(batch_requests) - successful_count
            
            self.logger.info(
                f"Ray router completed batch for {self.model_name} in {batch_time:.2f}s: "
                f"{successful_count} successful, {failed_count} failed"
            )
            
            return processed_results
            
        except Exception as e:
            batch_time = time.time() - batch_start
            self.failed_requests += len(batch_requests)
            self.total_routing_time += batch_time
            
            self.logger.error(f"Ray router batch processing failed for {self.model_name}: {e}")
            
            # Return error results for all requests in the batch
            error_results = []
            for request_data in batch_requests:
                error_result = {
                    "status": "error",
                    "ma_don_vi": request_data.get("ma_don_vi", "UNKNOWN"),
                    "ma_bao_cao": request_data.get("ma_bao_cao", "UNKNOWN"),
                    "ky_du_lieu": request_data.get("ky_du_lieu", "UNKNOWN"),
                    "ma_tieu_chi": request_data.get("ma_tieu_chi", "UNKNOWN"),
                    "fld_code": request_data.get("fld_code", "UNKNOWN"),
                    "model_name": self.model_name,
                    "error_message": f"Ray router batch error: {str(e)}",
                    "is_anomaly": False,
                    "prediction": None,
                    "router_actor_id": ray.get_runtime_context().get_actor_id()
                }
                error_results.append(error_result)
                
            return error_results
    
    async def route_parallel_tasks(self, prediction_tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Route prediction tasks using Ray parallel processing with proper task management"""
        parallel_start = time.time()
        
        try:
            self.logger.info(f"Ray router executing {len(prediction_tasks)} parallel tasks for {self.model_name}")
            
            # Create Ray remote tasks for true parallel processing
            ray_tasks = []
            for task_data in prediction_tasks:
                # Each task is executed as a Ray remote call for parallel processing
                ray_task = self.deployment_actor.predict.remote(task_data)
                ray_tasks.append(ray_task)
            
            # Wait for all Ray tasks to complete in parallel
            parallel_results = await asyncio.gather(*[task for task in ray_tasks], return_exceptions=True)
            
            # Process results
            processed_results = []
            successful_tasks = 0
            failed_tasks = 0
            
            for i, result in enumerate(parallel_results):
                if isinstance(result, Exception):
                    # Handle Ray task exceptions
                    task_data = prediction_tasks[i]
                    error_result = {
                        "status": "error",
                        "ma_don_vi": task_data.get("ma_don_vi", "UNKNOWN"),
                        "ma_bao_cao": task_data.get("ma_bao_cao", "UNKNOWN"),
                        "ky_du_lieu": task_data.get("ky_du_lieu", "UNKNOWN"),
                        "ma_tieu_chi": task_data.get("ma_tieu_chi", "UNKNOWN"),
                        "fld_code": task_data.get("fld_code", "UNKNOWN"),
                        "model_name": self.model_name,
                        "error_message": f"Ray parallel task error: {str(result)}",
                        "is_anomaly": False,
                        "prediction": None,
                        "router_actor_id": ray.get_runtime_context().get_actor_id()
                    }
                    processed_results.append(error_result)
                    failed_tasks += 1
                else:
                    # Add parallel processing metadata
                    if isinstance(result, dict):
                        result["ray_parallel_processing"] = True
                        result["router_actor_id"] = ray.get_runtime_context().get_actor_id()
                    processed_results.append(result)
                    if result.get("status") == "success":
                        successful_tasks += 1
                    else:
                        failed_tasks += 1
            
            parallel_time = time.time() - parallel_start
            self.request_count += len(prediction_tasks)
            self.total_routing_time += parallel_time
            
            self.logger.info(
                f"Ray router parallel processing completed for {self.model_name} in {parallel_time:.2f}s: "
                f"{successful_tasks} successful, {failed_tasks} failed"
            )
            
            return processed_results
            
        except Exception as e:
            parallel_time = time.time() - parallel_start
            self.failed_requests += len(prediction_tasks)
            self.total_routing_time += parallel_time
            
            self.logger.error(f"Ray router parallel processing failed for {self.model_name}: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Ray router statistics"""
        avg_routing_time = (
            self.total_routing_time / max(1, self.request_count)
        )
        error_rate = self.failed_requests / max(1, self.request_count)
        
        return {
            "model_name": self.model_name,
            "request_count": self.request_count,
            "failed_requests": self.failed_requests,
            "total_routing_time": self.total_routing_time,
            "avg_routing_time": avg_routing_time,
            "error_rate": error_rate,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "router_actor_id": ray.get_runtime_context().get_actor_id(),
            "router_status": "active"
        }
    
    def update_concurrency_limit(self, new_limit: int):
        """Update the maximum concurrent tasks limit"""
        self.max_concurrent_tasks = new_limit
        self.task_semaphore = asyncio.Semaphore(new_limit)
        self.logger.info(f"Updated concurrency limit for {self.model_name} to {new_limit}")
    
    def cleanup(self):
        """Cleanup router resources"""
        try:
            self.deployment_actor = None
            self.logger.info(f"Ray router actor cleanup completed for {self.model_name}")
        except Exception as e:
            self.logger.error(f"Error during Ray router cleanup for {self.model_name}: {e}")