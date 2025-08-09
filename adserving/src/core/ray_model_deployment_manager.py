"""
Ray Model Deployment Manager

This module implements the requested workflow: load model > deploy model > ray model router
Each model gets its own dedicated Ray Serve deployment and router for optimal performance.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Set
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import ray
from ray import serve
import pandas as pd

from adserving.src.utils.logger import get_logger
from adserving.src.core.ray_actors import RayModelDeployment, RayModelRouter
from adserving.src.core.utils.model_tier import ModelTier


logger = get_logger()


def ray_serializable_model_loader(model_name: str):
    """
    Ray-serializable standalone model loading function
    This function can be serialized by Ray unlike bound methods
    FIXED: Use synchronous loading to avoid async/sync conflicts
    """
    try:
        # Import here to avoid circular imports
        from adserving.src.core.model_manager import ModelManager
        from adserving.src.config.config_manager import get_config
        
        # Create a temporary model manager instance for loading
        config = get_config()
        mlflow_uri = getattr(config, 'mlflow_tracking_uri', 'http://localhost:5000')
        temp_model_manager = ModelManager(mlflow_tracking_uri=mlflow_uri)
        
        # Use synchronous model loading to avoid event loop conflicts
        # FIXED: Use _load_model_sync instead of async version to prevent
        # "coroutines cannot be used with run_in_executor()" error
        return temp_model_manager._load_model_sync(model_name)
            
    except Exception as e:
        logger.error(f"Ray serializable model loader failed for {model_name}: {e}")
        return None


@dataclass
class ModelDeploymentInfo:
    """Information about a deployed model"""
    model_name: str
    deployment_name: str
    router_name: str
    deployment_handle: Any
    router_handle: Any
    loaded_at: float
    deployed_at: float
    status: str


@ray.remote(num_cpus=1, num_gpus=0)
class RayModelRouter:
    """Dedicated Ray router for a specific model"""
    
    def __init__(self, model_name: str, deployment_handle):
        self.model_name = model_name
        self.deployment_handle = deployment_handle
        self.request_count = 0
        self.total_response_time = 0.0
        self.error_count = 0
        self.logger = logger
        
    async def route_request(self, input_data: Dict) -> Dict:
        """Route request to the model's dedicated deployment"""
        start_time = time.time()
        self.request_count += 1
        
        try:
            # Route to the specific model deployment
            result = await self.deployment_handle.predict.remote(input_data)
            
            response_time = time.time() - start_time
            self.total_response_time += response_time
            
            self.logger.debug(f"Router for {self.model_name} processed request in {response_time:.3f}s")
            return result
            
        except Exception as e:
            self.error_count += 1
            response_time = time.time() - start_time
            self.total_response_time += response_time
            
            self.logger.error(f"Router for {self.model_name} failed: {e}")
            return {
                "status": "error",
                "error_message": f"Router error for {self.model_name}: {str(e)}",
                "model_name": self.model_name,
                "response_time": response_time
            }
    
    def get_stats(self) -> Dict:
        """Get router statistics"""
        avg_response_time = (
            self.total_response_time / self.request_count 
            if self.request_count > 0 else 0
        )
        error_rate = (
            self.error_count / self.request_count 
            if self.request_count > 0 else 0
        )
        
        return {
            "model_name": self.model_name,
            "request_count": self.request_count,
            "avg_response_time": avg_response_time,
            "error_count": self.error_count,
            "error_rate": error_rate
        }


@ray.remote
class RayModelDeployment:
    """Dedicated Ray deployment for a specific model"""
    
    def __init__(self, model_name: str, model_info):
        self.model_name = model_name
        self.model_info = model_info
        self.prediction_count = 0
        self.logger = logger
        
    async def initialize(self):
        """Initialize the deployment with pre-loaded model"""
        try:
            self.logger.info(f"Initializing Ray deployment for model {self.model_name}")
            
            if not self.model_info:
                raise Exception(f"Model info not provided for {self.model_name}")
                
            self.logger.info(f"Model {self.model_name} initialized successfully in Ray deployment")
            return {"status": "success", "model_name": self.model_name}
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model {self.model_name}: {e}")
            return {"status": "error", "error_message": str(e), "model_name": self.model_name}
    
    async def predict(self, input_data: Dict) -> Dict:
        """Make prediction using the loaded model"""
        start_time = time.time()
        self.prediction_count += 1
        
        try:
            if not self.model_info:
                raise Exception(f"Model {self.model_name} not loaded")
            
            # Convert input data to DataFrame format
            df_input = pd.DataFrame({
                "ma_tieu_chi": [input_data.get("ma_tieu_chi", "UNKNOWN")],
                "fld_code": [input_data.get("fld_code", "UNKNOWN")],
                "gia_tri": [float(input_data.get("gia_tri", 0.0))]
            })
            
            # Make prediction using model
            prediction_result = self.model_info.model.predict(df_input[["fld_code", "gia_tri"]])
            anomaly_score = float(prediction_result[0]) if prediction_result is not None else None
            
            processing_time = time.time() - start_time
            is_anomaly = (
                anomaly_score is not None and 
                anomaly_score > getattr(self.model_info, 'anomaly_threshold', 0.5)
            )
            
            # Return detailed prediction result
            return {
                "status": "success",
                "model_name": self.model_name,
                "ma_tieu_chi": input_data.get("ma_tieu_chi", "UNKNOWN"),
                "fld_code": input_data.get("fld_code", "UNKNOWN"),
                "is_anomaly": is_anomaly,
                "prediction": anomaly_score,
                "anomaly_threshold": getattr(self.model_info, 'anomaly_threshold', 0.5),
                "processing_time": processing_time,
                "model_version": getattr(self.model_info, 'version', None),
                "prediction_count": self.prediction_count
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Prediction failed for {self.model_name}: {e}")
            
            return {
                "status": "error",
                "model_name": self.model_name,
                "ma_tieu_chi": input_data.get("ma_tieu_chi", "UNKNOWN"),
                "fld_code": input_data.get("fld_code", "UNKNOWN"),
                "is_anomaly": False,
                "prediction": None,
                "error_message": str(e),
                "processing_time": processing_time,
                "prediction_count": self.prediction_count
            }
    
    def get_stats(self) -> Dict:
        """Get deployment statistics"""
        return {
            "model_name": self.model_name,
            "prediction_count": self.prediction_count,
            "model_loaded": self.model_info is not None,
            "model_version": getattr(self.model_info, 'version', None) if self.model_info else None
        }


class RayModelDeploymentManager:
    """Manager for Ray model deployments using pure Ray parallel processing (no workers)"""
    
    def __init__(self, model_manager, ray_deployment_manager=None):
        self.model_manager = model_manager
        self.logger = get_logger()
        
        # Load Ray optimization configuration
        from adserving.src.config.config_manager import get_config
        config = get_config()
        ray_config = getattr(config, 'ray', None)
        ray_optimization = getattr(ray_config, 'optimization', {}) if ray_config else {}
        
        # Extract configurable parameters with fallback defaults
        self.deployment_batch_size = ray_optimization.get('deployment_batch_size', 5)
        self.deployment_batch_interval = ray_optimization.get('deployment_batch_interval', 2.0)
        self.num_deployment_actors = ray_optimization.get('num_deployment_actors', 4)
        self.model_init_timeout = ray_optimization.get('model_initialization_timeout', 30.0)
        self.request_routing_timeout = ray_optimization.get('request_routing_timeout', 20.0)
        self.max_concurrent_tasks = ray_optimization.get('max_concurrent_tasks_per_router', 20)
        
        # Use Ray-based deployment manager (eliminates all worker mechanisms)
        from adserving.src.core.manager.ray_deployment_manager import RayDeploymentManager
        self.ray_deployment_manager = ray_deployment_manager or RayDeploymentManager(
            batch_size=self.deployment_batch_size,
            batch_interval=self.deployment_batch_interval,
            load_model_func=ray_serializable_model_loader,  # Use Ray-serializable function
            num_actors=self.num_deployment_actors  # Configurable Ray actors instead of workers
        )
        
        # Initialize Ray deployment infrastructure (no workers)
        self.ray_deployment_manager.initialize_infrastructure()
        
        # Tracking deployed models
        self.deployed_models: Dict[str, ModelDeploymentInfo] = {}
        
        # Ray initialization
        self._ensure_ray_initialized()
    
    def _load_model_sync(self, model_name: str):
        """Synchronous model loading for DeploymentManager integration"""
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.model_manager.load_model_async(model_name))
            finally:
                loop.close()
        except Exception as e:
            self.logger.error(f"Error in sync model loading for {model_name}: {e}")
            return None
    
    def _ensure_ray_initialized(self):
        """Ensure Ray is initialized"""
        try:
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
                self.logger.info("Ray initialized for model deployments")
            else:
                self.logger.info("Ray already initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize Ray: {e}")
            raise
    
    async def load_deploy_and_route_model(self, model_name: str) -> Dict[str, Any]:
        """
        Complete workflow using Ray parallel processing: Load model > Deploy model > Create Ray model router
        Returns deployment and router information (no workers used)
        """
        if model_name in self.deployed_models:
            self.logger.info(f"Model {model_name} already deployed with router")
            return {
                "status": "already_deployed", 
                "model_name": model_name,
                "deployment_info": self.deployed_models[model_name]
            }
        
        try:
            start_time = time.time()
            
            self.logger.info(f"Starting Ray parallel load > deploy > route workflow for {model_name}")
            
            # Step 1: Use Ray deployment manager to deploy model (replaces worker staging)
            self.logger.info(f"Step 1: Loading and deploying model {model_name} using Ray parallel processing")
            deployment_result = await self.ray_deployment_manager.deploy_single_model(model_name)
            if deployment_result.get("status") != "success":
                raise Exception(f"Failed to deploy model {model_name}: {deployment_result.get('error_message', 'Unknown error')}")
            
            # Step 2: Load model info directly (no staging area needed with Ray)
            model_info = await self.model_manager.load_model_async(model_name)
            if not model_info:
                raise Exception(f"Failed to load model info for {model_name}")
            
            load_time = time.time()
            
            # Step 3: Create Ray deployment for the loaded model
            self.logger.info(f"Step 2: Creating Ray deployment for {model_name}")
            deployment_name = f"model_deployment_{model_name.replace('.', '_')}"
            
            # Create Ray deployment with the loaded model info
            deployment_actor = RayModelDeployment.remote(model_name, model_info)
            try:
                # Add configurable timeout to prevent hanging on Ray remote initialization
                init_result = await asyncio.wait_for(
                    deployment_actor.initialize.remote(),
                    timeout=self.model_init_timeout  # Configurable timeout for model initialization
                )
            except asyncio.TimeoutError:
                raise Exception(f"Model {model_name} initialization timed out after {self.model_init_timeout} seconds")
            
            if init_result["status"] != "success":
                raise Exception(f"Deployment initialization failed: {init_result}")
            
            deploy_time = time.time()
            
            # Step 4: Create Ray model router
            self.logger.info(f"Step 3: Creating Ray router for {model_name}")
            router_name = f"model_router_{model_name.replace('.', '_')}"
            router_actor = RayModelRouter.remote(model_name, deployment_actor)
            
            route_time = time.time()
            
            # Store deployment information
            deployment_info = ModelDeploymentInfo(
                model_name=model_name,
                deployment_name=deployment_name,
                router_name=router_name,
                deployment_handle=deployment_actor,
                router_handle=router_actor,
                loaded_at=load_time,
                deployed_at=deploy_time,
                status="active"
            )
            
            self.deployed_models[model_name] = deployment_info
            
            total_time = time.time() - start_time
            self.logger.info(
                f"Ray parallel deployment completed for {model_name} "
                f"in {total_time:.2f}s (load: {load_time-start_time:.2f}s, "
                f"deploy: {deploy_time-load_time:.2f}s, route: {route_time-deploy_time:.2f}s)"
            )
            
            return {
                "status": "success",
                "model_name": model_name,
                "deployment_info": deployment_info,
                "timings": {
                    "total_time": total_time,
                    "load_time": load_time - start_time,
                    "deploy_time": deploy_time - load_time,
                    "route_time": route_time - deploy_time
                },
                "ray_based": True,
                "no_workers": True
            }
            
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            
            # Enhanced error logging for better debugging
            self.logger.error(
                f"Ray parallel deployment failed for {model_name}: {e}\n"
                f"Full traceback:\n{error_traceback}\n"
                f"Deployment stage context: model_name={model_name}, "
                f"elapsed_time={time.time()-start_time:.2f}s"
            )
            
            # Clean up Ray actors if deployment failed
            try:
                if model_name in self.deployed_models:
                    deployment_info = self.deployed_models[model_name]
                    ray.kill(deployment_info.deployment_handle)
                    ray.kill(deployment_info.router_handle)
                    del self.deployed_models[model_name]
                    self.logger.info(f"Cleaned up Ray actors for failed deployment: {model_name}")
            except Exception as cleanup_error:
                self.logger.warning(f"Failed to cleanup Ray actors for {model_name}: {cleanup_error}")
                
            return {
                "status": "error",
                "model_name": model_name,
                "error_message": str(e),
                "error_traceback": error_traceback,
                "deployment_context": {
                    "elapsed_time": time.time() - start_time,
                    "stage": "deployment_workflow"
                }
            }
    
    async def route_request_to_model(self, model_name: str, input_data: Dict) -> Dict:
        """Route request to the specific model's dedicated router"""
        if model_name not in self.deployed_models:
            return {
                "status": "error",
                "error_message": f"Model {model_name} not deployed with router",
                "model_name": model_name
            }
        
        deployment_info = self.deployed_models[model_name]
        try:
            # Use the dedicated router for this model with configurable timeout to prevent hanging
            result = await asyncio.wait_for(
                deployment_info.router_handle.route_request.remote(input_data),
                timeout=self.request_routing_timeout  # Configurable timeout to prevent hanging
            )
            return result
        except asyncio.TimeoutError:
            self.logger.error(f"Ray remote call timed out for {model_name} after {self.request_routing_timeout} seconds")
            return {
                "status": "error",
                "error_message": f"Request routing timed out for {model_name} ({self.request_routing_timeout}s timeout)",
                "model_name": model_name
            }
        except Exception as e:
            self.logger.error(f"Failed to route request to {model_name}: {e}")
            return {
                "status": "error",
                "error_message": f"Routing failed for {model_name}: {str(e)}",
                "model_name": model_name
            }
    
    def _sort_models_by_tier_priority(self, model_names: List[str]) -> List[str]:
        """Sort models by tier priority: HOT -> WARM -> COLD"""
        def get_tier_priority(model_name: str) -> int:
            try:
                # Try to get tier from model manager if available
                if hasattr(self.model_manager, 'get_model_tier'):
                    tier = self.model_manager.get_model_tier(model_name)
                    if tier == ModelTier.HOT:
                        return 0  # Highest priority
                    elif tier == ModelTier.WARM:
                        return 1  # Medium priority
                    else:
                        return 2  # Lowest priority (COLD)
                else:
                    # Fallback: HOT models based on naming convention
                    if 'hot' in model_name.lower():
                        return 0
                    elif 'warm' in model_name.lower():
                        return 1
                    else:
                        return 2
            except Exception:
                return 2  # Default to COLD priority if error
        
        sorted_models = sorted(model_names, key=get_tier_priority)
        
        # Log tier-based sorting
        hot_models = [m for m in sorted_models if get_tier_priority(m) == 0]
        warm_models = [m for m in sorted_models if get_tier_priority(m) == 1]
        cold_models = [m for m in sorted_models if get_tier_priority(m) == 2]
        
        self.logger.info(f"Tier-based model sorting:")
        self.logger.info(f"  HOT tier (priority 1): {len(hot_models)} models")
        self.logger.info(f"  WARM tier (priority 2): {len(warm_models)} models")
        self.logger.info(f"  COLD tier (priority 3): {len(cold_models)} models")
        
        return sorted_models
    
    async def _deploy_models_truly_parallel(self, model_names: List[str]) -> Dict[str, Any]:
        """Deploy models using MEMORY-SAFE BATCHED deployment to prevent OOM"""
        self.logger.info(f"Deploying {len(model_names)} models using BATCHED deployment (OOM prevention)")
        
        # Memory-safe batch configuration
        MAX_BATCH_SIZE = 10  # Maximum 10 models per batch to prevent OOM
        MAX_CONCURRENT_ACTORS = 20  # Maximum concurrent Ray actors at any time
        
        # Create semaphore to limit concurrent Ray actor creation
        deployment_semaphore = asyncio.Semaphore(MAX_CONCURRENT_ACTORS)
        
        async def _deploy_with_semaphore(model_name: str):
            """Deploy single model with semaphore to limit concurrency"""
            async with deployment_semaphore:
                try:
                    return await self._deploy_single_model_direct(model_name)
                except Exception as e:
                    self.logger.error(f"Deployment failed for {model_name}: {e}")
                    return {
                        "status": "error",
                        "error_message": f"Batched deployment failed: {str(e)}",
                        "model_name": model_name
                    }
        
        # Split models into batches
        batches = [model_names[i:i + MAX_BATCH_SIZE] for i in range(0, len(model_names), MAX_BATCH_SIZE)]
        
        self.logger.info(f"Split {len(model_names)} models into {len(batches)} batches of max {MAX_BATCH_SIZE} models each")
        self.logger.info(f"Maximum concurrent Ray actors limited to: {MAX_CONCURRENT_ACTORS}")
        
        start_time = time.time()
        deployment_results = {}
        successful_count = 0
        failed_count = 0
        
        # Process each batch with memory monitoring
        for batch_idx, batch in enumerate(batches):
            batch_start_time = time.time()
            self.logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} with {len(batch)} models...")
            
            # Check memory before batch deployment
            try:
                import psutil
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                
                if memory_percent > 90.0:
                    self.logger.warning(f"High memory usage detected: {memory_percent:.1f}% - Adding delay before batch")
                    await asyncio.sleep(2.0)  # Brief pause to let memory pressure subside
                    
                self.logger.debug(f"Memory status before batch {batch_idx + 1}: {memory_percent:.1f}%")
            except ImportError:
                self.logger.warning("psutil not available - skipping memory monitoring")
            
            # Deploy batch with semaphore-controlled concurrency
            batch_tasks = [_deploy_with_semaphore(model_name) for model_name in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process batch results
            for model_name, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    deployment_results[model_name] = {
                        "status": "error",
                        "error_message": f"Batch deployment exception: {str(result)}",
                        "model_name": model_name
                    }
                    failed_count += 1
                else:
                    deployment_results[model_name] = result
                    if result.get("status") == "success":
                        successful_count += 1
                    else:
                        failed_count += 1
            
            batch_time = time.time() - batch_start_time
            self.logger.info(f"Batch {batch_idx + 1}/{len(batches)} completed in {batch_time:.2f}s: "
                           f"{len([r for r in batch_results if not isinstance(r, Exception) and r.get('status') == 'success'])} successful, "
                           f"{len([r for r in batch_results if isinstance(r, Exception) or r.get('status') != 'success'])} failed")
            
            # Brief pause between batches to allow memory cleanup
            if batch_idx < len(batches) - 1:  # Don't pause after last batch
                await asyncio.sleep(0.5)
        
        total_time = time.time() - start_time
        self.logger.info(f"MEMORY-SAFE batched deployment completed in {total_time:.2f}s: "
                        f"{successful_count} successful, {failed_count} failed")
        
        return {
            "results": deployment_results,
            "total_time": total_time,
            "deployment_type": "memory_safe_batched",
            "batch_info": {
                "total_batches": len(batches),
                "max_batch_size": MAX_BATCH_SIZE,
                "max_concurrent_actors": MAX_CONCURRENT_ACTORS
            }
        }
    
    async def _deploy_single_model_direct(self, model_name: str) -> Dict[str, Any]:
        """Deploy single model directly without going through problematic deployment manager"""
        try:
            # Load model info
            model_info = await self.model_manager.load_model_async(model_name)
            if not model_info:
                return {
                    "status": "error",
                    "error_message": f"Failed to load model info for {model_name}",
                    "model_name": model_name
                }
            
            # Create Ray deployment and router directly
            deployment_name = f"model_deployment_{model_name.replace('.', '_')}"
            deployment_actor = RayModelDeployment.remote(model_name, model_info)
            router_actor = RayModelRouter.remote(model_name, deployment_actor)
            
            # Initialize deployment
            init_result = await deployment_actor.initialize.remote()
            if init_result.get("status") != "success":
                return {
                    "status": "error", 
                    "error_message": f"Ray deployment initialization failed for {model_name}",
                    "model_name": model_name
                }
            
            # Store deployment info
            deployment_info = ModelDeploymentInfo(
                model_name=model_name,
                deployment_name=deployment_name,
                router_name=f"model_router_{model_name.replace('.', '_')}",
                deployment_handle=deployment_actor,
                router_handle=router_actor,
                loaded_at=time.time(),
                deployed_at=time.time(),
                status="active"
            )
            self.deployed_models[model_name] = deployment_info
            
            return {
                "status": "success",
                "model_name": model_name,
                "deployment_info": deployment_info,
                "ray_based": True,
                "parallel_deployment": True
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error_message": f"Direct deployment failed for {model_name}: {str(e)}",
                "model_name": model_name
            }
    
    async def deploy_multiple_models_parallel(self, model_names: List[str]) -> Dict[str, Any]:
        """Deploy multiple models using true parallel processing with tier-based priority"""
        self.logger.info(f"Starting tier-based parallel deployment of {len(model_names)} models")
        start_time = time.time()
        
        # Sort models by tier priority (HOT first, then WARM, then COLD)
        sorted_models = self._sort_models_by_tier_priority(model_names)
        
        # Deploy all models in parallel using asyncio.gather (no sequential bottleneck)
        deployment_result = await self._deploy_models_truly_parallel(sorted_models)
        
        # Extract results from parallel deployment (models already deployed and routers created)
        deployment_results = deployment_result.get("results", {})
        successful_deployments = 0
        failed_deployments = 0
        
        # Count successes and failures
        for model_name, result in deployment_results.items():
            if result.get("status") == "success":
                successful_deployments += 1
            else:
                failed_deployments += 1
        
        total_time = time.time() - start_time
        
        self.logger.info(
            f"Ray parallel deployment completed in {total_time:.2f}s: "
            f"{successful_deployments} successful, {failed_deployments} failed"
        )
        
        return {
            "status": "completed",
            "total_models": len(model_names),
            "successful_deployments": successful_deployments,
            "failed_deployments": failed_deployments,
            "total_time": total_time,
            "results": deployment_results,
            "ray_based": True,
            "no_workers": True,
            "ray_deployment_stats": deployment_result
        }
    
    async def _ray_deploy_single_model(self, model_name: str):
        """Ray-based deployment method (replaces worker-based deployment)"""
        ray_start_time = time.time()
        try:
            self.logger.info(f"Ray deploying model {model_name} using pure Ray parallel processing")
            result = await self.load_deploy_and_route_model(model_name)
            
            if result.get("status") in ["success", "already_deployed"]:
                ray_time = time.time() - ray_start_time
                self.logger.info(f"Ray successfully deployed {model_name} with router in {ray_time:.2f}s")
                return result
            else:
                error_msg = result.get('error_message', 'Unknown error')
                error_traceback = result.get('error_traceback', 'No traceback available')
                
                self.logger.error(
                    f"Ray deployment failed for {model_name}: {error_msg}\n"
                    f"Ray execution time: {time.time() - ray_start_time:.2f}s\n"
                    f"Deployment result traceback:\n{error_traceback}"
                )
                raise Exception(f"Ray deployment failed: {error_msg}")
                
        except Exception as e:
            import traceback
            ray_time = time.time() - ray_start_time
            ray_traceback = traceback.format_exc()
            
            self.logger.error(
                f"Ray-based deployment failed for {model_name}: {e}\n"
                f"Ray execution time: {ray_time:.2f}s\n"
                f"Ray traceback:\n{ray_traceback}"
            )
            raise

    async def _deploy_single_model(self, model_name: str):
        """Legacy compatibility method - now uses Ray-based deployment"""
        self.logger.info(f"Redirecting to Ray-based deployment for {model_name}")
        return await self._ray_deploy_single_model(model_name)
    
    def get_deployed_models(self) -> List[str]:
        """Get list of deployed models"""
        return list(self.deployed_models.keys())
    
    def get_deployment_stats(self) -> Dict[str, Any]:
        """Get comprehensive deployment statistics"""
        return {
            "total_deployed_models": len(self.deployed_models),
            "loading_models": len(self.loading_models),
            "deployed_models": list(self.deployed_models.keys()),
            "deployment_details": {
                name: {
                    "deployment_name": info.deployment_name,
                    "router_name": info.router_name,
                    "status": info.status,
                    "loaded_at": info.loaded_at,
                    "deployed_at": info.deployed_at
                }
                for name, info in self.deployed_models.items()
            }
        }
    
    async def get_model_stats(self, model_name: str) -> Dict[str, Any]:
        """Get statistics for a specific model"""
        if model_name not in self.deployed_models:
            return {"error": f"Model {model_name} not deployed"}
        
        deployment_info = self.deployed_models[model_name]
        
        try:
            # Get stats from both deployment and router
            deployment_stats = await deployment_info.deployment_handle.get_stats.remote()
            router_stats = await deployment_info.router_handle.get_stats.remote()
            
            return {
                "model_name": model_name,
                "deployment_stats": deployment_stats,
                "router_stats": router_stats,
                "deployment_info": {
                    "deployment_name": deployment_info.deployment_name,
                    "router_name": deployment_info.router_name,
                    "status": deployment_info.status
                }
            }
        except Exception as e:
            import traceback
            stats_traceback = traceback.format_exc()
            stats_context = {
                "model_name": model_name,
                "deployment_exists": model_name in self.deployed_models,
                "total_deployed_models": len(self.deployed_models),
                "operation": "get_model_stats"
            }
            
            self.logger.error(
                f"Failed to get stats for {model_name}: {e}\n"
                f"Stats context: {stats_context}\n"
                f"Full traceback:\n{stats_traceback}"
            )
            return {"error": f"Failed to get stats for {model_name}: {str(e)}"}
    
    def cleanup(self):
        """Cleanup all deployments and routers"""
        self.logger.info("Cleaning up Ray model deployments and routers")
        
        for model_name, deployment_info in self.deployed_models.items():
            try:
                # Kill Ray actors
                ray.kill(deployment_info.deployment_handle)
                ray.kill(deployment_info.router_handle)
                self.logger.info(f"Cleaned up deployment and router for {model_name}")
            except Exception as e:
                import traceback
                cleanup_traceback = traceback.format_exc()
                cleanup_context = {
                    "model_name": model_name,
                    "deployment_name": deployment_info.deployment_name,
                    "router_name": deployment_info.router_name,
                    "operation": "cleanup_ray_actors",
                    "total_models_to_cleanup": len(self.deployed_models)
                }
                
                self.logger.error(
                    f"Error cleaning up {model_name}: {e}\n"
                    f"Cleanup context: {cleanup_context}\n"
                    f"Full traceback:\n{cleanup_traceback}"
                )
        
        self.deployed_models.clear()
        self.loading_models.clear()
        
        self.logger.info("Ray model deployment manager cleanup completed")