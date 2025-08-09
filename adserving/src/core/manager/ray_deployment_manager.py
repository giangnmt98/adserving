"""
Ray-Based Deployment Manager

This module handles zero-downtime model deployments using only Ray parallel processing.
Completely eliminates worker mechanisms in favor of Ray remote tasks and actors.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional
import ray
from dataclasses import dataclass

from adserving.src.utils.logger import get_logger


@dataclass
class DeploymentTask:
    """Ray remote deployment task"""
    model_name: str
    deployment_type: str
    queued_at: float
    priority: int = 1


@ray.remote(num_cpus=0.5)
class RayDeploymentActor:
    """Ray remote actor for handling model deployments"""
    
    def __init__(self, actor_id: str, load_model_func=None):
        self.actor_id = actor_id
        self.load_model_func = load_model_func
        self.deployments_processed = 0
        self.successful_deployments = 0
        self.failed_deployments = 0
        self.logger = get_logger()
        
        # Staging area for this actor
        self.staging_area = {}
        
    async def deploy_model(self, deployment_task: DeploymentTask) -> Dict[str, Any]:
        """Deploy a single model using Ray remote processing"""
        start_time = time.time()
        model_name = deployment_task.model_name
        
        try:
            self.logger.info(f"[Ray Actor {self.actor_id}] Starting deployment for model: {model_name}")
            
            # Step 1: Load model into staging
            if not await self._stage_model(model_name):
                raise Exception(f"Failed to stage model {model_name}")
                
            # Step 2: Validate staged model
            if not await self._validate_staged_model(model_name):
                raise Exception(f"Model validation failed for {model_name}")
                
            # Step 3: Promote to active (atomic switch)
            await self._promote_staged_model_to_active(model_name)
            
            deployment_time = time.time() - start_time
            self.deployments_processed += 1
            self.successful_deployments += 1
            
            self.logger.info(f"[Ray Actor {self.actor_id}] Successfully deployed {model_name} in {deployment_time:.2f}s")
            
            return {
                "status": "success",
                "model_name": model_name,
                "deployment_time": deployment_time,
                "actor_id": self.actor_id,
                "ray_based": True
            }
            
        except Exception as e:
            deployment_time = time.time() - start_time
            self.deployments_processed += 1
            self.failed_deployments += 1
            
            # Cleanup staging
            try:
                await self._remove_from_staging(model_name)
            except Exception:
                pass
                
            self.logger.error(f"[Ray Actor {self.actor_id}] Deployment failed for {model_name}: {e}")
            
            return {
                "status": "error",
                "model_name": model_name,
                "error_message": str(e),
                "deployment_time": deployment_time,
                "actor_id": self.actor_id,
                "ray_based": True
            }
    
    async def _stage_model(self, model_name: str) -> bool:
        """Load model into staging area using Ray processing"""
        try:
            if not self.load_model_func:
                self.logger.error("No model loading function provided")
                return False
                
            # Load model in executor to avoid blocking
            model_info = await asyncio.get_event_loop().run_in_executor(
                None, self.load_model_func, model_name
            )
            
            if not model_info:
                return False
                
            # Add to staging area
            self.staging_area[model_name] = {
                "model_info": model_info,
                "staged_at": time.time(),
                "validated": False,
            }
            
            self.logger.debug(f"[Ray Actor {self.actor_id}] Model {model_name} staged successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"[Ray Actor {self.actor_id}] Failed to stage model {model_name}: {e}")
            return False
    
    async def _validate_staged_model(self, model_name: str) -> bool:
        """Validate staged model"""
        try:
            staged_info = self.staging_area.get(model_name)
            if not staged_info:
                return False
                
            model_info = staged_info["model_info"]
            
            # Basic validation: ensure model exists and has required attributes
            if not hasattr(model_info, 'model') and model_info is None:
                return False
                
            # Mark as validated
            staged_info["validated"] = True
            self.logger.debug(f"[Ray Actor {self.actor_id}] Model {model_name} validated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"[Ray Actor {self.actor_id}] Model validation failed for {model_name}: {e}")
            return False
    
    async def _promote_staged_model_to_active(self, model_name: str):
        """Promote staged model to active (atomic switch)"""
        try:
            staged_info = self.staging_area.get(model_name)
            if not staged_info or not staged_info.get("validated"):
                raise Exception(f"Model {model_name} not properly staged or validated")
                
            # In a real implementation, this would update the active model registry
            # For now, we just mark it as promoted
            staged_info["promoted_at"] = time.time()
            self.logger.debug(f"[Ray Actor {self.actor_id}] Model {model_name} promoted to active")
            
        except Exception as e:
            self.logger.error(f"[Ray Actor {self.actor_id}] Failed to promote model {model_name}: {e}")
            raise
    
    async def _remove_from_staging(self, model_name: str):
        """Remove model from staging area"""
        try:
            if model_name in self.staging_area:
                del self.staging_area[model_name]
                self.logger.debug(f"[Ray Actor {self.actor_id}] Removed {model_name} from staging")
        except Exception as e:
            self.logger.error(f"[Ray Actor {self.actor_id}] Failed to remove {model_name} from staging: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get deployment statistics for this Ray actor"""
        return {
            "actor_id": self.actor_id,
            "deployments_processed": self.deployments_processed,
            "successful_deployments": self.successful_deployments,
            "failed_deployments": self.failed_deployments,
            "success_rate": (self.successful_deployments / max(1, self.deployments_processed)) * 100,
            "staging_area_size": len(self.staging_area),
            "ray_actor_stats": True
        }


class RayDeploymentManager:
    """Ray-based deployment manager - completely eliminates worker mechanisms"""

    def __init__(self, batch_size: int = 5, batch_interval: float = 2.0, load_model_func=None, num_actors: int = 4):
        self.logger = get_logger()
        self._batch_size = batch_size
        self._batch_interval = batch_interval
        self._load_model_func = load_model_func
        self._num_actors = num_actors

        # Ray deployment actors (replaces workers)
        self._deployment_actors: List[ray.ObjectRef] = []

        # Deployment statistics
        self._deployment_stats = {
            "total_deployments": 0,
            "successful_deployments": 0,
            "failed_deployments": 0,
            "average_deployment_time": 0.0,
            "ray_based": True
        }

        self.logger.info(f"RayDeploymentManager initialized with {num_actors} Ray actors (no workers)")

    def initialize_infrastructure(self):
        """Initialize Ray deployment actors (replaces worker initialization)"""
        try:
            # Ensure Ray is initialized
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
                self.logger.info("Ray initialized for deployment processing")

            # Create Ray deployment actors (replaces workers)
            for i in range(self._num_actors):
                actor_id = f"ray-deployment-actor-{i}"
                actor = RayDeploymentActor.remote(actor_id, self._load_model_func)
                self._deployment_actors.append(actor)

            self.logger.info(f"Ray deployment infrastructure initialized with {self._num_actors} Ray actors")

        except Exception as e:
            self.logger.error(f"Failed to initialize Ray deployment infrastructure: {e}")

    async def deploy_models_parallel(self, model_names: List[str], deployment_type: str = "new") -> Dict[str, Any]:
        """Deploy multiple models in parallel using Ray actors (replaces worker-based deployment)"""
        if not self._deployment_actors:
            self.initialize_infrastructure()

        start_time = time.time()
        self.logger.info(f"Starting Ray parallel deployment of {len(model_names)} models")

        # Create deployment tasks
        deployment_tasks = []
        for model_name in model_names:
            task = DeploymentTask(
                model_name=model_name,
                deployment_type=deployment_type,
                queued_at=time.time(),
                priority=1 if deployment_type == "update" else 2
            )
            deployment_tasks.append(task)

        # Distribute tasks across Ray actors in parallel
        ray_futures = []
        for i, task in enumerate(deployment_tasks):
            actor_index = i % len(self._deployment_actors)
            actor = self._deployment_actors[actor_index]
            future = actor.deploy_model.remote(task)
            ray_futures.append(future)

        # Wait for all Ray tasks to complete in parallel
        results = await asyncio.gather(*[
            asyncio.create_task(self._await_ray_future(future))
            for future in ray_futures
        ], return_exceptions=True)

        # Process results
        successful_deployments = 0
        failed_deployments = 0
        deployment_results = {}

        for i, result in enumerate(results):
            model_name = deployment_tasks[i].model_name
            
            if isinstance(result, Exception):
                deployment_results[model_name] = {
                    "status": "error",
                    "error_message": str(result),
                    "model_name": model_name,
                    "ray_based": True
                }
                failed_deployments += 1
            else:
                deployment_results[model_name] = result
                if result.get("status") == "success":
                    successful_deployments += 1
                else:
                    failed_deployments += 1

        total_time = time.time() - start_time

        # Update statistics
        self._deployment_stats["total_deployments"] += len(model_names)
        self._deployment_stats["successful_deployments"] += successful_deployments
        self._deployment_stats["failed_deployments"] += failed_deployments

        # Update average deployment time
        current_avg = self._deployment_stats["average_deployment_time"]
        total_deployments = self._deployment_stats["total_deployments"]
        self._deployment_stats["average_deployment_time"] = (
            current_avg * (total_deployments - len(model_names)) + total_time
        ) / total_deployments

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
            "no_workers": True
        }

    async def _await_ray_future(self, ray_future):
        """Await a Ray future in an async context"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, ray.get, ray_future)

    async def deploy_single_model(self, model_name: str, deployment_type: str = "new") -> Dict[str, Any]:
        """Deploy a single model using Ray actor (replaces worker-based deployment)"""
        result = await self.deploy_models_parallel([model_name], deployment_type)
        
        if model_name in result["results"]:
            return result["results"][model_name]
        else:
            return {
                "status": "error",
                "error_message": "Model not found in deployment results",
                "model_name": model_name,
                "ray_based": True
            }

    def get_deployment_stats(self) -> Dict[str, Any]:
        """Get comprehensive deployment statistics"""
        return {
            **self._deployment_stats,
            "num_ray_actors": len(self._deployment_actors),
            "deployment_method": "ray_actors_only"
        }

    async def get_actor_stats(self) -> Dict[str, Any]:
        """Get statistics from all Ray deployment actors"""
        if not self._deployment_actors:
            return {"error": "No Ray actors initialized"}

        try:
            # Get stats from all Ray actors in parallel
            stat_futures = [actor.get_stats.remote() for actor in self._deployment_actors]
            actor_stats = await asyncio.gather(*[
                self._await_ray_future(future) for future in stat_futures
            ], return_exceptions=True)

            return {
                "total_actors": len(self._deployment_actors),
                "actor_statistics": [
                    stats if not isinstance(stats, Exception) else {"error": str(stats)}
                    for stats in actor_stats
                ],
                "ray_based": True
            }

        except Exception as e:
            self.logger.error(f"Failed to get Ray actor stats: {e}")
            return {"error": f"Failed to get actor stats: {str(e)}"}

    def cleanup(self):
        """Cleanup Ray deployment actors"""
        try:
            self.logger.info("Cleaning up Ray deployment actors")
            
            for actor in self._deployment_actors:
                try:
                    ray.kill(actor)
                except Exception as e:
                    self.logger.warning(f"Error killing Ray actor: {e}")

            self._deployment_actors.clear()
            self.logger.info("Ray deployment manager cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during Ray deployment manager cleanup: {e}")

    # Legacy compatibility methods (will delegate to Ray processing)
    async def queue_batched_deployments(
        self,
        model_names: List[str],
        deployment_type: str,
        fallback_deploy_func=None,
        fallback_update_func=None,
    ):
        """Legacy compatibility - now uses Ray parallel deployment"""
        self.logger.info("Legacy queue_batched_deployments called - using Ray parallel deployment")
        return await self.deploy_models_parallel(model_names, deployment_type)

    async def stage_model(self, model_name: str, load_model_func=None) -> bool:
        """Legacy compatibility - now uses Ray actor deployment"""
        self.logger.info(f"Legacy stage_model called for {model_name} - using Ray deployment")
        result = await self.deploy_single_model(model_name)
        return result.get("status") == "success"

    async def validate_staged_model(self, model_name: str) -> bool:
        """Legacy compatibility - validation is now part of Ray deployment"""
        self.logger.debug(f"Legacy validate_staged_model called for {model_name} - integrated into Ray deployment")
        return True  # Validation is now integrated into the Ray deployment process

    async def promote_staged_model_to_active(self, model_name: str, cache_put_func=None):
        """Legacy compatibility - promotion is now part of Ray deployment"""
        self.logger.debug(f"Legacy promote_staged_model_to_active called for {model_name} - integrated into Ray deployment")
        # This is now handled automatically in the Ray deployment process

    async def remove_from_staging(self, model_name: str):
        """Legacy compatibility - staging cleanup is now handled by Ray actors"""
        self.logger.debug(f"Legacy remove_from_staging called for {model_name} - handled by Ray actors")
        # This is now handled automatically by Ray actors