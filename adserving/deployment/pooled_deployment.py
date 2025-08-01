"""
Pooled Model Deployment with Shared Resources

This module implements a pooled deployment strategy where multiple models
share deployment resources instead of having individual deployments.

Key improvements:
- Shared resource pool for multiple models
- Dynamic model loading/unloading within deployments
- GPU sharing across models
- Intelligent load balancing
- Reduced resource overhead
"""

import logging
from typing import Any, Dict, List, Optional

import ray
from ray import serve
from ray.serve.config import AutoscalingConfig

from ..core.model_manager import ModelManager
from .pooled_actor import PooledModelActor
from .resource_config import (
    AutoscalingSettings,
    PooledDeploymentConfig,
    PooledResourceConfig,
)


class PooledModelDeployment:
    """Manages pooled model deployments for hundreds of models"""

    def __init__(self, model_manager: ModelManager, config: Optional[Any] = None):
        self.model_manager = model_manager
        self.config = config  # Store the main config object
        self.deployments: Dict[str, Any] = {}
        self.deployment_configs: Dict[str, PooledDeploymentConfig] = {}
        self.logger = logging.getLogger(__name__)

    def _get_available_gpus(self) -> float:
        """Get the number of available GPUs in the Ray cluster"""
        try:
            cluster_resources = ray.cluster_resources()
            available_gpus = cluster_resources.get("GPU", 0.0)
            self.logger.info(f"Available GPUs in cluster: {available_gpus}")
            return available_gpus
        except Exception as e:
            self.logger.warning(f"Could not detect GPU resources: {e}")
            return 0.0

    def create_pooled_deployment(self, config: PooledDeploymentConfig) -> bool:
        """Create a new pooled deployment"""
        try:
            # Configure Ray Serve deployment
            deployment_config = {
                "ray_actor_options": {
                    "num_cpus": config.resource_config.num_cpus,
                    "num_gpus": config.resource_config.num_gpus,
                    "memory": config.resource_config.memory * 1024 * 1024,
                },
                "autoscaling_config": AutoscalingConfig(
                    min_replicas=config.autoscaling.min_replicas,
                    max_replicas=config.autoscaling.max_replicas,
                    target_num_ongoing_requests_per_replica=(
                        config.autoscaling.target_num_ongoing_requests_per_replica
                    ),
                    metrics_interval_s=config.autoscaling.metrics_interval_s,
                    look_back_period_s=config.autoscaling.look_back_period_s,
                    smoothing_factor=config.autoscaling.smoothing_factor,
                ),
                "health_check_period_s": config.health_check_period_s,
                "health_check_timeout_s": config.health_check_timeout_s,
            }

            # Create and deploy the pooled actor
            pooled_actor = PooledModelActor.options(**deployment_config).bind(
                deployment_name=config.deployment_name,
                mlflow_tracking_uri=self.model_manager.mlflow_tracking_uri,
                config=config,
            )

            # Deploy with Ray Serve
            serve.run(
                pooled_actor,
                name=config.deployment_name,
                route_prefix=f"/{config.deployment_name}",
            )

            self.deployments[config.deployment_name] = pooled_actor
            self.deployment_configs[config.deployment_name] = config

            self.logger.debug(f"Created pooled deployment: {config.deployment_name}")
            return True

        except Exception as e:
            self.logger.error(
                f"Error creating pooled deployment {config.deployment_name}: {e}"
            )
            return False

    def create_default_pools(self, num_pools: Optional[int] = None) -> List[str]:
        """Create default pooled deployments for
        load distribution using config values"""
        created_pools = []

        # Use config values if available, otherwise fallback to defaults
        if self.config and hasattr(self.config, "pooled_deployment"):
            pool_config = self.config.pooled_deployment
            actual_num_pools = (
                num_pools if num_pools is not None else pool_config.default_pool_count
            )
            resource_config = pool_config.pool_resource_config
            autoscaling_config = pool_config.autoscaling_config
            self.logger.debug(
                f"Using config values: {actual_num_pools} pools, "
                f"{resource_config.num_cpus} CPUs, "
                f"{resource_config.memory}MB memory"
            )
        else:
            # Fallback to hardcoded values if no config
            actual_num_pools = num_pools if num_pools is not None else 3
            resource_config = PooledResourceConfig(
                num_cpus=4.0,
                num_gpus=1.0,  # Will be adjusted based on availability
                memory=8192,
                object_store_memory=4096,
            )
            autoscaling_config = AutoscalingSettings(
                min_replicas=2,
                max_replicas=20,
                target_num_ongoing_requests_per_replica=8,
            )
            self.logger.warning("No config provided, using fallback hardcoded values")

        # Check available GPU resources
        available_gpus = self._get_available_gpus()
        self.logger.debug(
            f"Creating {actual_num_pools} pools with {available_gpus} GPUs available"
        )

        for i in range(actual_num_pools):
            # Only allocate GPU to first pool and only if GPUs are available
            gpu_allocation = 0.0
            if i == 0 and available_gpus > 0:
                # Use configured GPU amount or available GPUs, whichever is smaller
                configured_gpus = (
                    resource_config.num_gpus
                    if hasattr(resource_config, "num_gpus")
                    else 1.0
                )
                gpu_allocation = min(configured_gpus, available_gpus)
                self.logger.debug(
                    f"Allocating {gpu_allocation} GPU to pool {i} "
                    f"(configured: {configured_gpus}, "
                    f"available: {available_gpus})"
                )
            else:
                self.logger.debug(
                    f"No GPU allocated to pool {i} (available: {available_gpus})"
                )

            # Create deployment config using config values
            deployment_config = PooledDeploymentConfig(
                deployment_name=f"model_pool_{i}",
                resource_config=PooledResourceConfig(
                    num_cpus=resource_config.num_cpus,
                    num_gpus=gpu_allocation,  # Conditional GPU allocation
                    memory=resource_config.memory,
                    object_store_memory=getattr(
                        resource_config, "object_store_memory", 4096
                    ),
                    max_models_per_replica=getattr(
                        resource_config, "max_models_per_replica", 10
                    ),
                ),
                autoscaling=AutoscalingSettings(
                    min_replicas=autoscaling_config.min_replicas,
                    max_replicas=autoscaling_config.max_replicas,
                    target_num_ongoing_requests_per_replica=(
                        autoscaling_config.target_num_ongoing_requests_per_replica
                    ),
                    metrics_interval_s=getattr(
                        autoscaling_config, "metrics_interval_s", 5.0
                    ),
                    look_back_period_s=getattr(
                        autoscaling_config, "look_back_period_s", 30.0
                    ),
                    smoothing_factor=getattr(
                        autoscaling_config, "smoothing_factor", 0.8
                    ),
                ),
                model_pool_size=(
                    getattr(self.config.pooled_deployment, "models_per_pool", 50)
                    if self.config and hasattr(self.config, "pooled_deployment")
                    else 50
                ),
            )

            if self.create_pooled_deployment(deployment_config):
                created_pools.append(deployment_config.deployment_name)
                self.logger.debug(
                    f"Created pool {i}: {deployment_config.deployment_name} "
                    f"with {deployment_config.resource_config.num_cpus} CPUs, "
                    f"{deployment_config.resource_config.memory}MB memory, "
                    f"{deployment_config.autoscaling.min_replicas}-"
                    f"{deployment_config.autoscaling.max_replicas} replicas"
                )
            else:
                self.logger.error(f"Failed to create pool {i}: model_pool_{i}")

        self.logger.info(
            f"Successfully created {len(created_pools)}/{actual_num_pools} "
            f"pools: {created_pools}"
        )
        return created_pools

    def get_deployment_status(self, deployment_name: str) -> Dict[str, Any]:
        """Get status of a pooled deployment"""
        if deployment_name not in self.deployments:
            return {"status": "not_found"}

        try:
            # Get health check from deployment
            handle = serve.get_deployment_handle(deployment_name).get_handle()
            health = ray.get(handle.health_check.remote())
            return health
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def get_all_deployment_stats(self) -> Dict[str, Any]:
        """Get statistics for all pooled deployments"""
        stats = {}
        for deployment_name in self.deployments:
            stats[deployment_name] = self.get_deployment_status(deployment_name)
        return stats

    def cleanup(self):
        """Cleanup all deployments"""
        for deployment_name in list(self.deployments.keys()):
            try:
                serve.delete(deployment_name)
                self.logger.debug(f"Cleaned up deployment: {deployment_name}")
            except Exception as e:
                self.logger.error(
                    f"Error cleaning up deployment {deployment_name}: {e}"
                )

        self.deployments.clear()
        self.deployment_configs.clear()
