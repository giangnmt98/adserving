"""
Model routing utilities for pooled deployments
"""

from typing import Any, Optional

from adserving.src.deployment.model_pool import ModelPool
from adserving.src.utils.logger import FrameworkLogger, get_logger


class ModelPoolRouter:
    """Handles routing requests to appropriate model pools"""

    def __init__(self, logger: Optional[FrameworkLogger] = None):
        self.logger = logger or get_logger()

    def select_optimal_pool(self, model_name: str, model_pools: list) -> ModelPool:
        """
        Select the optimal pool for a model based on load and
        existing models
        """
        # Check if model is already loaded in any pool
        for pool in model_pools:
            if model_name in pool.loaded_models:
                return pool

        # Select pool with least load
        return min(model_pools, key=lambda p: len(p.loaded_models))

    async def route_and_predict(
        self,
        model_name: str,
        input_data: Any,
        model_manager: Any,
        model_pools: list,
        config: Any,
    ) -> Any:
        """Route request to appropriate model pool and perform prediction"""
        # Find the best pool for this model (load balancing)
        target_pool = self.select_optimal_pool(model_name, model_pools)

        # Check if model is already loaded in the pool
        model_info = target_pool.get_model(model_name)

        if not model_info:
            # Load model using enhanced model manager
            model_info = await model_manager.load_model_async(model_name)
            if not model_info:
                raise ValueError(f"Could not load model: {model_name}")

            # Add to pool
            target_pool.add_model(model_name, model_info, model_info.tier)

        # Perform prediction
        if config.enable_batching and len(input_data) == 1:
            # Use batching for single requests
            return await self._batched_predict(
                model_name, input_data, target_pool, model_manager
            )
        else:
            # Direct prediction for batch requests
            return await model_manager.predict(model_name, input_data)

    async def _batched_predict(
        self, model_name: str, input_data: Any, pool: ModelPool, model_manager: Any
    ) -> Any:
        """Perform batched prediction for better throughput"""
        # For now, perform direct prediction
        # In a full implementation, this would queue requests and batch them
        return await model_manager.predict(model_name, input_data)
