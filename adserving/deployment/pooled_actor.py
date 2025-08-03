"""
Pooled model actor implementation
"""

import logging
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List

from ray import serve

from adserving.deployment.utils.metrics_collector import MetricsCollector
from adserving.deployment.utils.model_router import ModelPoolRouter
from adserving.deployment.utils.request_processor import RequestProcessor
from adserving.deployment.utils.task_processor import TaskProcessor

from ..core.model_manager import ModelManager
from .model_pool import ModelPool
from .resource_config import PooledDeploymentConfig


@serve.deployment
class PooledModelActor:
    """Pooled model actor that serves multiple models"""

    def __init__(
        self,
        deployment_name: str,
        mlflow_tracking_uri: str,
        config: PooledDeploymentConfig,
    ):

        self.deployment_name = deployment_name
        self.config = config

        # Initialize model manager
        self.model_manager = ModelManager(
            mlflow_tracking_uri=mlflow_tracking_uri,
            hot_cache_size=config.model_pool_size // 3,
            warm_cache_size=config.model_pool_size // 2,
            cold_cache_size=config.model_pool_size,
            max_workers=8,
        )

        # Model pools for load distribution
        self.model_pools: List[ModelPool] = [
            ModelPool(f"pool_{i}", config.model_pool_size)
            for i in range(max(1, int(config.resource_config.num_gpus) or 1))
        ]

        # Request batching
        self.batch_queues: Dict[str, List] = defaultdict(list)
        self.batch_timers: Dict[str, threading.Timer] = {}

        # Initialize helper components
        self.logger = logging.getLogger(f"{__name__}.{deployment_name}")
        self.request_processor = RequestProcessor(self.logger)
        self.model_router = ModelPoolRouter(self.logger)
        self.task_processor = TaskProcessor(self.logger)
        self.metrics_collector = MetricsCollector()

        # Start model manager monitoring
        self.model_manager.start_monitoring()

    def __del__(self):
        """
        Cleanup resources when actor is destroyed to prevent memory leaks
        """
        try:
            if hasattr(self, "model_manager"):
                self.model_manager.cleanup()
        except Exception as e:
            # Use print since logger might not be available during destruction
            print(f"Error during PooledModelActor cleanup: {e}")

    def cleanup(self):
        """Explicit cleanup method for graceful shutdown"""
        try:
            if hasattr(self, "model_manager"):
                self.model_manager.cleanup()

            # Cancel any pending batch timers
            if hasattr(self, "batch_timers"):
                for timer in self.batch_timers.values():
                    try:
                        timer.cancel()
                    except Exception:
                        pass
                self.batch_timers.clear()

            # Clear batch queues
            if hasattr(self, "batch_queues"):
                self.batch_queues.clear()

            self.logger.info("PooledModelActor cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during PooledModelActor cleanup: {e}")

    async def __call__(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle prediction requests with intelligent routing - batch support
        """
        start_time = time.time()
        self.metrics_collector.increment_request_count()

        try:
            # Get prediction tasks from request
            prediction_tasks = self.request_processor.prepare_input_data(request)

            # Process prediction tasks in parallel for better performance
            results = await self.task_processor.process_prediction_tasks(
                prediction_tasks=prediction_tasks,
                model_router=self.model_router,
                model_manager=self.model_manager,
                model_pools=self.model_pools,
                config=self.config,
                metrics_lock=self.metrics_collector._metrics_lock,
                model_request_counts=self.metrics_collector.model_request_counts,
            )

            # Update metrics
            inference_time = time.time() - start_time
            self.metrics_collector.add_inference_time(inference_time)

            # Return consolidated results
            return self.task_processor.create_consolidated_response(
                results, prediction_tasks, inference_time
            )

        except Exception as e:
            self.metrics_collector.increment_error_count()
            self.logger.error(f"Prediction error: {e}")

            return {
                "error": str(e),
                "status": "error",
                "inference_time": time.time() - start_time,
            }

    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        try:
            # Check model manager health
            cache_stats = self.model_manager.get_cache_stats()

            # Check pool health
            pool_stats = [pool.get_pool_stats() for pool in self.model_pools]

            # Get metrics snapshot
            metrics_snapshot = self.metrics_collector.get_metrics_snapshot()
            derived_metrics = self.metrics_collector.calculate_derived_metrics()

            # Combine metrics
            metrics_snapshot.update(derived_metrics)

            # Get top models
            top_models = dict(
                sorted(
                    metrics_snapshot["model_request_counts"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:10]
            )

            return {
                "status": "healthy",
                "deployment_name": self.deployment_name,
                "request_count": metrics_snapshot["request_count"],
                "error_count": metrics_snapshot["error_count"],
                "error_rate": metrics_snapshot["error_rate"],
                "avg_inference_time": metrics_snapshot["avg_inference_time"],
                "cache_stats": cache_stats,
                "pool_stats": pool_stats,
                "top_models": top_models,
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "deployment_name": self.deployment_name,
            }

    async def preload_model(self, model_name: str) -> Dict[str, Any]:
        """Pre-load a specific model into this deployment pool"""
        try:
            self.logger.info(
                f"Pre-loading model {model_name} into deployment {self.deployment_name}"
            )

            # Load the model through the model manager
            model_info = await self.model_manager.load_model_async(model_name)

            if model_info:
                self.logger.info(
                    f"Successfully pre-loaded model "
                    f"{model_name} into deployment {self.deployment_name}"
                )
                return {
                    "status": "success",
                    "message": f"Model {model_name}" f" pre-loaded successfully",
                    "model_info": model_info,
                }
            else:
                self.logger.error(f"Failed to pre-load model {model_name}")
                return {
                    "status": "error",
                    "message": f"Failed to pre-load model {model_name}",
                }

        except Exception as e:
            self.logger.error(f"Error pre-loading model {model_name}: {e}")
            return {
                "status": "error",
                "message": f"Error pre-loading model" f" {model_name}: {str(e)}",
            }

    def get_metrics(self) -> Dict[str, Any]:
        """Get detailed metrics"""
        metrics_snapshot = self.metrics_collector.get_metrics_snapshot()

        return {
            "deployment_name": self.deployment_name,
            "request_count": metrics_snapshot["request_count"],
            "error_count": metrics_snapshot["error_count"],
            "total_inference_time": metrics_snapshot["total_inference_time"],
            "model_request_counts": metrics_snapshot["model_request_counts"],
            "cache_stats": self.model_manager.get_cache_stats(),
            "pool_stats": [pool.get_pool_stats() for pool in self.model_pools],
        }
