"""
Model Manager with Tiered Loading Strategy

This module implements an improved model management system that supports:
- Tiered model loading (Hot/Warm/Cold)
- Increased cache capacity for hundreds of models
- Dynamic model loading/unloading based on demand
- Model warming strategies
- Batch inference optimization
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from adserving.core.manager.deployment_manager import DeploymentManager
from adserving.core.manager.production_model_manager import ProductionModelManager
from adserving.core.service.model_loader_service import ModelLoaderService
from adserving.core.service.monitoring_service import ModelMonitoringService
from adserving.core.service.prediction_service import PredictionService
from adserving.core.utils.mlflow_client import MLflowClient
from adserving.core.utils.model_info import ModelInfo, ModelTier
from adserving.core.utils.tiered_model_cache import TieredModelCache


class ModelManager:
    """Model manager with tiered loading and improved caching"""

    def __init__(
        self,
        mlflow_tracking_uri: str,
        hot_cache_size: int = 50,
        warm_cache_size: int = 200,
        cold_cache_size: int = 500,
        update_interval: int = 10,
        max_workers: int = 8,
        enable_model_warming: bool = True,
    ):
        # Initialize logger first before any other operations
        self.logger = logging.getLogger(__name__)

        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.update_interval = update_interval
        print("A$$$$$$$$$$$$$$$$$$$$$$$$$$$$$", self.update_interval)
        self.max_workers = max_workers
        self.enable_model_warming = enable_model_warming

        # Tiered cache
        self.cache = TieredModelCache(hot_cache_size, warm_cache_size, cold_cache_size)

        # MLflow client with connection pooling optimization
        self.mlflow_client = MLflowClient(mlflow_tracking_uri)

        # Threading - Limit worker count to prevent OOM issues
        cpu_count = os.cpu_count() or 1
        actual_workers = min(max_workers, 16, cpu_count * 2)
        self.executor = ThreadPoolExecutor(max_workers=actual_workers)
        self.logger.info(
            f"ThreadPoolExecutor initialized with "
            f"{actual_workers} workers (CPU cores: {cpu_count}, "
            f"limited to prevent OOM, requested: {max_workers})"
        )

        # Model tier configuration
        self.tier_config = self._load_tier_config()

        # Deployment manager
        self.deployment_manager = DeploymentManager(
            load_model_func=self._load_model_sync
        )

        # Initialize services
        self.monitoring_service = ModelMonitoringService(
            self, update_interval, enable_model_warming
        )
        self.model_loader = ModelLoaderService(self, self.executor)
        self.production_model_manager = ProductionModelManager(self)
        self.prediction_service = PredictionService(self)

    def _load_tier_config(self) -> Dict[str, ModelTier]:
        """Load model tier configuration"""
        # This could be loaded from a configuration file
        # For now, return the default configuration
        return {}

    def start_monitoring(self):
        """Start background monitoring and optimization with zero-downtime deployment"""
        self.monitoring_service.start_monitoring()

    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_service.stop_monitoring()

    async def _check_and_update_production_models(self):
        """Check for production model changes
        and update models for zero-downtime deployment"""
        await self.production_model_manager.check_and_update_production_models()

    def get_production_models(self) -> List[str]:
        """Get list of production models from MLflow"""
        return self.production_model_manager.get_production_models()

    async def load_model_async(self, model_name: str) -> Optional[ModelInfo]:
        """Asynchronously load model with non-blocking optimization"""
        return await self.model_loader.load_model_async(model_name)

    def _load_model_sync(self, model_name: str) -> Optional[ModelInfo]:
        """Synchronously load model with metadata"""
        return self.model_loader.load_model_sync(model_name)

    def _is_model_current(self, model_info: ModelInfo) -> bool:
        """Check if cached model is current"""
        return self.model_loader._is_model_current(model_info)

    async def predict(self, model_name: str, input_data) -> Optional[Any]:
        """Prediction with batch processing and performance tracking"""
        return await self.prediction_service.predict(model_name, input_data)

    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get model information"""
        return self.prediction_service.get_model_info(model_name)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        return self.cache.get_cache_stats()

    def get_deployment_stats(self) -> Dict[str, Any]:
        """Get deployment statistics for zero-downtime deployment monitoring"""
        return self.deployment_manager.get_deployment_stats()

    def cleanup(self):
        """Cleanup resources to prevent memory leaks and OOM issues"""
        try:
            self._stop_monitoring_service()
            self._cleanup_model_caches()
            self._cleanup_prediction_service()
            self._cleanup_thread_executor()
            self._cleanup_loading_models()

            self.logger.info("ModelManager cleanup completed successfully")

        except Exception as e:
            self.logger.error(f"Error during ModelManager cleanup: {e}")

    def _stop_monitoring_service(self):
        """Stop the monitoring service"""
        self.stop_monitoring()

    def _cleanup_model_caches(self):
        """Clear all model caches"""
        if hasattr(self, "cache"):
            try:
                self.cache.hot_cache.clear()
                self.cache.warm_cache.clear()
                self.cache.cold_cache.clear()
                self.logger.info("Cleared all model caches to prevent memory leaks")
            except Exception as e:
                self.logger.error(f"Error clearing model cache: {e}")

    def _cleanup_prediction_service(self):
        """Cleanup the prediction service"""
        if hasattr(self, "prediction_service"):
            self.prediction_service.cleanup_batch_processing()

    def _cleanup_thread_executor(self):
        """Shutdown and cleanup thread executor"""
        if hasattr(self, "executor") and self.executor:
            try:
                self.executor.shutdown(wait=True)
                self.logger.info("ThreadPoolExecutor shutdown completed")
            except Exception as e:
                self.logger.error(f"Error shutting down executor: {e}")
                try:
                    self.executor._threads.clear()
                except Exception:
                    pass

    def _cleanup_loading_models(self):
        """Clear loading models set"""
        if hasattr(self.model_loader, "_loading_models"):
            self.model_loader._loading_models.clear()
