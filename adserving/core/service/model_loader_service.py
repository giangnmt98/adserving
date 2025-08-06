"""
Model Loader Service

This module handles asynchronous model loading with optimization.
"""

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Set

from mlflow.pyfunc import load_model

from adserving.core.utils.model_info import ModelInfo, ModelTier


class ModelLoaderService:
    """Handles asynchronous model loading with non-blocking optimization"""

    def __init__(self, model_manager, executor: ThreadPoolExecutor):
        self.model_manager = model_manager
        self.executor = executor
        self.logger = logging.getLogger(__name__)

        # Async loading optimization
        self._loading_models: Set[str] = set()  # Track models being loaded
        self._model_load_queue: Optional[asyncio.Queue] = (
            asyncio.Queue() if self._is_event_loop_running() else None
        )
        self._background_loader_task = None
        self._loading_lock = threading.RLock()
        # Events for async waiting
        self._loading_events: Dict[str, asyncio.Event] = {}

    def _is_event_loop_running(self) -> bool:
        """Check if event loop is running"""
        try:
            asyncio.get_event_loop()
            return True
        except RuntimeError:
            return False

    async def load_model_async(self, model_name: str) -> Optional[ModelInfo]:
        """Asynchronously load model with non-blocking optimization"""
        # Check if already cached
        cached_model = self.model_manager.cache.get(model_name)
        if cached_model and self._is_model_current(cached_model):
            return cached_model

        # Check if currently being loaded
        should_wait = False
        with self._loading_lock:
            if model_name in self._loading_models:
                # Another thread/coroutine is loading this model, wait for it
                self.logger.info(
                    f"Model {model_name} is already being loaded, waiting..."
                )
                should_wait = True
                if model_name not in self._loading_events:
                    self._loading_events[model_name] = asyncio.Event()
            else:
                # We will load this model, add to loading set and create event
                self._loading_models.add(model_name)
                self._loading_events[model_name] = asyncio.Event()

        # If another coroutine is loading, wait for completion
        if should_wait:
            await self._loading_events[model_name].wait()
            # Return the now-loaded model
            return self.model_manager.cache.get(model_name)

        try:
            # Load in background thread
            loop = asyncio.get_event_loop()
            model_info = await loop.run_in_executor(
                self.executor, self.load_model_sync, model_name
            )
            return model_info
        finally:
            # Remove from loading set and signal completion
            with self._loading_lock:
                self._loading_models.discard(model_name)
                if model_name in self._loading_events:
                    event = self._loading_events.pop(model_name)
                    event.set()  # Signal that loading is complete

    def load_model_sync(self, model_name: str) -> Optional[ModelInfo]:
        """Synchronously load model with metadata"""
        try:
            # Check if model is already cached
            cached_model = self.model_manager.cache.get(model_name)
            if cached_model and self._is_model_current(cached_model):
                return cached_model

            # Get latest production version
            latest_version = None
            anomaly_threshold = 0.0

            try:
                mlflow_client = self.model_manager.mlflow_client

                #Lấy ModelVersion object, KHÔNG lấy .version ngay
                production_versions_list = mlflow_client.client.get_latest_versions(
                    name=model_name,
                    stages=["Production"]
                )

                if not production_versions_list:
                    self.logger.warning(f"No Production stage version found for model: {model_name}")
                    return None

                #Lấy ModelVersion object từ list
                latest_version = production_versions_list[0]

                # Lấy threshold parameters
                try:
                    anomaly_threshold = mlflow_client.get_threshold_parameters(model_name)
                except Exception as e:
                    self.logger.debug(f"No threshold parameters found for {model_name}: {e}")
                    anomaly_threshold = {}

            except Exception as e:
                self.logger.error(f"Error getting production version for model {model_name}: {e}")
                return None

            if not latest_version:
                self.logger.error(f"Could not find any suitable version for model {model_name}")
                return None

            #ĐÚNG: Construct model URI với version number
            model_uri = f"models:/{model_name}/{latest_version.version}"

            # Load model
            start_time = time.time()
            model = load_model(model_uri)
            load_time = time.time() - start_time

            # Determine tier
            tier = self._determine_model_tier(model_name)

            # ĐÚNG: Create model info với version string
            model_info = ModelInfo(
                model_name=model_name,
                model_version=latest_version.version,
                model_uri=model_uri,
                anomaly_threshold=anomaly_threshold,
                model=model,
                loaded_at=time.time(),
                last_accessed=time.time(),
                access_count=1,
                tier=tier,
                memory_usage=0.0,
                avg_inference_time=load_time,
                error_count=0,
                success_count=0,
            )

            # Cache the model
            self.model_manager.cache.put(model_name, model_info)
            self.model_manager.cache.record_load_time(load_time)

            self.logger.debug(
                f"Loaded model {model_name} (v{latest_version.version}) "
                f"in {load_time:.2f}s, tier: {tier.value}"
            )

            return model_info

        except Exception as e:
            self.logger.error(f"Error loading model {model_name}: {e}")
            return None

    def _determine_model_tier(self, model_name: str) -> ModelTier:
        """Determine appropriate tier for model"""
        if model_name in self.model_manager.tier_config:
            return self.model_manager.tier_config[model_name]
        else:
            # Auto-assign tier based on current model count
            cache_stats = self.model_manager.cache.get_cache_stats()
            total_models = cache_stats["total_models"]

            if total_models < self.model_manager.cache.hot_cache_size:
                return ModelTier.HOT
            elif total_models < (
                self.model_manager.cache.hot_cache_size
                + self.model_manager.cache.warm_cache_size
            ):
                return ModelTier.WARM
            else:
                return ModelTier.COLD

    def _is_model_current(self, model_info: ModelInfo) -> bool:
        """Check if cached model is current"""
        try:
            # Try the new alias-based approach first (recommended)
            try:
                latest_version = (
                    self.model_manager.mlflow_client.get_model_version_by_alias(
                        model_info.model_name, "production"
                    )
                )
            except Exception:
                # Fallback to search_model_versions for models still using
                # stages
                versions = self.model_manager.mlflow_client.search_model_versions(
                    filter_string=f"name='{model_info.model_name}'",
                    order_by=["version_number DESC"],
                    max_results=1,
                )
                if not versions:
                    return False
                latest_version = versions[0]
            return model_info.model_version == latest_version.version
        except Exception:
            return False
