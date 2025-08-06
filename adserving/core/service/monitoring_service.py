"""
Model Monitoring Service

This module handles background monitoring and optimization of models.
"""

import asyncio
import logging
import threading
import time


class ModelMonitoringService:
    """Handles background monitoring and optimization of models"""

    def __init__(
        self,
        model_manager,
        update_interval: int = 10,
        enable_model_warming: bool = True,
        production_check_interval: int = 10,
    ):
        self.model_manager = model_manager
        self.update_interval = update_interval
        self.enable_model_warming = enable_model_warming
        self.production_check_interval = production_check_interval

        self.logger = logging.getLogger(__name__)
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        self._last_production_check = 0

    def start_monitoring(self):
        """Start background monitoring and optimization with zero-downtime deployment"""
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            self._stop_monitoring.clear()

            # Initialize deployment infrastructure
            self.model_manager.deployment_manager.initialize_infrastructure()

            self._monitoring_thread = threading.Thread(
                target=self._monitor_and_optimize, daemon=True
            )
            self._monitoring_thread.start()
            self.logger.info("Model monitoring started with zero-downtime deployment")

    def stop_monitoring(self):
        """Stop background monitoring"""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._stop_monitoring.set()
            self.model_manager.deployment_manager.stop()
            self._monitoring_thread.join(timeout=5)
            self.logger.info("Model monitoring stopped")

    def _monitor_and_optimize(self):
        """Background monitoring and optimization loop"""
        while not self._stop_monitoring.wait(self.update_interval):
            try:
                # Check for production model changes for zero-downtime deployment
                current_time = time.time()
                if (
                    current_time - self._last_production_check
                    >= self.production_check_interval
                ):
                    self.logger.info("Checking for production model changes...")
                    # Run async production model check in thread pool
                    try:
                        # Create new event loop for this thread if needed
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(
                            self.model_manager._check_and_update_production_models()
                        )
                        loop.close()
                    except Exception as e:
                        self.logger.error(f"Error checking production models: {e}")
                    finally:
                        self._last_production_check = current_time

                self._optimize_model_tiers()
                self._warm_popular_models()
                self._cleanup_unused_models()
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")

    def _optimize_model_tiers(self):
        """Optimize model tier assignments based on usage patterns"""
        # This would analyze usage patterns and adjust tier assignments
        pass

    def _warm_popular_models(self):
        """Pre-warm popular models based on usage patterns"""
        if not self.enable_model_warming:
            return

        # Get production models and warm the most popular ones
        try:
            production_models = self.model_manager.get_production_models()
            # Warm top models that aren't already loaded
            for model_name in production_models[:10]:  # Top 10 models
                if not self.model_manager.cache.get(model_name):
                    self.logger.info(f"Pre-warming model: {model_name}")
                    # Use thread pool to avoid blocking the monitoring loop
                    self.model_manager.executor.submit(
                        self.model_manager._load_model_sync, model_name
                    )
        except Exception as e:
            self.logger.error(f"Error warming models: {e}")

    def _cleanup_unused_models(self):
        """Clean up models that haven't been used recently"""
        current_time = time.time()
        cleanup_threshold = 3600  # 1 hour

        try:
            # Get all cached models
            models_to_remove = []

            # Check each tier for unused models
            for tier_name in ["hot_cache", "warm_cache", "cold_cache"]:
                tier_cache = getattr(self.model_manager.cache, tier_name, {})
                for model_name, model_info in list(tier_cache.items()):
                    if hasattr(model_info, "last_accessed"):
                        time_since_access = current_time - model_info.last_accessed
                        if time_since_access > cleanup_threshold:
                            models_to_remove.append(model_name)

            # Remove unused models to free memory
            removed_count = 0
            for model_name in models_to_remove:
                try:
                    self.model_manager.cache.remove(model_name)
                    removed_count += 1
                    self.logger.info(f"Cleaned up unused model: {model_name}")
                except Exception as e:
                    self.logger.error(f"Error removing model {model_name}: {e}")

            if removed_count > 0:
                self.logger.info(
                    f"Cleaned up {removed_count}"
                    f" unused models to prevent memory issues"
                )

        except Exception as e:
            self.logger.error(f"Error in cleanup_unused_models: {e}")
