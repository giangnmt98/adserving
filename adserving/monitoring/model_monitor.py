"""
Model monitor with advanced capabilities.
"""

import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

from .metrics_collector import MetricsCollector


class ModelMonitor:
    """Model monitor with advanced capabilities"""

    def __init__(
        self,
        collection_interval: int = 5,
        optimization_interval: int = 60,
        enable_prometheus: bool = True,
        prometheus_port: int = 9090,
    ):

        self.collection_interval = collection_interval
        self.optimization_interval = optimization_interval

        # Components
        self.metrics_collector = MetricsCollector(
            collection_interval=collection_interval,
            enable_prometheus=enable_prometheus,
            prometheus_port=prometheus_port,
        )

        # Optimization callbacks
        self.optimization_callbacks: List[Callable] = []

        # Threading
        self._optimization_thread: Optional[threading.Thread] = None
        self._stop_optimization = threading.Event()

        self.logger = logging.getLogger(__name__)

    def start_monitoring(self):
        """Start comprehensive monitoring"""
        self.metrics_collector.start_collection()

        if (
            self._optimization_thread is None
            or not self._optimization_thread.is_alive()
        ):
            self._stop_optimization.clear()
            self._optimization_thread = threading.Thread(
                target=self._optimization_loop, daemon=True
            )
            self._optimization_thread.start()

        self.logger.info("Monitoring started")

    def stop_monitoring(self):
        """Stop monitoring"""
        self.metrics_collector.stop_collection()

        if self._optimization_thread and self._optimization_thread.is_alive():
            self._stop_optimization.set()
            self._optimization_thread.join(timeout=10)

        self.logger.info("Monitoring stopped")

    def _optimization_loop(self):
        """Optimization and analysis loop"""
        while not self._stop_optimization.wait(self.optimization_interval):
            try:
                self._run_optimization_callbacks()
                self._analyze_performance_trends()
            except Exception as e:
                self.logger.error(f"Error in optimization loop: {e}")

    def _run_optimization_callbacks(self):
        """Run registered optimization callbacks"""
        for callback in self.optimization_callbacks:
            try:
                callback(self.metrics_collector)
            except Exception as e:
                self.logger.error(f"Error in optimization callback: {e}")

    def _analyze_performance_trends(self):
        """Analyze performance trends and log insights"""
        try:
            model_metrics = self.metrics_collector.get_all_model_metrics()

            # Find models with performance issues
            slow_models = []
            error_prone_models = []

            for name, metrics in model_metrics.items():
                if metrics.avg_response_time > 2.0:  # Slow models
                    slow_models.append((name, metrics.avg_response_time))

                if metrics.total_requests > 10:
                    error_rate = metrics.failed_requests / metrics.total_requests
                    if error_rate > 0.05:  # High error rate
                        error_prone_models.append((name, error_rate))

            # Log insights
            if slow_models:
                self.logger.warning(f"Slow models detected: {slow_models[:5]}")

            if error_prone_models:
                self.logger.warning(
                    f"Error-prone models detected: {error_prone_models[:5]}"
                )

        except Exception as e:
            self.logger.error(f"Error analyzing performance trends: {e}")

    def add_optimization_callback(self, callback: Callable):
        """Add optimization callback"""
        self.optimization_callbacks.append(callback)

    def get_scaling_recommendation(
        self, deployment_name: str, current_replicas: int
    ) -> Tuple[int, str]:
        """Get scaling recommendation for deployment"""
        return self.metrics_collector.capacity_planner.predict_scaling_needs(
            deployment_name, current_replicas
        )

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data"""
        return self.metrics_collector.get_dashboard_data()

    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics"""
        return self.metrics_collector.prometheus_exporter.get_metrics()

    def cleanup(self):
        """Cleanup resources"""
        self.stop_monitoring()
