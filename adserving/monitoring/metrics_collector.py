"""
Metrics collector with model-level tracking.
"""

import logging
import statistics
import threading
import time
from collections import defaultdict, deque
from typing import Any, Dict, Optional

import numpy as np

from .capacity_planner import CapacityPlanner
from .deployment_metrics import DeploymentMetrics
from .model_metrics import ModelPerformanceMetrics
from .prometheus_exporter import PrometheusExporter
from .resource_metrics import collect_system_metrics


class MetricsCollector:
    """Metrics collector with model-level tracking"""

    def __init__(
        self,
        collection_interval: int = 5,
        history_size: int = 10000,
        enable_prometheus: bool = True,
        prometheus_port: int = 9090,
    ):

        self.collection_interval = collection_interval
        self.history_size = history_size

        # Metrics storage
        self.resource_history: deque = deque(maxlen=history_size)
        self.model_metrics: Dict[str, ModelPerformanceMetrics] = {}
        self.deployment_metrics: Dict[str, DeploymentMetrics] = {}

        # Response time tracking for percentiles
        self.response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Threading
        self._collection_thread: Optional[threading.Thread] = None
        self._stop_collection = threading.Event()
        self._lock = threading.RLock()

        # Components
        self.prometheus_exporter = PrometheusExporter(
            enable_prometheus, prometheus_port
        )
        self.capacity_planner = CapacityPlanner()

        self.logger = logging.getLogger(__name__)

    def start_collection(self):
        """Start metrics collection"""
        if self._collection_thread is None or not self._collection_thread.is_alive():
            self._stop_collection.clear()
            self._collection_thread = threading.Thread(
                target=self._collection_loop, daemon=True
            )
            self._collection_thread.start()
            self.logger.info("Metrics collection started")

    def stop_collection(self):
        """Stop metrics collection"""
        if self._collection_thread and self._collection_thread.is_alive():
            self._stop_collection.set()
            self._collection_thread.join(timeout=10)
            self.logger.info("Metrics collection stopped")

    def _collection_loop(self):
        """Main collection loop"""
        while not self._stop_collection.wait(self.collection_interval):
            try:
                self._collect_system_metrics()
                self._update_model_percentiles()
                self._cleanup_old_data()
            except Exception as e:
                self.logger.error(f"Error in metrics collection: {e}")

    def _collect_system_metrics(self):
        """Collect system resource metrics"""
        try:
            metrics = collect_system_metrics()

            with self._lock:
                self.resource_history.append(metrics)

            # Update Prometheus metrics
            self.prometheus_exporter.update_system_metrics(metrics)

            # Record for capacity planning
            self.capacity_planner.record_resource_usage(metrics)

        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")

    def record_model_request(
        self,
        model_name: str,
        response_time: float,
        success: bool,
        memory_usage: Optional[float] = None,
        cpu_usage: Optional[float] = None,
        cache_hit: bool = False,
    ):
        """Record model request metrics"""
        current_time = time.time()

        with self._lock:
            # Initialize model metrics if not exists
            if model_name not in self.model_metrics:
                self.model_metrics[model_name] = ModelPerformanceMetrics(
                    model_name=model_name, first_request_time=current_time
                )

            metrics = self.model_metrics[model_name]

            # Update request counts
            metrics.total_requests += 1
            if success:
                metrics.successful_requests += 1
            else:
                metrics.failed_requests += 1

            # Update response time metrics
            if success:
                self.response_times[model_name].append(response_time)

                metrics.min_response_time = min(
                    metrics.min_response_time, response_time
                )
                metrics.max_response_time = max(
                    metrics.max_response_time, response_time
                )

                # Update average (exponential moving average)
                alpha = 0.1
                if metrics.avg_response_time == 0:
                    metrics.avg_response_time = response_time
                else:
                    metrics.avg_response_time = (
                        alpha * response_time + (1 - alpha) * metrics.avg_response_time
                    )

            # Update resource usage
            if memory_usage is not None:
                if metrics.avg_memory_usage == 0:
                    metrics.avg_memory_usage = memory_usage
                else:
                    metrics.avg_memory_usage = (
                        0.1 * memory_usage + 0.9 * metrics.avg_memory_usage
                    )
                metrics.peak_memory_usage = max(metrics.peak_memory_usage, memory_usage)

            if cpu_usage is not None:
                if metrics.avg_cpu_usage == 0:
                    metrics.avg_cpu_usage = cpu_usage
                else:
                    metrics.avg_cpu_usage = (
                        0.1 * cpu_usage + 0.9 * metrics.avg_cpu_usage
                    )

            # Update cache metrics
            if cache_hit:
                metrics.cache_hits += 1
            else:
                metrics.cache_misses += 1

            total_cache_requests = metrics.cache_hits + metrics.cache_misses
            if total_cache_requests > 0:
                metrics.cache_hit_rate = metrics.cache_hits / total_cache_requests

            # Update timestamps
            metrics.last_request_time = current_time
            metrics.last_updated = current_time

            # Calculate RPS (requests in last minute)
            recent_requests = sum(
                1
                for req_time in self.response_times[model_name]
                if current_time - req_time <= 60
            )
            metrics.requests_per_second = recent_requests / 60
            metrics.peak_rps = max(metrics.peak_rps, metrics.requests_per_second)

        # Update Prometheus metrics
        self.prometheus_exporter.record_request(model_name, response_time, success)

        # Record for capacity planning
        self.capacity_planner.record_request(current_time, model_name, response_time)

    def _update_model_percentiles(self):
        """Update response time percentiles for all models"""
        with self._lock:
            for model_name, times in self.response_times.items():
                if len(times) >= 10 and model_name in self.model_metrics:
                    times_list = list(times)
                    metrics = self.model_metrics[model_name]

                    metrics.p50_response_time = np.percentile(times_list, 50)
                    metrics.p95_response_time = np.percentile(times_list, 95)
                    metrics.p99_response_time = np.percentile(times_list, 99)

    def update_deployment_metrics(
        self,
        deployment_name: str,
        replica_count: int,
        current_load: int,
        queue_length: int,
        healthy_replicas: int,
        loaded_models: int,
    ):
        """Update deployment metrics"""
        current_time = time.time()

        with self._lock:
            if deployment_name not in self.deployment_metrics:
                self.deployment_metrics[deployment_name] = DeploymentMetrics(
                    deployment_name=deployment_name
                )

            metrics = self.deployment_metrics[deployment_name]

            metrics.replica_count = replica_count
            metrics.current_load = current_load
            metrics.queue_length = queue_length
            metrics.healthy_replicas = healthy_replicas
            metrics.unhealthy_replicas = replica_count - healthy_replicas
            metrics.loaded_models = loaded_models
            metrics.peak_load = max(metrics.peak_load, current_load)

            # Update average load (exponential moving average)
            alpha = 0.1
            if metrics.avg_load == 0:
                metrics.avg_load = current_load
            else:
                metrics.avg_load = alpha * current_load + (1 - alpha) * metrics.avg_load

            metrics.last_updated = current_time

        # Update Prometheus metrics
        if self.prometheus_exporter.enabled:
            self.prometheus_exporter.deployment_replicas_gauge.labels(
                deployment_name=deployment_name, status="healthy"
            ).set(healthy_replicas)

            self.prometheus_exporter.deployment_replicas_gauge.labels(
                deployment_name=deployment_name, status="unhealthy"
            ).set(metrics.unhealthy_replicas)

            self.prometheus_exporter.queue_length_gauge.labels(
                deployment_name=deployment_name
            ).set(queue_length)

    def _cleanup_old_data(self):
        """Clean up old metrics data"""
        current_time = time.time()
        cleanup_threshold = 24 * 3600  # 24 hours

        with self._lock:
            # Clean up old response times
            for model_name in list(self.response_times.keys()):
                times = self.response_times[model_name]
                # Keep only recent times
                while times and current_time - times[0] > cleanup_threshold:
                    times.popleft()

                # Remove empty entries
                if not times:
                    del self.response_times[model_name]

    def get_model_metrics(self, model_name: str) -> Optional[ModelPerformanceMetrics]:
        """Get metrics for specific model"""
        with self._lock:
            return self.model_metrics.get(model_name)

    def get_all_model_metrics(self) -> Dict[str, ModelPerformanceMetrics]:
        """Get metrics for all models"""
        with self._lock:
            return dict(self.model_metrics)

    def get_deployment_metrics(
        self, deployment_name: str
    ) -> Optional[DeploymentMetrics]:
        """Get metrics for specific deployment"""
        with self._lock:
            return self.deployment_metrics.get(deployment_name)

    def get_system_health_score(self) -> float:
        """Calculate overall system health score (0-100)"""
        try:
            if not self.resource_history:
                return 50.0  # Neutral score if no data

            recent_metrics = list(self.resource_history)[-10:]  # Last 10

            # Calculate component scores
            cpu_score = 100 - statistics.mean([m.cpu_percent for m in recent_metrics])
            memory_score = 100 - statistics.mean(
                [m.memory_percent for m in recent_metrics]
            )

            # GPU score
            gpu_score = 100
            if recent_metrics[0].gpu_count > 0:
                avg_gpu_util = statistics.mean(
                    [
                        statistics.mean(m.gpu_utilization) if m.gpu_utilization else 0
                        for m in recent_metrics
                    ]
                )
                gpu_score = 100 - avg_gpu_util

            # Model performance score
            model_score = 100
            if self.model_metrics:
                error_rates = []
                response_times = []

                for metrics in self.model_metrics.values():
                    if metrics.total_requests > 0:
                        error_rate = metrics.failed_requests / metrics.total_requests
                        error_rates.append(error_rate)
                        response_times.append(metrics.avg_response_time)

                if error_rates:
                    avg_error_rate = statistics.mean(error_rates)
                    avg_response_time = statistics.mean(response_times)

                    # Penalize high error rates and slow response times
                    model_score = int(
                        max(0, 100 - (avg_error_rate * 1000) - (avg_response_time * 10))
                    )

            # Weighted average
            health_score = (
                cpu_score * 0.3
                + memory_score * 0.3
                + gpu_score * 0.2
                + model_score * 0.2
            )

            return max(0, min(100, health_score))

        except Exception as e:
            self.logger.error(f"Error calculating health score: {e}")
            return 50.0

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        with self._lock:
            # System overview
            latest_resource = (
                self.resource_history[-1] if self.resource_history else None
            )

            # Top models by requests
            top_models = sorted(
                self.model_metrics.items(),
                key=lambda x: x[1].total_requests,
                reverse=True,
            )[:10]

            # Deployment overview
            deployment_summary = {}
            for name, metrics in self.deployment_metrics.items():
                deployment_summary[name] = {
                    "replicas": metrics.replica_count,
                    "healthy": metrics.healthy_replicas,
                    "load": metrics.current_load,
                    "queue": metrics.queue_length,
                    "models": metrics.loaded_models,
                }

            return {
                "timestamp": time.time(),
                "system": {
                    "health_score": self.get_system_health_score(),
                    "cpu_percent": (
                        latest_resource.cpu_percent if latest_resource else 0
                    ),
                    "memory_percent": (
                        latest_resource.memory_percent if latest_resource else 0
                    ),
                    "gpu_count": (latest_resource.gpu_count if latest_resource else 0),
                    "gpu_utilization": (
                        latest_resource.gpu_utilization if latest_resource else []
                    ),
                    "total_models": len(self.model_metrics),
                    "total_deployments": len(self.deployment_metrics),
                },
                "top_models": [
                    {
                        "name": name,
                        "requests": metrics.total_requests,
                        "success_rate": metrics.successful_requests
                        / max(1, metrics.total_requests),
                        "avg_response_time": metrics.avg_response_time,
                        "rps": metrics.requests_per_second,
                        "tier": metrics.current_tier,
                    }
                    for name, metrics in top_models
                ],
                "deployments": deployment_summary,
                "capacity_planning": self.capacity_planner.get_scaling_history(5),
            }
