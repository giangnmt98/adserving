"""
Prometheus metrics exporter for monitoring integration.
"""

from typing import Any, Dict

from .model_metrics import ModelPerformanceMetrics
from .resource_metrics import ResourceMetrics

try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class PrometheusExporter:
    """Prometheus metrics exporter"""

    def __init__(self, enabled: bool = True, port: int = 9090):
        self.enabled = enabled and PROMETHEUS_AVAILABLE
        self.port = port

        if self.enabled:
            self.registry = CollectorRegistry()
            self._setup_metrics()

    def _setup_metrics(self):
        """Setup Prometheus metrics"""
        # Request metrics
        self.request_counter = Counter(
            "mlops_requests_total",
            "Total number of requests",
            ["model_name", "status"],
            registry=self.registry,
        )

        self.response_time_histogram = Histogram(
            "mlops_response_time_seconds",
            "Response time in seconds",
            ["model_name"],
            registry=self.registry,
        )

        # Model metrics
        self.model_cache_gauge = Gauge(
            "mlops_model_cache_size",
            "Number of models in cache",
            ["tier"],
            registry=self.registry,
        )

        self.model_memory_gauge = Gauge(
            "mlops_model_memory_usage_bytes",
            "Memory usage per model",
            ["model_name"],
            registry=self.registry,
        )

        # System metrics
        self.cpu_usage_gauge = Gauge(
            "mlops_cpu_usage_percent", "CPU usage percentage", registry=self.registry
        )

        self.memory_usage_gauge = Gauge(
            "mlops_memory_usage_percent",
            "Memory usage percentage",
            registry=self.registry,
        )

        self.gpu_usage_gauge = Gauge(
            "mlops_gpu_usage_percent",
            "GPU usage percentage",
            ["gpu_id"],
            registry=self.registry,
        )

        # Deployment metrics
        self.deployment_replicas_gauge = Gauge(
            "mlops_deployment_replicas",
            "Number of deployment replicas",
            ["deployment_name", "status"],
            registry=self.registry,
        )

        self.queue_length_gauge = Gauge(
            "mlops_queue_length",
            "Current queue length",
            ["deployment_name"],
            registry=self.registry,
        )

    def record_request(self, model_name: str, response_time: float, success: bool):
        """Record request metrics"""
        if not self.enabled:
            return

        status = "success" if success else "error"
        self.request_counter.labels(model_name=model_name, status=status).inc()
        self.response_time_histogram.labels(model_name=model_name).observe(
            response_time
        )

    def update_system_metrics(self, metrics: ResourceMetrics):
        """Update system metrics"""
        if not self.enabled:
            return

        self.cpu_usage_gauge.set(metrics.cpu_percent)
        self.memory_usage_gauge.set(metrics.memory_percent)

        for i, gpu_util in enumerate(metrics.gpu_utilization):
            self.gpu_usage_gauge.labels(gpu_id=str(i)).set(gpu_util)

    def update_model_metrics(self, model_metrics: Dict[str, ModelPerformanceMetrics]):
        """Update model-level metrics"""
        if not self.enabled:
            return

        for model_name, metrics in model_metrics.items():
            self.model_memory_gauge.labels(model_name=model_name).set(
                metrics.avg_memory_usage
            )

    def update_cache_metrics(self, cache_stats: Dict[str, Any]):
        """Update cache metrics"""
        if not self.enabled:
            return

        for tier, stats in cache_stats.items():
            if isinstance(stats, dict) and "size" in stats:
                self.model_cache_gauge.labels(tier=tier).set(stats["size"])

    def get_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        if not self.enabled:
            return ""

        return generate_latest(self.registry).decode("utf-8")
