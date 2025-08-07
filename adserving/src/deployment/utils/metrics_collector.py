"""
Metrics collection utilities for pooled deployments
"""

import threading
from collections import defaultdict
from typing import Any, Dict


class MetricsCollector:
    """Thread-safe metrics collection for pooled deployments"""

    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.total_inference_time = 0.0
        self.model_request_counts: Dict[str, int] = defaultdict(int)
        self._metrics_lock = threading.Lock()

    def increment_request_count(self):
        """Increment total request count"""
        with self._metrics_lock:
            self.request_count += 1

    def increment_error_count(self):
        """Increment error count"""
        with self._metrics_lock:
            self.error_count += 1

    def add_inference_time(self, time_taken: float):
        """Add inference time to total"""
        with self._metrics_lock:
            self.total_inference_time += time_taken

    def increment_model_request_count(self, model_name: str):
        """Increment request count for specific model"""
        with self._metrics_lock:
            self.model_request_counts[model_name] += 1

    def get_metrics_snapshot(self) -> Dict[str, Any]:
        """Get thread-safe snapshot of current metrics"""
        with self._metrics_lock:
            return {
                "request_count": self.request_count,
                "error_count": self.error_count,
                "total_inference_time": self.total_inference_time,
                "model_request_counts": dict(self.model_request_counts),
            }

    def calculate_derived_metrics(self) -> Dict[str, Any]:
        """Calculate derived metrics like averages and rates"""
        snapshot = self.get_metrics_snapshot()

        successful_requests = max(
            1, snapshot["request_count"] - snapshot["error_count"]
        )
        avg_inference_time = snapshot["total_inference_time"] / successful_requests
        error_rate = snapshot["error_count"] / max(1, snapshot["request_count"])

        return {
            "avg_inference_time": avg_inference_time,
            "error_rate": error_rate,
            "successful_requests": successful_requests,
        }
