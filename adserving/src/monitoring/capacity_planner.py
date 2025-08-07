"""
Intelligent capacity planning for auto-scaling.
"""

import statistics
import time
from collections import deque
from typing import Any, Dict, List, Tuple

from .resource_metrics import ResourceMetrics
from adserving.src.utils.logger import get_logger


class CapacityPlanner:
    """Intelligent capacity planning for auto-scaling"""

    def __init__(self, history_window: int = 3600):
        self.history_window = history_window  # seconds
        self.request_history: deque = deque(maxlen=10000)
        self.resource_history: deque = deque(maxlen=1000)
        self.scaling_decisions: deque = deque(maxlen=100)

        self.logger = get_logger()

    def record_request(self, timestamp: float, model_name: str, response_time: float):
        """Record request for capacity planning"""
        self.request_history.append(
            {
                "timestamp": timestamp,
                "model_name": model_name,
                "response_time": response_time,
            }
        )

    def record_resource_usage(self, metrics: ResourceMetrics):
        """Record resource usage"""
        self.resource_history.append(metrics)

    def predict_scaling_needs(
        self,
        deployment_name: str,
        current_replicas: int,
        target_response_time: float = 1.0,
        target_cpu_usage: float = 70.0,
    ) -> Tuple[int, str]:
        """Predict scaling needs based on historical data"""
        try:
            current_time = time.time()

            recent_metrics = self._get_recent_metrics(current_time)
            if not recent_metrics:
                return current_replicas, "No recent data"

            avg_metrics = self._calculate_average_metrics(recent_metrics)
            scaling_reasons = self._evaluate_scaling_conditions(
                avg_metrics, target_response_time, target_cpu_usage
            )

            new_replicas, reason = self._make_scaling_decision(
                scaling_reasons, current_replicas, avg_metrics["current_rps"]
            )

            self._record_scaling_decision(
                deployment_name, current_replicas, new_replicas, reason, avg_metrics
            )

            return new_replicas, reason

        except Exception as e:
            self.logger.error(f"Error in capacity planning: {e}")
            return current_replicas, f"Error: {e}"

    def _get_recent_metrics(self, current_time: float) -> Dict:
        recent_requests = [
            req
            for req in self.request_history
            if current_time - req["timestamp"] <= 300
        ]

        recent_resources = [
            res for res in self.resource_history if current_time - res.timestamp <= 300
        ]

        return {"requests": recent_requests, "resources": recent_resources}

    def _calculate_average_metrics(self, recent_metrics: Dict) -> Dict:
        requests = recent_metrics["requests"]
        resources = recent_metrics["resources"]

        avg_response_time = statistics.mean([req["response_time"] for req in requests])
        current_rps = len(requests) / 300

        if resources:
            avg_cpu = statistics.mean([res.cpu_percent for res in resources])
            avg_memory = statistics.mean([res.memory_percent for res in resources])
        else:
            avg_cpu = avg_memory = 0

        predicted_rps = self._predict_future_rps(current_rps)

        return {
            "avg_response_time": avg_response_time,
            "current_rps": current_rps,
            "predicted_rps": predicted_rps,
            "avg_cpu": avg_cpu,
            "avg_memory": avg_memory,
        }

    def _evaluate_scaling_conditions(
        self, metrics: Dict, target_response_time: float, target_cpu_usage: float
    ) -> Dict:
        scale_up_reasons = []
        scale_down_reasons = []

        # Response time based scaling
        if metrics["avg_response_time"] > target_response_time * 1.5:
            scale_up_reasons.append(
                f"High response time: {metrics['avg_response_time']:.2f}s"
            )
        elif metrics["avg_response_time"] < target_response_time * 0.5:
            scale_down_reasons.append(
                f"Low response time: {metrics['avg_response_time']:.2f}s"
            )

        # CPU based scaling
        if metrics["avg_cpu"] > target_cpu_usage:
            scale_up_reasons.append(f"High CPU usage: {metrics['avg_cpu']:.1f}%")
        elif metrics["avg_cpu"] < target_cpu_usage * 0.3:
            scale_down_reasons.append(f"Low CPU usage: {metrics['avg_cpu']:.1f}%")

        # Memory based scaling
        if metrics["avg_memory"] > 85:
            scale_up_reasons.append(f"High memory usage: {metrics['avg_memory']:.1f}%")

        return {"up": scale_up_reasons, "down": scale_down_reasons}

    def _make_scaling_decision(
        self, scaling_reasons: Dict, current_replicas: int, current_rps: float
    ) -> Tuple[int, str]:
        capacity_per_replica = 10
        needed_replicas = max(1, int(current_rps / capacity_per_replica))

        if needed_replicas > current_replicas:
            scaling_reasons["up"].append(f"Predicted RPS: {current_rps:.1f}")
        elif needed_replicas < current_replicas * 0.7:
            scaling_reasons["down"].append(f"Low predicted RPS: {current_rps:.1f}")

        if scaling_reasons["up"]:
            new_replicas = min(current_replicas + 1, needed_replicas + 2)
            reason = f"Scale up: {', '.join(scaling_reasons['up'])}"
        elif scaling_reasons["down"] and current_replicas > 1:
            new_replicas = max(1, current_replicas - 1)
            reason = f"Scale down: {', '.join(scaling_reasons['down'])}"
        else:
            new_replicas = current_replicas
            reason = "No scaling needed"

        return new_replicas, reason

    def _record_scaling_decision(
        self,
        deployment_name: str,
        current_replicas: int,
        new_replicas: int,
        reason: str,
        metrics: Dict,
    ):
        self.scaling_decisions.append(
            {
                "timestamp": time.time(),
                "deployment": deployment_name,
                "from_replicas": current_replicas,
                "to_replicas": new_replicas,
                "reason": reason,
                "metrics": metrics,
            }
        )

    def _predict_future_rps(self, current_rps: float) -> float:
        """Simple RPS prediction based on recent trends"""
        if len(self.request_history) < 100:
            return current_rps

        # Get RPS for different time windows
        current_time = time.time()
        windows = [60, 300, 900]  # 1min, 5min, 15min
        rps_values = []

        for window in windows:
            window_requests = [
                req
                for req in self.request_history
                if current_time - req["timestamp"] <= window
            ]
            if window_requests:
                rps_values.append(len(window_requests) / window)

        if not rps_values:
            return current_rps

        # Simple trend analysis - if recent RPS is higher, predict increase
        if len(rps_values) >= 2:
            trend = rps_values[0] - rps_values[-1]  # Short term - long term
            return max(0, current_rps + trend * 0.5)  # Conservative prediction

        return current_rps

    def get_scaling_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent scaling decisions"""
        return list(self.scaling_decisions)[-limit:]
