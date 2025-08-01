"""
Intelligent request queue with priority and batching
"""

import logging
import threading
import time
from collections import defaultdict, deque
from typing import Any, Dict, Optional


class RequestQueue:
    """Intelligent request queue with priority and batching"""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.queues: Dict[str, deque] = defaultdict(deque)  # Per-model queues
        self.priorities: Dict[str, int] = {}  # Model priorities
        self._total_queued = 0  # Running total for O(1) stats
        self._lock = threading.RLock()
        self.logger = logging.getLogger(f"{__name__}.RequestQueue")

    def enqueue(
        self, model_name: str, request_data: Dict[str, Any], priority: int = 0
    ) -> bool:
        """Enqueue request with priority and proper size limits"""
        with self._lock:
            # Check per-model queue size limit (max_size per model)
            if len(self.queues[model_name]) >= self.max_size:
                self.logger.warning(
                    f"Queue full for model {model_name} "
                    f"(size: {len(self.queues[model_name])})"
                )
                return False

            # Check total queue size using running counter
            if (
                self._total_queued >= self.max_size * 10
            ):  # Global limit: 10x per-model limit
                self.logger.warning(
                    f"Global queue limit reached (total: {self._total_queued})"
                )
                return False

            self.queues[model_name].append(
                {
                    "data": request_data,
                    "priority": priority,
                    "timestamp": time.time(),
                }
            )

            # Update counters
            self._total_queued += 1
            self.priorities[model_name] = len(self.queues[model_name])
            return True

    def dequeue(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Dequeue highest priority request for model"""
        with self._lock:
            if model_name not in self.queues or not self.queues[model_name]:
                return None

            # For now, simple FIFO. Could implement priority queue
            request = self.queues[model_name].popleft()
            self._total_queued -= 1  # Decrement running total
            self.priorities[model_name] = len(self.queues[model_name])
            return request

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics with O(1) total calculation"""
        with self._lock:
            return {
                "total_queued": self._total_queued,  # O(1) instead of O(n)
                "per_model_counts": {
                    model: len(queue) for model, queue in self.queues.items()
                },
                "priorities": dict(self.priorities),
            }
