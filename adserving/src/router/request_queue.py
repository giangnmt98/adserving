"""
Intelligent request queue with priority and batching
"""

import threading
import time
from collections import defaultdict, deque
from typing import Any, Dict, Optional
from adserving.src.utils.logger import get_logger


class RequestQueue:
    """Intelligent request queue with priority and batching"""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.queues: Dict[str, deque] = defaultdict(deque)  # Per-model queues
        self.priorities: Dict[str, int] = {}  # Model priorities
        self._total_queued = 0  # Running total for O(1) stats
        # Fine-grained locking to reduce contention
        self._model_locks: Dict[str, threading.RLock] = defaultdict(threading.RLock)
        self._counter_lock = threading.RLock()
        self.logger = get_logger()

    def enqueue(
        self, model_name: str, request_data: Dict[str, Any], priority: int = 0
    ) -> bool:
        """Enqueue request with priority and proper size limits"""
        # Check global queue limit first (lightweight)
        with self._counter_lock:
            if self._total_queued >= self.max_size * 10:  # Global limit: 10x per-model
                self.logger.warning(
                    f"Global queue limit reached (total: {self._total_queued})"
                )
                return False

        # Lock only this model's queue
        model_lock = self._model_locks[model_name]
        with model_lock:
            # Check per-model queue size limit (max_size per model)
            if len(self.queues[model_name]) >= self.max_size:
                self.logger.warning(
                    f"Queue full for model {model_name} "
                    f"(size: {len(self.queues[model_name])})"
                )
                return False

            self.queues[model_name].append(
                {
                    "data": request_data,
                    "priority": priority,
                    "timestamp": time.time(),
                }
            )
            # Update per-model priority/length
            self.priorities[model_name] = len(self.queues[model_name])

        # Update global counter outside model lock
        with self._counter_lock:
            self._total_queued += 1

        return True

    def dequeue(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Dequeue highest priority request for model"""
        model_lock = self._model_locks[model_name]
        with model_lock:
            if model_name not in self.queues or not self.queues[model_name]:
                return None

            # For now, simple FIFO. Could implement priority queue
            request = self.queues[model_name].popleft()
            self.priorities[model_name] = len(self.queues[model_name])

        # Decrement global counter outside model lock
        with self._counter_lock:
            self._total_queued -= 1

        return request

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics using fine-grained locks (O(1) for total)."""
        # Snapshot global counter
        with self._counter_lock:
            total = self._total_queued

        # Snapshot per-model counts and priorities with per-model locks
        per_model_counts: Dict[str, int] = {}
        priorities: Dict[str, int] = {}
        for model in list(self.queues.keys()):
            lock = self._model_locks[model]
            with lock:
                per_model_counts[model] = len(self.queues[model])
                priorities[model] = self.priorities.get(model, 0)

        return {
            "total_queued": total,
            "per_model_counts": per_model_counts,
            "priorities": priorities,
        }
