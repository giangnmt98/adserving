"""
Model pool management for pooled deployments
"""

import threading
import time
from collections import defaultdict, deque
from typing import Any, Dict, Optional

from ..core.model_manager import ModelTier
from adserving.src.utils.logger import get_logger


class ModelPool:
    """Manages a pool of models within a single deployment"""

    def __init__(self, pool_id: str, max_models: int = 50):
        self.pool_id = pool_id
        self.max_models = max_models
        self.loaded_models: Dict[str, Any] = {}
        self.model_usage: Dict[str, float] = {}  # Last access time
        self.model_tiers: Dict[str, ModelTier] = {}
        self.request_queue: deque = deque()
        self._lock = threading.RLock()
        self.logger = get_logger()

    def add_model(self, model_name: str, model_info: Any, tier: ModelTier):
        """Add model to pool with eviction if needed"""
        with self._lock:
            if len(self.loaded_models) >= self.max_models:
                self._evict_least_used_model()

            self.loaded_models[model_name] = model_info
            self.model_usage[model_name] = time.time()
            self.model_tiers[model_name] = tier
            self.logger.info(f"Added model {model_name} to pool {self.pool_id}")

    def get_model(self, model_name: str) -> Optional[Any]:
        """Get model from pool and update usage"""
        with self._lock:
            if model_name in self.loaded_models:
                self.model_usage[model_name] = time.time()
                return self.loaded_models[model_name]
            return None

    def remove_model(self, model_name: str) -> bool:
        """Remove model from pool"""
        with self._lock:
            if model_name in self.loaded_models:
                del self.loaded_models[model_name]
                del self.model_usage[model_name]
                del self.model_tiers[model_name]
                self.logger.info(f"Removed model {model_name} from pool {self.pool_id}")
                return True
            return False

    def _evict_least_used_model(self):
        """Evict the least recently used model using optimized approach"""
        if not self.loaded_models:
            return

        # Separate models by tier for efficient eviction
        cold_models = []
        warm_models = []
        hot_models = []

        for model_name, last_used in self.model_usage.items():
            tier = self.model_tiers.get(model_name, ModelTier.COLD)
            if tier == ModelTier.COLD:
                cold_models.append((last_used, model_name))
            elif tier == ModelTier.WARM:
                warm_models.append((last_used, model_name))
            else:  # HOT
                hot_models.append((last_used, model_name))

        # Evict from COLD tier first (least priority), then WARM, then HOT
        lru_model = None
        if cold_models:
            lru_model = min(cold_models)[1]
        elif warm_models:
            lru_model = min(warm_models)[1]
        elif hot_models:
            lru_model = min(hot_models)[1]

        if lru_model:
            self.remove_model(lru_model)

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get pool statistics with memory optimization"""
        with self._lock:
            tier_counts = defaultdict(int)
            for tier in self.model_tiers.values():
                tier_counts[tier.value] += 1

            return {
                "pool_id": self.pool_id,
                "loaded_models": len(self.loaded_models),
                "max_models": self.max_models,
                "model_count": len(self.loaded_models),  # Use count instead
                "tier_distribution": dict(tier_counts),
            }
