"""
Tiered Model Cache

This module implements a tiered model cache with intelligent eviction
and promotion strategies for optimal model management.
"""

import threading
import time
from collections import OrderedDict, defaultdict
from typing import Any, Dict, Optional

from .model_info import ModelInfo
from .model_tier import ModelTier


class TieredModelCache:
    """Model cache with tiered storage and intelligent eviction"""

    def __init__(
        self,
        hot_cache_size: int = 50,
        warm_cache_size: int = 200,
        cold_cache_size: int = 500,
    ):
        self.hot_cache_size = hot_cache_size
        self.warm_cache_size = warm_cache_size
        self.cold_cache_size = cold_cache_size

        # Separate caches for each tier
        self.hot_cache: OrderedDict[str, ModelInfo] = OrderedDict()
        self.warm_cache: OrderedDict[str, ModelInfo] = OrderedDict()
        self.cold_cache: OrderedDict[str, ModelInfo] = OrderedDict()

        # Model tier assignments
        self.model_tiers: Dict[str, ModelTier] = {}

        # Usage statistics for tier promotion/demotion
        self.usage_stats: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {
                "hourly_requests": 0,
                "daily_requests": 0,
                "avg_response_time": 0,
                "error_rate": 0,
            }
        )

        # Real-time cache statistics
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_load_time = 0.0
        self._load_count = 0
        self._stats_lock = threading.Lock()

        self._lock = threading.RLock()

    def get(self, model_key: str) -> Optional[ModelInfo]:
        """Get model from appropriate tier cache with statistics tracking"""
        model_info = None
        needs_promotion = False

        # Fast path: check caches with minimal lock time
        with self._lock:
            # Check hot cache first
            if model_key in self.hot_cache:
                model_info = self.hot_cache[model_key]
                model_info.update_access()
                self.hot_cache.move_to_end(model_key)
            # Check warm cache
            elif model_key in self.warm_cache:
                model_info = self.warm_cache[model_key]
                model_info.update_access()
                self.warm_cache.move_to_end(model_key)
                needs_promotion = self._should_promote(model_info)
            # Check cold cache
            elif model_key in self.cold_cache:
                model_info = self.cold_cache[model_key]
                model_info.update_access()
                self.cold_cache.move_to_end(model_key)
                needs_promotion = self._should_promote(model_info)

        # Update stats outside of main lock
        if model_info:
            with self._stats_lock:
                self._cache_hits += 1
            # Handle promotion outside of main lock if needed
            if needs_promotion:
                self._promote_model_async(model_key, model_info)
        else:
            with self._stats_lock:
                self._cache_misses += 1

        return model_info

    def put(self, model_key: str, model_info: ModelInfo):
        """Put model in appropriate tier cache"""
        with self._lock:
            tier = model_info.tier

            if tier == ModelTier.HOT:
                self._put_in_hot_cache(model_key, model_info)
            elif tier == ModelTier.WARM:
                self._put_in_warm_cache(model_key, model_info)
            else:  # COLD
                self._put_in_cold_cache(model_key, model_info)

            self.model_tiers[model_key] = tier

    def _put_in_hot_cache(self, model_key: str, model_info: ModelInfo):
        """Put model in hot cache with eviction if needed"""
        self.hot_cache[model_key] = model_info
        if len(self.hot_cache) > self.hot_cache_size:
            # Evict least recently used, but demote to warm instead of removing
            lru_key, lru_model = self.hot_cache.popitem(last=False)
            lru_model.tier = ModelTier.WARM
            self._put_in_warm_cache(lru_key, lru_model)

    def _put_in_warm_cache(self, model_key: str, model_info: ModelInfo):
        """Put model in warm cache with eviction if needed"""
        self.warm_cache[model_key] = model_info
        if len(self.warm_cache) > self.warm_cache_size:
            # Evict least recently used, demote to cold
            lru_key, lru_model = self.warm_cache.popitem(last=False)
            lru_model.tier = ModelTier.COLD
            self._put_in_cold_cache(lru_key, lru_model)

    def _put_in_cold_cache(self, model_key: str, model_info: ModelInfo):
        """Put model in cold cache with eviction if needed"""
        self.cold_cache[model_key] = model_info
        if len(self.cold_cache) > self.cold_cache_size:
            # Evict least recently used completely
            self.cold_cache.popitem(last=False)

    def _should_promote(self, model_info: ModelInfo) -> bool:
        """Check if model should be promoted (lightweight check)"""
        current_time = time.time()

        # Check promotion criteria without doing the actual promotion
        if (
            model_info.tier == ModelTier.COLD
            and model_info.access_count > 5
            and current_time - model_info.loaded_at < 3600
        ):
            return True

        if (
            model_info.tier == ModelTier.WARM
            and model_info.access_count > 20
            and current_time - model_info.last_accessed < 300
        ):
            return True

        return False

    def _promote_model_async(self, model_key: str, model_info: ModelInfo):
        """Promote model to higher tier (can be called outside main lock)"""
        with self._lock:
            # Double-check the model is still in the expected tier
            current_tier = model_info.tier

            if (
                current_tier == ModelTier.COLD
                and model_key in self.cold_cache
                and self._should_promote(model_info)
            ):

                self.cold_cache.pop(model_key, None)
                model_info.tier = ModelTier.WARM
                self._put_in_warm_cache(model_key, model_info)

            elif (
                current_tier == ModelTier.WARM
                and model_key in self.warm_cache
                and self._should_promote(model_info)
            ):

                self.warm_cache.pop(model_key, None)
                model_info.tier = ModelTier.HOT
                self._put_in_hot_cache(model_key, model_info)

    def _consider_tier_promotion(self, model_key: str, model_info: ModelInfo):
        """Consider promoting model to higher tier based on usage (legacy method)"""
        # Keep for backward compatibility, but use the new async approach
        if self._should_promote(model_info):
            self._promote_model_async(model_key, model_info)

    def remove(self, model_key: str) -> bool:
        """Remove model from all caches"""
        with self._lock:
            removed = False
            if model_key in self.hot_cache:
                del self.hot_cache[model_key]
                removed = True
            if model_key in self.warm_cache:
                del self.warm_cache[model_key]
                removed = True
            if model_key in self.cold_cache:
                del self.cold_cache[model_key]
                removed = True

            self.model_tiers.pop(model_key, None)
            return removed

    def record_load_time(self, load_time: float):
        """Record model load time for statistics"""
        with self._stats_lock:
            self._total_load_time += load_time
            self._load_count += 1

    def get_real_cache_stats(self) -> Dict[str, float]:
        """Get real-time cache statistics"""
        with self._stats_lock:
            total_requests = self._cache_hits + self._cache_misses
            return {
                "hit_rate": (
                    self._cache_hits / total_requests if total_requests > 0 else 0.0
                ),
                "miss_rate": (
                    self._cache_misses / total_requests if total_requests > 0 else 0.0
                ),
                "avg_load_time": (
                    self._total_load_time / self._load_count
                    if self._load_count > 0
                    else 0.0
                ),
                "total_hits": self._cache_hits,
                "total_misses": self._cache_misses,
                "total_loads": self._load_count,
            }

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics with real-time metrics"""
        with self._lock:
            # Get real-time statistics
            real_time_stats = self.get_real_cache_stats()

            return {
                "hot_cache": {
                    "size": len(self.hot_cache),
                    "capacity": self.hot_cache_size,
                    "models": list(self.hot_cache.keys()),
                },
                "warm_cache": {
                    "size": len(self.warm_cache),
                    "capacity": self.warm_cache_size,
                    "models": list(self.warm_cache.keys()),
                },
                "cold_cache": {
                    "size": len(self.cold_cache),
                    "capacity": self.cold_cache_size,
                    "models": list(self.cold_cache.keys()),
                },
                "total_models": len(self.hot_cache)
                + len(self.warm_cache)
                + len(self.cold_cache),
                "performance": {
                    "hit_rate": real_time_stats["hit_rate"],
                    "miss_rate": real_time_stats["miss_rate"],
                    "avg_load_time": real_time_stats["avg_load_time"],
                    "total_hits": real_time_stats["total_hits"],
                    "total_misses": real_time_stats["total_misses"],
                    "total_loads": real_time_stats["total_loads"],
                },
            }

    def keys(self):
        """Get all cached model names from all tiers"""
        with self._lock:
            all_keys = []
            all_keys.extend(self.hot_cache.keys())
            all_keys.extend(self.warm_cache.keys())
            all_keys.extend(self.cold_cache.keys())
            return all_keys
