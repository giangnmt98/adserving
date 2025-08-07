"""
Model performance metrics data structures.
"""

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelPerformanceMetrics:
    """Detailed performance metrics for individual models"""

    model_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    # Response time metrics
    avg_response_time: float = 0.0
    min_response_time: float = float("inf")
    max_response_time: float = 0.0
    p50_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0

    # Throughput metrics
    requests_per_second: float = 0.0
    peak_rps: float = 0.0

    # Resource usage
    avg_memory_usage: float = 0.0
    peak_memory_usage: float = 0.0
    avg_cpu_usage: float = 0.0

    # Cache metrics
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0

    # Tier information
    current_tier: str = "warm"
    tier_promotions: int = 0
    tier_demotions: int = 0

    # Timestamps
    first_request_time: Optional[float] = None
    last_request_time: Optional[float] = None
    last_updated: float = field(default_factory=time.time)
