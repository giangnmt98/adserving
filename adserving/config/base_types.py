"""
Base types and enums for configuration system
"""

from enum import Enum


class ResourceSharingStrategy(Enum):
    """Resource sharing strategies"""

    NONE = "none"
    CPU_ONLY = "cpu_only"
    GPU_SHARED = "gpu_shared"
    MEMORY_MAPPED = "memory_mapped"
    FULL_SHARING = "full_sharing"


class ModelTier:
    """Model tier constants"""

    WARM = "warm"
    HOT = "hot"
    COLD = "cold"


class RoutingStrategy:
    """Routing strategy constants"""

    LEAST_LOADED = "least_loaded"
    ROUND_ROBIN = "round_robin"
    FASTEST_RESPONSE = "fastest_response"
    MODEL_AFFINITY = "model_affinity"
