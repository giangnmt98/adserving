"""
Routing strategies for model selection
"""

from enum import Enum


class RoutingStrategy(Enum):
    """Routing strategies for model selection"""

    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    FASTEST_RESPONSE = "fastest_response"
    MODEL_AFFINITY = "model_affinity"
