"""
Ray remote actors for parallel model prediction processing
"""

from .ray_model_deployment import RayModelDeployment
from .ray_model_router import RayModelRouter

__all__ = ["RayModelDeployment", "RayModelRouter"]