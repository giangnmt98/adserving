"""
Unified prediction endpoint that routes to appropriate models
"""

from typing import Any, Dict

from ray import serve
from adserving.src.utils.logger import get_logger


@serve.deployment
class UnifiedPredictionEndpoint:
    """Unified prediction endpoint that routes to appropriate models"""

    def __init__(self, model_router):
        self.model_router = model_router
        self.logger = get_logger()

    async def __call__(self, request) -> Dict[str, Any]:
        """Handle all prediction requests through single endpoint"""
        try:
            # Parse request data
            if hasattr(request, "json"):
                request_data = await request.json()
            else:
                request_data = request

            # Route request
            result = await self.model_router.route_request(request_data)

            return result

        except Exception as e:
            self.logger.error(f"Endpoint error: {e}")
            return {"error": str(e), "status": "endpoint_error"}

    async def health_check(self) -> Dict[str, Any]:
        """Health check for the unified endpoint"""
        try:
            stats = self.model_router.get_routing_stats()
            return {
                "status": "healthy",
                "endpoint": "/predict",
                "routing_stats": stats,
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def get_stats(self) -> Dict[str, Any]:
        """Get detailed statistics"""
        return self.model_router.get_routing_stats()

    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get routing information for specific model"""
        return self.model_router.get_model_routing_info(model_name)
