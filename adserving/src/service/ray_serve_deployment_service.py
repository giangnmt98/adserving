"""
Ray Serve Deployment Service

This service handles the deployment of Ray Serve applications and deployments
to make them visible on the Ray dashboard.
"""

import asyncio
from typing import Optional

from ray import serve
from adserving.src.utils.logger import get_logger
from adserving.src.router.unified_endpoint import UnifiedPredictionEndpoint


logger = get_logger()


class RayServeDeploymentService:
    """Service for deploying Ray Serve applications"""
    
    def __init__(self):
        self.logger = get_logger()
        self.deployed_applications = {}
        
    def deploy_unified_endpoint(self, model_router) -> bool:
        """Deploy the UnifiedPredictionEndpoint to Ray Serve"""
        try:
            self.logger.info("Deploying UnifiedPredictionEndpoint to Ray Serve...")
            
            # Create a Ray-serializable router to avoid pickle errors with threading objects
            from adserving.src.router.serializable_model_router import create_serializable_router_from_model_router
            serializable_router = create_serializable_router_from_model_router(model_router)
            self.logger.info("Created Ray-serializable router for deployment")
            
            # Create the deployment instance with the serializable model router
            endpoint = UnifiedPredictionEndpoint.bind(serializable_router)
            
            # Deploy to Ray Serve with a specific name
            serve.run(endpoint, name="unified_prediction_endpoint")
            
            # Store the deployment info
            self.deployed_applications["unified_prediction_endpoint"] = {
                "endpoint": endpoint,
                "status": "deployed",
                "name": "unified_prediction_endpoint"
            }
            
            self.logger.info("✓ UnifiedPredictionEndpoint successfully deployed to Ray Serve")
            self.logger.info("Ray Serve deployments should now be visible on Ray dashboard")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to deploy UnifiedPredictionEndpoint to Ray Serve: {e}")
            import traceback
            self.logger.error(f"Deployment traceback:\n{traceback.format_exc()}")
            return False
    
    def deploy_model_endpoints(self, ray_model_deployment_manager) -> bool:
        """Deploy individual model endpoints to Ray Serve"""
        try:
            deployed_models = ray_model_deployment_manager.get_deployed_models()
            
            if not deployed_models:
                self.logger.info("No Ray model deployments found to deploy to Ray Serve")
                return True
            
            self.logger.info(f"Deploying {len(deployed_models)} model endpoints to Ray Serve...")
            
            for model_name in deployed_models:
                try:
                    # Create a serve deployment for each model
                    model_deployment_info = ray_model_deployment_manager.deployed_models[model_name]
                    
                    # Create a wrapper deployment for the model
                    @serve.deployment(name=f"model_{model_name.replace('.', '_')}")
                    class ModelEndpoint:
                        def __init__(self, model_deployment_info):
                            self.model_deployment_info = model_deployment_info
                            self.logger = get_logger()
                            
                        async def __call__(self, request):
                            # Route to the Ray model deployment
                            return await self.model_deployment_info.router_handle.route_request.remote(request)
                    
                    # Deploy the model endpoint
                    model_endpoint = ModelEndpoint.bind(model_deployment_info)
                    serve.run(model_endpoint, name=f"model_{model_name.replace('.', '_')}")
                    
                    self.deployed_applications[f"model_{model_name}"] = {
                        "endpoint": model_endpoint,
                        "status": "deployed",
                        "name": f"model_{model_name.replace('.', '_')}"
                    }
                    
                    self.logger.info(f"✓ Model {model_name} endpoint deployed to Ray Serve")
                    
                except Exception as model_error:
                    self.logger.error(f"Failed to deploy model {model_name} to Ray Serve: {model_error}")
                    continue
            
            self.logger.info(f"✓ Model endpoints deployment completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to deploy model endpoints to Ray Serve: {e}")
            import traceback
            self.logger.error(f"Model deployment traceback:\n{traceback.format_exc()}")
            return False
    
    def get_deployment_status(self) -> dict:
        """Get status of all Ray Serve deployments"""
        try:
            # Get Ray Serve status
            serve_status = serve.status()
            
            return {
                "deployed_applications": list(self.deployed_applications.keys()),
                "ray_serve_status": serve_status,
                "total_deployments": len(self.deployed_applications)
            }
        except Exception as e:
            self.logger.error(f"Failed to get Ray Serve deployment status: {e}")
            return {
                "deployed_applications": list(self.deployed_applications.keys()),
                "ray_serve_status": "error",
                "total_deployments": len(self.deployed_applications),
                "error": str(e)
            }
    
    def cleanup(self):
        """Cleanup Ray Serve deployments"""
        try:
            self.logger.info("Cleaning up Ray Serve deployments...")
            
            # Shutdown Ray Serve deployments
            for app_name in self.deployed_applications:
                try:
                    serve.delete(app_name)
                    self.logger.info(f"Deleted Ray Serve deployment: {app_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to delete Ray Serve deployment {app_name}: {e}")
            
            self.deployed_applications.clear()
            self.logger.info("Ray Serve deployment cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during Ray Serve deployment cleanup: {e}")