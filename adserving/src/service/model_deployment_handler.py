"""
Model deployment handler with Ray Model Router System
Handles production model deployment using load > deploy > ray model router workflow
"""

import asyncio
from typing import Dict, List
from adserving.src.core.ray_model_deployment_manager import RayModelDeploymentManager
from adserving.src.utils.logger import get_logger

logger = get_logger()


class ModelDeploymentHandler:
    """Handles model deployment operations with Ray Model Router System"""

    def deploy_models(
        self, model_manager, deployment_manager, model_router
    ) -> Dict[str, int]:
        """Deploy production models using Ray Model Router workflow."""
        try:
            logger.info("Deploying production models with Ray Model Router System...")

            # Get production models
            production_models = model_manager.get_production_models()

            if not production_models:
                logger.warning("No production models found")
                return {"loaded": 0, "failed": 0}

            # Initialize Ray Model Deployment Manager using pure Ray parallel processing
            logger.info("Using Ray parallel processing for model deployment (no workers)")
            ray_deployment_manager = RayModelDeploymentManager(
                model_manager=model_manager  # Pure Ray processing, no worker infrastructure
            )

            # Deploy all models with Ray Model Router workflow
            logger.info(f"Starting Load > Deploy > Route workflow for {len(production_models)} models")
            
            # Run the async deployment workflow with proper event loop handling
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, create a new one in a thread
                    import concurrent.futures
                    
                    def run_deployment():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            return new_loop.run_until_complete(
                                ray_deployment_manager.deploy_multiple_models_parallel(production_models)
                            )
                        finally:
                            new_loop.close()
                    
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_deployment)
                        deployment_results = future.result()
                else:
                    deployment_results = asyncio.run(
                        ray_deployment_manager.deploy_multiple_models_parallel(production_models)
                    )
            except RuntimeError as e:
                if "asyncio.run() cannot be called from a running event loop" in str(e):
                    # Create new event loop to avoid conflict
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        deployment_results = new_loop.run_until_complete(
                            ray_deployment_manager.deploy_multiple_models_parallel(production_models)
                        )
                    finally:
                        new_loop.close()
                else:
                    raise

            loaded_count = deployment_results.get("successful_deployments", 0)
            failed_count = deployment_results.get("failed_deployments", 0)
            total_time = deployment_results.get("total_time", 0)

            logger.info(
                f"Ray Model Router deployment completed in {total_time:.2f}s: "
                f"{loaded_count} successful, {failed_count} failed"
            )

            # Log deployment details
            self._log_deployment_details(deployment_results)

            return {"loaded": loaded_count, "failed": failed_count}

        except Exception as e:
            logger.error(f"Failed to deploy production models with Ray Model Router: {e}")
            return {"loaded": 0, "failed": 0}

    def _log_deployment_details(self, deployment_results: Dict) -> None:
        """Log detailed deployment results."""
        try:
            results = deployment_results.get("results", {})
            
            if not results:
                logger.warning("No deployment results to log")
                return
            
            # Count successful and failed deployments by type
            successful_models = []
            failed_models = []
            
            for model_name, result in results.items():
                if result.get("status") == "success":
                    successful_models.append(model_name)
                else:
                    failed_models.append(model_name)
            
            # Log successful deployments
            if successful_models:
                logger.info(f"Successfully deployed models with Ray routers:")
                for model_name in successful_models[:5]:  # Show first 5
                    logger.info(f"  ✅ {model_name} - Ray deployment and router created")
                if len(successful_models) > 5:
                    logger.info(f"  ... and {len(successful_models) - 5} more models")
            
            # Log failed deployments
            if failed_models:
                logger.warning(f"Failed to deploy models with Ray routers:")
                for model_name in failed_models[:3]:  # Show first 3 failures
                    result = results[model_name]
                    error_msg = result.get("error_message", "Unknown error")
                    logger.warning(f"  ❌ {model_name} - {error_msg}")
                if len(failed_models) > 3:
                    logger.warning(f"  ... and {len(failed_models) - 3} more failures")
            
            # Log summary statistics
            total_time = deployment_results.get("total_time", 0)
            logger.info(
                f"Ray Model Router deployment summary: "
                f"{len(successful_models)} deployed, {len(failed_models)} failed "
                f"in {total_time:.2f}s"
            )
            
        except Exception as e:
            logger.error(f"Error logging deployment details: {e}")
