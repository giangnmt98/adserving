"""
Model deployment handler
Handles production model deployment with parallel loading
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
import ray
from ray import serve

logger = logging.getLogger(__name__)


class ModelDeploymentHandler:
    """Handles model deployment operations"""

    def deploy_models(
        self, model_manager, deployment_manager, model_router
    ) -> Dict[str, int]:
        """Deploy production models with parallel loading."""
        try:
            logger.info("Deploying production models from MLflow...")

            # Get production models
            production_models = model_manager.get_production_models()

            if not production_models:
                logger.warning("No production models found")
                return {"loaded": 0, "failed": 0}

            # Create deployment pools
            self._create_deployment_pools(deployment_manager, model_router)

            # Load models in parallel
            loaded_count, failed_count = self._load_models_parallel(
                production_models, model_manager
            )

            # Warm up deployment pools
            if loaded_count > 0:
                self._warm_up_deployment_pools()

            logger.info(
                f"Models deployed: {loaded_count} loaded, " f"{failed_count} failed"
            )

            return {"loaded": loaded_count, "failed": failed_count}

        except Exception as e:
            logger.error(f"Failed to deploy production models: {e}")
            return {"loaded": 0, "failed": 0}

    def _create_deployment_pools(self, deployment_manager, model_router) -> None:
        """Create default deployment pools."""
        existing_deployments = deployment_manager.get_all_deployment_stats()
        if existing_deployments:
            return

        logger.info("Creating default deployment pools...")
        created_pools = deployment_manager.create_default_pools()

        # Register pools with router
        for pool_name in created_pools:
            model_router.register_deployment(pool_name)

    def _load_models_parallel(
        self, production_models: List[str], model_manager
    ) -> tuple[int, int]:
        """Load models in parallel."""
        loaded_count = 0
        failed_count = 0

        max_concurrent = min(4, len(production_models))

        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            future_to_model = {
                executor.submit(self._load_single_model, model, model_manager): model
                for model in production_models
            }

            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    success = future.result()
                    if success:
                        loaded_count += 1
                        logger.debug(f"Loaded model: {model_name}")
                    else:
                        failed_count += 1
                        logger.warning(f"Failed to load: {model_name}")
                except Exception as e:
                    failed_count += 1
                    logger.warning(f"Error loading {model_name}: {e}")

        return loaded_count, failed_count

    def _load_single_model(self, model_name: str, model_manager) -> bool:
        """Load a single model."""
        try:
            model_info = model_manager._load_model_sync(model_name)
            return model_info is not None
        except Exception:
            return False

    def _warm_up_deployment_pools(self) -> None:
        """Warm up deployment pools to prevent cold starts."""
        try:
            deployments = serve.list_deployments()
            if not deployments:
                return

            # Find suitable deployment for warmup
            deployment_name = self._find_warmup_deployment(deployments)
            if not deployment_name:
                return

            self._perform_warmup_requests(deployment_name)

        except Exception as e:
            logger.warning(f"Pool warmup failed (non-critical): {e}")

    def _find_warmup_deployment(self, deployments: dict) -> Optional[str]:
        """Find a suitable deployment for warmup."""
        for name in deployments:
            if "pooled" in name.lower() or "anomaly" in name.lower():
                return name
        return list(deployments.keys())[0] if deployments else None

    def _perform_warmup_requests(self, deployment_name: str) -> None:
        """Perform warmup requests to a deployment."""
        deployment_handle = serve.get_app_handle(deployment_name)

        warmup_payload = {
            "ma_don_vi": "UBND.0019",
            "ma_bao_cao": "10628953_CT",
            "ky_du_lieu": "2024-01-01",
            "data": [{"ma_tieu_chi": "TONGCONG", "FN01": 1000000}],
        }

        # Make multiple warmup requests
        result_refs = []
        for _ in range(3):
            result_ref = deployment_handle.remote(warmup_payload)
            result_refs.append(result_ref)

        try:
            ray.get(result_refs, timeout=30)
            logger.info("Pool warmup completed successfully")
        except Exception as e:
            logger.warning(f"Warmup requests failed: {e}")
