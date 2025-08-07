"""
Production Model Manager

This module handles production model lifecycle management.
"""

import asyncio
from typing import Dict, List

import pandas as pd
from adserving.src.utils.logger import get_logger


class ProductionModelManager:
    """Manages production model lifecycle and updates"""

    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.logger = get_logger()

        # Zero-downtime deployment tracking
        # Track production model versions
        self._production_model_versions: Dict[str, str] = {}

    async def check_and_update_production_models(self):
        """Check for production model changes
        and update models for zero-downtime deployment"""
        try:
            current_production_models = await self._get_current_production_models()

            self.logger.debug(
                f"Found {len(current_production_models)} production models "
                f"in MLflow"
            )

            changes = self._detect_model_changes(current_production_models)
            new_models, updated_models, removed_models = changes

            await self._handle_new_models(new_models)
            await self._handle_updated_models(updated_models)
            await self._handle_removed_models(removed_models)

            if not any(changes):
                self.logger.debug(
                    "No production model changes detected - all models are "
                    "up to date"
                )

        except Exception as e:
            self.logger.error(f"Error checking production model changes: {e}")

    async def _get_current_production_models(self):
        return await asyncio.get_event_loop().run_in_executor(
            self.model_manager.executor,
            self.model_manager.mlflow_client.get_production_models_with_versions,
        )

    def _detect_model_changes(self, current_production_models):
        new_models = []
        updated_models = []
        removed_models = []

        for model_name, version in current_production_models.items():
            if model_name not in self._production_model_versions:
                new_models.append(model_name)
                self._production_model_versions[model_name] = version
            elif self._production_model_versions[model_name] != version:
                updated_models.append(
                    (
                        model_name,
                        self._production_model_versions[model_name],
                        version,
                    )
                )
                self._production_model_versions[model_name] = version

        for model_name in list(self._production_model_versions.keys()):
            if model_name not in current_production_models:
                removed_models.append(model_name)
                del self._production_model_versions[model_name]

        return new_models, updated_models, removed_models

    async def _handle_new_models(self, new_models):
        if new_models:
            self.logger.info(
                f"Detected {len(new_models)} new production models"
                f" - queueing for batched deployment"
            )
            await self.model_manager.deployment_manager.queue_batched_deployments(
                new_models,
                "new",
                fallback_deploy_func=self._deploy_new_production_model,
            )

    async def _handle_updated_models(self, updated_models):
        if updated_models:
            model_names = [name for name, _, _ in updated_models]
            self.logger.info(
                f"Detected {len(model_names)} updated production models"
                f" - queueing for batched deployment"
            )
            await self.model_manager.deployment_manager.queue_batched_deployments(
                model_names,
                "update",
                fallback_update_func=self._update_production_model,
            )

    async def _handle_removed_models(self, removed_models):
        if removed_models:
            self.logger.info(
                f"Detected {len(removed_models)} removed production "
                f"models: {removed_models}"
            )
            for model_name in removed_models:
                try:
                    self.logger.info(f"Removing production model: {model_name}")
                    await self._remove_production_model(model_name)
                    self.logger.info(
                        f"Successfully removed production model: {model_name}"
                    )
                except Exception as e:
                    self.logger.error(
                        f"Failed to remove production model {model_name}: {e}"
                    )

    async def _deploy_new_production_model(self, model_name: str):
        """Deploy a newly promoted production model"""
        self.logger.debug(f"Deploying new production model: {model_name}")

        # Load the model asynchronously
        model_info = await self.model_manager.model_loader.load_model_async(model_name)
        if model_info:
            self.logger.debug(
                f"Successfully deployed new production model: {model_name} "
                f"v{model_info.model_version}"
            )
        else:
            self.logger.error(f"Failed to load new production model: {model_name}")

    async def _update_production_model(
        self, model_name: str, old_version: str, new_version: str
    ):
        """Update an existing production model to a new version with zero-downtime"""
        self.logger.info(
            f"Starting zero-downtime update for model {model_name} "
            f"from v{old_version} to v{new_version}"
        )

        # Load the new version
        self.logger.info(f"Loading new model version {new_version} for {model_name}")
        new_model_info = await self.model_manager.model_loader.load_model_async(
            model_name
        )

        if new_model_info and new_model_info.model_version == new_version:
            # The load_model_async will automatically update the cache with
            # the new version due to the _is_model_current check
            self.logger.info(
                f"Model cache updated with new version: {model_name} " f"v{new_version}"
            )

            # Optional: Trigger a prediction with dummy data to warm up the new model
            try:
                self.logger.info(
                    f" Warming up new model version: {model_name} " f"v{new_version}"
                )
                # Create minimal dummy data for warming
                dummy_data = pd.DataFrame({"dummy": [1]})
                await self.model_manager.predict(model_name, dummy_data)
                self.logger.info(
                    f"Model warming completed: {model_name} v{new_version}"
                )
            except Exception as e:
                # Warming failure is not critical
                self.logger.warning(
                    f" Model warming failed for {model_name} " f"(non-critical): {e}"
                )

            self.logger.info(
                f"Zero-downtime model update completed: {model_name} is now "
                f"serving v{new_version}"
            )
        else:
            self.logger.error(
                f"Failed to update production model {model_name} to " f"v{new_version}"
            )

    async def _remove_production_model(self, model_name: str):
        """Handle removal of a model from production stage"""
        self.logger.info(f"Removing model from production tracking: {model_name}")

        # Note: We don't immediately remove from cache as it might still be
        # serving requests. The cleanup process will handle removal of unused
        # models. Just log the change for monitoring purposes
        self.logger.info(f"Model {model_name} is no longer in production stage")

    def get_production_models(self) -> List[str]:
        """Get list of production models from MLflow"""
        return self.model_manager.mlflow_client.get_production_models()
