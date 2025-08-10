# python
"""MLflow Parameter Updater - Zero-downtime parameter updates for ML models."""

import logging
from typing import Any, Dict, List, Optional

import mlflow

from .helpers.history_helpers import (compare_parameter_versions,
                                      get_model_version_parameters,
                                      get_parameter_update_history)
from .helpers.model_version_helpers import (
    create_model_version_from_run, transition_to_production_with_archive)
from .helpers.run_helpers import create_run_with_updated_parameters
from .helpers.validation_helpers import validate_parameter_update


class MLflowParameterUpdater:
    """Handles zero-downtime ML model parameter updates via MLflow."""

    def __init__(self, mlflow_client) -> None:
        """Initialize with existing MLflow client."""
        self.mlflow_client = mlflow_client
        self.client = mlflow_client.client
        self.tracking_uri = mlflow_client.tracking_uri
        mlflow.set_tracking_uri(self.tracking_uri)
        self.logger = logging.getLogger(__name__)

    def create_parameter_version(
        self,
        model_name: str,
        parameter_updates: Dict[str, Any],
        comment: Optional[str] = None,
    ) -> bool:
        """Create new model version with updated parameters."""
        try:
            # Get current Production version
            production_versions = self.client.get_latest_versions(
                name=model_name, stages=["Production"]
            )

            if not production_versions:
                self.logger.error(f"No Production version found: {model_name}")
                return False

            current_version = production_versions[0]
            current_run_id = current_version.run_id
            current_source = current_version.source

            self.logger.info(
                f"Found Production v{current_version.version} "
                f"for {model_name} (run: {current_run_id})"
            )

            # Create new run with updated parameters
            new_run_id = create_run_with_updated_parameters(
                client=self.client,
                logger=self.logger,
                base_run_id=current_run_id,
                parameter_updates=parameter_updates,
                comment=comment or "Parameter update",
                model_name=model_name,
                current_version=str(current_version.version),
            )

            if not new_run_id:
                return False

            # Create model version from new run (using original source)
            new_version = create_model_version_from_run(
                client=self.client,
                logger=self.logger,
                model_name=model_name,
                run_id=new_run_id,
                original_source=current_source,
                comment=comment or "Updated parameters",
            )

            if not new_version:
                return False

            # Transition new version to Production
            success = transition_to_production_with_archive(
                client=self.client,
                logger=self.logger,
                model_name=model_name,
                new_version=new_version.version,
            )

            if success:
                self.logger.debug(
                    f"Successfully deployed v{new_version.version} "
                    f"for {model_name} with updates: {parameter_updates}"
                )

            return success

        except Exception as e:
            self.logger.error(f"Error creating parameter version: {e}")
            return False

    def update_anomaly_threshold(
        self, model_name: str, new_threshold: float, comment: Optional[str] = None
    ) -> bool:
        """Update anomaly_threshold parameter."""
        return self.create_parameter_version(
            model_name=model_name,
            parameter_updates={"anomaly_threshold": new_threshold},
            comment=comment or f"Updated anomaly_threshold to {new_threshold}",
        )

    def validate_parameter_update(
        self, model_name: str, parameter_updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate parameter updates before applying."""
        return validate_parameter_update(
            client=self.client,
            model_name=model_name,
            parameter_updates=parameter_updates,
        )

    def get_current_parameters(self, model_name: str) -> Dict[str, Any]:
        """Get current parameters from Production version."""
        return self.mlflow_client.get_model_parameters_by_stage(
            model_name, "Production"
        )

    def rollback_to_version(self, model_name: str, target_version: str) -> bool:
        """Rollback to specific model version."""
        try:
            # Validate target version exists
            target_model_version = self.client.get_model_version(
                model_name, target_version
            )

            if target_model_version.current_stage == "Production":
                self.logger.info(f"v{target_version} already in Production")
                return True

            # Transition to Production
            self.client.transition_model_version_stage(
                name=model_name,
                version=target_version,
                stage="Production",
                archive_existing_versions=True,
            )

            self.logger.info(f"Rolled back {model_name} to v{target_version}")
            return True

        except Exception as e:
            self.logger.error(f"Error rolling back to v{target_version}: {e}")
            return False

    def get_parameter_update_history(
        self, model_name: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get parameter update history for a model."""
        return get_parameter_update_history(
            client=self.client, logger=self.logger, model_name=model_name, limit=limit
        )

    def get_model_version_parameters(
        self, model_name: str, version: str
    ) -> Dict[str, Any]:
        """Get parameters for a specific model version."""
        return get_model_version_parameters(
            client=self.client,
            logger=self.logger,
            model_name=model_name,
            version=version,
        )

    def compare_parameter_versions(
        self, model_name: str, version1: str, version2: str
    ) -> Dict[str, Any]:
        """Compare parameters between two model versions."""
        return compare_parameter_versions(
            client=self.client,
            logger=self.logger,
            model_name=model_name,
            version1=version1,
            version2=version2,
        )
