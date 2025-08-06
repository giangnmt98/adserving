"""MLflow Parameter Updater - Zero-downtime parameter updates for ML models."""

import json
import logging
import time
from typing import Any, Dict, Optional, List

import mlflow
from mlflow.tracking import MlflowClient as MLflowTrackingClient


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
            new_run_id = self._create_run_with_updated_parameters(
                current_run_id, parameter_updates, comment, model_name,
                current_version.version
            )

            if not new_run_id:
                return False

            # Create model version from new run (using original source)
            new_version = self._create_model_version_from_run(
                model_name, new_run_id, current_source, comment
            )

            if not new_version:
                return False

            # Transition new version to Production
            success = self._transition_to_production_with_archive(
                model_name, new_version.version
            )

            if success:
                self.logger.info(
                    f"Successfully deployed v{new_version.version} "
                    f"for {model_name} with updates: {parameter_updates}"
                )

            return success

        except Exception as e:
            self.logger.error(f"Error creating parameter version: {e}")
            return False

    def _create_run_with_updated_parameters(
        self,
        base_run_id: str,
        parameter_updates: Dict[str, Any],
        comment: str,
        model_name: str,
        current_version: str,
    ) -> Optional[str]:
        """Create new MLflow run with updated parameters."""
        try:
            base_run = self.client.get_run(base_run_id)

            with mlflow.start_run(
                experiment_id=base_run.info.experiment_id
            ) as new_run:
                # Log updated parameters FIRST
                for param_key, param_value in parameter_updates.items():
                    mlflow.log_param(param_key, param_value)

                # Copy existing parameters (if not overridden)
                if hasattr(base_run.data, 'params') and base_run.data.params:
                    for param_key, param_value in base_run.data.params.items():
                        if param_key not in parameter_updates:
                            try:
                                mlflow.log_param(param_key, param_value)
                            except Exception as e:
                                self.logger.debug(f"Skipped param {param_key}: {e}")

                # Copy metrics
                if hasattr(base_run.data, 'metrics') and base_run.data.metrics:
                    for metric_key, metric_value in base_run.data.metrics.items():
                        try:
                            mlflow.log_metric(metric_key, metric_value)
                        except Exception as e:
                            self.logger.debug(f"Skipped metric {metric_key}: {e}")

                # Copy tags (exclude system tags)
                if hasattr(base_run.data, 'tags') and base_run.data.tags:
                    for tag_key, tag_value in base_run.data.tags.items():
                        if not tag_key.startswith('mlflow.'):
                            try:
                                mlflow.set_tag(tag_key, tag_value)
                            except Exception as e:
                                self.logger.debug(f"Skipped tag {tag_key}: {e}")

                # Add update metadata
                mlflow.set_tag("parameter_update_timestamp", str(int(time.time())))
                mlflow.set_tag("parameter_update_comment", comment or "Parameter update")
                mlflow.set_tag("original_run_id", base_run_id)
                mlflow.set_tag("updated_parameters", json.dumps(parameter_updates))
                mlflow.set_tag("parameter_update", "true")

                # Create placeholder artifact
                self._create_model_placeholder(model_name, current_version)

                self.logger.info(f"Created run {new_run.info.run_id}")
                return new_run.info.run_id

        except Exception as e:
            self.logger.error(f"Error creating run: {e}")
            return None

    def _create_model_placeholder(self, model_name: str, current_version: str) -> None:
        """Create minimal model placeholder for versioning."""
        try:
            # Create minimal text artifact
            mlflow.log_text(
                f"Parameter update for {model_name} v{current_version}",
                "parameter_update.txt"
            )

            # Add metadata tags
            mlflow.set_tag("model_placeholder", "true")
            mlflow.set_tag("original_model_name", model_name)
            mlflow.set_tag("original_model_version", current_version)

            self.logger.debug(f"Created placeholder for {model_name} v{current_version}")

        except Exception as e:
            self.logger.error(f"Error creating placeholder: {e}")

    def _create_model_version_from_run(
        self,
        model_name: str,
        run_id: str,
        original_source: str,
        comment: str,
    ) -> Optional[Any]:
        """Create model version using original source but new run."""
        try:
            new_version = self.client.create_model_version(
                name=model_name,
                source=original_source,  # Original model artifacts
                run_id=run_id,  # New run with updated parameters
                description=f"Parameter update: {comment or 'Updated parameters'}"
            )

            self.logger.info(
                f"Created v{new_version.version} from run {run_id} "
                f"using source {original_source}"
            )
            return new_version

        except Exception as e:
            self.logger.error(f"Error creating model version: {e}")
            return None

    def _transition_to_production_with_archive(
        self, model_name: str, new_version: str
    ) -> bool:
        """Transition new version to Production."""
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=new_version,
                stage="Production",
                archive_existing_versions=True,
            )

            self.logger.info(f"Transitioned {model_name} v{new_version} to Production")
            return True

        except Exception as e:
            self.logger.error(f"Error transitioning to Production: {e}")
            return False

    def update_anomaly_threshold(
        self, model_name: str, new_threshold: float, comment: Optional[str] = None
    ) -> bool:
        """Update anomaly_threshold parameter."""
        return self.create_parameter_version(
            model_name=model_name,
            parameter_updates={"anomaly_threshold": new_threshold},
            comment=comment or f"Updated anomaly_threshold to {new_threshold}"
        )

    def validate_parameter_update(
        self, model_name: str, parameter_updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate parameter updates before applying."""
        validation_result = {"valid": True, "errors": [], "warnings": []}

        try:
            # Check if model exists
            try:
                self.client.get_registered_model(model_name)
            except Exception:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Model {model_name} not found")
                return validation_result

            # Check Production version exists
            production_versions = self.client.get_latest_versions(
                name=model_name, stages=["Production"]
            )

            if not production_versions:
                validation_result["valid"] = False
                validation_result["errors"].append(
                    f"No Production version for {model_name}"
                )
                return validation_result

            # Validate parameter values
            for param_key, param_value in parameter_updates.items():
                if param_key == "anomaly_threshold":
                    if not isinstance(param_value, (int, float)):
                        validation_result["valid"] = False
                        validation_result["errors"].append(
                            f"anomaly_threshold must be numeric, "
                            f"got {type(param_value).__name__}"
                        )
                    elif not (0.0 <= param_value <= 1.0):
                        validation_result["warnings"].append(
                            f"anomaly_threshold {param_value} "
                            f"outside range [0.0, 1.0]"
                        )

                elif param_key == "contamination":
                    if not isinstance(param_value, (int, float)):
                        validation_result["valid"] = False
                        validation_result["errors"].append(
                            f"contamination must be numeric, "
                            f"got {type(param_value).__name__}"
                        )
                    elif not (0.0 < param_value < 0.5):
                        validation_result["warnings"].append(
                            f"contamination {param_value} "
                            f"outside range (0.0, 0.5)"
                        )

            return validation_result

        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Validation error: {e}")
            return validation_result

    def get_current_parameters(self, model_name: str) -> Dict[str, Any]:
        """Get current parameters from Production version."""
        return self.mlflow_client.get_model_parameters_by_stage(
            model_name, "Production"
        )

    def rollback_to_version(
        self, model_name: str, target_version: str
    ) -> bool:
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
        """
        Get parameter update history for a model.

        Args:
            model_name: Name of the registered model
            limit: Maximum number of updates to return

        Returns:
            List of parameter update records with timestamps and changes
        """
        try:
            # Get all versions for the model
            model_versions = self.client.search_model_versions(
                f"name='{model_name}'", max_results=limit * 2
            )

            if not model_versions:
                self.logger.info(f"No versions found for model {model_name}")
                return []

            parameter_updates = []

            for version in sorted(model_versions, key=lambda x: int(x.version), reverse=True):
                try:
                    # Get run details to check for parameter updates
                    run = self.client.get_run(version.run_id)

                    # Check if this is a parameter update (has our special tag)
                    tags = run.data.tags if hasattr(run.data, 'tags') else {}

                    if tags.get("parameter_update") == "true":
                        # Extract update information
                        update_record = {
                            "version": version.version,
                            "created_timestamp": version.creation_timestamp,
                            "last_updated_timestamp": version.last_updated_timestamp,
                            "status": version.status,
                            "stage": version.current_stage,
                            "run_id": version.run_id,
                            "comment": tags.get("parameter_update_comment", "No comment"),
                            "update_timestamp": tags.get("parameter_update_timestamp"),
                            "original_run_id": tags.get("original_run_id"),
                        }

                        # Extract updated parameters
                        updated_params_str = tags.get("updated_parameters", "{}")
                        try:
                            updated_params = json.loads(updated_params_str)
                            update_record["updated_parameters"] = updated_params
                        except json.JSONDecodeError:
                            update_record["updated_parameters"] = {}

                        # Add current parameters from run
                        if hasattr(run.data, 'params') and run.data.params:
                            update_record["current_parameters"] = dict(run.data.params)
                        else:
                            update_record["current_parameters"] = {}

                        parameter_updates.append(update_record)

                        # Stop if we have enough updates
                        if len(parameter_updates) >= limit:
                            break

                except Exception as e:
                    self.logger.debug(f"Error processing version {version.version}: {e}")
                    continue

            self.logger.info(f"Found {len(parameter_updates)} parameter updates for {model_name}")
            return parameter_updates

        except Exception as e:
            self.logger.error(f"Error getting parameter history for {model_name}: {e}")
            return []

    def get_model_version_parameters(
            self, model_name: str, version: str
    ) -> Dict[str, Any]:
        """
        Get parameters for a specific model version.

        Args:
            model_name: Name of the registered model
            version: Version number

        Returns:
            Dictionary of parameters for the version
        """
        try:
            model_version = self.client.get_model_version(model_name, version)
            run = self.client.get_run(model_version.run_id)

            if hasattr(run.data, 'params') and run.data.params:
                return dict(run.data.params)
            else:
                return {}

        except Exception as e:
            self.logger.error(f"Error getting parameters for {model_name} v{version}: {e}")
            return {}

    def compare_parameter_versions(
            self, model_name: str, version1: str, version2: str
    ) -> Dict[str, Any]:
        """
        Compare parameters between two model versions.

        Args:
            model_name: Name of the registered model
            version1: First version to compare
            version2: Second version to compare

        Returns:
            Dictionary showing parameter differences
        """
        try:
            params1 = self.get_model_version_parameters(model_name, version1)
            params2 = self.get_model_version_parameters(model_name, version2)

            # Find differences
            all_keys = set(params1.keys()) | set(params2.keys())
            differences = {}
            unchanged = {}

            for key in all_keys:
                val1 = params1.get(key, "<not set>")
                val2 = params2.get(key, "<not set>")

                if val1 != val2:
                    differences[key] = {
                        f"version_{version1}": val1,
                        f"version_{version2}": val2
                    }
                else:
                    unchanged[key] = val1

            return {
                "model_name": model_name,
                "version1": version1,
                "version2": version2,
                "differences": differences,
                "unchanged": unchanged,
                "total_params": len(all_keys),
                "changed_params": len(differences),
                "unchanged_params": len(unchanged)
            }

        except Exception as e:
            self.logger.error(f"Error comparing versions {version1} and {version2}: {e}")
            return {
                "model_name": model_name,
                "version1": version1,
                "version2": version2,
                "error": str(e),
                "differences": {},
                "unchanged": {},
                "total_params": 0,
                "changed_params": 0,
                "unchanged_params": 0
            }
