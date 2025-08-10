# python
import json
import time
from typing import Any, Dict, Optional

import mlflow


def create_model_placeholder(model_name: str, current_version: str, logger) -> None:
    """Create minimal model placeholder for versioning."""
    try:
        mlflow.log_text(
            f"Parameter update for {model_name} v{current_version}",
            "parameter_update.txt",
        )
        mlflow.set_tag("model_placeholder", "true")
        mlflow.set_tag("original_model_name", model_name)
        mlflow.set_tag("original_model_version", current_version)
        logger.debug(f"Created placeholder for {model_name} v{current_version}")
    except Exception as e:
        logger.error(f"Error creating placeholder: {e}")


def create_run_with_updated_parameters(
    client,
    logger,
    base_run_id: str,
    parameter_updates: Dict[str, Any],
    comment: Optional[str],
    model_name: str,
    current_version: str,
) -> Optional[str]:
    """Create new MLflow run with updated parameters."""
    try:
        base_run = client.get_run(base_run_id)

        with mlflow.start_run(experiment_id=base_run.info.experiment_id) as new_run:
            _log_updated_parameters(parameter_updates)
            _copy_existing_parameters(base_run, parameter_updates, logger)
            _copy_metrics(base_run, logger)
            _copy_tags(base_run, logger)
            _add_update_metadata(base_run_id, parameter_updates, comment)

            create_model_placeholder(model_name, current_version, logger)

            logger.info(f"Created run {new_run.info.run_id}")
            return new_run.info.run_id

    except Exception as e:
        logger.error(f"Error creating run: {e}")
        return None


def _log_updated_parameters(parameter_updates: Dict[str, Any]) -> None:
    """Log updated parameters."""
    for param_key, param_value in parameter_updates.items():
        mlflow.log_param(param_key, param_value)


def _copy_existing_parameters(
    base_run, parameter_updates: Dict[str, Any], logger
) -> None:
    """Copy existing parameters if not overridden."""
    if hasattr(base_run.data, "params") and base_run.data.params:
        for param_key, param_value in base_run.data.params.items():
            if param_key not in parameter_updates:
                try:
                    mlflow.log_param(param_key, param_value)
                except Exception as e:
                    logger.debug(f"Skipped param {param_key}: {e}")


def _copy_metrics(base_run, logger) -> None:
    """Copy metrics from base run."""
    if hasattr(base_run.data, "metrics") and base_run.data.metrics:
        for metric_key, metric_value in base_run.data.metrics.items():
            try:
                mlflow.log_metric(metric_key, metric_value)
            except Exception as e:
                logger.debug(f"Skipped metric {metric_key}: {e}")


def _copy_tags(base_run, logger) -> None:
    """Copy non-system tags from base run."""
    if hasattr(base_run.data, "tags") and base_run.data.tags:
        for tag_key, tag_value in base_run.data.tags.items():
            if not tag_key.startswith("mlflow."):
                try:
                    mlflow.set_tag(tag_key, tag_value)
                except Exception as e:
                    logger.debug(f"Skipped tag {tag_key}: {e}")


def _add_update_metadata(
    base_run_id: str, parameter_updates: Dict[str, Any], comment: Optional[str]
) -> None:
    """Add metadata about the parameter update."""
    mlflow.set_tag("parameter_update_timestamp", str(int(time.time())))
    mlflow.set_tag("parameter_update_comment", comment or "Parameter update")
    mlflow.set_tag("original_run_id", base_run_id)
    mlflow.set_tag("updated_parameters", json.dumps(parameter_updates))
    mlflow.set_tag("parameter_update", "true")
