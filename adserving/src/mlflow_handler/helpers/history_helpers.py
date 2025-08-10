# python
import json
from typing import Any, Dict, List


def get_parameter_update_history(
    client, logger, model_name: str, limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Get parameter update history for a model.
    Returns:
        List of parameter update records with timestamps and changes
    """
    try:
        # Get all versions for the model
        model_versions = client.search_model_versions(
            f"name='{model_name}'", max_results=limit * 2
        )

        if not model_versions:
            logger.info(f"No versions found for model {model_name}")
            return []

        parameter_updates: List[Dict[str, Any]] = []

        for version in sorted(
            model_versions, key=lambda x: int(x.version), reverse=True
        ):
            try:
                # Get run details to check for parameter updates
                run = client.get_run(version.run_id)

                # Check if this is a parameter update (has our special tag)
                tags = run.data.tags if hasattr(run.data, "tags") else {}

                if tags.get("parameter_update") == "true":
                    # Extract update information
                    update_record: Dict[str, Any] = {
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
                    if hasattr(run.data, "params") and run.data.params:
                        update_record["current_parameters"] = dict(run.data.params)
                    else:
                        update_record["current_parameters"] = {}

                    parameter_updates.append(update_record)

                    # Stop if we have enough updates
                    if len(parameter_updates) >= limit:
                        break

            except Exception as e:
                logger.debug(f"Error processing version {version.version}: {e}")
                continue

        logger.info(
            f"Found {len(parameter_updates)} parameter updates for {model_name}"
        )
        return parameter_updates

    except Exception as e:
        logger.error(f"Error getting parameter history for {model_name}: {e}")
        return []


def get_model_version_parameters(
    client, logger, model_name: str, version: str
) -> Dict[str, Any]:
    """Get parameters for a specific model version."""
    try:
        model_version = client.get_model_version(model_name, version)
        run = client.get_run(model_version.run_id)

        if hasattr(run.data, "params") and run.data.params:
            return dict(run.data.params)
        else:
            return {}

    except Exception as e:
        logger.error(f"Error getting parameters for {model_name} v{version}: {e}")
        return {}


def compare_parameter_versions(
    client, logger, model_name: str, version1: str, version2: str
) -> Dict[str, Any]:
    """Compare parameters between two model versions."""
    try:
        params1 = get_model_version_parameters(client, logger, model_name, version1)
        params2 = get_model_version_parameters(client, logger, model_name, version2)

        # Find differences
        all_keys = set(params1.keys()) | set(params2.keys())
        differences: Dict[str, Dict[str, Any]] = {}
        unchanged: Dict[str, Any] = {}

        for key in all_keys:
            val1 = params1.get(key, "<not set>")
            val2 = params2.get(key, "<not set>")

            if val1 != val2:
                differences[key] = {
                    f"version_{version1}": val1,
                    f"version_{version2}": val2,
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
            "unchanged_params": len(unchanged),
        }

    except Exception as e:
        logger.error(f"Error comparing versions {version1} and {version2}: {e}")
        return {
            "model_name": model_name,
            "version1": version1,
            "version2": version2,
            "error": str(e),
            "differences": {},
            "unchanged": {},
            "total_params": 0,
            "changed_params": 0,
            "unchanged_params": 0,
        }
