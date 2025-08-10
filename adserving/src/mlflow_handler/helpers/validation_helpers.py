from typing import Any, Dict


def validate_parameter_update(
    client, model_name: str, parameter_updates: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate parameter updates before applying."""
    validation_result = {"valid": True, "errors": [], "warnings": []}

    try:
        if not _validate_model_exists(client, model_name, validation_result):
            return validation_result

        if not _validate_production_version_exists(
            client, model_name, validation_result
        ):
            return validation_result

        _validate_parameter_values(parameter_updates, validation_result)

        return validation_result

    except Exception:
        return validation_result


def _validate_model_exists(
    client, model_name: str, validation_result: Dict[str, Any]
) -> bool:
    try:
        client.get_registered_model(model_name)
        return True
    except Exception:
        validation_result["valid"] = False
        validation_result["errors"].append(f"Model {model_name} not found")
        return False


def _validate_production_version_exists(
    client, model_name: str, validation_result: Dict[str, Any]
) -> bool:
    production_versions = client.get_latest_versions(
        name=model_name, stages=["Production"]
    )
    if not production_versions:
        validation_result["valid"] = False
        validation_result["errors"].append(f"No Production version for {model_name}")
        return False
    return True


def _validate_parameter_values(
    parameter_updates: Dict[str, Any], validation_result: Dict[str, Any]
) -> None:
    for param_key, param_value in parameter_updates.items():
        if param_key == "anomaly_threshold":
            _validate_anomaly_threshold(param_value, validation_result)
        elif param_key == "contamination":
            _validate_contamination(param_value, validation_result)


def _validate_anomaly_threshold(
    param_value: Any, validation_result: Dict[str, Any]
) -> None:
    if not isinstance(param_value, (int, float)):
        validation_result["valid"] = False
        validation_result["errors"].append(
            f"anomaly_threshold must be numeric, got {type(param_value).__name__}"
        )
    elif 0.0 <= float(param_value) <= 1.0:
        validation_result["warnings"].append(
            f"anomaly_threshold {param_value} outside range [0.0, 1.0]"
        )


def _validate_contamination(
    param_value: Any, validation_result: Dict[str, Any]
) -> None:
    if not isinstance(param_value, (int, float)):
        validation_result["valid"] = False
        validation_result["errors"].append(
            f"contamination must be numeric, got {type(param_value).__name__}"
        )
    elif not 0.0 < float(param_value) < 0.5:
        validation_result["warnings"].append(
            f"contamination {param_value} outside range (0.0, 0.5)"
        )
