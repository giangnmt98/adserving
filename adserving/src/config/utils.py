from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Type

import yaml


def load_data_from_file(config_path: Path) -> Dict[str, Any]:
    """Load configuration data from a YAML or JSON file.

    Parameters
    ----------
    config_path: Path
        Path to the configuration file. Supported suffixes: .yaml, .yml, .json

    Returns
    -------
    Dict[str, Any]
        Parsed configuration as a dictionary.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        suffix = config_path.suffix.lower()
        if suffix in [".yaml", ".yml"]:
            return yaml.safe_load(f)
        if suffix == ".json":
            return json.load(f)
        raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")


def wrap_dataclass(data: Dict[str, Any], cls: Type[Any]) -> Any:
    """Safely construct a dataclass/config object from a dict.

    Falls back to default construction if the dict is invalid for the target class.
    """
    try:
        return cls(**(data or {}))
    except Exception:
        return cls()


def process_config_section(
    key: str, value: Dict[str, Any], config_map: Mapping[str, Type[Any]]
) -> Any:
    """Return an instantiated config section for a given key using config_map.

    If the key isn't in the map, returns None.
    """
    if key in config_map:
        return wrap_dataclass(value, config_map[key])
    return None
