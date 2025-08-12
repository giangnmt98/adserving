"""
Main configuration module - Backward compatibility wrapper

This file maintains backward compatibility by importing all configuration
classes and functions from the modular structure.
"""

# Import everything from the new modular structure
from . import *

# Maintain backward compatibility
from .config_manager import (
    Config,
    _global_config,
    create_sample_config,
    get_config,
    load_config,
    set_config,
)

# Module docstring for documentation
__doc__ = """
Configuration for Anomaly Detection Serve
"""
