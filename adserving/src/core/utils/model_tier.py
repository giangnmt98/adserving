"""
Model Tier Enum

This module defines the ModelTier enum for tiered loading strategy.
"""

from enum import Enum


class ModelTier(Enum):
    """Model tier classification for tiered loading strategy"""

    HOT = "hot"  # Always loaded, high replicas
    WARM = "warm"  # Loaded on-demand, medium replicas
    COLD = "cold"  # Load when requested, minimal replicas
