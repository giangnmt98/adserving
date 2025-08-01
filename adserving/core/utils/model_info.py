"""
Model Info Dataclass

This module defines the ModelInfo dataclass for storing model information
with tier and usage statistics.
"""

import time
from dataclasses import dataclass
from typing import Any

from .model_tier import ModelTier


@dataclass
class ModelInfo:
    """Model information with tier and usage statistics"""

    model_name: str
    model_version: str
    model_uri: str
    model: Any
    loaded_at: float
    last_accessed: float
    access_count: int
    tier: ModelTier
    memory_usage: float
    avg_inference_time: float
    error_count: int
    success_count: int

    def update_access(self):
        """Update access statistics"""
        self.last_accessed = time.time()
        self.access_count += 1

    def update_performance(self, inference_time: float, success: bool):
        """Update performance statistics"""
        if success:
            self.success_count += 1
            # Update rolling average
            total_time = (
                self.avg_inference_time * (self.success_count - 1) + inference_time
            )
            self.avg_inference_time = total_time / self.success_count
        else:
            self.error_count += 1
