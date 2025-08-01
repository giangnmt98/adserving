"""
Prediction Service

This module handles model predictions with performance tracking.
"""

import asyncio
import logging
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from adserving.core.utils.model_info import ModelInfo


class PredictionService:
    """Handles model predictions with batch processing and performance tracking"""

    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.logger = logging.getLogger(__name__)

        # Batch processing
        self.batch_queues: Dict[str, List[Tuple[pd.DataFrame, asyncio.Future]]] = (
            defaultdict(list)
        )
        self.batch_timers: Dict[str, threading.Timer] = {}
        self.batch_size_threshold = 10
        self.batch_timeout = 0.1  # 100ms

    async def predict(self, model_name: str, input_data: pd.DataFrame) -> Optional[Any]:
        """Prediction with batch processing and performance tracking"""
        try:
            # Load model if not cached
            model_info = self.model_manager.cache.get(model_name)
            if not model_info:
                model_info = await self.model_manager.model_loader.load_model_async(
                    model_name
                )
                if not model_info:
                    return None

            # Record prediction attempt
            start_time = time.time()

            try:
                # Perform prediction
                result = model_info.model.predict(input_data)

                # Update performance metrics
                inference_time = time.time() - start_time
                model_info.update_performance(inference_time, True)

                return result

            except Exception as e:
                # Update error metrics
                model_info.update_performance(0, False)
                self.logger.error(f"Prediction error for {model_name}: {e}")
                raise

        except Exception as e:
            self.logger.error(f"Error in predict for {model_name}: {e}")
            return None

    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get model information"""
        return self.model_manager.cache.get(model_name)

    def cleanup_batch_processing(self):
        """Cleanup batch processing resources"""
        # Clear batch queues
        if hasattr(self, "batch_queues"):
            self.batch_queues.clear()

        # Cancel any pending batch timers
        if hasattr(self, "batch_timers"):
            for timer in self.batch_timers.values():
                try:
                    timer.cancel()
                except Exception:
                    pass
            self.batch_timers.clear()
