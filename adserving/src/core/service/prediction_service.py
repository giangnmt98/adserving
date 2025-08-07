"""
Prediction Service

This module handles model predictions with performance tracking.
"""

import asyncio
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from adserving.src.core.utils.model_info import ModelInfo
from adserving.src.datahandler.models import DetailedPredictionResult
from adserving.src.utils.logger import get_logger


class PredictionService:
    """Handles model predictions with batch processing and performance tracking"""

    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.logger = get_logger()

        # Batch processing
        self.batch_queues: Dict[str, List[Tuple[pd.DataFrame, asyncio.Future]]] = (
            defaultdict(list)
        )
        self.batch_timers: Dict[str, threading.Timer] = {}
        self.batch_size_threshold = 10
        self.batch_timeout = 0.1  # 100ms

    async def predict(
        self,
        model_name: str,
        input_data: pd.DataFrame,
    ) -> Optional[Any]:
        """Prediction with batch processing and performance tracking"""
        start_time = time.time()

        try:
            # Get model info from cache AND check if current
            model_info = self.model_manager.cache.get(model_name)
            if not model_info or not self.model_manager._is_model_current(model_info):
                # Load/reload model if not in cache or outdated
                model_info = await self.model_manager.model_loader.load_model_async(
                    model_name
                )
                if not model_info:
                    processing_time = time.time() - start_time
                    return DetailedPredictionResult(
                        model_name=model_name,
                        ma_tieu_chi=input_data["ma_tieu_chi"][0],
                        fld_code=input_data["fld_code"][0],
                        is_anomaly=False,
                        anomaly_score=None,
                        anomaly_threshold=None,
                        processing_time=processing_time,
                        model_version=None,
                        error_message="Model not found or failed to load",
                        status="error",
                    )

            # Make prediction
            try:
                # Get anomaly score from model prediction
                anomaly_score_array = model_info.model.predict(
                    input_data[["fld_code", "gia_tri"]]
                )
                anomaly_score = (
                    float(anomaly_score_array[0])
                    if anomaly_score_array is not None
                    else None
                )

                # Calculate processing time
                processing_time = time.time() - start_time

                # Update model performance metrics
                model_info.update_performance(processing_time, True)

                # Determine if anomaly based on a threshold
                is_anomaly = (
                    anomaly_score is not None
                    and anomaly_score > model_info.anomaly_threshold
                )
                return DetailedPredictionResult(
                    model_name=model_name,
                    ma_tieu_chi=input_data["ma_tieu_chi"][0],
                    fld_code=input_data["fld_code"][0],
                    is_anomaly=is_anomaly,
                    anomaly_score=anomaly_score,
                    anomaly_threshold=model_info.anomaly_threshold,
                    processing_time=processing_time,
                    model_version=getattr(model_info, "model_version", None),
                    error_message=None,
                    status="success",
                )

            except Exception as e:
                # Log prediction error and update error count
                processing_time = time.time() - start_time
                model_info.update_performance(processing_time, False)
                self.logger.error(f"Prediction error for {model_name}: {e}")
                return DetailedPredictionResult(
                    model_name=model_name,
                    ma_tieu_chi=input_data["ma_tieu_chi"][0],
                    fld_code=input_data["fld_code"][0],
                    is_anomaly=False,
                    anomaly_score=None,
                    anomaly_threshold=getattr(model_info, "anomaly_threshold", None),
                    processing_time=processing_time,
                    model_version=getattr(model_info, "version", None),
                    error_message=f"Prediction failed: {str(e)}",
                    status="error",
                )

        except Exception as e:
            # Log any other errors
            processing_time = time.time() - start_time
            self.logger.error(f"Error in predict_detailed for {model_name}: {e}")

            return DetailedPredictionResult(
                model_name=model_name,
                ma_tieu_chi=input_data["ma_tieu_chi"][0],
                fld_code=input_data["fld_code"][0],
                is_anomaly=False,
                anomaly_score=None,
                anomaly_threshold=None,
                processing_time=processing_time,
                model_version=None,
                error_message=f"Service error: {str(e)}",
                status="error",
            )

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
