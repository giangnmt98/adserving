"""
Ray remote actor for model deployment and prediction processing
"""

import asyncio
import time
import ray
import pandas as pd
from typing import Dict, Any, Optional

from adserving.src.utils.logger import get_logger


@ray.remote(num_cpus=0.5)
class RayModelDeployment:
    """Ray remote actor for model deployment and parallel prediction processing"""
    
    def __init__(self, model_name: str, model_info: Any):
        self.model_name = model_name
        self.model_info = model_info
        self.model = None
        self.prediction_count = 0
        self.total_prediction_time = 0.0
        self.logger = get_logger()
        
    async def initialize(self) -> Dict[str, Any]:
        """Initialize the Ray actor with the model"""
        try:
            start_time = time.time()
            self.logger.info(f"Initializing Ray model deployment actor for {self.model_name}")
            
            # The model is already loaded in model_info
            if hasattr(self.model_info, 'model'):
                self.model = self.model_info.model
            else:
                self.model = self.model_info
                
            init_time = time.time() - start_time
            self.logger.info(f"Ray deployment actor initialized for {self.model_name} in {init_time:.2f}s")
            
            return {
                "status": "success",
                "model_name": self.model_name,
                "initialization_time": init_time,
                "actor_id": ray.get_runtime_context().get_actor_id()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Ray deployment actor for {self.model_name}: {e}")
            return {
                "status": "error",
                "model_name": self.model_name,
                "error_message": str(e)
            }
    
    async def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute model prediction using Ray remote processing"""
        prediction_start = time.time()
        
        try:
            self.logger.debug(f"Ray actor predicting for {self.model_name}: {input_data}")
            
            # Convert input data to DataFrame format expected by model
            input_df = pd.DataFrame({
                "ma_tieu_chi": [input_data.get("ma_tieu_chi", "UNKNOWN")],
                "fld_code": [input_data.get("fld_code", "UNKNOWN")],
                "gia_tri": [float(input_data.get("gia_tri", 0.0))]
            })
            
            # Execute model prediction
            if hasattr(self.model, 'predict'):
                prediction_result = self.model.predict(input_df)
            else:
                # Handle case where model_info contains the actual prediction method
                prediction_result = await self._execute_prediction(input_df)
            
            prediction_time = time.time() - prediction_start
            self.prediction_count += 1
            self.total_prediction_time += prediction_time
            
            # Format result based on prediction output
            if hasattr(prediction_result, 'status') and prediction_result.status == "success":
                result = {
                    "status": "success",
                    "ma_don_vi": input_data.get("ma_don_vi", "UNKNOWN"),
                    "ma_bao_cao": input_data.get("ma_bao_cao", "UNKNOWN"),
                    "ky_du_lieu": input_data.get("ky_du_lieu", "UNKNOWN"),
                    "ma_tieu_chi": prediction_result.ma_tieu_chi,
                    "is_anomaly": prediction_result.is_anomaly,
                    "fld_code": prediction_result.fld_code,
                    "prediction": prediction_result.anomaly_score,
                    "anomaly_threshold": prediction_result.anomaly_threshold,
                    "model_name": prediction_result.model_name,
                    "model_version": prediction_result.model_version,
                    "processing_time": prediction_result.processing_time,
                    "ray_actor_prediction": True,
                    "actor_id": ray.get_runtime_context().get_actor_id()
                }
            else:
                result = {
                    "status": "error",
                    "ma_don_vi": input_data.get("ma_don_vi", "UNKNOWN"),
                    "ma_bao_cao": input_data.get("ma_bao_cao", "UNKNOWN"),
                    "ky_du_lieu": input_data.get("ky_du_lieu", "UNKNOWN"),
                    "ma_tieu_chi": input_data.get("ma_tieu_chi", "UNKNOWN"),
                    "fld_code": input_data.get("fld_code", "UNKNOWN"),
                    "model_name": self.model_name,
                    "error_message": getattr(prediction_result, 'error_message', 'Prediction failed'),
                    "is_anomaly": False,
                    "prediction": None,
                    "ray_actor_prediction": True,
                    "actor_id": ray.get_runtime_context().get_actor_id()
                }
            
            self.logger.debug(f"Ray actor prediction completed for {self.model_name} in {prediction_time:.2f}s")
            return result
            
        except Exception as e:
            prediction_time = time.time() - prediction_start
            self.logger.error(f"Ray actor prediction failed for {self.model_name}: {e}")
            
            return {
                "status": "error",
                "ma_don_vi": input_data.get("ma_don_vi", "UNKNOWN"),
                "ma_bao_cao": input_data.get("ma_bao_cao", "UNKNOWN"),
                "ky_du_lieu": input_data.get("ky_du_lieu", "UNKNOWN"),
                "ma_tieu_chi": input_data.get("ma_tieu_chi", "UNKNOWN"),
                "fld_code": input_data.get("fld_code", "UNKNOWN"),
                "model_name": self.model_name,
                "error_message": f"Ray actor prediction error: {str(e)}",
                "is_anomaly": False,
                "prediction": None,
                "processing_time": prediction_time,
                "ray_actor_prediction": True,
                "actor_id": ray.get_runtime_context().get_actor_id()
            }
    
    async def _execute_prediction(self, input_df: pd.DataFrame):
        """Execute prediction using the model info's prediction capability"""
        # This method handles different ways the model might be stored in model_info
        try:
            if hasattr(self.model_info, 'model') and hasattr(self.model_info.model, 'predict'):
                return self.model_info.model.predict(input_df)
            elif hasattr(self.model_info, 'predict'):
                return self.model_info.predict(input_df)
            else:
                raise Exception("No prediction method found in model_info")
        except Exception as e:
            self.logger.error(f"Error executing prediction for {self.model_name}: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Ray actor statistics"""
        avg_prediction_time = (
            self.total_prediction_time / max(1, self.prediction_count)
        )
        
        return {
            "model_name": self.model_name,
            "prediction_count": self.prediction_count,
            "total_prediction_time": self.total_prediction_time,
            "avg_prediction_time": avg_prediction_time,
            "actor_id": ray.get_runtime_context().get_actor_id(),
            "actor_status": "active"
        }
    
    def cleanup(self):
        """Cleanup model resources"""
        try:
            self.model = None
            self.model_info = None
            self.logger.info(f"Ray deployment actor cleanup completed for {self.model_name}")
        except Exception as e:
            self.logger.error(f"Error during Ray actor cleanup for {self.model_name}: {e}")