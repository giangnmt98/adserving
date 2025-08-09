"""
Ray remote tasks for prediction processing
"""

from typing import Dict
import ray
import pandas as pd


@ray.remote
class PredictionTask:
    """Ray remote task for parallel model predictions"""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
        
    async def predict_field(self, model_name: str, prediction_data: Dict) -> Dict:
        """Execute a single prediction task for a field"""
        try:
            # Extract data from prediction_data
            ma_tieu_chi = prediction_data["ma_tieu_chi"]
            field_name = prediction_data["field_name"]
            field_value = prediction_data["field_value"]
            request_metadata = prediction_data["request_metadata"]
            
            # Convert to pandas DataFrame format expected by prediction service
            input_df = pd.DataFrame({
                "ma_tieu_chi": [ma_tieu_chi],
                "fld_code": [field_name],
                "gia_tri": [float(field_value)]
            })
            
            # Call actual model prediction
            prediction_result = await self.model_manager.predict(model_name, input_df)
            
            if prediction_result and hasattr(prediction_result, 'status') and prediction_result.status == "success":
                # Convert DetailedPredictionResult to dict format
                return {
                    "status": "success",
                    "ma_don_vi": request_metadata["ma_don_vi"],
                    "ma_bao_cao": request_metadata["ma_bao_cao"],
                    "ky_du_lieu": request_metadata["ky_du_lieu"],
                    "ma_tieu_chi": prediction_result.ma_tieu_chi,
                    "is_anomaly": prediction_result.is_anomaly,
                    "fld_code": prediction_result.fld_code,
                    "prediction": prediction_result.anomaly_score,
                    "anomaly_threshold": prediction_result.anomaly_threshold,
                    "model_name": prediction_result.model_name,
                    "model_version": prediction_result.model_version,
                    "processing_time": prediction_result.processing_time,
                    "task_id": prediction_data.get("task_id"),
                }
            else:
                # Handle prediction error
                return {
                    "status": "error",
                    "ma_don_vi": request_metadata["ma_don_vi"],
                    "ma_bao_cao": request_metadata["ma_bao_cao"],
                    "ky_du_lieu": request_metadata["ky_du_lieu"],
                    "ma_tieu_chi": ma_tieu_chi,
                    "fld_code": field_name,
                    "model_name": model_name,
                    "error_message": prediction_result.error_message if prediction_result else "Prediction failed",
                    "is_anomaly": False,
                    "prediction": None,
                    "task_id": prediction_data.get("task_id"),
                }
                
        except Exception as field_error:
            return {
                "status": "error",
                "ma_don_vi": request_metadata["ma_don_vi"],
                "ma_bao_cao": request_metadata["ma_bao_cao"],
                "ky_du_lieu": request_metadata["ky_du_lieu"],
                "ma_tieu_chi": ma_tieu_chi,
                "fld_code": field_name,
                "model_name": model_name,
                "error_message": f"Field processing error: {str(field_error)}",
                "is_anomaly": False,
                "prediction": None,
                "task_id": prediction_data.get("task_id"),
            }