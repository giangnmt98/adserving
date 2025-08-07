"""
Task processing utilities for parallel execution
"""

import asyncio
from typing import Any, Dict, List, Optional
from adserving.src.utils.logger import FrameworkLogger, get_logger


class TaskProcessor:
    """Handles parallel processing of prediction tasks"""

    def __init__(self, logger: Optional[FrameworkLogger] = None):
        self.logger = logger or get_logger()

    async def process_prediction_tasks(
        self,
        prediction_tasks: List[Dict[str, Any]],
        model_router: Any,
        model_manager: Any,
        model_pools: list,
        config: Any,
        metrics_lock: Any,
        model_request_counts: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        """Process prediction tasks in parallel for better performance"""

        async def process_task(task):
            """Process individual prediction task"""
            model_name = task["model_name"]
            input_data = task["input_data"]

            # Update model request statistics
            with metrics_lock:
                model_request_counts[model_name] += 1

            try:
                # Route to appropriate model pool and perform prediction
                prediction = await model_router.route_and_predict(
                    model_name, input_data, model_manager, model_pools, config
                )

                prediction_result = prediction.to_dict()
                prediction_result["status"] = "success"
                # Create a result for this task
                return prediction_result

            except Exception as task_error:
                # Handle individual task errors
                self.logger.error(f"Task error for {model_name}: {task_error}")
                return {
                    "model_name": model_name,
                    "ma_tieu_chi": input_data["ma_tieu_chi"].iloc[0],
                    "fld_code": input_data["fld_code"].iloc[0],
                    "error_message": str(task_error).split("(")[0],
                    "status": "error",
                }

        # Process all tasks in parallel using asyncio.gather
        tasks = [process_task(task) for task in prediction_tasks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions that occurred during parallel processing
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Convert exception to error result
                task = prediction_tasks[i]
                self.logger.error(
                    f"Parallel task error for "
                    f"{task.get('model_name', 'unknown')}: {result}"
                )
                processed_results.append(
                    {
                        "model_name": task.get("model_name", "unknown"),
                        "ma_don_vi": task.get("ma_don_vi"),
                        "ma_bao_cao": task.get("ma_bao_cao"),
                        "ma_tieu_chi": task.get("ma_tieu_chi"),
                        "fld_code": task.get("fld_code"),
                        "gia_tri": task.get("gia_tri"),
                        "ky_du_lieu": task.get("ky_du_lieu"),
                        "error": str(result),
                        "status": "error",
                    }
                )
            else:
                processed_results.append(result)

        return processed_results

    def create_consolidated_response(
        self,
        results: List[Dict[str, Any]],
        prediction_tasks: List[Dict[str, Any]],
        inference_time: float,
    ) -> Dict[str, Any]:
        """Create consolidated response from processed results"""
        successful_results = [r for r in results if r["status"] == "success"]
        error_results = [r for r in results if r["status"] == "error"]

        return {
            "results": results,
            "total_tasks": len(prediction_tasks),
            "successful_tasks": len(successful_results),
            "failed_tasks": len(error_results),
            "inference_time": inference_time,
            "status": "success" if len(successful_results) > 0 else "error",
        }
