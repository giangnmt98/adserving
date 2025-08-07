"""
Request processing utilities for pooled model deployments
"""

from typing import Any, Dict, List, Optional

import pandas as pd
from adserving.src.utils.logger import FrameworkLogger, get_logger


class RequestProcessor:
    """Handles request processing and validation for pooled deployments"""

    def __init__(self, logger: Optional[FrameworkLogger] = None):
        self.logger = logger or get_logger()

    def extract_model_names(self, request: Dict[str, Any]) -> List[str]:
        """
        Extract model names from request data - returns list for batch
        processing
        """
        model_names = []

        # Handle the new format: ma_don_vi, ma_bao_cao, data with
        # ma_tieu_chi and FN fields
        if all(key in request for key in ["ma_don_vi", "ma_bao_cao", "data"]):
            ma_don_vi = request["ma_don_vi"]
            ma_bao_cao = request["ma_bao_cao"]

            if isinstance(request["data"], list):
                for data_item in request["data"]:
                    if "ma_tieu_chi" in data_item:
                        ma_tieu_chi = data_item["ma_tieu_chi"]

                        # Extract FN fields from data item
                        fn_fields = {
                            k: v
                            for k, v in data_item.items()
                            if k.startswith("FN") and k != "ma_tieu_chi"
                        }

                        # Create model name for each FN field
                        for fld_code in fn_fields.keys():
                            normalized_fld = self._normalize_field_code(fld_code)
                            model_name = (
                                f"{ma_don_vi}_{ma_bao_cao}_"
                                f"{ma_tieu_chi}_{normalized_fld}"
                            )
                            model_names.append(model_name)

        # Fallback to direct model_name specification
        if not model_names and "model_name" in request:
            model_names.append(request["model_name"])

        return model_names

    def _normalize_field_code(self, field_name: str) -> str:
        """Normalize field code similar to input_handler"""
        if not field_name.upper().startswith("FN"):
            return field_name.upper()

        # Extract number part
        number_part = field_name[2:]
        if number_part.isdigit():
            # Pad with zero if single digit
            if len(number_part) == 1:
                return f"FN0{number_part}"
            else:
                return f"FN{number_part}"

        return field_name.upper()

    def prepare_input_data(self, request: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Prepare input data for model prediction - handles only specified format
        """
        prediction_tasks = []
        self._validate_request(request)

        ma_don_vi, ma_bao_cao, ky_du_lieu, data_list = self._extract_request_fields(
            request
        )
        self._validate_field_values(ma_don_vi, ma_bao_cao, ky_du_lieu)
        self._validate_data_list(data_list)

        prediction_tasks = self._create_prediction_tasks(
            data_list, ma_don_vi, ma_bao_cao, ky_du_lieu
        )
        return prediction_tasks

    def _validate_request(self, request: Dict[str, Any]) -> None:
        if not request:
            self.logger.error("Empty request received")
            raise ValueError(
                "Request cannot be empty. Expected format: "
                "{'ma_don_vi': '...', 'ma_bao_cao': '...', "
                "'ky_du_lieu': '...', 'data': [...]}"
            )

        required_fields = ["ma_don_vi", "ma_bao_cao", "ky_du_lieu", "data"]
        missing_fields = [field for field in required_fields if field not in request]
        if missing_fields:
            self.logger.error(f"Missing required fields: {missing_fields}")
            raise ValueError(
                f"Missing required fields: {missing_fields}. Expected format: "
                f"{{'ma_don_vi': '...', 'ma_bao_cao': '...', "
                f"'ky_du_lieu': '...', 'data': [...]}}"
            )

    def _extract_request_fields(self, request: Dict[str, Any]) -> tuple:
        ma_don_vi = request["ma_don_vi"]
        ma_bao_cao = request["ma_bao_cao"]
        ky_du_lieu = request["ky_du_lieu"]
        data_list = request["data"]
        return ma_don_vi, ma_bao_cao, ky_du_lieu, data_list

    def _validate_field_values(
        self, ma_don_vi: str, ma_bao_cao: str, ky_du_lieu: str
    ) -> None:
        if not ma_don_vi or not ma_bao_cao or not ky_du_lieu:
            self.logger.error(
                f"Empty values in required fields: "
                f"ma_don_vi='{ma_don_vi}', ma_bao_cao='{ma_bao_cao}', "
                f"ky_du_lieu='{ky_du_lieu}'"
            )
            raise ValueError("ma_don_vi, ma_bao_cao, and ky_du_lieu cannot be empty")

    def _validate_data_list(self, data_list: List) -> None:
        if not isinstance(data_list, list):
            self.logger.error(
                f"Invalid data format: expected list, got {type(data_list)}"
            )
            raise ValueError(
                "'data' field must be a list of objects with "
                "ma_tieu_chi and FN fields"
            )

        if not data_list:
            self.logger.error("Empty data list received")
            raise ValueError("'data' list cannot be empty")

    def _create_prediction_tasks(
        self, data_list: List, ma_don_vi: str, ma_bao_cao: str, ky_du_lieu: str
    ) -> List[Dict[str, Any]]:
        prediction_tasks = []
        for i, data_item in enumerate(data_list):
            self._validate_data_item(data_item, i)
            ma_tieu_chi = self._get_ma_tieu_chi(data_item, i)
            fn_fields = self._get_fn_fields(data_item, i)

            for fn_field, gia_tri in fn_fields.items():
                if gia_tri is None:
                    self.logger.warning(
                        f"Skipping {fn_field} with None value in "
                        f"data item at index {i}"
                    )
                    continue

                prediction_tasks.append(
                    self._create_prediction_task(
                        ma_don_vi,
                        ma_bao_cao,
                        ma_tieu_chi,
                        fn_field,
                        gia_tri,
                        ky_du_lieu,
                    )
                )

        return prediction_tasks

    def _validate_data_item(self, data_item: Dict, index: int) -> None:
        if not isinstance(data_item, dict):
            self.logger.error(
                f"Invalid data item at index {index}: expected dict, "
                f"got {type(data_item)}"
            )
            raise ValueError(f"Data item at index {index} must be a dictionary")

    def _get_ma_tieu_chi(self, data_item: Dict, index: int) -> str:
        if "ma_tieu_chi" not in data_item:
            self.logger.error(f"Missing ma_tieu_chi in data item at index {index}")
            raise ValueError(
                f"Data item at index {index} is missing required field "
                f"'ma_tieu_chi'"
            )

        ma_tieu_chi = data_item["ma_tieu_chi"]
        if not ma_tieu_chi:
            self.logger.error(f"Empty ma_tieu_chi in data item at index {index}")
            raise ValueError(
                f"ma_tieu_chi cannot be empty in data item at index {index}"
            )
        return ma_tieu_chi

    def _get_fn_fields(self, data_item: Dict, index: int) -> Dict:
        fn_fields = {
            k: v
            for k, v in data_item.items()
            if k.startswith("FN") and k != "ma_tieu_chi"
        }

        if not fn_fields:
            self.logger.error(
                f"No FN fields found in data item at index {index}. "
                f"Available fields: {list(data_item.keys())}"
            )
            raise ValueError(
                f"Data item at index {index} must contain at least one "
                f"FN field (e.g., FN01, FN02, etc.)"
            )

        return fn_fields

    def _create_prediction_task(
        self,
        ma_don_vi: str,
        ma_bao_cao: str,
        ma_tieu_chi: str,
        fn_field: str,
        gia_tri: Any,
        ky_du_lieu: str,
    ) -> Dict[str, Any]:
        normalized_fn = self._normalize_field_code(fn_field)
        model_name = f"{ma_don_vi}_{ma_bao_cao}_" f"{ma_tieu_chi}_{normalized_fn}"

        input_features = {
            "ma_don_vi": ma_don_vi,
            "ma_bao_cao": ma_bao_cao,
            "ma_tieu_chi": ma_tieu_chi,
            "fld_code": normalized_fn,
            "ky_du_lieu": ky_du_lieu,
            fn_field: gia_tri,
            "gia_tri": gia_tri,
        }

        return {
            "model_name": model_name,
            "input_data": pd.DataFrame([input_features]),
        }
