"""
Request processing utilities for pooled model deployments
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from adserving.src.utils.logger import FrameworkLogger, get_logger


class RequestProcessor:
    """Handles request processing and validation for pooled deployments"""

    def __init__(self, logger: Optional[FrameworkLogger] = None):
        self.logger = logger or get_logger()

    def _normalize_field_code(self, field_name: str) -> str:
        if not field_name.upper().startswith("FN"):
            return field_name.upper()
        number_part = field_name[2:]
        if number_part.isdigit():
            if len(number_part) == 1:
                return f"FN0{number_part}"
            return f"FN{number_part}"
        return field_name.upper()

    def prepare_input_data(self, request: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chuẩn hóa và validate request. Trả HTTPException(400) ngay khi invalid.
        """
        prediction_tasks: List[Dict[str, Any]] = []
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
        """Validate request. Raise HTTPException(400) if invalid."""
        if not request:
            self.logger.error("Empty request received")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "error_code": "EMPTY_REQUEST",
                        "error_message": (
                            "Request không được rỗng. Định dạng hợp lệ: "
                            "{'ma_don_vi': '...', 'ma_bao_cao': '...', "
                            "'ky_du_lieu': '...', 'data': [...]}"
                        ),
                    },
                    "field_path": "",
                    "timestamp": datetime.now().isoformat(),
                },
            )

        required_fields = ["ma_don_vi", "ma_bao_cao", "ky_du_lieu", "data"]
        missing_fields = [field for field in required_fields if field not in request]
        if missing_fields:
            self.logger.error(f"Missing required fields: {missing_fields}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "error_code": "MISSING_REQUIRED_FIELD",
                        "error_message": f"Thiếu trường bắt buộc:"
                        f" {', '.join(missing_fields)}",
                    },
                    "field_path": ",".join(missing_fields),
                    "timestamp": datetime.now().isoformat(),
                },
            )

    def _extract_request_fields(self, request: Dict[str, Any]) -> tuple:
        """Extract required fields from request.
        Raise HTTPException(400) if invalid."""
        ma_don_vi = request["ma_don_vi"]
        ma_bao_cao = request["ma_bao_cao"]
        ky_du_lieu = request["ky_du_lieu"]
        data_list = request["data"]
        return ma_don_vi, ma_bao_cao, ky_du_lieu, data_list

    def _validate_field_values(
        self, ma_don_vi: Any, ma_bao_cao: Any, ky_du_lieu: Any
    ) -> None:
        # ma_don_vi
        if not isinstance(ma_don_vi, str):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "error_code": "INVALID_DATA_TYPE",
                        "error_message": (
                            f"Trường ma_don_vi phải là kiểu chuỗi (string), "
                            f"nhận được kiểu {type(ma_don_vi).__name__}"
                        ),
                    },
                    "field_path": "ma_don_vi",
                    "timestamp": datetime.now().isoformat(),
                },
            )

        if not ma_don_vi.strip():
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "error_code": "EMPTY_REQUIRED_FIELD",
                        "error_message": "Trường ma_don_vi không được rỗng",
                    },
                    "field_path": "ma_don_vi",
                    "timestamp": datetime.now().isoformat(),
                },
            )

        # ma_bao_cao
        if not isinstance(ma_bao_cao, str):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "error_code": "INVALID_DATA_TYPE",
                        "error_message": (
                            f"Trường ma_bao_cao phải là kiểu chuỗi (string), "
                            f"nhận được kiểu {type(ma_bao_cao).__name__}"
                        ),
                    },
                    "field_path": "ma_bao_cao",
                    "timestamp": datetime.now().isoformat(),
                },
            )

        if not ma_bao_cao.strip():
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "error_code": "EMPTY_REQUIRED_FIELD",
                        "error_message": "Trường ma_bao_cao không được rỗng",
                    },
                    "field_path": "ma_bao_cao",
                    "timestamp": datetime.now().isoformat(),
                },
            )

        # ky_du_lieu
        if not isinstance(ky_du_lieu, str):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "error_code": "INVALID_DATA_TYPE",
                        "error_message": "Kiểu dữ liệu không hợp lệ",
                        "error_details": (
                            f"Trường ky_du_lieu phải là kiểu chuỗi (string), "
                            f"nhận được kiểu {type(ky_du_lieu).__name__}"
                        ),
                    },
                    "field_path": "ky_du_lieu",
                    "timestamp": datetime.now().isoformat(),
                },
            )

        # Validate format YYYY-MM-DD
        try:
            datetime.strptime(ky_du_lieu, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "error_code": "INVALID_DATE_FORMAT",
                        "error_message": "Định dạng ngày không hợp lệ",
                        "error_details": (
                            f"Trường ky_du_lieu phải có format YYYY-MM-DD "
                            f"(ví dụ: 2024-01-01), nhận được: {ky_du_lieu}"
                        ),
                    },
                    "field_path": "ky_du_lieu",
                    "timestamp": datetime.now().isoformat(),
                },
            )

    def _validate_data_list(self, data_list: Any) -> None:
        """Validate data list"""
        if not isinstance(data_list, list):
            self.logger.error(
                f"Invalid data format: expected list, got {type(data_list)}"
            )
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "error_code": "INVALID_DATA_TYPE",
                        "error_message": (
                            f"Trường data phải là kiểu mảng (array), "
                            f"nhận được kiểu {type(data_list).__name__}"
                        ),
                    },
                    "field_path": "data",
                    "timestamp": datetime.now().isoformat(),
                },
            )

        if not data_list:
            self.logger.error("Empty data list received")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "error_code": "EMPTY_REQUIRED_FIELD",
                        "error_message": (
                            "Trường data không được rỗng, phải chứa ít nhất một phần tử"
                        ),
                    },
                    "field_path": "data",
                    "timestamp": datetime.now().isoformat(),
                },
            )

    def _create_prediction_tasks(
        self, data_list: List, ma_don_vi: str, ma_bao_cao: str, ky_du_lieu: str
    ) -> List[Dict[str, Any]]:
        """Create prediction tasks from data list"""
        prediction_tasks: List[Dict[str, Any]] = []
        for i, data_item in enumerate(data_list):
            self._validate_data_item(data_item, i)
            ma_tieu_chi = self._get_ma_tieu_chi(data_item, i)
            fn_fields = self._get_fn_fields(data_item, i)

            for fn_field, gia_tri in fn_fields.items():
                if gia_tri is None:
                    self.logger.debug(
                        f"Skipping {fn_field} with None value in data item at index {i}"
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
                f"Invalid data item at index {index}: expected dict,"
                f" got {type(data_item)}"
            )
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "error_code": "INVALID_DATA_TYPE",
                        "error_message": (
                            f"Phần tử thứ {index + 1} "
                            f"trong data phải là kiểu đối tượng "
                            f"(object), nhận được kiểu {type(data_item).__name__}"
                        ),
                    },
                    "field_path": f"data[{index}]",
                    "timestamp": datetime.now().isoformat(),
                },
            )

    def _get_ma_tieu_chi(self, data_item: Dict, index: int) -> str:
        if "ma_tieu_chi" not in data_item:
            self.logger.error(f"Missing ma_tieu_chi in data item at index {index}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "error_code": "MISSING_REQUIRED_FIELD",
                        "error_message": (
                            f"Data item tại vị trí {index + 1} thiếu trường bắt buộc "
                            "'ma_tieu_chi'"
                        ),
                    },
                    "field_path": f"data[{index}].ma_tieu_chi",
                    "timestamp": datetime.now().isoformat(),
                },
            )
        ma_tieu_chi = data_item["ma_tieu_chi"]
        if not isinstance(ma_tieu_chi, str) or not ma_tieu_chi.strip():
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "error_code": "INVALID_DATA_TYPE",
                        "error_message": (
                            f"Trường ma_tieu_chi tại vị trí {index + 1} phải là chuỗi "
                            "không rỗng"
                        ),
                    },
                    "field_path": f"data[{index}].ma_tieu_chi",
                    "timestamp": datetime.now().isoformat(),
                },
            )
        return ma_tieu_chi.strip()

    def _get_fn_fields(self, data_item: Dict, index: int) -> Dict:
        fn_fields = {k: v for k, v in data_item.items() if k.startswith("FN")}
        if not fn_fields:
            self.logger.error(
                f"No FN fields found in data item at index {index}. "
                f"Available: {list(data_item.keys())}"
            )
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "error_code": "MISSING_REQUIRED_FIELD",
                        "error_message": (
                            f"Phần tử thứ {index + 1} phải chứa ít nhất một trường FN "
                            "(FN01, FN02, ...)"
                        ),
                    },
                    "field_path": f"data[{index}]",
                    "timestamp": datetime.now().isoformat(),
                },
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
        model_name = f"{ma_don_vi}_{ma_bao_cao}_{ma_tieu_chi}_{normalized_fn}"

        input_features = {
            "ma_don_vi": ma_don_vi.strip(),
            "ma_bao_cao": ma_bao_cao.strip(),
            "ma_tieu_chi": ma_tieu_chi,
            "fld_code": normalized_fn,
            "ky_du_lieu": ky_du_lieu,
            fn_field: gia_tri,
            "gia_tri": gia_tri,
        }
        return {"model_name": model_name, "input_data": input_features}
