# Python
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import HTTPException


def ensure_list_non_empty(v: Any, *, field_path: str) -> List[Any]:
    """
    - Nếu không phải list -> 400 ngay
    - Nếu rỗng -> 400 ngay
    """
    if not isinstance(v, list):
        err_msg = (
            "Trường data phải là kiểu mảng (array), nhận được kiểu "
            f"{type(v).__name__}"
        )
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "error_code": "INVALID_DATA_TYPE",
                    "error_message": err_msg,
                },
                "field_path": field_path,
                "timestamp": datetime.now().isoformat(),
            },
        )
    if not v:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "error_code": "EMPTY_REQUIRED_FIELD",
                    "error_message": (
                        "Trường data không được rỗng, " "phải chứa ít nhất một phần tử"
                    ),
                },
                "field_path": field_path,
                "timestamp": datetime.now().isoformat(),
            },
        )
    return v


def validate_item_is_dict(
    index: int,
    item: Any,
    errors: List[Dict[str, Any]],
) -> bool:
    """
    Xác nhận phần tử data[index] là dict.
    Nếu sai, append lỗi và trả False.
    """
    if isinstance(item, dict):
        return True

    msg = (
        f"Phần tử thứ {index + 1} trong data phải là kiểu đối tượng (object), "
        f"nhận được kiểu {type(item).__name__}"
    )
    errors.append(
        {
            "error_message": msg,
            "ma_tieu_chi": None,
        }
    )
    return False


def extract_ma_tieu_chi_with_errors(
    *,
    i: int,
    item: Dict[str, Any],
    errors: List[Dict[str, Any]],
) -> Optional[str]:
    """
    Lấy ma_tieu_chi. Nếu thiếu/sai kiểu/rỗng, đẩy lỗi phù hợp.
    Trả về chuỗi đã strip() hoặc None.
    """
    if "ma_tieu_chi" not in item:
        msg = f"Phần tử thứ {i + 1} trong data thiếu trường bắt buộc " "ma_tieu_chi"
        errors.append(
            {
                "error_message": msg,
                "ma_tieu_chi": None,
            }
        )
        return None

    value = item.get("ma_tieu_chi")
    if not isinstance(value, str):
        msg = (
            f"Trường ma_tieu_chi trong phần tử thứ {i + 1} phải là kiểu "
            f"chuỗi (string), nhận được kiểu {type(value).__name__}"
        )
        errors.append(
            {
                "error_message": msg,
                "ma_tieu_chi": None,
            }
        )
        return None

    if not value.strip():
        msg = "Trường ma_tieu_chi trong phần tử " f"thứ {i + 1} không được rỗng"
        errors.append(
            {
                "error_message": msg,
                "ma_tieu_chi": None,
            }
        )
        return None

    return value.strip()


def collect_and_validate_fn_fields(
    *,
    i: int,
    item: Dict[str, Any],
    ma_tieu_chi: Optional[str],
    errors: List[Dict[str, Any]],
) -> List[str]:
    """
    - Thu thập các trường FNxx trong item
    - Nếu không có, đẩy lỗi
    - Kiểm tra kiểu dữ liệu từng FNxx phải là số
    """
    fn_fields = [k for k in item.keys() if k.startswith("FN")]
    if not fn_fields:
        msg = (
            f"Phần tử thứ {i + 1} trong data "
            f"(ma_tieu_chi: {ma_tieu_chi or 'UNKNOWN'}) "
            "phải có ít nhất một trường FN (ví dụ: FN01, FN02, ...)"
        )
        errors.append(
            {
                "error_message": msg,
                "ma_tieu_chi": ma_tieu_chi,
            }
        )
        return []

    for fn_field in fn_fields:
        fn_value = item[fn_field]
        if not isinstance(fn_value, (int, float)):
            msg = (
                f"Trường {fn_field} phải là kiểu số (number), nhận được "
                f"kiểu {type(fn_value).__name__}"
            )
            errors.append(
                {
                    "error_message": msg,
                    "ma_tieu_chi": ma_tieu_chi,
                    "fn_field": fn_field,
                }
            )
    return fn_fields


def finalize_validated_list_with_errors(
    v: List[Dict[str, Any]],
    validation_errors: List[Dict[str, Any]],
):
    """
    - Gắn _validation_errors vào list và từng item
    - Nhóm lỗi theo ma_tieu_chi và đưa về item tương ứng
    """

    class ValidatedList(list):
        """Custom list class to store validation errors"""

        def __init__(self, v):
            self._validation_errors = None

    validated_data = ValidatedList(v)
    validated_data._validation_errors = validation_errors  # type: ignore[attr-defined]

    # Build error lookup dict
    error_map: Dict[str, List[Dict[str, Any]]] = {}
    for error in validation_errors:
        mtc = error.get("ma_tieu_chi")
        if mtc:
            if mtc not in error_map:
                error_map[mtc] = []
            error_map[mtc].append(error)

    # Initialize error arrays and assign errors to items
    for item in validated_data:
        if isinstance(item, dict):
            item["_validation_errors"] = []
            if mtc := item.get("ma_tieu_chi"):
                if mtc in error_map:
                    item["_validation_errors"].extend(error_map[mtc])
    return validated_data
