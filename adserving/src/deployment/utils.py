from typing import Any, Dict, Optional, Tuple


def _build_model_uri(name: str, version: str | int) -> str:
    """Build MLflow model URI from name and version."""
    return f"models:/{name}/{version}"


def _parse_model_name(model_name: str) -> Tuple[str, str]:
    """Parse model name to extract ma_tieu_chi and fld_code."""
    parts = model_name.split("_")
    if len(parts) < 4:
        return "", ""
    fld_code = parts[-1]
    ma_tieu_chi = "_".join(parts[2:-1]) if len(parts) > 3 else ""
    return ma_tieu_chi, fld_code


def decide_anomaly(score: float, threshold: Optional[float]) -> bool:
    """Decide anomaly based on score and threshold.

    If threshold is None, returns False.
    If threshold <= 0: anomaly when score < threshold.
    Else: anomaly when score > threshold.
    """
    if threshold is None:
        return False
    return bool(score < threshold) if threshold <= 0 else bool(score > threshold)


def build_unknown_model_response(
    model_name: str,
    ma_tieu_chi: str,
    fld_code: str,
    threshold: Optional[float],
    model_version: Optional[str],
) -> Dict[str, Any]:
    """Build unknown model response."""
    return {
        "model_name": model_name,
        "ma_tieu_chi": ma_tieu_chi,
        "fld_code": fld_code,
        "is_anomaly": False,
        "anomaly_score": None,
        "anomaly_threshold": threshold,
        "processing_time": 0.0,
        "model_version": model_version,
        "error_message": f"Unknown model: {model_name}",
        "status": "error",
    }


def build_exception_response(
    model_name: str,
    ma_tieu_chi: str,
    fld_code: str,
    threshold: Optional[float],
    processing_time: float,
    model_version: Optional[str],
    error_message: str,
) -> Dict[str, Any]:
    """Build exception response."""
    return {
        "model_name": model_name,
        "ma_tieu_chi": ma_tieu_chi,
        "fld_code": fld_code,
        "is_anomaly": False,
        "anomaly_score": None,
        "anomaly_threshold": threshold,
        "processing_time": processing_time,
        "model_version": model_version,
        "error_message": error_message,
        "status": "error",
    }


def build_success_response(
    model_name: str,
    ma_tieu_chi: str,
    fld_code: str,
    is_anomaly: bool,
    score: float,
    threshold: Optional[float],
    processing_time: float,
    model_version: Optional[str],
) -> Dict[str, Any]:
    """Build success response."""
    return {
        "model_name": model_name,
        "ma_tieu_chi": ma_tieu_chi,
        "fld_code": fld_code,
        "is_anomaly": is_anomaly,
        "anomaly_score": score,
        "anomaly_threshold": threshold,
        "processing_time": processing_time,
        "model_version": model_version,
        "error_message": None,
        "status": "success",
    }


def validate_task_element(
    t: Dict[str, Any],
    index: int,
    models_active: Dict[str, Any],
) -> Tuple[Optional[Tuple[int, str, float]], Optional[Dict[str, Any]]]:
    """Validate a single task element.

    Returns (valid_tuple, failed_dict).
    valid_tuple: (index, model_name, value) if valid else None.
    failed_dict: error description if invalid else None.
    """
    try:
        m = str(t["model_name"])  # may raise
        v = float(t["value"])  # may raise
    except Exception:
        return None, {
            "element_index": index,
            "model_name": t.get("model_name"),
            "ma_tieu_chi": "",
            "fld_code": "",
            "error": "invalid_task",
            "error_details": "Thiếu hoặc sai định dạng model_name / value",
        }

    if m not in models_active:
        mtc, fld = _parse_model_name(m)
        return None, {
            "element_index": index,
            "model_name": m,
            "ma_tieu_chi": mtc,
            "fld_code": fld,
            "error": "model_not_found",
            "error_details": "Model không tồn tại trong preload.",
        }

    return (index, m, v), None
