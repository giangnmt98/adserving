# Python
from typing import Any, Dict, Optional
from datetime import datetime

from adserving.src.audit.redis_emitter import get_emitter
from adserving.src.utils.logger import get_logger

logger = get_logger()


def emit_training_request(
    *,
    request_id: str,
    ma_don_vi: str,
    ma_bao_cao: str,
    ky_du_lieu: str,
    features: Dict[str, Any],
    consent_flags: Optional[Dict[str, Any]] = None,
    user_id_hash: Optional[str] = None,
    schema_version: str = "v1",
) -> None:
    """Ghi record training tối thiểu vào Redis Streams (best-effort, non-blocking)."""
    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "request_id": request_id,
        "ma_don_vi": ma_don_vi,
        "ma_bao_cao": ma_bao_cao,
        "ky_du_lieu": ky_du_lieu,
        "features": features,
        "consent_flags": consent_flags or {},
        "user_id_hash": user_id_hash,
        "label": None,
        "source": "online_serving",
        "schema_version": schema_version,
    }
    try:
        get_emitter().emit_training(payload)
    except Exception:
        # Không bao giờ để lỗi này ảnh hưởng tới luồng chính
        logger.debug("emit_training_request failed silently.", exc_info=False)


def emit_inference_result(
    *,
    request_id: str,
    model_name: str,
    processing_time: float,
    status: str = "success",
    model_version: Optional[str] = None,
    error_message: Optional[str] = None,
    summary: Optional[Dict[str, Any]] = None,
    threshold: Optional[float] = None,
    schema_version: str = "v1",
) -> None:
    """Ghi kết quả inference rút gọn vào Redis Streams (best-effort, non-blocking)."""
    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "request_id": request_id,
        "model_name": model_name,
        "model_version": model_version,
        "processing_time": processing_time,
        "status": status,
        "error_message": error_message,
        "summary": summary or {},  # ví dụ: {ma_tieu_chi, fld_code, is_anomaly, anomaly_score}
        "threshold": threshold,
        "schema_version": schema_version,
    }
    print("XXXXXXXXXX", payload)
    try:
        get_emitter().emit_inference(payload)
    except Exception:
        logger.debug("emit_inference_result failed silently.", exc_info=False)