# Python
from __future__ import annotations

from typing import Any, Dict, Optional

from adserving.src.audit.hooks import emit_inference_result, emit_training_request
from adserving.src.utils.logger import get_logger

logger = get_logger()


def hash_user(user_id: Optional[str]) -> Optional[str]:
    """
    Hash/ẩn danh user_id để tuân thủ bảo mật/PII.
    Lưu ý: Đây chỉ là placeholder. Hãy thay bằng hàm băm thực sự (có salt/pepper).
    """
    if not user_id or not isinstance(user_id, str):
        return None
    # Ví dụ giả lập: chỉ lấy 6 ký tự cuối để tránh lộ PII
    return f"hash:{user_id[-6:]}"


def build_training_features(
    req_body: Dict[str, Any],
    *,
    max_fields_per_item: int = 50,
) -> Dict[str, Any]:
    """
    Trích xuất features nhẹ phục vụ training từ request đã parse.
    - Không đưa PII thô (để phần emit quyết định redaction).
    - Chỉ giữ các thông tin chung để train mô hình.
    - Có tham số giới hạn số trường để tránh payload quá lớn.

    req_body kỳ vọng có các trường:
      - ma_don_vi, ma_bao_cao, ky_du_lieu
      - data: List[Dict[str, Any]] với các trường FNxx và/hoặc dữ liệu đầu vào.

    Trả về: dict features tối thiểu.
    """
    features: Dict[str, Any] = {}

    # Ví dụ rút gọn: đếm số item, gom một vài khóa top-level phổ biến
    data_list = req_body.get("data") or []
    features["num_items"] = min(len(data_list), 10_000)

    # Lấy các khóa FNxx từ 1-2 item đầu (giới hạn số trường) để có “schema hint”
    sample_schema = {}
    for idx, item in enumerate(data_list[:2]):
        if not isinstance(item, dict):
            continue
        keys = sorted([k for k in item.keys() if isinstance(k, str)])
        if max_fields_per_item > 0:
            keys = keys[:max_fields_per_item]
        sample_schema[f"item_{idx}"] = keys
    if sample_schema:
        features["sample_schema"] = sample_schema

    # Có thể thêm các thống kê nhẹ khác nếu cần (ví dụ số lượng FNxx xuất hiện)
    # Nhưng phải đảm bảo không tăng kích thước payload quá nhiều.

    return features


def on_request_parsed(
    request_id: str,
    req_body: Dict[str, Any],
    features: Dict[str, Any],
    consent: Dict[str, Any],
) -> None:
    """
    Gọi ngay sau khi parse/validate request thành công, để emit training record.
    Không chặn luồng chính. Nếu emitter gặp lỗi, sẽ im lặng bỏ qua.
    """
    try:
        emit_training_request(
            request_id=request_id,
            ma_don_vi=str(req_body.get("ma_don_vi", "")),
            ma_bao_cao=str(req_body.get("ma_bao_cao", "")),
            ky_du_lieu=str(req_body.get("ky_du_lieu", "")),
            features=features,
            consent_flags=consent or {},
            user_id_hash=hash_user(req_body.get("user_id")),
        )
    except Exception:
        # Không để ảnh hưởng luồng chính
        logger.debug("on_request_parsed emit failed silently.", exc_info=False)


def on_inference_done(
    request_id: str,
    result: Dict[str, Any],
    total_time: float,
) -> None:
    """
    Gọi ngay trước khi trả response, sau khi có kết quả inference (hoặc lỗi).
    Gửi bản tóm tắt nhẹ để trace (non-blocking).
    """
    try:
        summary = {
            "ma_tieu_chi": result.get("ma_tieu_chi"),
            "fld_code": result.get("fld_code"),
            "is_anomaly": result.get("is_anomaly"),
            "anomaly_score": result.get("anomaly_score"),
        }
        print("AAAAAAAAAAA", summary)
        emit_inference_result(
            request_id=request_id,
            model_name=str(result.get("model_name", "unknown")),
            model_version=result.get("model_version"),
            processing_time=float(total_time or 0.0),
            status=str(result.get("status", "success")),
            error_message=result.get("error_message"),
            summary=summary,
            threshold=result.get("anomaly_threshold"),
        )
    except Exception:
        # Không để ảnh hưởng luồng chính
        logger.debug("on_inference_done emit failed silently.", exc_info=False)