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
    data,
) -> None:
    """
    Gọi ngay sau khi parse/validate request thành công, để emit training record.
    Không chặn luồng chính. Nếu emitter gặp lỗi, sẽ im lặng bỏ qua.
    """
    try:
        emit_training_request(
            ma_don_vi=data[0]["input_data"]["ma_don_vi"],
            ma_bao_cao=data[0]["input_data"]["ma_bao_cao"],
            ky_du_lieu=data[0]["input_data"]["ky_du_lieu"],
            data=data,
        )
    except Exception as e:
        # Không để ảnh hưởng luồng chính
        logger.error(f"on_request_parsed emit failed: {e}")


def on_inference_done(
    request_id: str,
    timestamp,
    details_result,
) -> None:
    """
    Gọi ngay trước khi trả response, sau khi có kết quả inference (hoặc lỗi).
    Gửi bản tóm tắt nhẹ để trace (non-blocking).
    #"""
    data = {
        "request_id": request_id,
        "timestamp": timestamp,
        "details_result": details_result,
    }
    try:
        emit_inference_result(data)
    except Exception as e:
        # Không để ảnh hưởng luồng chính
        logger.error(f"on_inference_done emit failed silently. {e}")
