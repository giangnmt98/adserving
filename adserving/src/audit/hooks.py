from adserving.src.audit.redis_emitter import get_emitter
from adserving.src.utils.logger import get_logger

logger = get_logger()


def emit_training_request(
    ma_don_vi: str,
    ma_bao_cao: str,
    ky_du_lieu: str,
    data,
) -> None:
    """Ghi record training tối thiểu vào Redis Streams (best-effort, non-blocking)."""

    try:
        get_emitter().emit_training(ma_don_vi, ma_bao_cao, ky_du_lieu, data)
    except Exception as e:
        # Không bao giờ để lỗi này ảnh hưởng tới luồng chính
        logger.debug(f"emit_training_request failed silently: {e}")


def emit_inference_result(data) -> None:
    """Ghi kết quả inference rút gọn vào Redis Streams (best-effort, non-blocking)."""
    try:
        get_emitter().emit_inference(data)
    except Exception as e:
        print("emit_inference_result failed:", e)
        logger.debug("emit_inference_result failed silently.")
