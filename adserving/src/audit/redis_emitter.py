"""Redis emitter for audit data streaming using best-effort delivery."""

# Python
import json
import os
import random
from typing import Any, Dict, Optional

import redis

from adserving.src.config.config import get_config
from adserving.src.utils.logger import get_logger

logger = get_logger()


class RedisEmitter:
    """Emitter gửi record vào Redis Streams theo cơ
    chế best-effort, không chặn luồng chính."""

    def __init__(self) -> None:
        """Initialize Redis emitter with configuration settings."""
        cfg = get_config()
        # Đọc cấu hình đã qua config_manager
        audit_cfg = cfg.audit
        redis_cfg = cfg.redis
        self.enabled: bool = bool(audit_cfg.enabled)
        self.training_rate: float = audit_cfg.training_rate
        self.infer_rate: float = audit_cfg.inference_rate
        self.redact_pii: bool = bool(getattr(audit_cfg, "redact_pii", True))

        streams_cfg = redis_cfg.streams
        self.stream_training: str = (
            getattr(streams_cfg, "training_data", "training_data")
            if streams_cfg
            else "training_data"
        )
        self.stream_infer: str = (
            getattr(streams_cfg, "inference_results", "inference_results")
            if streams_cfg
            else "inference_results"
        )
        self.maxlen: int = int(
            getattr(streams_cfg, "max_stream_length", 1_000_000)
            if streams_cfg
            else 1_000_000
        )

        self._client = None
        if not self.enabled:
            logger.info("Audit disabled; RedisEmitter will be no-op.")
            return

        if redis is None:
            logger.warning("Package 'redis' not installed. RedisEmitter disabled.")
            self.enabled = False
            return

        try:
            # Redis chỉ requirepass: không truyền username
            host = redis_cfg.host or "127.0.0.1"
            port = redis_cfg.port or 6379
            db = redis_cfg.db or 0
            password = os.getenv("REDIS_PASSWORD") or redis_cfg.password or None
            socket_timeout = redis_cfg.socket_timeout or 0.05
            connect_timeout = redis_cfg.socket_connect_timeout or 0.05
            retry_on_timeout = redis_cfg.retry_on_timeout or True
            health_check_interval = int(redis_cfg.health_check_interval) or 15

            self._client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password or None,  # requirepass only
                socket_timeout=socket_timeout,
                socket_connect_timeout=connect_timeout,
                retry_on_timeout=retry_on_timeout,
                health_check_interval=health_check_interval,
                decode_responses=True,
            )
            try:
                self._client.ping()
            except Exception:
                # Không ping được cũng không chặn
                pass
        except Exception as e:
            logger.warning(f"RedisEmitter init failed: {e}. Emitter disabled.")
            self.enabled = False

    def _xadd(self, stream: str, list_record) -> None:
        """Add records to Redis stream with best-effort delivery."""
        if not self.enabled or not self._client:
            return
        for record in list_record:
            fields = {
                k: (
                    json.dumps(v, ensure_ascii=False)
                    if isinstance(v, (dict, list))
                    else ("" if v is None else str(v))
                )
                for k, v in record.items()
            }
            try:
                self._client.xadd(stream, fields, maxlen=self.maxlen, approximate=True)
            except Exception as e:
                logger.warning(f"RedisEmitter xadd failed: {e}")
                # Best-effort: không ảnh hưởng luồng chính

    def emit_training(
        self, ma_don_vi, ma_bao_cao, ky_du_lieu, payload: Dict[str, Any]
    ) -> None:
        """Emit training data to Redis stream with sampling rate."""
        if not self.enabled or self.training_rate <= 0:
            return
        if random.random() > self.training_rate:
            return
        payload = [
            {
                **dict(item["input_data"]),
                "ma_don_vi": ma_don_vi,
                "ma_bao_cao": ma_bao_cao,
                "ky_du_lieu": ky_du_lieu,
            }
            for item in payload
        ]
        self._xadd(self.stream_training, payload)

    def emit_inference(self, payload: Dict[str, Any]) -> None:
        """Emit inference results to Redis stream with sampling rate."""
        if not self.enabled or self.infer_rate <= 0:
            return
        if random.random() > self.infer_rate:
            return
        request_id = payload["request_id"]
        timestamp = payload["timestamp"]
        payload = [
            {**item, "request_id": request_id, "timestamp": timestamp}
            for item in payload["details_result"]
        ]
        self._xadd(self.stream_infer, payload)


_EMITTER: Optional[RedisEmitter] = None


def get_emitter() -> RedisEmitter:
    """Get or create global Redis emitter instance."""
    global _EMITTER
    if _EMITTER is None:
        _EMITTER = RedisEmitter()
    return _EMITTER
