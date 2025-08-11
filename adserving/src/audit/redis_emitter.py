# Python
import json
import os
import random
from typing import Any, Dict, Optional

try:
    import redis  # Optional: nếu thiếu, emitter sẽ tự vô hiệu hóa
except Exception:  # pragma: no cover
    redis = None  # type: ignore

from adserving.src.config.config import get_config
from adserving.src.utils.logger import get_logger

logger = get_logger()


class RedisEmitter:
    """Emitter gửi record vào Redis Streams theo cơ chế best-effort, không chặn luồng chính."""

    def __init__(self) -> None:
        cfg = get_config()
        # Đọc cấu hình đã qua config_manager
        audit_cfg = getattr(cfg, "audit", None)
        redis_cfg = getattr(cfg, "redis", None)
        print(999999999999999999999999)
        print(audit_cfg)
        print(redis_cfg)
        self.enabled: bool = bool(getattr(audit_cfg, "enabled", False))
        self.training_rate: float = float(getattr(getattr(audit_cfg, "sampling", None), "training_rate", 0.0) or 0.0)
        self.infer_rate: float = float(getattr(getattr(audit_cfg, "sampling", None), "inference_rate", 1.0) or 1.0)
        self.redact_pii: bool = bool(getattr(audit_cfg, "redact_pii", True))

        streams_cfg = getattr(redis_cfg, "streams", None)
        self.stream_training: str = getattr(streams_cfg, "training_data", "training_data") if streams_cfg else "training_data"
        self.stream_infer: str = getattr(streams_cfg, "inference_results", "inference_results") if streams_cfg else "inference_results"
        self.maxlen: int = int(getattr(streams_cfg, "max_stream_length", 1_000_000) if streams_cfg else 1_000_000)

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
            host = getattr(redis_cfg, "host", "127.0.0.1")
            port = int(getattr(redis_cfg, "port", 6379))
            db = int(getattr(redis_cfg, "db", 0))
            password = os.getenv("REDIS_PASSWORD") or getattr(redis_cfg, "password", None)
            socket_timeout = float(getattr(redis_cfg, "socket_timeout", 0.1))
            connect_timeout = float(getattr(redis_cfg, "socket_connect_timeout", 0.05))
            retry_on_timeout = bool(getattr(redis_cfg, "retry_on_timeout", True))
            health_check_interval = int(getattr(redis_cfg, "health_check_interval", 15))

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

    def _xadd(self, stream: str, record: Dict[str, Any]) -> None:
        if not self.enabled or not self._client:
            return
        fields = {
            k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else ("" if v is None else str(v))
            for k, v in record.items()
        }
        try:
            self._client.xadd(stream, fields, maxlen=self.maxlen, approximate=True)
        except Exception as e:
            # Best-effort: không ảnh hưởng luồng chính
            print("$$$$$$$$$$$$$$$$$", e)

    def emit_training(self, payload: Dict[str, Any]) -> None:
        self.enabled = 1
        print("CCCCCCCCCCCCCCCCCCC", self.training_rate)
        if not self.enabled or self.training_rate <= 0:
            return
        if random.random() > self.training_rate:
            return
        self._xadd(self.stream_training, payload)

    def emit_inference(self, payload: Dict[str, Any]) -> None:
        print("AAAAAAAAAAAXXXXXXXXXXXXXXX", self.infer_rate)
        print("AAAAAAAAAAA", self.enabled)
        self.enabled = 1
        if not self.enabled or self.infer_rate <= 0:
            print("BBBBBBBBBB")
            return
        if random.random() > self.infer_rate:
            print(111111111111111)
            return

        self._xadd(self.stream_infer, payload)


_EMITTER: Optional[RedisEmitter] = None


def get_emitter() -> RedisEmitter:
    global _EMITTER
    if _EMITTER is None:
        _EMITTER = RedisEmitter()
    return _EMITTER