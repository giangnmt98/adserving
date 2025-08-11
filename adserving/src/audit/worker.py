# Python
import json
import time
from typing import Any, Dict, List, Tuple

import ray

try:
    import redis
except Exception:  # pragma: no cover
        redis = None  # type: ignore

from sqlalchemy import text
from adserving.src.db.postgres import make_pg_engine
from adserving.src.config.config import get_config
from adserving.src.utils.logger import get_logger

logger = get_logger()


def _prepare_json(v: Any) -> str:
    return json.dumps(v) if v is not None else "null"


@ray.remote
class StreamConsumer:
    """Consumer đọc từ Redis Streams (XREADGROUP) và ghi vào PostgreSQL."""

    def __init__(self, stream: str, group: str, consumer_name: str, mode: str) -> None:
        self.mode = mode
        self.stream = stream
        self.group = group
        self.consumer_name = consumer_name

        cfg = get_config()
        rc = getattr(cfg, "redis", None)

        cons = getattr(rc, "consumer", None)
        self.batch_size = int(getattr(cons, "batch_size", 200) if cons else 200)
        self.block_ms = int(getattr(cons, "block_time_ms", 100) if cons else 100)

        if redis is None:
            raise RuntimeError("redis package not installed; cannot run StreamConsumer.")

        host = getattr(rc, "host", "127.0.0.1")
        port = int(getattr(rc, "port", 6379))
        db = int(getattr(rc, "db", 0))
        password = getattr(rc, "password", None)
        health_check_interval = int(getattr(rc, "health_check_interval", 15))

        self.r = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password or None,  # requirepass only
            decode_responses=True,
            health_check_interval=health_check_interval,
        )

        # Tạo group nếu chưa có
        try:
            self.r.xgroup_create(self.stream, self.group, id="$", mkstream=True)
        except Exception as e:
            if "BUSYGROUP" not in str(e):
                raise

        self.engine = make_pg_engine()
        logger.info(
            f"StreamConsumer started: stream={self.stream}, group={self.group}, consumer={self.consumer_name}, mode={self.mode}"
        )

    def _parse(self, chunks) -> List[Tuple[str, Dict[str, Any]]]:
        res: List[Tuple[str, Dict[str, Any]]] = []
        for _, entries in chunks:
            for msg_id, fields in entries:
                try:
                    obj = {
                        k: (json.loads(v) if v and (v.startswith("{") or v.startswith("[")) else v)
                        for k, v in fields.items()
                    }
                except Exception:
                    obj = fields
                res.append((msg_id, obj))
        return res

    def _persist_training(self, batch: List[Tuple[str, Dict[str, Any]]]) -> None:
        if not batch:
            return
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO queue_status(queue_name, message_id, status)
                    VALUES (:q, :m, 'processing')
                    ON CONFLICT DO NOTHING
                    """
                ),
                [{"q": self.stream, "m": mid} for mid, _ in batch],
            )
            conn.execute(
                text(
                    """
                    INSERT INTO training_events(
                        request_id, ma_don_vi, ma_bao_cao, ky_du_lieu, features,
                        consent_flags, user_id_hash, label, source, schema_version, message_id
                    )
                    VALUES (
                        :request_id, :ma_don_vi, :ma_bao_cao, :ky_du_lieu, CAST(:features AS JSONB),
                        CAST(:consent_flags AS JSONB), :user_id_hash, CAST(:label AS JSONB),
                        :source, :schema_version, :message_id
                    )
                    ON CONFLICT (message_id) DO NOTHING
                    """
                ),
                [
                    {
                        "request_id": r.get("request_id"),
                        "ma_don_vi": r.get("ma_don_vi"),
                        "ma_bao_cao": r.get("ma_bao_cao"),
                        "ky_du_lieu": r.get("ky_du_lieu"),
                        "features": _prepare_json(r.get("features")),
                        "consent_flags": _prepare_json(r.get("consent_flags")),
                        "user_id_hash": r.get("user_id_hash"),
                        "label": _prepare_json(r.get("label")),
                        "source": r.get("source", "online_serving"),
                        "schema_version": r.get("schema_version", "v1"),
                        "message_id": mid,
                    }
                    for mid, r in batch
                ],
            )
            conn.execute(
                text(
                    """
                    UPDATE queue_status
                    SET status='completed', processing_completed_at=NOW(), updated_at=NOW()
                    WHERE queue_name=:q AND message_id = ANY(:mids)
                    """
                ),
                {"q": self.stream, "mids": [mid for mid, _ in batch]},
            )

    def _persist_inference(self, batch: List[Tuple[str, Dict[str, Any]]]) -> None:
        if not batch:
            return
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO queue_status(queue_name, message_id, status)
                    VALUES (:q, :m, 'processing')
                    ON CONFLICT DO NOTHING
                    """
                ),
                [{"q": self.stream, "m": mid} for mid, _ in batch],
            )
            conn.execute(
                text(
                    """
                    INSERT INTO infer_result(
                        model_name, request_id, ma_tieu_chi, fld_code, is_anomaly,
                        anomaly_score, threshold, processing_time, model_version, status
                    )
                    VALUES (
                        :model_name, :request_id, :ma_tieu_chi, :fld_code, :is_anomaly,
                        :anomaly_score, :threshold, :processing_time, :model_version, :status
                    )
                    """
                ),
                [
                    {
                        "model_name": r.get("model_name"),
                        "request_id": r.get("request_id"),
                        "ma_tieu_chi": (r.get("summary") or {}).get("ma_tieu_chi"),
                        "fld_code": (r.get("summary") or {}).get("fld_code"),
                        "is_anomaly": (r.get("summary") or {}).get("is_anomaly"),
                        "anomaly_score": (r.get("summary") or {}).get("anomaly_score"),
                        "threshold": r.get("threshold"),
                        "processing_time": int(float(r.get("processing_time") or 0)),
                        "model_version": r.get("model_version"),
                        "status": r.get("status", "success"),
                    }
                    for _, r in batch
                ],
            )
            conn.execute(
                text(
                    """
                    UPDATE queue_status
                    SET status='completed', processing_completed_at=NOW(), updated_at=NOW()
                    WHERE queue_name=:q AND message_id = ANY(:mids)
                    """
                ),
                {"q": self.stream, "mids": [mid for mid, _ in batch]},
            )

    def run_forever(self) -> None:
        while True:
            try:
                chunks = self.r.xreadgroup(
                    groupname=self.group,
                    consumername=self.consumer_name,
                    streams={self.stream: ">"},
                    count=self.batch_size,
                    block=self.block_ms,
                )
                if not chunks:
                    continue
                batch = self._parse(chunks)
                if not batch:
                    continue

                if self.mode == "training":
                    self._persist_training(batch)
                else:
                    self._persist_inference(batch)

                for mid, _ in batch:
                    try:
                        self.r.xack(self.stream, self.group, mid)
                    except Exception:
                        pass

            except Exception as e:
                logger.warning(f"StreamConsumer loop error ({self.consumer_name}): {e}")
                time.sleep(0.1)