# Python
import json
import time
from typing import Any, Dict, List, Tuple

import ray
import redis
from sqlalchemy import text

from adserving.src.config.config import get_config
from adserving.src.db.postgres import make_pg_engine
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
        rc = cfg.redis
        self.audit_cfg = cfg.audit
        cons = getattr(rc, "consumer", None)
        self.batch_size = int(getattr(cons, "batch_size", 200) if cons else 200)
        self.block_ms = int(getattr(cons, "block_time_ms", 100) if cons else 100)

        if redis is None:
            raise RuntimeError(
                "redis package not installed; cannot run StreamConsumer."
            )

        host = rc.host or "127.0.0.1"
        port = rc.port or 6379
        db = rc.db or 0
        password = rc.password or None
        health_check_interval = rc.health_check_interval or 15

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
            f"StreamConsumer started: stream={self.stream}, "
            f"group={self.group}, consumer={self.consumer_name}, mode={self.mode}"
        )

    def _parse(self, chunks) -> List[Tuple[str, Dict[str, Any]]]:
        res: List[Tuple[str, Dict[str, Any]]] = []
        for _, entries in chunks:
            for msg_id, fields in entries:
                try:
                    obj = {
                        k: (
                            json.loads(v)
                            if v and (v.startswith("{") or v.startswith("["))
                            else v
                        )
                        for k, v in fields.items()
                    }
                except Exception:
                    obj = fields
                res.append((msg_id, obj))
        return res

    def _persist_training(self, batch: List[Tuple[str, Dict[str, Any]]]) -> None:
        if not batch:
            return
        time_now = str(time.time() * 1000)
        with self.engine.begin() as conn:
            # Queue status vẫn giữ nguyên
            conn.execute(
                text(
                    """
                    INSERT INTO queue_status(queue_name, message_id, status)
                    VALUES (:q, :m, 'processing') ON CONFLICT DO NOTHING
                    """
                ),
                [{"q": self.stream, "m": mid} for mid, _ in batch],
            )

            # UPSERT cho training events
            conn.execute(
                text(
                    f"""
                    INSERT INTO {self.audit_cfg.training_database_table}(
                        ma_don_vi, ma_bao_cao,
                        ky_du_lieu, ma_tieu_chi, fld_code, gia_tri,
                        created_at, updated_at
                    )
                    VALUES (
                        :ma_don_vi, :ma_bao_cao,
                        :ky_du_lieu, :ma_tieu_chi, :fld_code, :gia_tri,
                        {time_now},
                        {time_now}

                    )
                    ON CONFLICT (ma_don_vi, ma_bao_cao, ky_du_lieu,
                     ma_tieu_chi, fld_code)
                    DO UPDATE SET
                        gia_tri = EXCLUDED.gia_tri,
                        updated_at = {time_now}
                    """
                ),
                [
                    {
                        "ma_don_vi": record_data.get("ma_don_vi"),
                        "ma_bao_cao": record_data.get("ma_bao_cao"),
                        "ky_du_lieu": record_data.get("ky_du_lieu"),
                        "ma_tieu_chi": record_data.get("ma_tieu_chi"),
                        "fld_code": record_data.get("fld_code"),
                        "gia_tri": (
                            float(record_data.get("gia_tri", 0))
                            if record_data.get("gia_tri")
                            else None
                        ),
                    }
                    for _, record_data in batch
                ],
            )

            # Update queue status
            conn.execute(
                text(
                    f"""
                    UPDATE queue_status
                    SET status='completed',
                        processing_completed_at={time_now},
                        updated_at={time_now}
                    WHERE queue_name = :q
                      AND message_id = ANY (:mids)
                    """
                ),
                {"q": self.stream, "mids": [mid for mid, _ in batch]},
            )

    def _persist_inference(self, batch: List[Tuple[str, Dict[str, Any]]]) -> None:
        if not batch:
            return
        time_now = str(time.time() * 1000)
        with self.engine.begin() as conn:
            # Queue status
            conn.execute(
                text(
                    """
                    INSERT INTO queue_status(queue_name, message_id, status)
                    VALUES (:q, :m, 'processing') ON CONFLICT DO NOTHING
                    """
                ),
                [{"q": self.stream, "m": mid} for mid, _ in batch],
            )

            # UPSERT cho inference results
            conn.execute(
                text(
                    f"""
                    INSERT INTO {self.audit_cfg.inference_database_table}(
                    model_name, request_id, timestamp, ma_tieu_chi,
                    fld_code,is_anomaly,anomaly_score,
                    anomaly_threshold, processing_time,
                     model_version
                    )
                    VALUES (
                        :model_name, :request_id, :timestamp, :ma_tieu_chi,
                        :fld_code, :is_anomaly, :anomaly_score,
                        :anomaly_threshold, :processing_time,
                        :model_version
                    )
                    """
                ),
                [
                    {
                        "model_name": r.get("model_name"),
                        "request_id": r.get("request_id"),
                        "timestamp": r.get("timestamp"),
                        "ma_tieu_chi": r.get("ma_tieu_chi"),
                        "fld_code": r.get("fld_code"),
                        "is_anomaly": r.get("is_anomaly"),
                        "anomaly_score": r.get("anomaly_score"),
                        "anomaly_threshold": r.get("anomaly_threshold"),
                        "processing_time": r.get("processing_time"),
                        "model_version": r.get("model_version"),
                    }
                    for _, r in batch
                ],
            )

            # Update queue status
            conn.execute(
                text(
                    f"""
                    UPDATE queue_status
                    SET status='completed',
                        processing_completed_at={time_now},
                        updated_at={time_now}
                    WHERE queue_name = :q
                      AND message_id = ANY (:mids)
                    """
                ),
                {"q": self.stream, "mids": [mid for mid, _ in batch]},
            )

    def run_forever(self) -> None:
        """
        Continuously processes messages from Redis stream and persists to a database.
        """
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
                try:
                    self.r.xack(self.stream, self.group, *[mid for mid, _ in batch])
                except Exception:
                    pass
            except Exception as e:
                logger.warning(f"StreamConsumer loop error ({self.consumer_name}): {e}")
                time.sleep(0.1)
