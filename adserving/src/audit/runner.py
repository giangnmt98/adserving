# Python
import ray
from adserving.src.config.config import get_config
from adserving.src.audit.worker import StreamConsumer
from adserving.src.utils.logger import get_logger

logger = get_logger()


def start_audit_workers() -> None:
    cfg = get_config()
    raw = getattr(cfg, "raw", {}) if hasattr(cfg, "raw") else {}

    rc = raw.get("redis", {}) or {}
    streams = rc.get("streams", {}) or {}
    group = (rc.get("consumer", {}) or {}).get("group_name", "adserving_consumers")

    training_stream = streams.get("training_data", "training_data")
    inference_stream = streams.get("inference_results", "inference_results")

    rworkers = raw.get("ray_workers", {}) or {}
    tw = int(rworkers.get("training_workers", 2))
    iw = int(rworkers.get("inference_workers", 2))

    # Khởi tạo Ray nếu chưa
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    actors = []
    for i in range(tw):
        a = StreamConsumer.remote(
            stream=training_stream,
            group=group,
            consumer_name=f"trainw-{i}",
            mode="training",
        )
        actors.append(a)

    for i in range(iw):
        a = StreamConsumer.remote(
            stream=inference_stream,
            group=group,
            consumer_name=f"inferw-{i}",
            mode="inference",
        )
        actors.append(a)

    # Chạy nền: gửi các task không blocking (không ray.get)
    for a in actors:
        a.run_forever.remote()

    logger.info(
        f"Audit workers started: training={tw} on {training_stream}, inference={iw} on {inference_stream}, group={group}"
    )