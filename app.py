# python
import asyncio
import os
import threading

import uvicorn
from fastapi import FastAPI
from ray import serve

from adserving.src.api import api_dependencies
from adserving.src.api.core_endpoints import router as core_router
from adserving.src.api.model_endpoints import router as model_router
from adserving.src.api.prediction_endpoint import router as prediction_router
from adserving.src.audit.runner import start_audit_workers
from adserving.src.config.config import get_config
from adserving.src.datahandler.data_handler import DataHandler
from adserving.src.deployment.preloaded_model_server import PreloadedModelServer
from adserving.src.utils.exception_handlers import setup_exception_handlers
from adserving.src.utils.logger import get_logger

logger = get_logger()
app = FastAPI(title="Preloaded MLflow Serving", version="1.0.0")
setup_exception_handlers(app)

# Cờ đảm bảo không khởi động workers nhiều lần trong cùng process
_AUDIT_WORKERS_STARTED = False
_AUDIT_LOCK = threading.Lock()


def _start_audit_workers_once() -> None:
    """Khởi động audit workers một lần duy nhất."""
    global _AUDIT_WORKERS_STARTED
    with _AUDIT_LOCK:
        if _AUDIT_WORKERS_STARTED:
            return
        if os.getenv("AUDIT_WORKERS_ENABLED", "true").lower() not in (
            "1",
            "true",
            "yes",
            "y",
        ):
            logger.info("Audit workers are disabled by AUDIT_WORKERS_ENABLED.")
            return
        try:
            # Khởi động Ray Actors tiêu thụ Redis Streams và ghi PostgreSQL (non-blocking)
            start_audit_workers()
            _AUDIT_WORKERS_STARTED = True
            logger.info("Audit workers started on app startup.")
        except Exception as e:
            # Không để lỗi worker ảnh hưởng quá trình khởi động app
            logger.warning(f"Failed to start audit workers: {e}")


@app.on_event("startup")
async def on_startup() -> None:
    """Xử lý sự kiện khởi động ứng dụng."""
    cfg = get_config()
    api_dependencies.update_service_readiness(
        ready=False, models_loaded=0, models_failed=0, initialization_complete=False
    )

    try:
        serve.start(
            detached=True,
            http_options={
                "host": cfg.http.host,
                "port": cfg.http.port,
            },
        )
    except RuntimeError as e:
        if "already started" not in str(e):
            raise

    input_handler = DataHandler()
    api_dependencies.initialize_dependencies(handler=input_handler)

    api_prefix = (
        (cfg.api.prefix if getattr(cfg, "api", None) else None) or cfg.api_prefix or ""
    )
    app.include_router(prediction_router, prefix=api_prefix, tags=["Prediction"])
    app.include_router(model_router, prefix=api_prefix, tags=["Model"])
    app.include_router(core_router, prefix=api_prefix, tags=["Core"])

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", cfg.mlflow.tracking_uri)
    app_graph = PreloadedModelServer.bind(
        tracking_uri=tracking_uri,
        max_load_concurrency=cfg.preload.max_load_concurrency,
        watcher_interval_seconds=cfg.watcher.interval_seconds,
        sanity_check_enabled=cfg.watcher.sanity_check_enabled,
        sanity_inputs=cfg.watcher.sanity_inputs,
    )
    serve.run(app_graph, name="preloaded_model_server")

    handle = serve.get_app_handle("preloaded_model_server")
    logger.info("Waiting for preloaded_model_server to be ready...")

    # Khởi chạy audit workers ở nền, không chặn luồng khởi tạo model server
    threading.Thread(target=_start_audit_workers_once, daemon=True).start()

    ready = False
    models_loaded = 0
    for _ in range(600):
        try:
            if await handle.ready.remote():
                names = await handle.list_models.remote()
                models_loaded = len(names or [])
                ready = True
                break
        except Exception:
            pass
        await asyncio.sleep(0.5)

    if not ready:
        api_dependencies.update_service_readiness(
            ready=False,
            models_loaded=models_loaded,
            models_failed=0,
            initialization_complete=False,
        )
        raise RuntimeError("Model server not ready in time.")

    api_dependencies.update_service_readiness(
        ready=True,
        models_loaded=models_loaded,
        models_failed=0,
        initialization_complete=True,
    )

    logger.info("Application startup completed successfully")


def main() -> None:
    """Hàm main để khởi chạy ứng dụng."""
    # Cài đặt cleanup handlers ngay từ đầu
    # install_termination_handlers()

    try:
        cfg = get_config()
        host = (cfg.api.host if getattr(cfg, "api", None) else None) or cfg.api_host
        port = (cfg.api.port if getattr(cfg, "api", None) else None) or cfg.api_port

        logger.info(f"Starting application on {host}:{port}")
        # logger.info(f"Ray cleanup status: {get_cleanup_status()}")

        uvicorn.run(
            "app:app",  # Updated module reference
            host=host,
            port=port,
            reload=True,
            workers=1,
            log_level="info",
        )
    except Exception as e:
        logger.error(f"Lỗi trong main: {e}")
        raise
    finally:
        # Đảm bảo dọn dẹp khi main kết thúc
        logger.info("Main function ending, performing final cleanup")


if __name__ == "__main__":
    main()
