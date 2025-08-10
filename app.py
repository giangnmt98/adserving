# Python
import asyncio
import os

import uvicorn
from fastapi import FastAPI
from ray import serve

from adserving.src.deployment.preloaded_model_server import PreloadedModelServer
from adserving.src.utils.logger import get_logger

from adserving.src.config.config_manager import get_config
from adserving.src.api.prediction_endpoint import router as prediction_router
from adserving.src.api.model_endpoints import router as model_router
from adserving.src.api.core_endpoints import router as core_router
from adserving.src.api import api_dependencies
from adserving.src.utils.exception_handlers import setup_exception_handlers
from adserving.src.datahandler.data_handler import DataHandler


logger = get_logger()
app = FastAPI(title="Preloaded MLflow Serving", version="1.0.0")
setup_exception_handlers(app)


@app.on_event("startup")
async def on_startup() -> None:
    cfg = get_config()

    api_dependencies.update_service_readiness(
        ready=False, models_loaded=0, models_failed=0, initialization_complete=False
    )

    try:
        serve.start(
            detached=True,
            http_options={
                "host": cfg.serve.http.host,
                "port": cfg.serve.http.port,
            },
        )
    except RuntimeError as e:
        if "already started" not in str(e):
            raise

    input_handler = DataHandler()
    api_dependencies.initialize_dependencies(handler=input_handler)

    api_prefix = (cfg.api.prefix if getattr(cfg, "api", None) else None) or cfg.api_prefix or ""
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



@app.on_event("shutdown")
async def on_shutdown() -> None:
    try:
        sys = globals().get('RUNTIME_ASYNC_SYS')
        if sys:
            await sys.close()
    except Exception:
        pass


def main() -> None:
    cfg = get_config()
    host = (cfg.api.host if getattr(cfg, "api", None) else None) or cfg.api_host
    port = (cfg.api.port if getattr(cfg, "api", None) else None) or cfg.api_port
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=False,
        workers=1,
        log_level="info",
    )