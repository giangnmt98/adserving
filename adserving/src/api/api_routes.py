"""
FastAPI Application Factory
"""

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from adserving.src.config.config_manager import get_config
from adserving.src.utils.logger import get_logger

from . import (
    core_endpoints,
    exception_handlers,
    model_endpoints,
    prediction_endpoint,
    tier_management_endpoints,
)

logger = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global service_start_time
    service_start_time = datetime.now()

    logger.info("Anomaly Detection Serve API starting up...")
    yield
    logger.info("Anomaly Detection Serve API shutting down...")


def create_app(api_prefix: str = "") -> FastAPI:
    """Create and configure FastAPI application"""
    # Create base FastAPI config
    api_config = {
        "title": "Anomaly Detection API",
        "description": ("Enhanced MLOps serving system for hundreds of models"),
        "version": get_config().api_version,
        "lifespan": lifespan,
    }

    # Add root_path only if api_prefix is provided
    if api_prefix and api_prefix.strip():
        api_config["root_path"] = api_prefix

    app = FastAPI(**api_config)

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_exception_handler(Exception, exception_handlers.general_exception_handler)

    # Include routers
    app.include_router(core_endpoints.router, tags=["Core"])
    app.include_router(prediction_endpoint.router, tags=["Prediction"])
    app.include_router(model_endpoints.router, tags=["Models"])
    app.include_router(tier_management_endpoints.router, tags=["Tier Management"])

    return app


# Global app variable
app: Optional[FastAPI] = None


def initialize_app_with_config(api_prefix: str = "") -> FastAPI:
    """Initialize app with configuration"""
    global app
    app = create_app(api_prefix)
    return app
