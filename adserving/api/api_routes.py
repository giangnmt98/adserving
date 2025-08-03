"""
FastAPI Application Factory
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from adserving.config.config import get_config

from . import (
    core_endpoints,
    exception_handlers,
    model_endpoints,
    prediction_endpoint,
    tier_management_endpoints,
)

logger = logging.getLogger(__name__)


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

    # Register exception handlers
    app.add_exception_handler(HTTPException, exception_handlers.http_exception_handler)
    app.add_exception_handler(
        RequestValidationError, exception_handlers.validation_exception_handler
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
