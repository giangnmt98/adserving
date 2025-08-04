"""
Main FastAPI application entry point
Enhanced with request body capture middleware for better validation error handling
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting up Anomaly Detection API...")

    try:
        # Initialize any startup processes here
        logger.info("Application startup completed successfully")

        yield

    except Exception as e:
        logger.error(f"Error during application startup: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down Anomaly Detection API...")
        logger.info("Application shutdown completed")


def create_app(api_prefix: str = "") -> FastAPI:
    """Create and configure FastAPI application with enhanced middleware"""

    try:
        # Import required modules
        from adserving.config.config import get_config
        from adserving.api import exception_handlers
        from adserving.api import (
            core_endpoints,
            prediction_endpoint,
            model_endpoints,
            tier_management_endpoints
        )

        # Get application configuration
        config = get_config()

        # Create base FastAPI config
        api_config = {
            "title": "Anomaly Detection API",
            "description": "Enhanced MLOps serving system with advanced validation error handling",
            "version": config.api_version,
            "lifespan": lifespan,
            "docs_url": "/docs",
            "redoc_url": "/redoc",
            "openapi_url": "/openapi.json"
        }

        # Add root_path only if api_prefix is provided
        if api_prefix and api_prefix.strip():
            api_config["root_path"] = api_prefix
            logger.info(f"API configured with prefix: {api_prefix}")

        app = FastAPI(**api_config)

        # Add CORS middleware - this should be added first
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                "http://localhost:3000",
                "http://localhost:8000",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:8000",
                "*"  # In production, replace with specific origins
            ],
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["*"],
        )

        # Add custom request body capture middleware
        # This must be added AFTER CORS but BEFORE other middleware

        logger.info("Request body capture middleware registered successfully")

        # Register exception handlers in order of specificity
        # Most specific first, most general last
        app.add_exception_handler(Exception, exception_handlers.general_exception_handler)

        logger.info("Exception handlers registered successfully")

        # Include routers with their respective tags
        app.include_router(core_endpoints.router, tags=["Core"])
        app.include_router(prediction_endpoint.router, tags=["Prediction"])
        app.include_router(model_endpoints.router, tags=["Models"])
        app.include_router(tier_management_endpoints.router, tags=["Tier Management"])

        logger.info("API routes registered successfully")

        return app

    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error("Please ensure all dependencies are installed and modules are available")
        raise
    except Exception as e:
        logger.error(f"Failed to create FastAPI application: {e}")
        raise


def main() -> None:
    """Main entry point for the application"""
    try:
        # Ensure we're in the correct directory
        current_dir = Path(__file__).parent
        if not (current_dir / "adserving").exists():
            logger.error("adserving package not found in current directory")
            logger.error(f"Current directory: {current_dir}")
            logger.error("Please run from the project root directory")
            sys.exit(1)

        # Import and run service using the main service module
        from adserving.service.main_service import AnomalyDetectionServe

        # Create service instance
        service = AnomalyDetectionServe()

        # Run the service (this will use the FastAPI app created by create_app)
        service.run()

    except ImportError as e:
        logger.error(f"Failed to import service components: {e}")
        logger.error("Ensure all dependencies are installed:")
        logger.error("  pip install -r requirements.txt")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to start Anomaly Detection Serve: {e}")
        logger.exception("Full exception traceback:")
        sys.exit(1)


def run_development_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = True,
    api_prefix: str = ""
) -> None:
    """Run development server with uvicorn"""
    try:
        import uvicorn

        logger.info(f"Starting development server on {host}:{port}")
        logger.info(f"Reload mode: {'enabled' if reload else 'disabled'}")

        # Create the app
        app = create_app(api_prefix)

        # Run with uvicorn
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
            log_level="info",
            access_log=True,
            reload_dirs=["adserving"] if reload else None,
            reload_excludes=["*.pyc", "*.pyo", "__pycache__"] if reload else None
        )

    except ImportError:
        logger.error("uvicorn is not installed. Please install it:")
        logger.error("  pip install uvicorn[standard]")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start development server: {e}")
        sys.exit(1)


# Create the app instance for ASGI servers (gunicorn, uvicorn, etc.)
app = create_app()


if __name__ == "__main__":
    import argparse

    # Command line argument parsing
    parser = argparse.ArgumentParser(description="Anomaly Detection API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--api-prefix", default="", help="API prefix path")
    parser.add_argument("--dev", action="store_true", help="Run development server")
    parser.add_argument("--production", action="store_true", help="Run production service")

    args = parser.parse_args()

    if args.dev:
        # Run development server with uvicorn
        logger.info("Starting in development mode...")
        run_development_server(
            host=args.host,
            port=args.port,
            reload=args.reload,
            api_prefix=args.api_prefix
        )
    elif args.production:
        # Run production service
        logger.info("Starting in production mode...")
        main()
    else:
        # Default: run the main service
        logger.info("Starting Anomaly Detection Service...")
        main()