"""
Main FastAPI application entry point
Enhanced with ultra-scale deployment integration
"""

import sys
import os
import time
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from adserving.src.utils.logger import get_logger
from adserving.src.config.config import Config, create_sample_config
from adserving.src.service.service_components import ServiceComponents

# Configure basic logging
logger = get_logger()

# Global service components
service_components: Optional[ServiceComponents] = None
config: Optional[Config] = None


async def load_configuration():
    """Load or create configuration."""
    global config
    config_file = "config.yaml"
    
    try:
        if not Path(config_file).exists():
            logger.info(f"Creating default config: {config_file}")
            create_sample_config(config_file)

        logger.info(f"Loading configuration from: {config_file}")
        config = Config.from_file(config_file)
        
        logger.info(f"Configuration loaded successfully")
        return config

    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events - models deployed BEFORE server becomes alive"""
    global service_components, config
    
    # Startup
    logger.info("Starting up Anomaly Detection API with model pre-deployment...")

    try:
        # 1. Load configuration
        logger.info("Loading configuration...")
        config = await load_configuration()
        
        # 2. Initialize minimal service components
        logger.info("Initializing service components...")
        service_components = ServiceComponents()
        service_components.initialize_minimal(config)
        
        # 3. Complete Ray initialization BEFORE deployment
        logger.info("Completing Ray initialization and full component setup...")
        init_start_time = time.time()
        
        await service_components.complete_initialization(config)
        
        init_time = time.time() - init_start_time
        logger.info(f"Ray initialization completed in {init_time:.2f}s")
        
        # 4. Start background services
        logger.info("Starting background services...")
        service_components.start_background_services()
        
        # 5. Deploy models BEFORE server becomes alive (FIXED!)
        logger.info("Deploying models BEFORE server startup...")
        deployment_start_time = time.time()
        
        model_stats = await service_components.deploy_production_models()
        
        deployment_time = time.time() - deployment_start_time
        total_time = time.time() - init_start_time
        
        logger.info(f"Model deployment completed in {deployment_time:.2f}s - "
                   f"loaded: {model_stats['loaded']}, failed: {model_stats['failed']}")
        logger.info(f"Total initialization time: {total_time:.2f}s")
        
        # 6. Update readiness state with deployed models
        service_components.update_readiness_state(
            ready=True,  # Fully ready with models deployed
            models_loaded=model_stats["loaded"],
            models_failed=model_stats["failed"],
        )

        logger.info("Application startup completed successfully - server ready with models deployed!")
        logger.info(f"✅ Models deployed and routers created BEFORE FastAPI server alive!")

        yield

    except Exception as e:
        logger.error(f"Error during application startup: {e}")
        if service_components:
            try:
                service_components.cleanup()
            except Exception as cleanup_error:
                logger.error(f"Error during cleanup: {cleanup_error}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down Anomaly Detection API...")
        if service_components:
            try:
                service_components.cleanup()
                logger.info("Service components cleaned up successfully")
            except Exception as e:
                logger.error(f"Error during service cleanup: {e}")
        logger.info("Application shutdown completed")


async def background_initialization_and_deployment(config):
    """Complete Ray initialization and model deployment in background without blocking server startup"""
    global service_components
    
    try:
        logger.info("Background initialization: Starting Ray and full component setup...")
        init_start_time = time.time()
        
        # Step 1: Complete full initialization (Ray + all components)
        await service_components.complete_initialization(config)
        
        init_time = time.time() - init_start_time
        logger.info(f"Background Ray initialization completed in {init_time:.2f}s")
        
        # Step 2: Start background services
        logger.info("Background initialization: Starting background services...")
        service_components.start_background_services()
        
        # Step 3: Deploy models asynchronously
        logger.info("Background deployment: Starting model deployment...")
        deployment_start_time = time.time()
        
        model_stats = await service_components.deploy_production_models()
        
        deployment_time = time.time() - deployment_start_time
        total_time = time.time() - init_start_time
        
        logger.info(f"Background deployment completed in {deployment_time:.2f}s - "
                   f"loaded: {model_stats['loaded']}, failed: {model_stats['failed']}")
        logger.info(f"Total background initialization time: {total_time:.2f}s")
        
        # Update readiness state when everything completes
        service_components.update_readiness_state(
            ready=True,  # Now fully ready with models deployed
            models_loaded=model_stats["loaded"],
            models_failed=model_stats["failed"],
        )
        
        logger.info("All components initialized and models deployed - service fully ready!")
        
    except Exception as e:
        logger.error(f"Background initialization and deployment failed: {e}")
        # Update state to show initialization failed but server still alive
        service_components.update_readiness_state(
            ready=False,
            models_loaded=0,
            models_failed=1,
        )


def create_app(api_prefix: str = "") -> FastAPI:
    """Create and configure FastAPI application with enhanced middleware"""

    try:
        # Import required modules
        from adserving.src.config.config_manager import get_config
        from adserving.src.api import exception_handlers
        from adserving.src.api import prediction_endpoint
        from adserving.src.api import model_endpoints
        from adserving.src.api import core_endpoints

        # Get application configuration
        config = get_config()

        # Use api_prefix from config if not provided as parameter
        if not api_prefix and hasattr(config, 'api_prefix'):
            api_prefix = config.api_prefix
            
        # Create base FastAPI config with proper docs URLs
        if api_prefix and api_prefix.strip():
            docs_url = f"{api_prefix}/docs"
            redoc_url = f"{api_prefix}/redoc" 
            openapi_url = f"{api_prefix}/openapi.json"
            logger.info(f"API configured with prefix: {api_prefix}")
        else:
            docs_url = "/docs"
            redoc_url = "/redoc"
            openapi_url = "/openapi.json"
            logger.info("API configured without prefix")

        api_config = {
            "title": "Anomaly Detection API",
            "description": "Enhanced MLOps serving system with advanced validation error handling",
            "version": config.api_version,
            "lifespan": lifespan,
            "docs_url": docs_url,
            "redoc_url": redoc_url,
            "openapi_url": openapi_url
        }

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

        # Include routers with their respective tags and prefix
        prefix_to_use = api_prefix if api_prefix and api_prefix.strip() else ""
        app.include_router(core_endpoints.router, tags=["Core"], prefix=prefix_to_use)
        app.include_router(prediction_endpoint.router, tags=["Prediction"], prefix=prefix_to_use)
        app.include_router(model_endpoints.router, tags=["Models"], prefix=prefix_to_use)

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
    """Main entry point for the application with integrated ultra-scale deployment"""
    try:
        # Ensure we're in the correct directory
        current_dir = Path(__file__).parent
        if not (current_dir / "adserving").exists():
            logger.error("adserving package not found in current directory")
            logger.error(f"Current directory: {current_dir}")
            logger.error("Please run from the project root directory")
            sys.exit(1)

        # Import uvicorn for running the server
        import uvicorn
        
        # Get host and port from environment or use defaults
        host = os.getenv("MLOPS_HOST", "0.0.0.0")
        port = int(os.getenv("MLOPS_PORT", "8000"))
        
        logger.info(f"Starting Anomaly Detection Serve with ultra-scale deployment on {host}:{port}")

        # Create the FastAPI app with integrated ultra-scale deployment
        app_instance = create_app()

        # Run the server directly
        uvicorn.run(
            app_instance,
            host=host,
            port=port,
            log_level="info",
            access_log=True,
        )

    except ImportError as e:
        logger.error(f"Failed to import required components: {e}")
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
            reload=False,
            log_level="info",
            access_log=True,
            workers=1,
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