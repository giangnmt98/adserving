"""
Ray Serve initialization and management
Handles Ray Serve startup with proper configuration
"""

import os
import logging
import ray
from ray import serve
from ..config.config import Config
from .port_manager import PortManager
from adserving.src.utils.logger import get_logger

logger = get_logger()


class RayManager:
    """Manages Ray Serve initialization"""

    def __init__(self) -> None:
        """Initialize Ray Serve manager."""
        self.port_manager = PortManager()

    def initialize(self, config: Config) -> None:
        """Initialize Ray server and Ray Serve with custom configuration."""
        logger.info("Initializing Ray server and Ray Serve...")

        try:
            # Step 1: Initialize Ray server first
            self._initialize_ray_server(config)
            
            # Step 2: Configure Ray logging
            self._configure_ray_logging(config)

            # Step 3: Find available port for Ray Serve
            api_port = getattr(config, "api_port", 8000)
            ray_port = self.port_manager.find_available_port(api_port + 1, "127.0.0.1")

            logger.info(f"Ray Serve HTTP proxy using port {ray_port}")

            # Step 4: Start Ray Serve (after Ray server is running)
            serve.start(http_options={"host": "127.0.0.1", "port": ray_port})

            logger.info(f"Ray server and Ray Serve initialized successfully on port {ray_port}")

        except Exception as e:
            logger.warning(f"Ray server/Serve init failed/already running: {e}")
            logger.info("Continuing with existing Ray instance...")
    
    def _initialize_ray_server(self, config: Config) -> None:
        """Initialize Ray server with configuration parameters."""
        try:
            # Check if Ray is already initialized
            if ray.is_initialized():
                logger.info("Ray server already initialized")
                return
            
            # Get Ray configuration from config
            ray_config = getattr(config, 'ray', None)
            if not ray_config:
                logger.warning("No Ray configuration found, using defaults")
                ray.init(ignore_reinit_error=True)
                return
            
            # Extract Ray configuration parameters
            init_params = {}
            
            # Basic Ray parameters
            if hasattr(ray_config, 'address') and ray_config.address:
                init_params['address'] = ray_config.address
            
            if hasattr(ray_config, 'num_cpus') and ray_config.num_cpus:
                init_params['num_cpus'] = ray_config.num_cpus
                
            if hasattr(ray_config, 'num_gpus') and ray_config.num_gpus:
                init_params['num_gpus'] = ray_config.num_gpus
                
            if hasattr(ray_config, 'object_store_memory') and ray_config.object_store_memory:
                # Convert object store memory to proper format
                object_store_memory = ray_config.object_store_memory
                # If it's a number, treat it as MB; if string, parse it
                if isinstance(object_store_memory, (int, float)):
                    init_params['object_store_memory'] = int(object_store_memory * 1024 * 1024)  # Convert MB to bytes
                elif isinstance(object_store_memory, str):
                    # Handle string formats like "8000MB", "8GB", etc.
                    import re
                    match = re.match(r'(\d+)\s*(GB|MB|B)?', str(object_store_memory).upper())
                    if match:
                        value, unit = match.groups()
                        value = int(value)
                        if unit == 'GB':
                            init_params['object_store_memory'] = value * 1024 * 1024 * 1024
                        elif unit == 'MB' or unit is None:
                            init_params['object_store_memory'] = value * 1024 * 1024
                        else:  # bytes
                            init_params['object_store_memory'] = value
                    else:
                        # Fallback: treat as bytes
                        init_params['object_store_memory'] = int(object_store_memory)
                
            if hasattr(ray_config, 'runtime_env') and ray_config.runtime_env:
                init_params['runtime_env'] = ray_config.runtime_env
            
            # Dashboard configuration
            if hasattr(ray_config, 'dashboard_host') and ray_config.dashboard_host:
                init_params['dashboard_host'] = ray_config.dashboard_host
                
            if hasattr(ray_config, 'dashboard_port') and ray_config.dashboard_port:
                init_params['dashboard_port'] = ray_config.dashboard_port
            
            # Always allow reinit
            init_params['ignore_reinit_error'] = True
            
            logger.info(f"Starting Ray server with parameters: {init_params}")
            
            # Initialize Ray server
            ray.init(**init_params)
            
            logger.info("Ray server initialized successfully")
            logger.info(f"Ray dashboard available at: http://{init_params.get('dashboard_host', '127.0.0.1')}:{init_params.get('dashboard_port', 8265)}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Ray server: {e}")
            # Fallback to basic Ray init
            logger.info("Falling back to basic Ray initialization...")
            ray.init(ignore_reinit_error=True)

    def _configure_ray_logging(self, config: Config) -> None:
        """Configure Ray logging levels."""
        ray_log_level = config.ray.log_level
        logger.info(f"Setting Ray log level to: {ray_log_level}")

        # Set environment variable
        os.environ["RAY_LOG_LEVEL"] = ray_log_level

        # Configure Ray loggers directly
        ray_logger = logging.getLogger("ray")
        ray_logger.setLevel(getattr(logging, ray_log_level.upper()))

        ray_serve_logger = logging.getLogger("ray.serve")
        ray_serve_logger.setLevel(getattr(logging, ray_log_level.upper()))
