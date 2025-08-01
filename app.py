import logging
import os
import sys
import signal
import time
import socket
import uvicorn
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import Anomaly Detection Serve components
try:
    from adserving.api.api_routes import app, initialize_dependencies, initialize_app_with_config, update_service_readiness
    from adserving.datahandler.data_handler import DataHandler
    from adserving.core.model_manager import ModelManager
    from adserving.router.model_router import ModelRouter
    from adserving.monitoring.model_monitor import ModelMonitor
    from adserving.deployment.pooled_deployment import PooledModelDeployment, PooledDeploymentConfig
    from adserving.config.config import Config, create_sample_config

    # Import MLflow for connection verification
    import mlflow
    from mlflow.tracking import MlflowClient
except ImportError as e:
    logger.error(f"Failed to import Anomaly Detection Serve components: {e}")
    logger.error("Make sure all dependencies are installed and you're in the correct directory")
    sys.exit(1)


class AnomalyDetectionServe:
    """Main service class for Anomaly Detection Serve"""

    def __init__(self):
        self.config = None
        self.model_manager = None
        self.model_router = None
        self.monitor = None
        self.input_handler = None
        self.deployment_manager = None

        # Service settings (will be updated from config)
        self.host = "0.0.0.0"
        self.port = 8000
        self.config_file = "config.yaml"
        self.models_directory = "individual_models"
        
        # Readiness tracking for cold start optimization
        self._models_loaded = False
        self._loaded_model_count = 0
        self._failed_model_count = 0
        self._service_ready = False

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("Anomaly Detection Serve Main initialized")

    def _is_port_available(self, host: str, port: int) -> bool:
        """Check if a port is available for binding"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                result = sock.bind((host, port))
                return True
        except OSError:
            return False

    def _find_available_port(self, start_port: int, host: str = "0.0.0.0") -> int:
        """Find an available port starting from start_port"""
        for port in range(start_port, start_port + 100):  # Try 100 ports
            if self._is_port_available(host, port):
                return port
        raise RuntimeError(f"No available ports found starting from {start_port}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.cleanup()
        sys.exit(0)

    def load_configuration(self):
        """Load or create configuration"""
        try:
            if not Path(self.config_file).exists():
                logger.info(f"Configuration file not found, creating default: {self.config_file}")
                create_sample_config(self.config_file)

            logger.info(f"Loading configuration from: {self.config_file}")
            self.config = Config.from_file(self.config_file)

            # Update host and port from config
            if hasattr(self.config, 'api_host'):
                self.host = self.config.api_host
            if hasattr(self.config, 'api_port'):
                self.port = self.config.api_port

            # Override with environment variables if present
            if os.getenv("MLOPS_HOST"):
                self.host = os.getenv("MLOPS_HOST")
            if os.getenv("MLOPS_PORT"):
                self.port = int(os.getenv("MLOPS_PORT"))
            if os.getenv("MLOPS_MODELS_DIR"):
                self.models_directory = os.getenv("MLOPS_MODELS_DIR")

            logger.info(f"Configuration loaded successfully - API will run on {self.host}:{self.port}")

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    def verify_mlflow_connection(self, max_retries: int = 3, retry_delay: float = 2.0, timeout: float = 10.0) -> bool:
        """
        Verify MLflow connection with retry mechanism

        Args:
            max_retries: Maximum number of connection attempts
            retry_delay: Delay between retry attempts in seconds
            timeout: Timeout for each connection attempt in seconds

        Returns:
            bool: True if connection successful, False otherwise
        """
        mlflow_uri = self.config.mlflow.tracking_uri
        logger.info(f"Verifying MLflow connection to: {mlflow_uri}")

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"MLflow connection attempt {attempt}/{max_retries}...")

                # Create MLflow client with timeout
                client = MlflowClient(tracking_uri=mlflow_uri)

                # Test connection by trying to list experiments
                # This is a lightweight operation that verifies connectivity
                start_time = time.time()
                experiments = client.search_experiments(max_results=1)
                connection_time = time.time() - start_time

                logger.info(f"MLflow connection successful!")
                return True

            except Exception as e:
                error_msg = str(e)
                logger.warning(f"MLflow connection attempt {attempt}/{max_retries} failed: {error_msg}")

                if attempt < max_retries:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    # Final attempt failed
                    logger.error("=" * 70)
                    logger.error("CRITICAL: MLflow Connection Failed!")
                    logger.error("=" * 70)
                    logger.error(f"MLflow server: {mlflow_uri}")
                    logger.error(f"Error: {error_msg}")
                    logger.error(f"Attempts made: {max_retries}")
                    logger.error("")
                    logger.error("Possible solutions:")
                    logger.error("1. Check if MLflow server is running")
                    logger.error("2. Verify the MLflow tracking URI in config.yaml")
                    logger.error("3. Check network connectivity to MLflow server")
                    logger.error("4. Ensure MLflow server is accessible from this machine")
                    logger.error("=" * 70)

                    return False

        return False

    def initialize_services(self):
        """Initialize all service components"""
        try:
            logger.info("Initializing Anomaly Detection Serve services...")

            # Initialize Ray Serve with custom HTTP options to avoid port conflict
            logger.info("Initializing Ray Serve with custom HTTP configuration...")
            try:
                import ray
                from ray import serve

                # Configure Ray logging level
                ray_log_level = self.config.ray.log_level
                logger.info(f"Setting Ray log level to: {ray_log_level}")
                
                # Set Ray logging level using environment variable and direct logger configuration
                import os
                os.environ["RAY_LOG_LEVEL"] = ray_log_level
                
                # Also configure Ray loggers directly
                ray_logger = logging.getLogger("ray")
                ray_logger.setLevel(getattr(logging, ray_log_level.upper()))
                
                # Configure Ray Serve logger as well
                ray_serve_logger = logging.getLogger("ray.serve")
                ray_serve_logger.setLevel(getattr(logging, ray_log_level.upper()))

                # Find available port for Ray Serve proxy (starting from FastAPI port + 1)
                ray_serve_port = self._find_available_port(self.port + 1, "127.0.0.1")
                logger.info(f"Configuring Ray Serve HTTP proxy to use port {ray_serve_port}")

                # Start Ray Serve with custom HTTP options
                serve.start(http_options={"host": "127.0.0.1", "port": ray_serve_port})
                logger.info(f"Ray Serve initialized successfully on port {ray_serve_port}")

            except Exception as e:
                logger.warning(f"Ray Serve initialization failed or already running: {e}")
                logger.info("Continuing with existing Ray Serve instance...")

            # Initialize Model Manager
            logger.info("Setting up Model Manager...")
            self.model_manager = ModelManager(
                mlflow_tracking_uri=self.config.mlflow.tracking_uri,
                hot_cache_size=self.config.tiered_loading.hot_cache_size,
                warm_cache_size=self.config.tiered_loading.warm_cache_size,
                cold_cache_size=self.config.tiered_loading.cold_cache_size,
                max_workers=self.config.max_workers,
                enable_model_warming=self.config.tiered_loading.enable_model_warming,
            )

            # Initialize Model Router
            logger.info("Setting up Model Router...")
            self.model_router = ModelRouter(
                model_manager=self.model_manager,
                pooled_deployment=self.config.pooled_deployment,
                routing_strategy=self.config.routing.strategy,
                enable_request_queuing=self.config.routing.enable_request_queuing,
                max_queue_size=self.config.routing.max_queue_size
            )

            # Initialize Monitor
            logger.info("Setting up Monitor...")
            self.monitor = ModelMonitor(
                collection_interval=self.config.monitoring.collection_interval,
                optimization_interval=self.config.monitoring.optimization_interval,
                enable_prometheus=self.config.monitoring.enable_prometheus_export,
                prometheus_port=self.config.monitoring.prometheus_port
            )

            # Initialize Input Handler
            logger.info("Setting up Input Handler...")
            self.input_handler = DataHandler()

            # Initialize Deployment Manager
            logger.info("Setting up Deployment Manager...")
            self.deployment_manager = PooledModelDeployment(self.model_manager, self.config)

            # Initialize FastAPI dependencies
            logger.info("Configuring FastAPI dependencies...")
            initialize_dependencies(
                self.model_manager,
                self.model_router,
                self.monitor,
                self.input_handler
            )

            logger.info("All services initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            raise

    def deploy_production_models(self):
        """Deploy production models from MLflow automatically with parallel loading"""
        try:
            logger.info("Deploying production models from MLflow...")

            # Get production models from MLflow
            production_models = self.model_manager.get_production_models()
            logger.debug(f"Found {len(production_models)} production models to deploy")

            if not production_models:
                logger.warning("No production models found in MLflow to deploy")
                return

            # Create default deployment pools if none exist
            existing_deployments = self.deployment_manager.get_all_deployment_stats()
            if not existing_deployments:
                logger.info("Creating default deployment pools...")
                # Use config value for pool count instead of hardcoded value
                if hasattr(self.config, 'pooled_deployment') and hasattr(self.config.pooled_deployment, 'default_pool_count'):
                    pool_count = self.config.pooled_deployment.default_pool_count
                    logger.info(f"Using configured pool count: {pool_count}")
                else:
                    pool_count = 3  # Fallback
                    logger.warning("No pool count configured, using fallback value: 3")

                created_pools = self.deployment_manager.create_default_pools()  # Let method use config internally
                logger.debug(f"Created deployment pools: {created_pools}")

                # Register deployment pools with model router
                logger.info("Registering deployment pools with model router...")
                for pool_name in created_pools:
                    self.model_router.register_deployment(pool_name)
                    logger.debug(f" Registered deployment pool: {pool_name}")

                # Log configuration summary
                if hasattr(self.config, 'pooled_deployment'):
                    pool_config = self.config.pooled_deployment
                    logger.debug(f"Pool configuration summary:")
                    logger.debug(f"  - Pool count: {pool_config.default_pool_count}")
                    logger.debug(f"  - Models per pool: {pool_config.models_per_pool}")
                    logger.debug(f"  - CPUs per pool: {pool_config.pool_resource_config.num_cpus}")
                    logger.debug(f"  - Memory per pool: {pool_config.pool_resource_config.memory}MB")
                    logger.debug(f"  - Min replicas: {pool_config.autoscaling_config.min_replicas}")
                    logger.debug(f"  - Max replicas: {pool_config.autoscaling_config.max_replicas}")
                else:
                    logger.warning("No pooled deployment configuration found, using defaults")

            # Pre-load production models into cache with parallel loading
            logger.info("Pre-loading production models into cache (parallel loading)...")
            loaded_models = []
            failed_models = []
            
            # Use ThreadPoolExecutor for parallel model loading
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import time
            
            def load_single_model(model_name):
                """Load a single model and return result"""
                try:
                    start_time = time.time()
                    model_info = self.model_manager._load_model_sync(model_name)
                    load_time = time.time() - start_time
                    
                    if model_info:
                        return {"model_name": model_name, "success": True, "load_time": load_time, "error": None}
                    else:
                        return {"model_name": model_name, "success": False, "load_time": load_time, "error": "Model loading returned None"}
                except Exception as e:
                    load_time = time.time() - start_time if 'start_time' in locals() else 0
                    return {"model_name": model_name, "success": False, "load_time": load_time, "error": str(e)}
            
            # Limit concurrent model loading to prevent resource exhaustion
            max_concurrent_loads = min(4, len(production_models))  # Max 4 concurrent loads
            
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=max_concurrent_loads) as executor:
                # Submit all model loading tasks
                future_to_model = {executor.submit(load_single_model, model_name): model_name 
                                 for model_name in production_models}
                
                # Collect results as they complete
                for future in as_completed(future_to_model):
                    result = future.result()
                    
                    if result["success"]:
                        loaded_models.append(result["model_name"])
                        logger.debug(f" Loaded model: {result['model_name']} ({result['load_time']:.2f}s)")
                    else:
                        failed_models.append(result["model_name"])
                        logger.warning(f" Failed to load model: {result['model_name']} ({result['load_time']:.2f}s) - {result['error']}")
            
            total_time = time.time() - start_time
            
            logger.info(f"Model loading completed in {total_time:.2f}s:")
            logger.info(f" Successfully loaded: {len(loaded_models)}/{len(production_models)} models")
            if failed_models:
                logger.warning(f"  Failed to load: {len(failed_models)} models: {failed_models}")
            
            if loaded_models:
                logger.debug(f"Successfully loaded models: {loaded_models}")
                
            # Pre-load models into Ray Serve deployment pools to prevent cold start
            if loaded_models:
                logger.info("Pre-loading models into Ray Serve deployment pools...")
                try:
                    import ray
                    from ray import serve
                    
                    # Get all Ray Serve deployments
                    deployments = serve.list_deployments()
                    
                    if deployments:
                        # Find our pooled deployment
                        pooled_deployment = None
                        for deployment_name, deployment_info in deployments.items():
                            if "pooled" in deployment_name.lower() or "anomaly" in deployment_name.lower():
                                pooled_deployment = deployment_name
                                break
                        
                        if not pooled_deployment and deployments:
                            # Use the first available deployment
                            pooled_deployment = list(deployments.keys())[0]
                        
                        if pooled_deployment:
                            logger.info(f"Using deployment '{pooled_deployment}' for model pool warmup")
                            
                            # Get deployment handle
                            deployment_handle = serve.get_deployment(pooled_deployment).get_handle()
                            
                            # Create a simple warmup request
                            warmup_payload = {
                                "ma_don_vi": "UBND.0019",
                                "ma_bao_cao": "10628953_CT", 
                                "ky_du_lieu": "2024-01-01",
                                "data": [{"ma_tieu_chi": "TONGCONG", "FN01": 1000000}]
                            }
                            
                            # Make multiple warmup requests to initialize all actor replicas
                            warmup_start = time.time()
                            warmup_requests = 5  # Warm up multiple replicas
                            successful_warmups = 0
                            
                            try:
                                # Make multiple concurrent requests to warm up all replicas
                                result_refs = []
                                for i in range(warmup_requests):
                                    result_ref = deployment_handle.remote(warmup_payload)
                                    result_refs.append(result_ref)
                                
                                # Wait for all requests to complete
                                results = ray.get(result_refs, timeout=45)  # 45 second timeout for multiple requests
                                
                                # Count successful warmups
                                for result in results:
                                    if isinstance(result, dict) and result.get('status') == 'success':
                                        successful_warmups += 1
                                
                                warmup_time = time.time() - warmup_start
                                logger.info(f" Pool warmup completed: {successful_warmups}/{warmup_requests} successful in {warmup_time:.2f}s")
                                
                            except Exception as e:
                                warmup_time = time.time() - warmup_start
                                logger.warning(f"Pool warmup requests failed after {warmup_time:.2f}s: {e}")
                                # Continue startup even if warmup fails
                        else:
                            logger.warning("No suitable Ray Serve deployment found for pool warmup")
                    else:
                        logger.warning("No Ray Serve deployments available for model pre-loading")
                        
                except Exception as e:
                    logger.warning(f"Pool pre-loading failed (non-critical): {e}")
                    # Continue startup even if pool pre-loading fails
            
            # Set service readiness flag
            self._models_loaded = True
            self._loaded_model_count = len(loaded_models)
            self._failed_model_count = len(failed_models)

        except Exception as e:
            logger.error(f"Failed to deploy production models: {e}")
            # Don't raise exception here to allow service to continue starting
            self._models_loaded = False
            self._loaded_model_count = 0
            self._failed_model_count = 0

    def start_background_services(self):
        """Start background monitoring and optimization services"""
        try:
            logger.info("Starting background services...")

            # Start model manager monitoring
            self.model_manager.start_monitoring()

            # Start system monitoring
            self.monitor.start_monitoring()

            logger.info("Background services started")

        except Exception as e:
            logger.error(f"Failed to start background services: {e}")
            raise

    def print_startup_info(self):
        """Print service startup information"""
        api_prefix = self.config.api_prefix if self.config.api_prefix else ""
        base_url = f"http://{self.host}:{self.port}{api_prefix}"

        print("\n" + "="*70)
        print("Anomaly Detection Serve - Service Started Successfully!")
        print("="*70)
        print(f"Service URL: {base_url}")
        print(f"API Documentation: {base_url}/docs")
        print(f"Health Check: {base_url}/health")
        print(f"Service Info: {base_url}/")
        print(f"Model Statistics: {base_url}/models/stats")
        print(f"Dashboard: {base_url}/dashboard")

        # if self.config.monitoring.enable_prometheus_export:
        #     print(f"Prometheus Metrics: http://localhost:{self.config.monitoring.prometheus_port}/metrics")

        print("="*70)

    def run(self):
        """Main service execution"""
        try:
            logger.info("Starting Anomaly Detection Serve service...")

            # Load configuration
            self.load_configuration()

            # Initialize app with api_prefix from configuration
            logger.info(f"Initializing FastAPI app with api_prefix: {self.config.api_prefix}")
            app_instance = initialize_app_with_config(self.config.api_prefix)

            # Verify MLflow connection before starting services
            logger.info("Verifying MLflow connectivity...")
            if not self.verify_mlflow_connection(max_retries=3, retry_delay=2.0):
                logger.error("Cannot start service without MLflow connection")
                logger.error("Please ensure MLflow server is running and accessible")
                sys.exit(1)

            # Initialize all services
            self.initialize_services()

            # Deploy production models from MLflow
            self.deploy_production_models()

            # Start background services
            self.start_background_services()
            
            # Mark service as ready after all initialization is complete
            self._service_ready = True
            logger.info(f"Service ready! Models loaded: {self._loaded_model_count}, Failed: {self._failed_model_count}")
            
            # Update readiness state for API endpoint
            update_service_readiness(
                ready=self._service_ready,
                models_loaded=self._loaded_model_count,
                models_failed=self._failed_model_count,
                initialization_complete=True
            )

            # Check port availability - fail if configured port is not available
            if not self._is_port_available(self.host, self.port):
                logger.error(f"Port {self.port} is already in use!")
                logger.error("=" * 70)
                logger.error("CRITICAL: Port Conflict Detected!")
                logger.error("=" * 70)
                logger.error(f"The configured port {self.port} on host {self.host} is already occupied by another service.")
                logger.error("")
                logger.error("Possible solutions:")
                logger.error("1. Stop the service currently using this port")
                logger.error("2. Change the api_port in config.yaml to a different port")
                logger.error("3. Set MLOPS_PORT environment variable to use a different port")
                logger.error("4. Use 'netstat -tulpn | grep :8000' (Linux) or 'netstat -an | findstr :8000' (Windows) to find the conflicting service")
                logger.error("=" * 70)
                sys.exit(1)
            else:
                logger.info(f"Port {self.port} is available")

            # Print startup information
            self.print_startup_info()

            # Start the FastAPI server
            logger.info(f"Starting FastAPI server on {self.host}:{self.port}")

            uvicorn.run(
                app_instance,
                host=self.host,
                port=self.port,
                log_level="info",
                access_log=True
            )

        except KeyboardInterrupt:
            logger.info("Service interrupted by user")
        except Exception as e:
            logger.error(f"Service failed to start: {e}")
            raise
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources on shutdown"""
        try:
            logger.info("Cleaning up resources...")

            if self.deployment_manager:
                logger.info("Cleaning up deployment manager...")
                self.deployment_manager.cleanup()

            if self.model_manager:
                self.model_manager.cleanup()

            if self.monitor:
                self.monitor.cleanup()

            logger.info("Cleanup completed successfully")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def main():
    """Main entry point"""
    try:
        service = AnomalyDetectionServe()
        service.run()
    except Exception as e:
        logger.error(f"Failed to start Anomaly Detection Serve: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()