"""
Core configuration classes
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .base_types import ResourceSharingStrategy, RoutingStrategy


@dataclass
class MLflowConfig:
    """MLflow configuration"""

    tracking_uri: str = "http://localhost:5000"
    registry_uri: Optional[str] = None
    experiment_name: str = "anomalydetectionserving"
    artifact_location: Optional[str] = None
    model_stage: str = "Production"
    enable_model_versioning: bool = True
    model_cache_ttl: int = 3600  # seconds
    production_check_interval: int = 60  # seconds


@dataclass
class RayConfig:
    """Ray cluster configuration for hundreds of models with comprehensive hardware control"""

    # Basic Ray cluster settings
    address: Optional[str] = None  # null for local mode
    runtime_env: Optional[Dict[str, Any]] = None
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8265
    log_level: str = "ERROR"  # Ray logging level
    
    # Memory configuration
    object_store_memory: Optional[str] = "50GB"  # Shared object store memory
    memory_management: Optional[Dict[str, Any]] = None  # Memory management settings
    plasma_store_socket_name: Optional[str] = None
    raylet_socket_name: Optional[str] = None
    
    # CPU Hardware Configuration
    num_cpus: Optional[int] = None  # Auto-detect if None
    cpu_scheduling_policy: str = "SCHED_OTHER"  # SCHED_OTHER/SCHED_FIFO/SCHED_RR/SCHED_BATCH
    enable_cpu_affinity: bool = False  # Enable CPU core affinity for Ray workers
    cpu_affinity_cores: Optional[str] = None  # Specific cores like "0-7" or "0,2,4,6"
    cpu_nice_priority: int = 0  # Process nice priority (-20 to 19, lower = higher priority)
    enable_numa_optimization: bool = False  # NUMA topology optimization
    numa_nodes: Optional[str] = None  # Specific NUMA nodes like "0,1" or "0-3"
    cpu_overcommit_factor: float = 1.0  # CPU overcommitment factor (1.0 = no overcommit)
    
    # GPU Hardware Configuration  
    num_gpus: Optional[int] = None  # Auto-detect if None
    gpu_devices: Optional[str] = None  # Specific GPU devices like "0,1,2" or "all"
    gpu_memory_fraction: float = 0.8  # Fraction of GPU memory to use per device
    enable_gpu_sharing: bool = True  # Allow multiple actors per GPU
    gpu_scheduling_strategy: str = "round_robin"  # round_robin/least_used/explicit
    gpu_memory_pool_size: Optional[str] = None  # Pre-allocated GPU memory pool
    enable_gpu_memory_fraction_per_device: bool = False  # Per-device memory fractions
    gpu_device_memory_fractions: Optional[Dict[int, float]] = None  # {gpu_id: fraction}
    enable_gpu_compute_capability_check: bool = True  # Check GPU compute capability
    min_gpu_compute_capability: float = 3.5  # Minimum required compute capability
    
    # Advanced Memory Management
    heap_memory_limit: Optional[str] = None  # Maximum heap memory per worker
    enable_memory_pressure_eviction: bool = True  # Evict objects under memory pressure
    memory_usage_threshold: float = 0.95  # Memory usage threshold for warnings
    enable_spill_to_disk: bool = False  # Spill objects to disk when memory is full
    disk_spill_directory: Optional[str] = None  # Directory for disk spilling
    max_disk_spill_size: Optional[str] = None  # Maximum disk spill size
    
    # Performance and Optimization
    optimization: Optional[Dict[str, Any]] = None  # Ray optimization settings
    enable_placement_group_bundle_cache: bool = True  # Cache placement group bundles
    placement_group_capture_child_tasks: bool = True  # Capture child tasks in placement groups
    enable_task_events: bool = False  # Enable detailed task event logging
    max_task_retries: int = 3  # Maximum automatic task retries
    task_retry_delay_ms: int = 100  # Delay between task retries
    
    # Network and Communication
    grpc_max_message_size: int = 100 * 1024 * 1024  # 100MB max gRPC message size
    grpc_keepalive_time_ms: int = 30000  # gRPC keepalive time
    grpc_keepalive_timeout_ms: int = 5000  # gRPC keepalive timeout
    raylet_heartbeat_timeout_ms: int = 30000  # Raylet heartbeat timeout
    
    # Resource Monitoring and Limits
    enable_resource_monitoring: bool = True  # Monitor resource usage
    resource_monitoring_interval_s: float = 1.0  # Resource monitoring interval
    enable_memory_monitor: bool = True  # Monitor memory usage
    memory_monitor_refresh_ms: int = 250  # Memory monitor refresh rate
    enable_automatic_cleanup: bool = True  # Automatic cleanup of unused resources
    cleanup_interval_s: int = 60  # Resource cleanup interval
    
    # Worker and Actor Configuration
    max_workers_per_process: int = 1  # Maximum workers per process
    enable_worker_preloading: bool = False  # Preload workers for faster startup
    worker_startup_timeout_s: int = 30  # Worker startup timeout
    actor_lifetime_timeout_s: int = 3600  # Actor lifetime timeout (1 hour)
    
    # Debug and Development
    enable_debug_mode: bool = False  # Enable debug mode with detailed logging
    profile_workers: bool = False  # Enable worker profiling
    profile_output_dir: Optional[str] = None  # Directory for profiling output
    enable_timeline: bool = False  # Enable Ray timeline for debugging


@dataclass
class TieredLoadingConfig:
    """Configuration for tiered model loading strategy"""

    enable_tiered_loading: bool = True
    hot_cache_size: int = 500  # Always loaded models
    warm_cache_size: int = 200  # On-demand loaded models
    cold_cache_size: int = 500  # Rarely used models

    # Tier promotion/demotion thresholds
    hot_promotion_threshold: int = 20  # requests within time window
    warm_promotion_threshold: int = 5
    hot_promotion_time_window: int = 300  # 5 minutes
    warm_promotion_time_window: int = 3600  # 1 hour

    # Model warming settings
    enable_model_warming: bool = True
    warm_popular_models_count: int = 10
    warming_interval: int = 300  # 5 minutes

    # Cleanup settings
    cleanup_interval: int = 3600  # 1 hour
    cold_model_ttl: int = 7200  # 2 hours


@dataclass
class ResourceSharingConfig:
    """Configuration for resource sharing optimization"""

    strategy: ResourceSharingStrategy = ResourceSharingStrategy.GPU_SHARED
    enable_memory_mapping: bool = True
    enable_model_weight_sharing: bool = True
    shared_memory_size: str = "10GB"

    # GPU sharing settings
    max_models_per_gpu: int = 5
    gpu_memory_reserve: float = 0.2  # Reserve 20% for system
    enable_dynamic_gpu_allocation: bool = True

    # CPU sharing settings
    cpu_oversubscription_factor: float = 2.0  # Allow 2x CPU oversubscription
    enable_cpu_affinity: bool = True

    # Memory optimization
    enable_model_compression: bool = True
    compression_ratio: float = 0.7  # Target 70% of original size




@dataclass
class RoutingConfig:
    """Configuration for intelligent routing"""

    strategy: RoutingStrategy = getattr(RoutingStrategy, "LEAST_LOADED")
    enable_request_queuing: bool = True
    max_queue_size: int = 50000  # Increased for hundreds of models
    queue_timeout: int = 30  # seconds

    # Load balancing settings
    enable_sticky_routing: bool = True
    routing_cache_ttl: int = 300  # 5 minutes
    max_retries: int = 3
    retry_backoff: float = 0.1  # seconds

    # Performance optimization
    enable_request_batching: bool = True
    batch_size: int = 32
    batch_timeout_ms: int = 50
    enable_async_processing: bool = True


@dataclass
class WorkerConfig:
    """Enhanced worker configuration for optimal deployment management"""
    
    # Core Worker Settings
    worker_count: Optional[int] = None  # Auto-calculate if None: min(max(cpu_cores, 4), 32)
    max_concurrent_tasks: int = 3  # Maximum tasks per worker
    task_timeout: int = 30  # Maximum time per deployment task (30 seconds - reduced to prevent hangs)
    queue_size: int = 100  # Size of task queue per worker
    
    # Resource Allocation Per Worker
    cpu_cores_per_worker: float = 1.0  # CPU cores allocated per worker
    memory_limit_mb: int = 2048  # Memory limit per worker (2GB)
    process_priority: str = "normal"  # OS process priority: low/normal/high
    enable_cpu_affinity: bool = True  # Pin workers to specific CPU cores
    cpu_affinity_strategy: str = "spread"  # sequential/spread/custom
    
    # Worker Pool Optimization
    load_balancing_strategy: str = "least_loaded"  # round_robin/least_loaded/weighted
    enable_worker_monitoring: bool = True  # Monitor worker performance and health
    worker_restart_threshold: int = 10  # Restart worker after N failed tasks
    enable_auto_scaling: bool = False  # Dynamically adjust worker count
    min_workers: int = 2  # Minimum number of workers to maintain
    max_workers: int = 50  # Maximum number of workers allowed
    
    # Performance Tuning
    batch_processing_enabled: bool = True  # Enable batching of similar tasks
    batch_size: int = 3  # Number of tasks to batch together
    batch_timeout_ms: int = 500  # Maximum wait time for batching
    enable_task_prioritization: bool = True  # Priority queue for urgent tasks
    enable_worker_specialization: bool = False  # Assign workers to specific model types
    
    # Monitoring and Metrics
    collect_worker_metrics: bool = True  # Enable detailed worker performance metrics
    metrics_interval_seconds: int = 30  # How often to collect metrics
    enable_worker_health_checks: bool = True  # Regular health checks for workers
    health_check_interval_seconds: int = 15  # Interval between health checks (reduced for faster hang detection)
    
    # Resource Sharing and Coordination
    enable_resource_sharing: bool = True  # Share resources between workers
    shared_memory_pool_mb: int = 1024  # Shared memory pool for model weights (1GB)
    enable_worker_coordination: bool = True  # Workers coordinate to avoid duplicate work
    coordination_strategy: str = "queue"  # leader/consensus/queue
    
    def get_optimal_worker_count(self) -> int:
        """Calculate optimal worker count based on system resources"""
        if self.worker_count is not None:
            return self.worker_count
            
        import os
        cpu_count = os.cpu_count() or 4
        # Scale with CPU but cap at reasonable limits
        optimal_count = min(max(cpu_count, 4), 32)
        
        # Adjust based on memory constraints
        import psutil
        try:
            available_memory_gb = psutil.virtual_memory().available // (1024**3)
            # Each worker needs at least memory_limit_mb + some overhead
            memory_per_worker_gb = (self.memory_limit_mb + 512) / 1024  # Add 512MB overhead
            max_workers_by_memory = max(1, int(available_memory_gb / memory_per_worker_gb))
            optimal_count = min(optimal_count, max_workers_by_memory)
        except ImportError:
            pass  # psutil not available, use CPU-based calculation
            
        return max(self.min_workers, min(optimal_count, self.max_workers))


@dataclass
class AnomalyDetectionConfig:
    """Anomaly detection configuration"""

    # Default threshold for probability-based models
    default_probability_threshold: float = 0.5

    # Default threshold for label-based models (usually 0 since labels are binary)
    default_label_threshold: float = 0.0

    # Model-specific thresholds (model_name -> threshold)
    model_specific_thresholds: Dict[str, float] = field(default_factory=dict)

    # Model output type configuration (model_name -> output_type)
    # output_type can be "probability" or "label"
    model_output_types: Dict[str, str] = field(default_factory=dict)

    # Enable automatic threshold detection from model metadata
    enable_metadata_threshold: bool = True

    # Fallback behavior when model metadata is not available
    fallback_to_config: bool = True
