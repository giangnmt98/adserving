"""
Resource metrics data structures and collection utilities.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

import psutil

try:
    import GPUtil

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    GPUtil = None


@dataclass
class ResourceMetrics:
    """Resource metrics with GPU and detailed memory tracking"""

    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used: int  # bytes
    memory_available: int  # bytes
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int

    # GPU metrics
    gpu_count: int = 0
    gpu_utilization: List[float] = field(default_factory=list)
    gpu_memory_used: List[int] = field(default_factory=list)  # MB
    gpu_memory_total: List[int] = field(default_factory=list)  # MB
    gpu_temperature: List[float] = field(default_factory=list)

    # Process-specific metrics
    process_cpu_percent: float = 0.0
    process_memory_mb: float = 0.0
    open_files: int = 0
    thread_count: int = 0


def collect_gpu_metrics() -> Dict[str, Any]:
    """Collect GPU metrics"""
    if not GPU_AVAILABLE:
        return {
            "gpu_count": 0,
            "gpu_utilization": [],
            "gpu_memory_used": [],
            "gpu_memory_total": [],
            "gpu_temperature": [],
        }

    try:
        gpus = GPUtil.getGPUs()
        return {
            "gpu_count": len(gpus),
            "gpu_utilization": [gpu.load * 100 for gpu in gpus],
            "gpu_memory_used": [gpu.memoryUsed for gpu in gpus],
            "gpu_memory_total": [gpu.memoryTotal for gpu in gpus],
            "gpu_temperature": [gpu.temperature for gpu in gpus],
        }
    except Exception:
        return {
            "gpu_count": 0,
            "gpu_utilization": [],
            "gpu_memory_used": [],
            "gpu_memory_total": [],
            "gpu_temperature": [],
        }


def collect_system_metrics() -> ResourceMetrics:
    """Collect system resource metrics"""
    # Basic system metrics
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    network = psutil.net_io_counters()

    # Process-specific metrics
    process = psutil.Process()
    process_cpu = process.cpu_percent()
    process_memory = process.memory_info().rss / 1024 / 1024  # MB

    # GPU metrics
    gpu_metrics = collect_gpu_metrics()

    return ResourceMetrics(
        timestamp=time.time(),
        cpu_percent=cpu_percent,
        memory_percent=memory.percent,
        memory_used=memory.used,
        memory_available=memory.available,
        disk_usage_percent=disk.percent,
        network_bytes_sent=network.bytes_sent,
        network_bytes_recv=network.bytes_recv,
        process_cpu_percent=process_cpu,
        process_memory_mb=process_memory,
        open_files=len(process.open_files()),
        thread_count=process.num_threads(),
        **gpu_metrics,
    )
