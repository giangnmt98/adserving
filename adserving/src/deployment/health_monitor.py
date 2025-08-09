"""
Enhanced Health Monitor and Graceful Shutdown Manager
Provides comprehensive health checking and graceful shutdown capabilities
"""

import asyncio
import time
import signal
import sys
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass, field
from threading import RLock, Event
import psutil
import logging

from ..utils.logger import get_logger

logger = get_logger()


class HealthStatus(Enum):
    """Health check status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ShutdownState(Enum):
    """Graceful shutdown states"""
    RUNNING = "running"
    SHUTDOWN_REQUESTED = "shutdown_requested"
    DRAINING = "draining"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    component: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    check_time: float = field(default_factory=time.time)
    response_time: float = 0.0


@dataclass
class SystemMetrics:
    """System performance metrics"""
    cpu_percent: float
    memory_usage_mb: float
    memory_percent: float
    disk_usage_percent: float
    active_connections: int
    request_rate: float
    error_rate: float
    avg_response_time: float
    timestamp: float = field(default_factory=time.time)


class HealthMonitor:
    """Enhanced health monitoring system"""
    
    def __init__(self, check_interval=30, degraded_threshold=0.8, unhealthy_threshold=0.9):
        self.check_interval = check_interval
        self.degraded_threshold = degraded_threshold
        self.unhealthy_threshold = unhealthy_threshold
        
        # Health check registry
        self.health_checks = {}
        self.health_results = {}
        self.system_metrics_history = []
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task = None
        self.shutdown_event = Event()
        
        # Thread safety
        self.monitor_lock = RLock()
        
        # Metrics
        self.request_count = 0
        self.error_count = 0
        self.response_times = []
        self.last_metrics_reset = time.time()
        
        logger.info("🏥 HealthMonitor initialized")
        logger.info(f"   - Check interval: {check_interval}s")
        logger.info(f"   - Degraded threshold: {degraded_threshold * 100}%")
        logger.info(f"   - Unhealthy threshold: {unhealthy_threshold * 100}%")
    
    def register_health_check(self, name: str, check_func: Callable, critical: bool = False):
        """Register a health check function"""
        with self.monitor_lock:
            self.health_checks[name] = {
                "func": check_func,
                "critical": critical,
                "enabled": True
            }
        
        logger.info(f"   Registered health check: {name} (critical: {critical})")
    
    def unregister_health_check(self, name: str):
        """Unregister a health check"""
        with self.monitor_lock:
            if name in self.health_checks:
                del self.health_checks[name]
                if name in self.health_results:
                    del self.health_results[name]
        
        logger.info(f"   Unregistered health check: {name}")
    
    async def start_monitoring(self):
        """Start continuous health monitoring"""
        if self.monitoring_active:
            logger.warning("Health monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("🏥 Health monitoring started")
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("🏥 Health monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        try:
            while self.monitoring_active and not self.shutdown_event.is_set():
                try:
                    # Perform health checks
                    await self._perform_health_checks()
                    
                    # Collect system metrics
                    await self._collect_system_metrics()
                    
                    # Wait for next interval
                    await asyncio.sleep(self.check_interval)
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    await asyncio.sleep(5)  # Short wait on error
                    
        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Monitoring loop failed: {e}")
    
    async def _perform_health_checks(self):
        """Perform all registered health checks"""
        check_results = {}
        
        with self.monitor_lock:
            checks_to_run = self.health_checks.copy()
        
        for name, check_config in checks_to_run.items():
            if not check_config["enabled"]:
                continue
            
            try:
                start_time = time.time()
                
                # Run health check (handle both sync and async functions)
                check_func = check_config["func"]
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                
                response_time = time.time() - start_time
                
                # Parse result
                if isinstance(result, bool):
                    status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                    message = "OK" if result else "Check failed"
                    details = {}
                elif isinstance(result, dict):
                    status = HealthStatus(result.get("status", "unknown"))
                    message = result.get("message", "No message")
                    details = result.get("details", {})
                else:
                    status = HealthStatus.UNKNOWN
                    message = str(result)
                    details = {}
                
                check_result = HealthCheckResult(
                    component=name,
                    status=status,
                    message=message,
                    details=details,
                    response_time=response_time
                )
                
                check_results[name] = check_result
                
            except Exception as e:
                logger.error(f"Health check '{name}' failed: {e}")
                check_results[name] = HealthCheckResult(
                    component=name,
                    status=HealthStatus.CRITICAL,
                    message=f"Check error: {e}",
                    details={"exception": str(e)}
                )
        
        # Update results
        with self.monitor_lock:
            self.health_results = check_results
        
        # Log critical issues
        critical_issues = [
            result for result in check_results.values()
            if result.status in [HealthStatus.CRITICAL, HealthStatus.UNHEALTHY]
        ]
        
        if critical_issues:
            logger.warning(f"Health check issues found: {len(critical_issues)} components")
            for issue in critical_issues:
                logger.warning(f"   {issue.component}: {issue.status.value} - {issue.message}")
    
    async def _collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network connections (approximate)
            try:
                connections = len(psutil.net_connections())
            except:
                connections = 0
            
            # Calculate request metrics
            current_time = time.time()
            time_window = current_time - self.last_metrics_reset
            
            request_rate = self.request_count / max(time_window, 1)
            error_rate = self.error_count / max(self.request_count, 1)
            avg_response_time = sum(self.response_times) / max(len(self.response_times), 1)
            
            metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_usage_mb=memory.used / 1024 / 1024,
                memory_percent=memory.percent,
                disk_usage_percent=disk.percent,
                active_connections=connections,
                request_rate=request_rate,
                error_rate=error_rate,
                avg_response_time=avg_response_time
            )
            
            # Store metrics (keep last 100 entries)
            self.system_metrics_history.append(metrics)
            if len(self.system_metrics_history) > 100:
                self.system_metrics_history = self.system_metrics_history[-100:]
            
            # Reset counters periodically
            if time_window > 300:  # Reset every 5 minutes
                self.request_count = 0
                self.error_count = 0
                self.response_times = []
                self.last_metrics_reset = current_time
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    def record_request(self, response_time: float, is_error: bool = False):
        """Record a request for metrics"""
        self.request_count += 1
        if is_error:
            self.error_count += 1
        
        self.response_times.append(response_time)
        
        # Keep only recent response times
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-500:]
    
    def get_overall_health_status(self) -> HealthStatus:
        """Get overall system health status"""
        with self.monitor_lock:
            if not self.health_results:
                return HealthStatus.UNKNOWN
            
            statuses = [result.status for result in self.health_results.values()]
            
            # Critical if any critical checks fail
            if HealthStatus.CRITICAL in statuses:
                return HealthStatus.CRITICAL
            
            # Unhealthy if any unhealthy checks fail
            if HealthStatus.UNHEALTHY in statuses:
                return HealthStatus.UNHEALTHY
            
            # Degraded if any degraded checks
            if HealthStatus.DEGRADED in statuses:
                return HealthStatus.DEGRADED
            
            # Healthy if all checks are healthy
            if all(status == HealthStatus.HEALTHY for status in statuses):
                return HealthStatus.HEALTHY
            
            return HealthStatus.UNKNOWN
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report"""
        with self.monitor_lock:
            overall_status = self.get_overall_health_status()
            
            # Get latest metrics
            latest_metrics = self.system_metrics_history[-1] if self.system_metrics_history else None
            
            report = {
                "overall_status": overall_status.value,
                "check_results": {
                    name: {
                        "status": result.status.value,
                        "message": result.message,
                        "response_time": result.response_time,
                        "check_time": result.check_time,
                        "details": result.details
                    }
                    for name, result in self.health_results.items()
                },
                "system_metrics": {
                    "cpu_percent": latest_metrics.cpu_percent if latest_metrics else 0,
                    "memory_usage_mb": latest_metrics.memory_usage_mb if latest_metrics else 0,
                    "memory_percent": latest_metrics.memory_percent if latest_metrics else 0,
                    "disk_usage_percent": latest_metrics.disk_usage_percent if latest_metrics else 0,
                    "active_connections": latest_metrics.active_connections if latest_metrics else 0,
                    "request_rate": latest_metrics.request_rate if latest_metrics else 0,
                    "error_rate": latest_metrics.error_rate if latest_metrics else 0,
                    "avg_response_time": latest_metrics.avg_response_time if latest_metrics else 0
                } if latest_metrics else {},
                "monitoring_active": self.monitoring_active,
                "total_checks": len(self.health_checks),
                "timestamp": time.time()
            }
            
            return report


class GracefulShutdownManager:
    """Manages graceful shutdown of services"""
    
    def __init__(self, shutdown_timeout=30, drain_timeout=60):
        self.shutdown_timeout = shutdown_timeout
        self.drain_timeout = drain_timeout
        
        # Shutdown state
        self.shutdown_state = ShutdownState.RUNNING
        self.shutdown_requested = Event()
        self.shutdown_complete = Event()
        
        # Registered shutdown handlers
        self.shutdown_handlers = []
        self.cleanup_handlers = []
        
        # Active requests tracking
        self.active_requests = 0
        self.request_lock = RLock()
        
        # Signal handlers
        self._setup_signal_handlers()
        
        logger.info("GracefulShutdownManager initialized")
        logger.info(f"   - Shutdown timeout: {shutdown_timeout}s")
        logger.info(f"   - Drain timeout: {drain_timeout}s")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def register_shutdown_handler(self, handler: Callable, priority: int = 100):
        """Register a shutdown handler (lower priority = runs first)"""
        self.shutdown_handlers.append((priority, handler))
        self.shutdown_handlers.sort(key=lambda x: x[0])
        
        logger.info(f"   Registered shutdown handler (priority: {priority})")
    
    def register_cleanup_handler(self, handler: Callable):
        """Register a cleanup handler (runs after shutdown)"""
        self.cleanup_handlers.append(handler)
        
        logger.info("   Registered cleanup handler")
    
    def enter_request(self):
        """Track entering a request"""
        with self.request_lock:
            if self.shutdown_state in [ShutdownState.DRAINING, ShutdownState.STOPPING]:
                return False  # Reject new requests during shutdown
            
            self.active_requests += 1
            return True
    
    def exit_request(self):
        """Track exiting a request"""
        with self.request_lock:
            self.active_requests = max(0, self.active_requests - 1)
    
    def is_shutting_down(self) -> bool:
        """Check if shutdown is in progress"""
        return self.shutdown_state != ShutdownState.RUNNING
    
    async def shutdown(self):
        """Perform graceful shutdown"""
        if self.shutdown_state != ShutdownState.RUNNING:
            logger.warning("Shutdown already in progress")
            return
        
        logger.info("Starting graceful shutdown...")
        self.shutdown_state = ShutdownState.SHUTDOWN_REQUESTED
        self.shutdown_requested.set()
        
        try:
            # Phase 1: Stop accepting new requests and drain existing ones
            logger.info("Phase 1: Draining active requests...")
            self.shutdown_state = ShutdownState.DRAINING
            
            # Wait for active requests to complete
            drain_start = time.time()
            while self.active_requests > 0 and time.time() - drain_start < self.drain_timeout:
                logger.info(f"   Waiting for {self.active_requests} active requests...")
                await asyncio.sleep(1)
            
            if self.active_requests > 0:
                logger.warning(f"Force proceeding with {self.active_requests} active requests")
            
            # Phase 2: Run shutdown handlers
            logger.info("Phase 2: Running shutdown handlers...")
            self.shutdown_state = ShutdownState.STOPPING
            
            for priority, handler in self.shutdown_handlers:
                try:
                    logger.info(f"   Running shutdown handler (priority: {priority})")
                    
                    if asyncio.iscoroutinefunction(handler):
                        await asyncio.wait_for(handler(), timeout=self.shutdown_timeout)
                    else:
                        handler()
                        
                except asyncio.TimeoutError:
                    logger.error(f"   Shutdown handler timeout (priority: {priority})")
                except Exception as e:
                    logger.error(f"   Shutdown handler error (priority: {priority}): {e}")
            
            # Phase 3: Run cleanup handlers
            logger.info("Phase 3: Running cleanup handlers...")
            
            for handler in self.cleanup_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler()
                    else:
                        handler()
                        
                except Exception as e:
                    logger.error(f"   Cleanup handler error: {e}")
            
            # Phase 4: Final cleanup
            self.shutdown_state = ShutdownState.STOPPED
            self.shutdown_complete.set()
            
            logger.info("Graceful shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during graceful shutdown: {e}")
            self.shutdown_state = ShutdownState.STOPPED
            self.shutdown_complete.set()
    
    async def wait_for_shutdown(self):
        """Wait for shutdown to complete"""
        await asyncio.get_event_loop().run_in_executor(None, self.shutdown_complete.wait)
    
    def get_shutdown_status(self) -> Dict[str, Any]:
        """Get shutdown status"""
        return {
            "state": self.shutdown_state.value,
            "active_requests": self.active_requests,
            "shutdown_handlers": len(self.shutdown_handlers),
            "cleanup_handlers": len(self.cleanup_handlers),
            "shutdown_timeout": self.shutdown_timeout,
            "drain_timeout": self.drain_timeout
        }