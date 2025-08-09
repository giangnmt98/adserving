"""
Comprehensive Monitoring and Observability System
Provides metrics collection, alerting, and deployment analytics for ultra-scale deployments
"""

import asyncio
import time
import json
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum
from dataclasses import dataclass, field
from threading import RLock
from collections import defaultdict, deque
import statistics
import logging

from .health_monitor import HealthStatus, HealthMonitor
from .service_discovery import ServiceDiscovery
from .blue_green_orchestrator import BlueGreenOrchestrator
from .ultra_scale_deployment_manager import UltraScaleDeploymentManager
from ..utils.logger import get_logger

logger = get_logger()


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricValue:
    """A single metric measurement"""
    name: str
    value: Union[int, float]
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class Alert:
    """System alert"""
    id: str
    level: AlertLevel
    title: str
    message: str
    component: str
    timestamp: float
    resolved: bool = False
    resolved_timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeploymentAnalytics:
    """Deployment analytics data"""
    deployment_id: str
    start_time: float
    end_time: Optional[float]
    success: bool
    environment: str
    models_deployed: int
    models_failed: int
    total_time: float
    phases: Dict[str, float] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)


class MonitoringSystem:
    """Comprehensive monitoring and observability system"""
    
    def __init__(self, metrics_retention_hours=24, alert_retention_hours=168):
        self.metrics_retention_hours = metrics_retention_hours
        self.alert_retention_hours = alert_retention_hours
        
        # Metrics storage
        self.metrics = defaultdict(lambda: deque(maxlen=10000))  # Per-metric circular buffer
        self.metric_summaries = {}  # Aggregated metric summaries
        
        # Alerting system
        self.alerts = {}  # alert_id -> Alert
        self.alert_rules = []  # List of alert rule functions
        self.alert_callbacks = []  # Alert notification callbacks
        
        # Analytics
        self.deployment_analytics = {}  # deployment_id -> DeploymentAnalytics
        self.performance_baselines = {}  # metric_name -> baseline_value
        
        # Component integrations
        self.health_monitor = None
        self.service_discovery = None
        self.blue_green_orchestrator = None
        self.deployment_managers = []
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task = None
        
        # Thread safety
        self.monitor_lock = RLock()
        
        # Custom metric handlers
        self.custom_collectors = []
        
        logger.info("MonitoringSystem initialized")
        logger.info(f"   - Metrics retention: {metrics_retention_hours}h")
        logger.info(f"   - Alert retention: {alert_retention_hours}h")
    
    def integrate_components(self, health_monitor: HealthMonitor = None,
                           service_discovery: ServiceDiscovery = None,
                           blue_green_orchestrator: BlueGreenOrchestrator = None,
                           deployment_managers: List[UltraScaleDeploymentManager] = None):
        """Integrate with deployment system components"""
        
        if health_monitor:
            self.health_monitor = health_monitor
            logger.info("   Integrated with HealthMonitor")
        
        if service_discovery:
            self.service_discovery = service_discovery
            # Register callback for service changes
            service_discovery.register_service_change_callback(self._on_service_change)
            logger.info("   Integrated with ServiceDiscovery")
        
        if blue_green_orchestrator:
            self.blue_green_orchestrator = blue_green_orchestrator
            logger.info("   Integrated with BlueGreenOrchestrator")
        
        if deployment_managers:
            self.deployment_managers = deployment_managers
            logger.info(f"   Integrated with {len(deployment_managers)} deployment managers")
    
    async def start_monitoring(self):
        """Start comprehensive monitoring"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        # Setup default alert rules
        self._setup_default_alert_rules()
        
        logger.info("Comprehensive monitoring started")
    
    async def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        try:
            while self.monitoring_active:
                try:
                    # Collect system metrics
                    await self._collect_system_metrics()
                    
                    # Collect component metrics
                    await self._collect_component_metrics()
                    
                    # Run custom collectors
                    await self._run_custom_collectors()
                    
                    # Check alert rules
                    await self._check_alert_rules()
                    
                    # Cleanup old data
                    await self._cleanup_old_data()
                    
                    # Update metric summaries
                    await self._update_metric_summaries()
                    
                    # Wait for next cycle
                    await asyncio.sleep(30)  # Collect every 30 seconds
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    await asyncio.sleep(5)
                    
        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Monitoring loop failed: {e}")
    
    def record_metric(self, name: str, value: Union[int, float], 
                     labels: Dict[str, str] = None, 
                     metric_type: MetricType = MetricType.GAUGE):
        """Record a metric value"""
        try:
            with self.monitor_lock:
                metric = MetricValue(
                    name=name,
                    value=value,
                    timestamp=time.time(),
                    labels=labels or {},
                    metric_type=metric_type
                )
                
                self.metrics[name].append(metric)
                
                # Update counter if it's a counter type
                if metric_type == MetricType.COUNTER:
                    counter_key = f"{name}_total"
                    if counter_key not in self.metric_summaries:
                        self.metric_summaries[counter_key] = 0
                    self.metric_summaries[counter_key] += value
                
        except Exception as e:
            logger.error(f"Failed to record metric {name}: {e}")
    
    def record_deployment_start(self, deployment_id: str, environment: str, models_count: int):
        """Record deployment start"""
        try:
            analytics = DeploymentAnalytics(
                deployment_id=deployment_id,
                start_time=time.time(),
                end_time=None,
                success=False,
                environment=environment,
                models_deployed=0,
                models_failed=0,
                total_time=0
            )
            
            self.deployment_analytics[deployment_id] = analytics
            
            # Record metric
            self.record_metric("deployment_started_total", 1, 
                             {"environment": environment}, MetricType.COUNTER)
            
            logger.info(f"Recorded deployment start: {deployment_id}")
            
        except Exception as e:
            logger.error(f"Failed to record deployment start: {e}")
    
    def record_deployment_end(self, deployment_id: str, success: bool, 
                            models_deployed: int, models_failed: int):
        """Record deployment completion"""
        try:
            if deployment_id not in self.deployment_analytics:
                logger.warning(f"No deployment analytics found for {deployment_id}")
                return
            
            analytics = self.deployment_analytics[deployment_id]
            analytics.end_time = time.time()
            analytics.success = success
            analytics.models_deployed = models_deployed
            analytics.models_failed = models_failed
            analytics.total_time = analytics.end_time - analytics.start_time
            
            # Record metrics
            status = "success" if success else "failure"
            self.record_metric("deployment_completed_total", 1,
                             {"environment": analytics.environment, "status": status}, 
                             MetricType.COUNTER)
            
            self.record_metric("deployment_duration_seconds", analytics.total_time,
                             {"environment": analytics.environment})
            
            self.record_metric("models_deployed_total", models_deployed,
                             {"environment": analytics.environment})
            
            if models_failed > 0:
                self.record_metric("models_failed_total", models_failed,
                               {"environment": analytics.environment}, MetricType.COUNTER)
            
            logger.info(f"Recorded deployment end: {deployment_id} ({status})")
            
        except Exception as e:
            logger.error(f"Failed to record deployment end: {e}")
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            import psutil
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric("system_cpu_percent", cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.record_metric("system_memory_used_bytes", memory.used)
            self.record_metric("system_memory_percent", memory.percent)
            self.record_metric("system_memory_available_bytes", memory.available)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.record_metric("system_disk_used_bytes", disk.used)
            self.record_metric("system_disk_percent", disk.percent)
            
            # Network connections
            try:
                connections = len(psutil.net_connections())
                self.record_metric("system_connections_active", connections)
            except:
                pass
                
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    async def _collect_component_metrics(self):
        """Collect metrics from integrated components"""
        try:
            # Health monitor metrics
            if self.health_monitor:
                health_report = self.health_monitor.get_health_report()
                
                # Overall health status
                health_score_map = {"healthy": 1, "degraded": 0.8, "unhealthy": 0.5, "critical": 0.2, "unknown": 0}
                health_score = health_score_map.get(health_report.get("overall_status", "unknown"), 0)
                self.record_metric("system_health_score", health_score)
                
                # System metrics from health monitor
                if "system_metrics" in health_report:
                    sys_metrics = health_report["system_metrics"]
                    for metric_name, value in sys_metrics.items():
                        if isinstance(value, (int, float)):
                            self.record_metric(f"health_monitor_{metric_name}", value)
            
            # Service discovery metrics
            if self.service_discovery:
                topology = self.service_discovery.get_service_topology()
                self.record_metric("service_instances_total", topology["total_instances"])
                self.record_metric("service_instances_active", topology["active_instances"])
                
                # Per-environment metrics
                for env_name, env_data in topology.get("environments", {}).items():
                    self.record_metric("service_instances_per_env", env_data["active_instances"],
                                     {"environment": env_name})
            
            # Blue-green orchestrator metrics
            if self.blue_green_orchestrator:
                status = self.blue_green_orchestrator.get_orchestrator_status()
                
                # Traffic weights
                for env_name, weight in status.get("traffic_weights", {}).items():
                    self.record_metric("traffic_weight", weight, {"environment": env_name})
                
                # Deployment state
                deployment_states = {"running": 0, "in_progress": 1}
                state_value = deployment_states.get(
                    "in_progress" if status.get("deployment_in_progress") else "running", 0
                )
                self.record_metric("blue_green_deployment_state", state_value)
            
            # Deployment manager metrics
            for i, manager in enumerate(self.deployment_managers):
                try:
                    status = manager.get_deployment_status()
                    env_label = f"manager_{i}"
                    
                    self.record_metric("deployment_manager_active", status["active_deployments"],
                                     {"manager": env_label})
                    self.record_metric("deployment_manager_staging", status["staging_deployments"],
                                     {"manager": env_label})
                    self.record_metric("deployment_manager_memory_mb", status["memory_usage_mb"],
                                     {"manager": env_label})
                except Exception as e:
                    logger.error(f"Failed to collect metrics from deployment manager {i}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to collect component metrics: {e}")
    
    async def _run_custom_collectors(self):
        """Run custom metric collectors"""
        for collector in self.custom_collectors:
            try:
                if asyncio.iscoroutinefunction(collector):
                    await collector(self)
                else:
                    collector(self)
            except Exception as e:
                logger.error(f"Custom collector failed: {e}")
    
    def register_custom_collector(self, collector: Callable):
        """Register a custom metric collector"""
        self.custom_collectors.append(collector)
        logger.info("   Registered custom metric collector")
    
    def create_alert(self, level: AlertLevel, title: str, message: str, 
                    component: str, metadata: Dict[str, Any] = None) -> str:
        """Create a new alert"""
        try:
            alert_id = f"alert_{int(time.time() * 1000)}_{component}"
            
            alert = Alert(
                id=alert_id,
                level=level,
                title=title,
                message=message,
                component=component,
                timestamp=time.time(),
                metadata=metadata or {}
            )
            
            with self.monitor_lock:
                self.alerts[alert_id] = alert
            
            # Record alert metric
            self.record_metric("alerts_total", 1, 
                             {"level": level.value, "component": component}, 
                             MetricType.COUNTER)
            
            # Notify callbacks
            self._notify_alert_callbacks(alert)
            
            logger.warning(f"🚨 Alert created: {title} ({level.value})")
            return alert_id
            
        except Exception as e:
            logger.error(f"Failed to create alert: {e}")
            return ""
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        try:
            with self.monitor_lock:
                if alert_id in self.alerts:
                    alert = self.alerts[alert_id]
                    if not alert.resolved:
                        alert.resolved = True
                        alert.resolved_timestamp = time.time()
                        
                        logger.info(f"✅ Alert resolved: {alert.title}")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to resolve alert {alert_id}: {e}")
            return False
    
    def _setup_default_alert_rules(self):
        """Setup default alerting rules"""
        
        def high_cpu_rule():
            cpu_metrics = self.get_recent_metrics("system_cpu_percent", minutes=5)
            if cpu_metrics and len(cpu_metrics) > 0:
                avg_cpu = statistics.mean([m.value for m in cpu_metrics])
                if avg_cpu > 90:
                    return {
                        "level": AlertLevel.CRITICAL,
                        "title": "High CPU Usage",
                        "message": f"CPU usage is {avg_cpu:.1f}% (threshold: 90%)",
                        "component": "system"
                    }
            return None
        
        def high_memory_rule():
            memory_metrics = self.get_recent_metrics("system_memory_percent", minutes=5)
            if memory_metrics and len(memory_metrics) > 0:
                avg_memory = statistics.mean([m.value for m in memory_metrics])
                if avg_memory > 95:
                    return {
                        "level": AlertLevel.CRITICAL,
                        "title": "High Memory Usage",
                        "message": f"Memory usage is {avg_memory:.1f}% (threshold: 95%)",
                        "component": "system"
                    }
            return None
        
        def deployment_failures_rule():
            failure_metrics = self.get_recent_metrics("models_failed_total", minutes=10)
            if failure_metrics and len(failure_metrics) > 0:
                total_failures = sum([m.value for m in failure_metrics])
                if total_failures > 10:
                    return {
                        "level": AlertLevel.ERROR,
                        "title": "High Deployment Failures",
                        "message": f"Multiple model deployment failures: {total_failures}",
                        "component": "deployment"
                    }
            return None
        
        def health_degradation_rule():
            health_metrics = self.get_recent_metrics("system_health_score", minutes=5)
            if health_metrics and len(health_metrics) > 0:
                avg_health = statistics.mean([m.value for m in health_metrics])
                if avg_health < 0.7:
                    return {
                        "level": AlertLevel.WARNING,
                        "title": "System Health Degradation",
                        "message": f"System health score is {avg_health:.2f} (threshold: 0.7)",
                        "component": "health"
                    }
            return None
        
        self.alert_rules = [high_cpu_rule, high_memory_rule, deployment_failures_rule, health_degradation_rule]
        logger.info(f"   Setup {len(self.alert_rules)} default alert rules")
    
    async def _check_alert_rules(self):
        """Check all alert rules"""
        for rule in self.alert_rules:
            try:
                result = rule()
                if result:
                    # Check if similar alert already exists and is unresolved
                    similar_alert = None
                    with self.monitor_lock:
                        for alert in self.alerts.values():
                            if (alert.component == result["component"] and 
                                alert.title == result["title"] and 
                                not alert.resolved):
                                similar_alert = alert
                                break
                    
                    if not similar_alert:
                        self.create_alert(**result)
                        
            except Exception as e:
                logger.error(f"Alert rule check failed: {e}")
    
    def register_alert_callback(self, callback: Callable[[Alert], None]):
        """Register callback for alert notifications"""
        self.alert_callbacks.append(callback)
        logger.info("   Registered alert callback")
    
    def _notify_alert_callbacks(self, alert: Alert):
        """Notify alert callbacks"""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def _on_service_change(self, service_name: str, event_type: str, instance):
        """Handle service change events"""
        try:
            self.record_metric("service_changes_total", 1,
                             {"service": service_name, "event": event_type, 
                              "environment": instance.environment},
                             MetricType.COUNTER)
            
            if event_type == "failed":
                self.create_alert(
                    AlertLevel.WARNING,
                    "Service Instance Failed",
                    f"Service instance {service_name}/{instance.id} has failed",
                    "service_discovery",
                    {"service": service_name, "instance": instance.id}
                )
                
        except Exception as e:
            logger.error(f"Failed to handle service change: {e}")
    
    def get_recent_metrics(self, metric_name: str, minutes: int = 5) -> List[MetricValue]:
        """Get recent metrics for a specific metric"""
        try:
            cutoff_time = time.time() - (minutes * 60)
            
            with self.monitor_lock:
                if metric_name in self.metrics:
                    return [m for m in self.metrics[metric_name] if m.timestamp >= cutoff_time]
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get recent metrics for {metric_name}: {e}")
            return []
    
    async def _update_metric_summaries(self):
        """Update metric summaries and baselines"""
        try:
            with self.monitor_lock:
                for metric_name, metric_deque in self.metrics.items():
                    if not metric_deque:
                        continue
                    
                    recent_values = [m.value for m in metric_deque if time.time() - m.timestamp < 3600]  # Last hour
                    
                    if recent_values:
                        summary = {
                            "count": len(recent_values),
                            "mean": statistics.mean(recent_values),
                            "min": min(recent_values),
                            "max": max(recent_values)
                        }
                        
                        if len(recent_values) > 1:
                            summary["stddev"] = statistics.stdev(recent_values)
                        
                        self.metric_summaries[metric_name] = summary
                        
        except Exception as e:
            logger.error(f"Failed to update metric summaries: {e}")
    
    async def _cleanup_old_data(self):
        """Clean up old metrics and alerts"""
        try:
            current_time = time.time()
            metrics_cutoff = current_time - (self.metrics_retention_hours * 3600)
            alerts_cutoff = current_time - (self.alert_retention_hours * 3600)
            
            # Clean old metrics
            with self.monitor_lock:
                for metric_name, metric_deque in self.metrics.items():
                    # Remove old entries
                    while metric_deque and metric_deque[0].timestamp < metrics_cutoff:
                        metric_deque.popleft()
                
                # Clean old alerts
                old_alerts = [
                    alert_id for alert_id, alert in self.alerts.items()
                    if alert.timestamp < alerts_cutoff and alert.resolved
                ]
                
                for alert_id in old_alerts:
                    del self.alerts[alert_id]
            
            # Clean old deployment analytics
            old_deployments = [
                dep_id for dep_id, analytics in self.deployment_analytics.items()
                if analytics.start_time < metrics_cutoff
            ]
            
            for dep_id in old_deployments:
                del self.deployment_analytics[dep_id]
                
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data"""
        try:
            with self.monitor_lock:
                # Active alerts
                active_alerts = [
                    {
                        "id": alert.id,
                        "level": alert.level.value,
                        "title": alert.title,
                        "message": alert.message,
                        "component": alert.component,
                        "timestamp": alert.timestamp
                    }
                    for alert in self.alerts.values()
                    if not alert.resolved
                ]
                
                # Recent deployment analytics
                recent_deployments = []
                cutoff_time = time.time() - 3600  # Last hour
                
                for analytics in self.deployment_analytics.values():
                    if analytics.start_time >= cutoff_time:
                        recent_deployments.append({
                            "id": analytics.deployment_id,
                            "environment": analytics.environment,
                            "success": analytics.success,
                            "models_deployed": analytics.models_deployed,
                            "models_failed": analytics.models_failed,
                            "duration": analytics.total_time,
                            "start_time": analytics.start_time
                        })
                
                # Key metrics summary
                key_metrics = {}
                for metric_name in ["system_cpu_percent", "system_memory_percent", "system_health_score",
                                  "service_instances_active", "deployment_manager_active"]:
                    if metric_name in self.metric_summaries:
                        key_metrics[metric_name] = self.metric_summaries[metric_name]
                
                dashboard = {
                    "timestamp": time.time(),
                    "monitoring_active": self.monitoring_active,
                    "active_alerts": active_alerts,
                    "alert_counts": {
                        "total": len(self.alerts),
                        "active": len(active_alerts),
                        "critical": len([a for a in active_alerts if a["level"] == "critical"])
                    },
                    "recent_deployments": recent_deployments,
                    "deployment_summary": {
                        "total_deployments": len(self.deployment_analytics),
                        "successful": len([a for a in self.deployment_analytics.values() if a.success]),
                        "failed": len([a for a in self.deployment_analytics.values() if not a.success and a.end_time])
                    },
                    "key_metrics": key_metrics,
                    "system_status": {
                        "components_integrated": {
                            "health_monitor": self.health_monitor is not None,
                            "service_discovery": self.service_discovery is not None,
                            "blue_green_orchestrator": self.blue_green_orchestrator is not None,
                            "deployment_managers": len(self.deployment_managers)
                        }
                    }
                }
                
                return dashboard
                
        except Exception as e:
            logger.error(f"Failed to generate monitoring dashboard: {e}")
            return {"error": str(e)}
    
    def export_metrics_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        try:
            lines = []
            
            with self.monitor_lock:
                for metric_name, metric_deque in self.metrics.items():
                    if not metric_deque:
                        continue
                    
                    latest_metric = metric_deque[-1]
                    
                    # Format labels
                    label_str = ""
                    if latest_metric.labels:
                        label_pairs = [f'{k}="{v}"' for k, v in latest_metric.labels.items()]
                        label_str = "{" + ",".join(label_pairs) + "}"
                    
                    # Add metric line
                    lines.append(f"{metric_name}{label_str} {latest_metric.value}")
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Failed to export Prometheus metrics: {e}")
            return ""