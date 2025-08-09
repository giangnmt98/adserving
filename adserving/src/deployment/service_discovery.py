"""
Advanced Service Discovery and Load Balancing
Manages service registration, discovery, and intelligent load balancing for ultra-scale deployments
"""

import asyncio
import time
import hashlib
from typing import Dict, List, Optional, Any, Set, Callable
from enum import Enum
from dataclasses import dataclass, field
from threading import RLock
import random
import logging

import ray
from ray import serve

from .health_monitor import HealthStatus, HealthMonitor
from ..utils.logger import get_logger

logger = get_logger()


class ServiceStatus(Enum):
    """Service instance status"""
    ACTIVE = "active"
    DRAINING = "draining"
    INACTIVE = "inactive"
    FAILED = "failed"
    STARTING = "starting"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    HASH_BASED = "hash_based"
    HEALTH_AWARE = "health_aware"


@dataclass
class ServiceInstance:
    """Service instance information"""
    id: str
    name: str
    environment: str
    address: str
    port: int
    status: ServiceStatus
    weight: float = 1.0
    current_connections: int = 0
    total_requests: int = 0
    avg_response_time: float = 0.0
    health_score: float = 1.0
    last_heartbeat: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    deployment_time: float = field(default_factory=time.time)


@dataclass
class ServiceEndpoint:
    """Service endpoint with load balancing"""
    name: str
    instances: List[ServiceInstance] = field(default_factory=list)
    load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.HEALTH_AWARE
    sticky_sessions: bool = False
    health_check_interval: float = 30.0
    last_health_check: float = 0.0


class ServiceDiscovery:
    """Advanced service discovery and load balancing system"""
    
    def __init__(self, heartbeat_interval=30, health_check_interval=60, stale_threshold=120):
        self.heartbeat_interval = heartbeat_interval
        self.health_check_interval = health_check_interval
        self.stale_threshold = stale_threshold
        
        # Service registry
        self.services = {}  # service_name -> ServiceEndpoint
        self.instances = {}  # instance_id -> ServiceInstance
        self.environments = {}  # env_name -> set of instance_ids
        
        # Load balancing state
        self.round_robin_counters = {}
        self.sticky_sessions = {}  # session_id -> instance_id
        
        # Discovery state
        self.discovery_active = False
        self.discovery_task = None
        
        # Thread safety
        self.registry_lock = RLock()
        
        # Health monitoring integration
        self.health_monitor = None
        
        # Callbacks
        self.service_change_callbacks = []
        
        logger.info("ServiceDiscovery initialized")
        logger.info(f"   - Heartbeat interval: {heartbeat_interval}s")
        logger.info(f"   - Health check interval: {health_check_interval}s")
        logger.info(f"   - Stale threshold: {stale_threshold}s")
    
    def set_health_monitor(self, health_monitor: HealthMonitor):
        """Set health monitor for integration"""
        self.health_monitor = health_monitor
        logger.info("   Health monitor integration enabled")
    
    def register_service_change_callback(self, callback: Callable[[str, str, ServiceInstance], None]):
        """Register callback for service changes (service_name, event_type, instance)"""
        self.service_change_callbacks.append(callback)
    
    async def start_discovery(self):
        """Start service discovery"""
        if self.discovery_active:
            logger.warning("Service discovery already active")
            return
        
        self.discovery_active = True
        self.discovery_task = asyncio.create_task(self._discovery_loop())
        
        logger.info("Service discovery started")
    
    async def stop_discovery(self):
        """Stop service discovery"""
        self.discovery_active = False
        
        if self.discovery_task:
            self.discovery_task.cancel()
            try:
                await self.discovery_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Service discovery stopped")
    
    async def _discovery_loop(self):
        """Main discovery loop"""
        try:
            while self.discovery_active:
                try:
                    # Check for stale instances
                    await self._cleanup_stale_instances()
                    
                    # Update health scores
                    await self._update_health_scores()
                    
                    # Perform service health checks
                    await self._perform_service_health_checks()
                    
                    # Wait for next cycle
                    await asyncio.sleep(self.heartbeat_interval)
                    
                except Exception as e:
                    logger.error(f"Error in discovery loop: {e}")
                    await asyncio.sleep(5)
                    
        except asyncio.CancelledError:
            logger.info("Discovery loop cancelled")
        except Exception as e:
            logger.error(f"Discovery loop failed: {e}")
    
    def register_service_instance(self, service_name: str, instance: ServiceInstance) -> bool:
        """Register a service instance"""
        try:
            with self.registry_lock:
                # Ensure service exists
                if service_name not in self.services:
                    self.services[service_name] = ServiceEndpoint(name=service_name)
                
                service = self.services[service_name]
                
                # Check if instance already exists
                existing_instance = next(
                    (inst for inst in service.instances if inst.id == instance.id),
                    None
                )
                
                if existing_instance:
                    # Update existing instance
                    existing_instance.status = instance.status
                    existing_instance.weight = instance.weight
                    existing_instance.last_heartbeat = time.time()
                    existing_instance.metadata = instance.metadata
                    logger.info(f"   Updated service instance: {service_name}/{instance.id}")
                else:
                    # Add new instance
                    service.instances.append(instance)
                    self.instances[instance.id] = instance
                    
                    # Track by environment
                    if instance.environment not in self.environments:
                        self.environments[instance.environment] = set()
                    self.environments[instance.environment].add(instance.id)
                    
                    logger.info(f"   Registered service instance: {service_name}/{instance.id} in {instance.environment}")
                
                # Initialize load balancing state
                if service_name not in self.round_robin_counters:
                    self.round_robin_counters[service_name] = 0
                
                # Notify callbacks
                self._notify_service_change(service_name, "registered", instance)
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to register service instance {service_name}/{instance.id}: {e}")
            return False
    
    def unregister_service_instance(self, service_name: str, instance_id: str) -> bool:
        """Unregister a service instance"""
        try:
            with self.registry_lock:
                if service_name not in self.services:
                    return False
                
                service = self.services[service_name]
                instance = next(
                    (inst for inst in service.instances if inst.id == instance_id),
                    None
                )
                
                if not instance:
                    return False
                
                # Remove from service
                service.instances.remove(instance)
                
                # Remove from global registry
                if instance_id in self.instances:
                    del self.instances[instance_id]
                
                # Remove from environment tracking
                for env_instances in self.environments.values():
                    env_instances.discard(instance_id)
                
                # Clean up empty environments
                self.environments = {
                    env: instances for env, instances in self.environments.items()
                    if instances
                }
                
                # Notify callbacks
                self._notify_service_change(service_name, "unregistered", instance)
                
                logger.info(f"   Unregistered service instance: {service_name}/{instance_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to unregister service instance {service_name}/{instance_id}: {e}")
            return False
    
    def update_instance_metrics(self, instance_id: str, metrics: Dict[str, Any]):
        """Update instance metrics"""
        try:
            with self.registry_lock:
                instance = self.instances.get(instance_id)
                if not instance:
                    return False
                
                # Update metrics
                if "current_connections" in metrics:
                    instance.current_connections = metrics["current_connections"]
                
                if "total_requests" in metrics:
                    instance.total_requests = metrics["total_requests"]
                
                if "avg_response_time" in metrics:
                    instance.avg_response_time = metrics["avg_response_time"]
                
                if "health_score" in metrics:
                    instance.health_score = metrics["health_score"]
                
                instance.last_heartbeat = time.time()
                return True
                
        except Exception as e:
            logger.error(f"Failed to update metrics for instance {instance_id}: {e}")
            return False
    
    async def get_service_instance(self, service_name: str, 
                                  session_id: Optional[str] = None,
                                  environment: Optional[str] = None) -> Optional[ServiceInstance]:
        """Get service instance using load balancing"""
        try:
            with self.registry_lock:
                if service_name not in self.services:
                    return None
                
                service = self.services[service_name]
                available_instances = [
                    inst for inst in service.instances
                    if inst.status == ServiceStatus.ACTIVE and
                    (not environment or inst.environment == environment)
                ]
                
                if not available_instances:
                    logger.warning(f"No available instances for service {service_name}")
                    return None
                
                # Handle sticky sessions
                if session_id and service.sticky_sessions:
                    if session_id in self.sticky_sessions:
                        sticky_instance_id = self.sticky_sessions[session_id]
                        sticky_instance = next(
                            (inst for inst in available_instances if inst.id == sticky_instance_id),
                            None
                        )
                        if sticky_instance:
                            return sticky_instance
                
                # Apply load balancing strategy
                selected_instance = await self._apply_load_balancing(
                    service, available_instances
                )
                
                if selected_instance and session_id and service.sticky_sessions:
                    self.sticky_sessions[session_id] = selected_instance.id
                
                return selected_instance
                
        except Exception as e:
            logger.error(f"Failed to get service instance for {service_name}: {e}")
            return None
    
    async def _apply_load_balancing(self, service: ServiceEndpoint, 
                                   instances: List[ServiceInstance]) -> Optional[ServiceInstance]:
        """Apply load balancing strategy"""
        if not instances:
            return None
        
        strategy = service.load_balancing_strategy
        
        if strategy == LoadBalancingStrategy.ROUND_ROBIN:
            counter = self.round_robin_counters.get(service.name, 0)
            selected = instances[counter % len(instances)]
            self.round_robin_counters[service.name] = (counter + 1) % len(instances)
            return selected
        
        elif strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return min(instances, key=lambda x: x.current_connections)
        
        elif strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            # Calculate weighted selection
            total_weight = sum(inst.weight for inst in instances)
            if total_weight == 0:
                return random.choice(instances)
            
            counter = self.round_robin_counters.get(service.name, 0)
            weighted_instances = []
            for inst in instances:
                weight_count = int(inst.weight * 100)  # Scale weights
                weighted_instances.extend([inst] * weight_count)
            
            if weighted_instances:
                selected = weighted_instances[counter % len(weighted_instances)]
                self.round_robin_counters[service.name] = (counter + 1) % len(weighted_instances)
                return selected
            
            return random.choice(instances)
        
        elif strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return min(instances, key=lambda x: x.avg_response_time)
        
        elif strategy == LoadBalancingStrategy.HASH_BASED:
            # Simple hash-based selection (would use request info in real implementation)
            hash_val = hash(service.name + str(time.time() // 60))  # Change every minute
            return instances[hash_val % len(instances)]
        
        elif strategy == LoadBalancingStrategy.HEALTH_AWARE:
            # Filter by health score and use weighted selection
            healthy_instances = [
                inst for inst in instances
                if inst.health_score >= 0.7  # Minimum health threshold
            ]
            
            if not healthy_instances:
                healthy_instances = instances  # Fallback to all instances
            
            # Weight by health score and inverse response time
            weights = []
            for inst in healthy_instances:
                weight = inst.health_score * inst.weight / max(inst.avg_response_time, 0.001)
                weights.append(weight)
            
            if sum(weights) == 0:
                return random.choice(healthy_instances)
            
            # Weighted random selection
            total_weight = sum(weights)
            r = random.uniform(0, total_weight)
            
            cumulative = 0
            for i, weight in enumerate(weights):
                cumulative += weight
                if r <= cumulative:
                    return healthy_instances[i]
            
            return healthy_instances[-1]  # Fallback
        
        # Default: random selection
        return random.choice(instances)
    
    async def _cleanup_stale_instances(self):
        """Remove stale service instances"""
        current_time = time.time()
        stale_instances = []
        
        with self.registry_lock:
            for instance in self.instances.values():
                if current_time - instance.last_heartbeat > self.stale_threshold:
                    stale_instances.append((instance.name, instance.id))
        
        # Remove stale instances
        for service_name, instance_id in stale_instances:
            logger.warning(f"Removing stale instance: {service_name}/{instance_id}")
            self.unregister_service_instance(service_name, instance_id)
    
    async def _update_health_scores(self):
        """Update health scores from health monitor"""
        if not self.health_monitor:
            return
        
        try:
            health_report = self.health_monitor.get_health_report()
            overall_status = health_report.get("overall_status", "unknown")
            
            # Map health status to score
            health_score_map = {
                "healthy": 1.0,
                "degraded": 0.8,
                "unhealthy": 0.5,
                "critical": 0.2,
                "unknown": 0.5
            }
            
            system_health_score = health_score_map.get(overall_status, 0.5)
            
            # Update all instances with system health score
            with self.registry_lock:
                for instance in self.instances.values():
                    # Combine instance health with system health
                    instance.health_score = min(instance.health_score, system_health_score)
                    
        except Exception as e:
            logger.error(f"Failed to update health scores: {e}")
    
    async def _perform_service_health_checks(self):
        """Perform health checks on service endpoints"""
        current_time = time.time()
        
        with self.registry_lock:
            services_to_check = [
                service for service in self.services.values()
                if current_time - service.last_health_check > service.health_check_interval
            ]
        
        for service in services_to_check:
            try:
                # Simple health check: mark instances as failed if no heartbeat
                current_time = time.time()
                
                for instance in service.instances:
                    if current_time - instance.last_heartbeat > self.stale_threshold * 0.5:
                        if instance.status == ServiceStatus.ACTIVE:
                            instance.status = ServiceStatus.FAILED
                            logger.warning(f"Marking instance as failed: {service.name}/{instance.id}")
                            
                            # Notify callbacks
                            self._notify_service_change(service.name, "failed", instance)
                
                service.last_health_check = current_time
                
            except Exception as e:
                logger.error(f"Health check failed for service {service.name}: {e}")
    
    def _notify_service_change(self, service_name: str, event_type: str, instance: ServiceInstance):
        """Notify registered callbacks of service changes"""
        for callback in self.service_change_callbacks:
            try:
                callback(service_name, event_type, instance)
            except Exception as e:
                logger.error(f"Service change callback error: {e}")
    
    def get_service_topology(self) -> Dict[str, Any]:
        """Get complete service topology"""
        with self.registry_lock:
            topology = {
                "services": {},
                "environments": {},
                "total_instances": len(self.instances),
                "active_instances": sum(
                    1 for inst in self.instances.values()
                    if inst.status == ServiceStatus.ACTIVE
                ),
                "timestamp": time.time()
            }
            
            # Service details
            for service_name, service in self.services.items():
                topology["services"][service_name] = {
                    "total_instances": len(service.instances),
                    "active_instances": sum(
                        1 for inst in service.instances
                        if inst.status == ServiceStatus.ACTIVE
                    ),
                    "load_balancing_strategy": service.load_balancing_strategy.value,
                    "sticky_sessions": service.sticky_sessions,
                    "instances": [
                        {
                            "id": inst.id,
                            "environment": inst.environment,
                            "status": inst.status.value,
                            "weight": inst.weight,
                            "current_connections": inst.current_connections,
                            "health_score": inst.health_score,
                            "avg_response_time": inst.avg_response_time,
                            "last_heartbeat": inst.last_heartbeat
                        }
                        for inst in service.instances
                    ]
                }
            
            # Environment details
            for env_name, instance_ids in self.environments.items():
                active_count = sum(
                    1 for inst_id in instance_ids
                    if self.instances.get(inst_id, {}).get("status") == ServiceStatus.ACTIVE
                )
                
                topology["environments"][env_name] = {
                    "total_instances": len(instance_ids),
                    "active_instances": active_count,
                    "instance_ids": list(instance_ids)
                }
            
            return topology
    
    def configure_service_load_balancing(self, service_name: str, 
                                       strategy: LoadBalancingStrategy,
                                       sticky_sessions: bool = False) -> bool:
        """Configure load balancing for a service"""
        try:
            with self.registry_lock:
                if service_name not in self.services:
                    self.services[service_name] = ServiceEndpoint(name=service_name)
                
                service = self.services[service_name]
                service.load_balancing_strategy = strategy
                service.sticky_sessions = sticky_sessions
                
                logger.info(f"   Configured load balancing for {service_name}: {strategy.value}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to configure load balancing for {service_name}: {e}")
            return False
    
    async def drain_service_instance(self, service_name: str, instance_id: str) -> bool:
        """Gracefully drain a service instance"""
        try:
            with self.registry_lock:
                instance = self.instances.get(instance_id)
                if not instance or instance.name != service_name:
                    return False
                
                if instance.status == ServiceStatus.ACTIVE:
                    instance.status = ServiceStatus.DRAINING
                    logger.info(f"   Draining service instance: {service_name}/{instance_id}")
                    
                    # Notify callbacks
                    self._notify_service_change(service_name, "draining", instance)
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to drain service instance {service_name}/{instance_id}: {e}")
            
        return False