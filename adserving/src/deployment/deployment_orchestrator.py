"""
Ultra-Scale Deployment Orchestrator
Integrates all deployment components for comprehensive model serving orchestration
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable
import logging

from .ultra_scale_deployment_manager import UltraScaleDeploymentManager
from .blue_green_orchestrator import BlueGreenOrchestrator, Environment
from .health_monitor import HealthMonitor, GracefulShutdownManager
from .service_discovery import ServiceDiscovery, ServiceInstance, ServiceStatus
from .monitoring_system import MonitoringSystem
from .deployment_config import DeploymentConfig, ConfigurationManager
from ..utils.logger import get_logger

logger = get_logger()


class UltraScaleDeploymentOrchestrator:
    """Master orchestrator for ultra-scale model deployment system"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        
        # Initialize core components
        self.health_monitor = HealthMonitor(
            check_interval=config.health_checks.check_interval,
            degraded_threshold=config.health_checks.degraded_threshold,
            unhealthy_threshold=config.health_checks.unhealthy_threshold
        )
        
        self.service_discovery = ServiceDiscovery(
            heartbeat_interval=config.service_discovery.heartbeat_interval,
            stale_threshold=config.service_discovery.stale_threshold
        )
        
        self.blue_green_orchestrator = BlueGreenOrchestrator(
            max_models_per_env=config.model_tiers.hot_tier_max_models,
            max_memory_per_env=config.model_tiers.hot_tier_memory_mb
        )
        
        self.monitoring_system = MonitoringSystem(
            metrics_retention_hours=config.monitoring.metrics_retention_hours,
            alert_retention_hours=config.monitoring.alert_retention_hours
        )
        
        self.shutdown_manager = GracefulShutdownManager(
            shutdown_timeout=30,
            drain_timeout=60
        )
        
        # Orchestrator state
        self.is_running = False
        self.initialization_complete = False
        self.deployment_stats = {
            "total_deployments": 0,
            "successful_deployments": 0,
            "failed_deployments": 0,
            "models_deployed": 0,
            "avg_deployment_time": 0.0
        }
        
        logger.info("UltraScaleDeploymentOrchestrator initialized")
        logger.info(f"   - Environment: {config.environment_name}")
        logger.info(f"   - Mode: {config.mode.value}")
        logger.info(f"   - Target models: {config.target_model_count}")
    
    async def initialize(self) -> bool:
        """Initialize the entire deployment system"""
        try:
            logger.info("Initializing ultra-scale deployment system...")
            start_time = time.time()
            
            # Step 1: Integrate components
            logger.info("Step 1: Integrating system components...")
            self.monitoring_system.integrate_components(
                health_monitor=self.health_monitor,
                service_discovery=self.service_discovery,
                blue_green_orchestrator=self.blue_green_orchestrator,
                deployment_managers=[]  # Will be populated later
            )
            
            # Set up cross-component integrations
            self.service_discovery.set_health_monitor(self.health_monitor)
            
            # Step 2: Start core services
            logger.info("Step 2: Starting core monitoring services...")
            await self.health_monitor.start_monitoring()
            await self.service_discovery.start_discovery()
            await self.monitoring_system.start_monitoring()
            
            # Step 3: Register shutdown handlers
            logger.info("Step 3: Registering shutdown handlers...")
            self.shutdown_manager.register_shutdown_handler(
                self._shutdown_monitoring_services, priority=10
            )
            self.shutdown_manager.register_shutdown_handler(
                self._shutdown_deployment_services, priority=20
            )
            
            # Step 4: Setup alert callbacks
            self.monitoring_system.register_alert_callback(self._handle_system_alert)
            
            # Step 5: Register health checks
            self._register_health_checks()
            
            initialization_time = time.time() - start_time
            self.initialization_complete = True
            self.is_running = True
            
            logger.info(f"System initialization completed in {initialization_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False
    
    async def deploy_models_ultra_scale(self, model_configs: List[Dict[str, Any]], 
                                      deployment_strategy: str = "blue_green") -> Dict[str, Any]:
        """Deploy models at ultra-scale with specified strategy"""
        if not self.is_running:
            raise RuntimeError("Orchestrator not initialized")
        
        deployment_id = f"deployment_{int(time.time() * 1000)}"
        logger.info(f"Starting ultra-scale deployment: {deployment_id}")
        logger.info(f"   - Models to deploy: {len(model_configs)}")
        logger.info(f"   - Strategy: {deployment_strategy}")
        
        # Record deployment start
        self.monitoring_system.record_deployment_start(
            deployment_id, self.config.environment_name, len(model_configs)
        )
        
        start_time = time.time()
        
        try:
            if deployment_strategy == "blue_green":
                result = await self.blue_green_orchestrator.deploy_models_blue_green(model_configs)
            else:
                # Direct deployment to single environment
                target_env = self.blue_green_orchestrator.current_active_env
                env_manager = self.blue_green_orchestrator.environments[target_env].deployment_manager
                result = await env_manager.deploy_models_ultra_scale(model_configs)
            
            # Update statistics
            deployment_time = time.time() - start_time
            success = result.get("success", result.get("deployed", 0) > 0)
            
            self._update_deployment_stats(deployment_time, success, result)
            
            # Record deployment completion
            self.monitoring_system.record_deployment_end(
                deployment_id,
                success,
                result.get("deployed", 0),
                result.get("failed", 0)
            )
            
            if success:
                logger.info(f"Ultra-scale deployment completed: {deployment_id}")
            else:
                logger.error(f"Ultra-scale deployment failed: {deployment_id}")
            
            return {
                "deployment_id": deployment_id,
                "success": success,
                "deployment_time": deployment_time,
                **result
            }
            
        except Exception as e:
            logger.error(f"Ultra-scale deployment error: {e}")
            
            # Record failed deployment
            self.monitoring_system.record_deployment_end(deployment_id, False, 0, len(model_configs))
            self._update_deployment_stats(time.time() - start_time, False, {"deployed": 0, "failed": len(model_configs)})
            
            return {
                "deployment_id": deployment_id,
                "success": False,
                "error": str(e),
                "deployed": 0,
                "failed": len(model_configs)
            }
    
    async def update_models(self, model_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Update existing models with zero-downtime"""
        logger.info(f"🔄 Starting model updates: {len(model_updates)} models")
        
        try:
            # Use blue-green deployment for updates
            result = await self.blue_green_orchestrator.deploy_models_blue_green(model_updates)
            
            if result["success"]:
                logger.info(f"Model updates completed successfully")
            else:
                logger.error(f"Model updates failed")
            
            return result
            
        except Exception as e:
            logger.error(f"Model update error: {e}")
            return {"success": False, "error": str(e)}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            # Get component statuses
            health_status = self.health_monitor.get_health_report()
            service_topology = self.service_discovery.get_service_topology()
            orchestrator_status = self.blue_green_orchestrator.get_orchestrator_status()
            monitoring_dashboard = self.monitoring_system.get_monitoring_dashboard()
            shutdown_status = self.shutdown_manager.get_shutdown_status()
            
            # Compile comprehensive status
            system_status = {
                "timestamp": time.time(),
                "orchestrator": {
                    "running": self.is_running,
                    "initialized": self.initialization_complete,
                    "environment": self.config.environment_name,
                    "mode": self.config.mode.value,
                    "version": self.config.version
                },
                "deployment_stats": self.deployment_stats,
                "health": {
                    "overall_status": health_status.get("overall_status", "unknown"),
                    "monitoring_active": health_status.get("monitoring_active", False),
                    "total_checks": health_status.get("total_checks", 0)
                },
                "services": {
                    "total_instances": service_topology.get("total_instances", 0),
                    "active_instances": service_topology.get("active_instances", 0),
                    "environments": len(service_topology.get("environments", {}))
                },
                "blue_green": {
                    "active_environment": orchestrator_status.get("current_active_env", "unknown"),
                    "deployment_in_progress": orchestrator_status.get("deployment_in_progress", False),
                    "rollback_available": orchestrator_status.get("rollback_available", False)
                },
                "monitoring": {
                    "active_alerts": len(monitoring_dashboard.get("active_alerts", [])),
                    "total_deployments": monitoring_dashboard.get("deployment_summary", {}).get("total_deployments", 0),
                    "monitoring_active": monitoring_dashboard.get("monitoring_active", False)
                },
                "shutdown": {
                    "state": shutdown_status.get("state", "running"),
                    "active_requests": shutdown_status.get("active_requests", 0)
                }
            }
            
            return system_status
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {"error": str(e), "timestamp": time.time()}
    
    async def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive system health check"""
        try:
            logger.info("Performing comprehensive health check...")
            
            health_results = {}
            
            # Check orchestrator health
            health_results["orchestrator"] = {
                "status": "healthy" if self.is_running and self.initialization_complete else "unhealthy",
                "details": {
                    "running": self.is_running,
                    "initialized": self.initialization_complete
                }
            }
            
            # Check component health
            health_results["health_monitor"] = {
                "status": "healthy" if self.health_monitor.monitoring_active else "unhealthy",
                "details": {"monitoring_active": self.health_monitor.monitoring_active}
            }
            
            health_results["service_discovery"] = {
                "status": "healthy" if self.service_discovery.discovery_active else "unhealthy",
                "details": {"discovery_active": self.service_discovery.discovery_active}
            }
            
            health_results["monitoring_system"] = {
                "status": "healthy" if self.monitoring_system.monitoring_active else "unhealthy",
                "details": {"monitoring_active": self.monitoring_system.monitoring_active}
            }
            
            # Overall health assessment
            component_statuses = [result["status"] for result in health_results.values()]
            overall_healthy = all(status == "healthy" for status in component_statuses)
            
            return {
                "overall_status": "healthy" if overall_healthy else "unhealthy",
                "components": health_results,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "overall_status": "critical",
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def shutdown(self) -> bool:
        """Perform graceful system shutdown"""
        try:
            logger.info("Starting orchestrator shutdown...")
            
            # Trigger shutdown manager
            await self.shutdown_manager.shutdown()
            
            # Wait for shutdown completion
            await self.shutdown_manager.wait_for_shutdown()
            
            self.is_running = False
            logger.info("Orchestrator shutdown completed")
            return True
            
        except Exception as e:
            logger.error(f"Shutdown failed: {e}")
            return False
    
    def _update_deployment_stats(self, deployment_time: float, success: bool, result: Dict[str, Any]):
        """Update deployment statistics"""
        try:
            self.deployment_stats["total_deployments"] += 1
            
            if success:
                self.deployment_stats["successful_deployments"] += 1
            else:
                self.deployment_stats["failed_deployments"] += 1
            
            self.deployment_stats["models_deployed"] += result.get("deployed", 0)
            
            # Update average deployment time
            total_deployments = self.deployment_stats["total_deployments"]
            current_avg = self.deployment_stats["avg_deployment_time"]
            self.deployment_stats["avg_deployment_time"] = (
                (current_avg * (total_deployments - 1) + deployment_time) / total_deployments
            )
            
        except Exception as e:
            logger.error(f"Failed to update deployment stats: {e}")
    
    def _register_health_checks(self):
        """Register system health checks"""
        try:
            # Orchestrator health check
            def orchestrator_health():
                return {
                    "status": "healthy" if self.is_running else "unhealthy",
                    "message": "Orchestrator running" if self.is_running else "Orchestrator not running"
                }
            
            # Deployment system health check
            def deployment_system_health():
                try:
                    bg_status = self.blue_green_orchestrator.get_orchestrator_status()
                    active_envs = sum(1 for env_data in bg_status["environments"].values() 
                                    if env_data["deployment_status"]["active_deployments"] > 0)
                    
                    if active_envs > 0:
                        return {"status": "healthy", "message": f"{active_envs} active environments"}
                    else:
                        return {"status": "degraded", "message": "No active deployments"}
                        
                except Exception as e:
                    return {"status": "unhealthy", "message": f"Health check error: {e}"}
            
            # Register health checks
            self.health_monitor.register_health_check("orchestrator", orchestrator_health, critical=True)
            self.health_monitor.register_health_check("deployment_system", deployment_system_health, critical=True)
            
            logger.info("   Registered system health checks")
            
        except Exception as e:
            logger.error(f"Failed to register health checks: {e}")
    
    def _handle_system_alert(self, alert):
        """Handle system alerts"""
        try:
            logger.warning(f"System alert: {alert.title} ({alert.level.value})")
            
            # Take action based on alert level and component
            if alert.level.value == "critical":
                if alert.component == "system":
                    logger.error("Critical system alert - considering emergency measures")
                elif alert.component == "deployment":
                    logger.error("Critical deployment alert - may trigger rollback")
            
        except Exception as e:
            logger.error(f"Failed to handle alert: {e}")
    
    async def _shutdown_monitoring_services(self):
        """Shutdown monitoring services"""
        try:
            logger.info("Shutting down monitoring services...")
            await self.monitoring_system.stop_monitoring()
            await self.health_monitor.stop_monitoring()
            await self.service_discovery.stop_discovery()
            
        except Exception as e:
            logger.error(f"Failed to shutdown monitoring services: {e}")
    
    async def _shutdown_deployment_services(self):
        """Shutdown deployment services"""
        try:
            logger.info("Shutting down deployment services...")
            # Add any deployment-specific shutdown logic here
            
        except Exception as e:
            logger.error(f"Failed to shutdown deployment services: {e}")


async def create_sample_deployment_scenario():
    """Create a sample deployment scenario for testing"""
    logger.info("Creating sample deployment scenario...")
    
    # Create configuration manager
    config_manager = ConfigurationManager()
    config_manager.create_preset_configs()
    
    # Load development configuration
    config = config_manager.load_config("development.yaml", environment="development")
    
    # Create orchestrator
    orchestrator = UltraScaleDeploymentOrchestrator(config)
    
    try:
        # Initialize system
        success = await orchestrator.initialize()
        if not success:
            logger.error("Failed to initialize orchestrator")
            return
        
        # Wait for initialization
        await asyncio.sleep(5)
        
        # Create sample model configurations
        sample_models = [
            {
                "name": f"sample_model_{i}",
                "type": "synthetic",
                "version": "1.0.0",
                "model_id": i
            }
            for i in range(50)  # Deploy 50 models for testing
        ]
        
        # Perform deployment
        logger.info("Starting sample deployment...")
        deployment_result = await orchestrator.deploy_models_ultra_scale(
            sample_models, 
            deployment_strategy="blue_green"
        )
        
        logger.info(f"Deployment result: {deployment_result}")
        
        # Check system status
        status = orchestrator.get_system_status()
        logger.info(f"System status: {status}")
        
        # Perform health check
        health_result = await orchestrator.perform_health_check()
        logger.info(f"Health check: {health_result}")
        
        # Wait for monitoring data
        await asyncio.sleep(30)
        
        # Get monitoring dashboard
        dashboard = orchestrator.monitoring_system.get_monitoring_dashboard()
        logger.info(f"Monitoring dashboard: {dashboard}")
        
        # Shutdown gracefully
        await orchestrator.shutdown()
        
        logger.info("Sample deployment scenario completed successfully")
        
    except Exception as e:
        logger.error(f"Sample deployment scenario failed: {e}")
        await orchestrator.shutdown()


if __name__ == "__main__":
    # Run sample deployment scenario
    asyncio.run(create_sample_deployment_scenario())