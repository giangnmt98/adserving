"""
Blue-Green Deployment Orchestrator
Manages multiple deployment environments for zero-downtime model updates
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from dataclasses import dataclass, field
from threading import RLock

import ray
from ray import serve

from .ultra_scale_deployment_manager import UltraScaleDeploymentManager, DeploymentStatus
from ..utils.logger import get_logger

logger = get_logger()


class Environment(Enum):
    """Deployment environment types"""
    BLUE = "blue"
    GREEN = "green"
    STAGING = "staging"


class TrafficState(Enum):
    """Traffic routing state"""
    BLUE_ACTIVE = "blue_active"
    GREEN_ACTIVE = "green_active"
    SWITCHING = "switching"
    TESTING = "testing"


@dataclass
class EnvironmentStatus:
    """Status of a deployment environment"""
    name: Environment
    deployment_manager: UltraScaleDeploymentManager
    active_models: Set[str] = field(default_factory=set)
    model_count: int = 0
    health_score: float = 1.0
    last_health_check: float = 0.0
    traffic_weight: float = 0.0
    status: DeploymentStatus = DeploymentStatus.INITIALIZING


class BlueGreenOrchestrator:
    """Orchestrates blue-green deployments for ultra-scale model serving"""
    
    def __init__(self, max_models_per_env=1000, max_memory_per_env=4000):
        self.max_models_per_env = max_models_per_env
        self.max_memory_per_env = max_memory_per_env
        
        # Initialize deployment environments
        self.environments = {
            Environment.BLUE: EnvironmentStatus(
                name=Environment.BLUE,
                deployment_manager=UltraScaleDeploymentManager(
                    max_loaded_models=max_models_per_env,
                    max_memory_mb=max_memory_per_env,
                    target_models=2000
                )
            ),
            Environment.GREEN: EnvironmentStatus(
                name=Environment.GREEN,
                deployment_manager=UltraScaleDeploymentManager(
                    max_loaded_models=max_models_per_env,
                    max_memory_mb=max_memory_per_env,
                    target_models=2000
                )
            ),
            Environment.STAGING: EnvironmentStatus(
                name=Environment.STAGING,
                deployment_manager=UltraScaleDeploymentManager(
                    max_loaded_models=500,  # Smaller staging environment
                    max_memory_mb=2000,
                    target_models=100
                )
            )
        }
        
        # Traffic management
        self.current_active_env = Environment.BLUE
        self.traffic_state = TrafficState.BLUE_ACTIVE
        self.traffic_weights = {
            Environment.BLUE: 1.0,
            Environment.GREEN: 0.0,
            Environment.STAGING: 0.0
        }
        
        # Orchestration state
        self.deployment_in_progress = False
        self.rollback_available = False
        self.switch_history = []
        
        # Thread safety
        self.orchestration_lock = RLock()
        
        logger.info("BlueGreenOrchestrator initialized")
        logger.info(f"   - Blue environment: {max_models_per_env} models, {max_memory_per_env}MB")
        logger.info(f"   - Green environment: {max_models_per_env} models, {max_memory_per_env}MB")
        logger.info(f"   - Staging environment: 500 models, 2000MB")
    
    async def deploy_models_blue_green(self, model_configs: List[Dict[str, Any]], 
                                     target_env: Optional[Environment] = None) -> Dict[str, Any]:
        """Deploy models using blue-green strategy"""
        
        with self.orchestration_lock:
            if self.deployment_in_progress:
                return {
                    "success": False,
                    "message": "Deployment already in progress",
                    "active_env": self.current_active_env.value
                }
            
            self.deployment_in_progress = True
        
        try:
            logger.info(f"Starting blue-green deployment of {len(model_configs)} models...")
            deployment_start = time.time()
            
            # Determine target environment
            if target_env is None:
                target_env = self._get_inactive_environment()
            
            target_status = self.environments[target_env]
            
            logger.info(f"   - Target environment: {target_env.value}")
            logger.info(f"   - Current active: {self.current_active_env.value}")
            
            # Phase 1: Deploy to target environment
            logger.info("Phase 1: Deploying models to target environment...")
            deployment_result = await target_status.deployment_manager.deploy_models_ultra_scale(model_configs)
            
            if deployment_result["failed"] > 0:
                logger.warning(f"Some models failed to deploy: {deployment_result['failed']} failures")
            
            # Update environment status
            target_status.model_count = deployment_result["deployed"]
            target_status.active_models = {config["name"] for config in model_configs}
            target_status.status = DeploymentStatus.READY
            
            # Phase 2: Health check new environment
            logger.info("Phase 2: Health checking new environment...")
            health_check_success = await self._comprehensive_health_check(target_env)
            
            if not health_check_success:
                logger.error("Health check failed, aborting deployment")
                return {
                    "success": False,
                    "message": "Health check failed",
                    "deployed": deployment_result["deployed"],
                    "failed": deployment_result["failed"]
                }
            
            # Phase 3: Gradual traffic switching
            logger.info("Phase 3: Gradual traffic switching...")
            switch_result = await self._gradual_traffic_switch(target_env)
            
            if not switch_result["success"]:
                logger.error("Traffic switching failed, initiating rollback")
                await self._rollback_deployment()
                return {
                    "success": False,
                    "message": "Traffic switching failed, rolled back",
                    "deployed": deployment_result["deployed"]
                }
            
            # Phase 4: Finalize deployment
            logger.info("Phase 4: Finalizing deployment...")
            await self._finalize_deployment(target_env)
            
            deployment_time = time.time() - deployment_start
            
            logger.info("Blue-green deployment completed successfully!")
            logger.info(f"   - New active environment: {target_env.value}")
            logger.info(f"   - Models deployed: {deployment_result['deployed']}")
            logger.info(f"   - Total time: {deployment_time:.2f}s")
            
            return {
                "success": True,
                "active_environment": target_env.value,
                "deployed": deployment_result["deployed"],
                "failed": deployment_result["failed"],
                "total_time": deployment_time,
                "traffic_weights": self.traffic_weights.copy()
            }
            
        except Exception as e:
            logger.error(f"Blue-green deployment failed: {e}")
            await self._rollback_deployment()
            return {
                "success": False,
                "message": f"Deployment failed: {e}",
                "active_environment": self.current_active_env.value
            }
        
        finally:
            with self.orchestration_lock:
                self.deployment_in_progress = False
    
    def _get_inactive_environment(self) -> Environment:
        """Get the inactive environment for deployment"""
        if self.current_active_env == Environment.BLUE:
            return Environment.GREEN
        else:
            return Environment.BLUE
    
    async def _comprehensive_health_check(self, env: Environment) -> bool:
        """Perform comprehensive health check on environment"""
        try:
            logger.info(f"   Checking health of {env.value} environment...")
            
            env_status = self.environments[env]
            deployment_manager = env_status.deployment_manager
            
            # Get deployment status
            status = deployment_manager.get_deployment_status()
            
            # Check deployment count
            if status["active_deployments"] == 0:
                logger.error(f"   No active deployments in {env.value}")
                return False
            
            # Check memory usage
            if status["memory_usage_mb"] > self.max_memory_per_env * 0.95:
                logger.error(f"   Memory usage too high in {env.value}: {status['memory_usage_mb']:.1f}MB")
                return False
            
            # Test sample predictions
            test_success = await self._test_sample_predictions(env)
            if not test_success:
                logger.error(f"   Sample prediction test failed in {env.value}")
                return False
            
            # Update health score
            env_status.health_score = 1.0
            env_status.last_health_check = time.time()
            
            logger.info(f"   {env.value} environment health check passed")
            return True
            
        except Exception as e:
            logger.error(f"Health check failed for {env.value}: {e}")
            return False
    
    async def _test_sample_predictions(self, env: Environment) -> bool:
        """Test sample predictions on environment"""
        try:
            # This would integrate with the actual Ray Serve endpoints
            # For now, we'll simulate successful predictions
            await asyncio.sleep(0.1)  # Simulate prediction time
            return True
            
        except Exception as e:
            logger.error(f"Sample prediction test failed: {e}")
            return False
    
    async def _gradual_traffic_switch(self, target_env: Environment) -> Dict[str, Any]:
        """Gradually switch traffic to new environment"""
        try:
            logger.info(f"   Switching traffic to {target_env.value}...")
            
            switch_start = time.time()
            self.traffic_state = TrafficState.SWITCHING
            
            # Traffic switching steps: 10% -> 50% -> 100%
            switch_steps = [
                {"target_weight": 0.1, "duration": 30},   # 10% for 30 seconds
                {"target_weight": 0.5, "duration": 60},   # 50% for 1 minute
                {"target_weight": 1.0, "duration": 0}     # 100% final
            ]
            
            old_env = self.current_active_env
            
            for step_idx, step in enumerate(switch_steps):
                logger.info(f"   Traffic switch step {step_idx + 1}: {step['target_weight']*100:.0f}% to {target_env.value}")
                
                # Update traffic weights
                self.traffic_weights[target_env] = step["target_weight"]
                self.traffic_weights[old_env] = 1.0 - step["target_weight"]
                
                # Wait for step duration
                if step["duration"] > 0:
                    await asyncio.sleep(step["duration"])
                
                # Monitor during switch
                health_ok = await self._monitor_during_switch(target_env)
                if not health_ok:
                    logger.error(f"   Health degraded during switch step {step_idx + 1}")
                    return {"success": False, "step": step_idx + 1}
            
            switch_time = time.time() - switch_start
            logger.info(f"   Traffic switch completed in {switch_time:.1f}s")
            
            return {"success": True, "switch_time": switch_time}
            
        except Exception as e:
            logger.error(f"Traffic switching failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _monitor_during_switch(self, target_env: Environment) -> bool:
        """Monitor system health during traffic switch"""
        try:
            # Quick health check during switch
            env_status = self.environments[target_env]
            status = env_status.deployment_manager.get_deployment_status()
            
            # Check for any critical issues
            if status.get("error"):
                return False
            
            if status["memory_usage_mb"] > self.max_memory_per_env:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Monitoring during switch failed: {e}")
            return False
    
    async def _finalize_deployment(self, new_active_env: Environment):
        """Finalize the blue-green deployment"""
        try:
            # Update active environment
            old_active = self.current_active_env
            self.current_active_env = new_active_env
            
            # Update traffic state
            if new_active_env == Environment.BLUE:
                self.traffic_state = TrafficState.BLUE_ACTIVE
            else:
                self.traffic_state = TrafficState.GREEN_ACTIVE
            
            # Final traffic weights
            self.traffic_weights = {env: 0.0 for env in Environment}
            self.traffic_weights[new_active_env] = 1.0
            
            # Update environment status
            self.environments[new_active_env].traffic_weight = 1.0
            self.environments[old_active].traffic_weight = 0.0
            
            # Enable rollback capability
            self.rollback_available = True
            
            # Record switch history
            self.switch_history.append({
                "timestamp": time.time(),
                "from_env": old_active.value,
                "to_env": new_active_env.value,
                "success": True
            })
            
            # Keep only last 10 switches in history
            if len(self.switch_history) > 10:
                self.switch_history = self.switch_history[-10:]
            
            logger.info(f"   Deployment finalized: {new_active_env.value} is now active")
            
        except Exception as e:
            logger.error(f"Failed to finalize deployment: {e}")
            raise
    
    async def _rollback_deployment(self):
        """Rollback to previous stable deployment"""
        try:
            if not self.rollback_available:
                logger.warning("No rollback available")
                return
            
            logger.info("Initiating deployment rollback...")
            
            # Get previous active environment
            if self.current_active_env == Environment.BLUE:
                rollback_env = Environment.GREEN
            else:
                rollback_env = Environment.BLUE
            
            # Quick traffic switch back
            self.traffic_weights = {env: 0.0 for env in Environment}
            self.traffic_weights[rollback_env] = 1.0
            self.current_active_env = rollback_env
            
            # Update traffic state
            if rollback_env == Environment.BLUE:
                self.traffic_state = TrafficState.BLUE_ACTIVE
            else:
                self.traffic_state = TrafficState.GREEN_ACTIVE
            
            # Record rollback
            self.switch_history.append({
                "timestamp": time.time(),
                "from_env": self.current_active_env.value,
                "to_env": rollback_env.value,
                "success": True,
                "rollback": True
            })
            
            logger.info(f"Rollback completed: {rollback_env.value} is now active")
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            raise
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator status"""
        try:
            status = {
                "current_active_env": self.current_active_env.value,
                "traffic_state": self.traffic_state.value,
                "deployment_in_progress": self.deployment_in_progress,
                "rollback_available": self.rollback_available,
                "traffic_weights": self.traffic_weights.copy(),
                "environments": {},
                "switch_history": self.switch_history[-5:]  # Last 5 switches
            }
            
            # Add environment details
            for env_name, env_status in self.environments.items():
                deployment_status = env_status.deployment_manager.get_deployment_status()
                status["environments"][env_name.value] = {
                    "model_count": env_status.model_count,
                    "health_score": env_status.health_score,
                    "last_health_check": env_status.last_health_check,
                    "traffic_weight": env_status.traffic_weight,
                    "status": env_status.status.value,
                    "deployment_status": deployment_status
                }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get orchestrator status: {e}")
            return {"error": str(e)}
    
    async def force_switch_environment(self, target_env: Environment) -> Dict[str, Any]:
        """Force switch to specified environment (emergency use)"""
        try:
            logger.warning(f"Force switching to {target_env.value} environment")
            
            # Immediate traffic switch
            self.traffic_weights = {env: 0.0 for env in Environment}
            self.traffic_weights[target_env] = 1.0
            self.current_active_env = target_env
            
            # Update traffic state
            if target_env == Environment.BLUE:
                self.traffic_state = TrafficState.BLUE_ACTIVE
            else:
                self.traffic_state = TrafficState.GREEN_ACTIVE
            
            logger.info(f"Force switch completed: {target_env.value} is now active")
            
            return {
                "success": True,
                "active_environment": target_env.value,
                "traffic_weights": self.traffic_weights.copy()
            }
            
        except Exception as e:
            logger.error(f"Force switch failed: {e}")
            return {"success": False, "error": str(e)}