"""
Ultra-Scale Deployment Manager
Based on main.py patterns for 2000+ model serving with zero-downtime deployment
"""

import ray
from ray import serve
import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging
import os
import json
import asyncio
import time
import pickle
import hashlib
import sqlite3
from threading import RLock
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import gc
from dataclasses import dataclass
from enum import Enum

from ..utils.logger import get_logger

logger = get_logger()


class DeploymentStatus(Enum):
    """Deployment status enum"""
    INITIALIZING = "initializing"
    READY = "ready"
    UPDATING = "updating"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class ModelDeploymentInfo:
    """Model deployment information"""
    name: str
    version: str
    status: DeploymentStatus
    tier: str  # hot, warm, cold
    memory_usage: Optional[float] = None
    last_used: Optional[float] = None
    prediction_count: int = 0
    avg_inference_time: float = 0.0
    model_type: str = "unknown"
    deployment_time: Optional[float] = None


class UltraScaleDeploymentManager:
    """Ultra-scale deployment manager for 2000+ models with zero-downtime deployment"""
    
    def __init__(self, max_loaded_models=500, max_memory_mb=8000, target_models=2000):
        self.max_loaded_models = max_loaded_models
        self.max_memory_mb = max_memory_mb
        self.target_models = target_models
        
        # Multi-tier model storage
        self.active_deployments = {}  # Currently serving models
        self.staging_deployments = {}  # Models being prepared
        self.deployment_metadata = {}  # All deployment metadata
        self.deployment_stats = {}  # Usage statistics
        
        # Thread-safe locks
        self.active_lock = RLock()
        self.staging_lock = RLock()
        self.stats_lock = RLock()
        
        # Configuration
        self.cache_dir = "ultra_deployment_cache"
        self.db_path = os.path.join(self.cache_dir, "deployments.db")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize database
        self._init_deployment_database()
        
        # Performance settings
        self.batch_size = 200
        self.parallel_deployers = 20
        
        logger.info(f"UltraScaleDeploymentManager initialized for "
                     f"{target_models} models")
    
    def _init_deployment_database(self):
        """Initialize deployment metadata database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS deployments (
                    name TEXT PRIMARY KEY,
                    version TEXT,
                    status TEXT,
                    tier TEXT,
                    deployment_time REAL,
                    last_used REAL,
                    prediction_count INTEGER DEFAULT 0,
                    total_time REAL DEFAULT 0.0,
                    metadata TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON deployments(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tier ON deployments(tier)")
            conn.commit()
        
        logger.info("Deployment database initialized")
    
    async def deploy_models_ultra_scale(self, model_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Deploy models at ultra-scale with zero-downtime"""
        logger.info(f"Starting ultra-scale deployment of {len(model_configs)} models...")
        
        deployment_start = time.time()
        deployed_count = 0
        failed_count = 0
        
        # Process in batches for optimal performance
        total_batches = (len(model_configs) + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(model_configs))
            batch_configs = model_configs[start_idx:end_idx]
            
            logger.info(f"Processing deployment batch {batch_idx + 1}/{total_batches} ({len(batch_configs)} models)")
            
            # Deploy batch in parallel
            batch_results = await self._deploy_batch_parallel(batch_configs, batch_idx)
            
            # Update counters
            deployed_count += batch_results["deployed"]
            failed_count += batch_results["failed"]
            
            # Memory management after each batch
            if (batch_idx + 1) % 5 == 0:
                await self._manage_deployment_memory()
        
        deployment_time = time.time() - deployment_start
        
        logger.info("Ultra-scale deployment complete!")
        logger.info(f"   - Deployed: {deployed_count}")
        logger.info(f"   - Failed: {failed_count}")
        logger.info(f"   - Total time: {deployment_time:.2f}s")
        
        return {
            "deployed": deployed_count,
            "failed": failed_count,
            "total_time": deployment_time,
            "batch_size": self.batch_size,
            "parallel_workers": self.parallel_deployers
        }
    
    async def _deploy_batch_parallel(self, batch_configs: List[Dict], batch_idx: int) -> Dict[str, int]:
        """Deploy a batch of models in parallel"""
        
        async def deploy_single_model(config):
            try:
                return await self._deploy_single_model_zero_downtime(config)
            except Exception as e:
                logger.error(f"Failed to deploy model {config.get('name', 'unknown')}: {e}")
                return False
        
        # Create semaphore to control concurrency
        semaphore = asyncio.Semaphore(self.parallel_deployers)
        
        async def deploy_with_semaphore(config):
            async with semaphore:
                return await deploy_single_model(config)
        
        # Deploy all models in batch concurrently
        tasks = [deploy_with_semaphore(config) for config in batch_configs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successes and failures
        deployed = sum(1 for result in results if result is True)
        failed = len(results) - deployed
        
        return {"deployed": deployed, "failed": failed}
    
    async def _deploy_single_model_zero_downtime(self, config: Dict[str, Any]) -> bool:
        """Deploy a single model with zero-downtime strategy"""
        model_name = config["name"]
        model_version = config.get("version", "latest")
        model_type = config.get("type", "mlflow")
        
        logger.debug(f"Deploying model {model_name} (v{model_version}) with zero-downtime")
        
        try:
            # Step 1: Stage the new model
            staged_successfully = await self._stage_model_for_deployment(model_name, config)
            if not staged_successfully:
                return False
            
            # Step 2: Validate staged model
            validation_success = await self._validate_staged_model(model_name)
            if not validation_success:
                await self._cleanup_failed_staging(model_name)
                return False
            
            # Step 3: Perform zero-downtime switch
            switch_success = await self._zero_downtime_switch(model_name, model_version)
            if not switch_success:
                await self._cleanup_failed_staging(model_name)
                return False
            
            # Step 4: Update deployment metadata
            await self._update_deployment_metadata(model_name, config, DeploymentStatus.READY)
            
            logger.debug(f"Zero-downtime deployment completed for {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Zero-downtime deployment failed for {model_name}: {e}")
            await self._cleanup_failed_staging(model_name)
            return False
    
    async def _stage_model_for_deployment(self, model_name: str, config: Dict[str, Any]) -> bool:
        """Stage model for zero-downtime deployment"""
        try:
            # Create staging key
            staging_key = f"{model_name}_staging"
            
            with self.staging_lock:
                if staging_key in self.staging_deployments:
                    logger.warning(f"Model {model_name} already in staging")
                    return False
                
                # Mark as staging
                self.staging_deployments[staging_key] = {
                    "config": config,
                    "status": DeploymentStatus.INITIALIZING,
                    "stage_time": time.time()
                }
            
            # Load model based on type
            if config.get("type") == "mlflow":
                model = await self._load_mlflow_model_async(config)
            elif config.get("type") == "synthetic":
                model = await self._create_synthetic_model_async(config)
            else:
                model = await self._load_custom_model_async(config)
            
            if not model:
                return False
            
            # Store staged model
            with self.staging_lock:
                self.staging_deployments[staging_key]["model"] = model
                self.staging_deployments[staging_key]["status"] = DeploymentStatus.READY
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to stage model {model_name}: {e}")
            return False
    
    async def _load_mlflow_model_async(self, config: Dict[str, Any]):
        """Load MLflow model asynchronously"""
        try:
            model_uri = config.get("uri") or f"models:/{config['name']}/{config.get('version', 'latest')}"
            loop = asyncio.get_event_loop()
            # Load model in thread pool to avoid blocking
            model = await loop.run_in_executor(None, mlflow.pyfunc.load_model, model_uri)
            return model
        except Exception as e:
            logger.error(f"Failed to load MLflow model: {e}")
            return None
    
    async def _create_synthetic_model_async(self, config: Dict[str, Any]):
        """Create synthetic model asynchronously"""
        try:
            class UltraFastSyntheticModel:
                def __init__(self, name, model_id=0):
                    self.name = name
                    self.id = model_id
                    # Pre-computed constants for speed (similar to main.py)
                    self.a = 1.0 + (model_id % 500) / 5000.0
                    self.b = (model_id % 200) / 2000.0
                    self.c = (model_id % 50) / 5000.0

                def predict(self, data):
                    if isinstance(data, pd.DataFrame):
                        values = data.iloc[:, 0].values
                        predictions = self.a * values + self.b * np.sqrt(np.abs(values)) + self.c
                        return predictions.tolist()
                    else:
                        pred = self.a * data + self.b * np.sqrt(abs(data)) + self.c
                        return [pred]
            
            model_id = config.get("model_id", hash(config["name"]) % 10000)
            return UltraFastSyntheticModel(config["name"], model_id)
            
        except Exception as e:
            logger.error(f"Failed to create synthetic model: {e}")
            return None
    
    async def _load_custom_model_async(self, config: Dict[str, Any]):
        """Load custom model asynchronously"""
        try:
            # Default to synthetic model for now
            return await self._create_synthetic_model_async(config)
        except Exception as e:
            logger.error(f"Failed to load custom model: {e}")
            return None
    
    async def _validate_staged_model(self, model_name: str) -> bool:
        """Validate staged model before deployment"""
        try:
            staging_key = f"{model_name}_staging"
            
            with self.staging_lock:
                staging_info = self.staging_deployments.get(staging_key)
                if not staging_info or "model" not in staging_info:
                    return False
                
                model = staging_info["model"]
            
            # Perform validation tests
            test_input = pd.DataFrame([[1.0]], columns=['value'])
            
            # Test prediction
            result = model.predict(test_input)
            if result is None or len(result) == 0:
                logger.error(f"Model {model_name} validation failed: no prediction")
                return False
            
            # Test prediction type
            if not isinstance(result, (list, np.ndarray)):
                logger.error(f"Model {model_name} validation failed: invalid prediction type")
                return False
            
            logger.debug(f"Model {model_name} validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Model {model_name} validation failed: {e}")
            return False
    
    async def _zero_downtime_switch(self, model_name: str, model_version: str) -> bool:
        """Perform zero-downtime switch from staging to active"""
        try:
            staging_key = f"{model_name}_staging"
            
            # Get staged model
            with self.staging_lock:
                staging_info = self.staging_deployments.get(staging_key)
                if not staging_info or "model" not in staging_info:
                    return False
                
                staged_model = staging_info["model"]
            
            # Atomic switch: move from staging to active
            with self.active_lock:
                # Store old model for rollback if needed
                old_model = self.active_deployments.get(model_name)
                
                # Switch to new model
                self.active_deployments[model_name] = {
                    "model": staged_model,
                    "version": model_version,
                    "status": DeploymentStatus.READY,
                    "activation_time": time.time()
                }
                
                # Update metadata
                self.deployment_metadata[model_name] = {
                    "version": model_version,
                    "status": DeploymentStatus.READY,
                    "tier": "hot",
                    "last_updated": time.time()
                }
            
            # Clean up staging
            with self.staging_lock:
                if staging_key in self.staging_deployments:
                    del self.staging_deployments[staging_key]
            
            logger.debug(f"Zero-downtime switch completed for {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Zero-downtime switch failed for {model_name}: {e}")
            return False
    
    async def _cleanup_failed_staging(self, model_name: str):
        """Clean up failed staging deployment"""
        try:
            staging_key = f"{model_name}_staging"
            with self.staging_lock:
                if staging_key in self.staging_deployments:
                    del self.staging_deployments[staging_key]
            
            logger.debug(f"Cleaned up failed staging for {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup staging for {model_name}: {e}")
    
    async def _update_deployment_metadata(self, model_name: str, config: Dict[str, Any], status: DeploymentStatus):
        """Update deployment metadata in database"""
        try:
            current_time = time.time()
            metadata_json = json.dumps(config)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO deployments 
                    (name, version, status, tier, deployment_time, last_used, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    model_name,
                    config.get("version", "latest"),
                    status.value,
                    "hot",  # Default to hot tier for new deployments
                    current_time,
                    current_time,
                    metadata_json
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to update deployment metadata for {model_name}: {e}")
    
    async def _manage_deployment_memory(self):
        """Advanced memory management for deployments"""
        try:
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            if current_memory > self.max_memory_mb * 0.9:  # 90% threshold
                logger.info(f"Deployment memory management triggered (current: {current_memory:.1f}MB)")
                
                # Move least recently used models to lower tiers
                with self.active_lock:
                    if len(self.active_deployments) > self.max_loaded_models // 2:
                        # Sort by last used time
                        lru_models = sorted(
                            self.active_deployments.keys(),
                            key=lambda k: self.deployment_stats.get(k, {}).get("last_used", 0)
                        )
                        
                        # Move 25% of active models to warm tier
                        models_to_demote = len(self.active_deployments) // 4
                        
                        for model_name in lru_models[:models_to_demote]:
                            await self._demote_to_warm_tier(model_name)
                
                # Force garbage collection
                gc.collect()
                
                new_memory = psutil.Process().memory_info().rss / 1024 / 1024
                logger.info(f"   Memory after cleanup: {new_memory:.1f}MB (saved: {current_memory - new_memory:.1f}MB)")
                
        except Exception as e:
            logger.error(f"Memory management failed: {e}")
    
    async def _demote_to_warm_tier(self, model_name: str):
        """Demote model from active to warm tier"""
        try:
            with self.active_lock:
                if model_name in self.active_deployments:
                    # Cache model to disk
                    model_data = self.active_deployments[model_name]
                    cache_file = os.path.join(
                        self.cache_dir, 
                        f"warm_{hashlib.md5(model_name.encode()).hexdigest()}.pkl"
                    )
                    
                    with open(cache_file, 'wb') as f:
                        pickle.dump(model_data["model"], f, protocol=pickle.HIGHEST_PROTOCOL)
                    
                    # Remove from active deployments
                    del self.active_deployments[model_name]
                    
                    # Update metadata
                    self.deployment_metadata[model_name]["tier"] = "warm"
                    
                    logger.debug(f"   Demoted {model_name} to warm tier")
                    
        except Exception as e:
            logger.error(f"Failed to demote {model_name} to warm tier: {e}")
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get comprehensive deployment status"""
        try:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent = psutil.cpu_percent()
            
            with self.active_lock:
                active_count = len(self.active_deployments)
            
            with self.staging_lock:
                staging_count = len(self.staging_deployments)
            
            total_deployments = len(self.deployment_metadata)
            
            return {
                "active_deployments": active_count,
                "staging_deployments": staging_count,
                "total_deployments": total_deployments,
                "memory_usage_mb": memory_usage,
                "cpu_percent": cpu_percent,
                "max_loaded_models": self.max_loaded_models,
                "target_models": self.target_models,
                "status": "ready" if active_count > 0 else "initializing"
            }
            
        except Exception as e:
            logger.error(f"Failed to get deployment status: {e}")
            return {"error": str(e)}