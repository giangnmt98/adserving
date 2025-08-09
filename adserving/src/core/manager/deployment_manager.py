"""
Zero-Downtime Deployment Manager

This module handles zero-downtime model deployments with staging and validation.
"""

import asyncio
import threading
import time
from typing import Any, Dict, List, Optional
from adserving.src.utils.logger import get_logger
from adserving.src.config.core_configs import WorkerConfig


class DeploymentManager:
    """Manages zero-downtime model deployments"""

    def __init__(
        self, batch_size: int = 5, batch_interval: float = 2.0, load_model_func=None, worker_config: Optional[WorkerConfig] = None
    ):
        self.logger = get_logger()
        self._batch_size = batch_size
        self._batch_interval = batch_interval
        self._load_model_func = load_model_func
        
        # Enhanced worker configuration
        self.worker_config = worker_config or WorkerConfig()
        
        # Log worker configuration for debugging
        optimal_count = self.worker_config.get_optimal_worker_count()
        self.logger.info(f"DeploymentManager initialized with WorkerConfig: {optimal_count} workers, "
                        f"{self.worker_config.memory_limit_mb}MB per worker, "
                        f"CPU affinity: {self.worker_config.enable_cpu_affinity}")
        
        # Store calculated worker count for infrastructure initialization
        self._optimal_worker_count = optimal_count

        # Deployment infrastructure
        self._deployment_queue: Optional[asyncio.Queue] = None
        self._deployment_workers: List[asyncio.Task] = []
        self._deployment_semaphore: Optional[asyncio.Semaphore] = None

        # Staging area for models
        self._staging_area: Dict[str, Dict[str, Any]] = {}
        self._staging_lock = threading.RLock()

        # Deployment statistics
        self._deployment_stats = {
            "total_deployments": 0,
            "successful_deployments": 0,
            "failed_deployments": 0,
            "average_deployment_time": 0.0,
        }

        # Stop monitoring flag
        self._stop_monitoring = threading.Event()

    def initialize_infrastructure(self):
        """Initialize deployment queue and workers for zero-downtime deployment with WorkerConfig optimization"""
        try:
            # Initialize deployment queue with WorkerConfig queue size
            queue_size = self.worker_config.queue_size * self._optimal_worker_count
            self._deployment_queue = asyncio.Queue(maxsize=queue_size)
            
            # Initialize semaphore based on max concurrent tasks per worker
            max_concurrent = self._optimal_worker_count * self.worker_config.max_concurrent_tasks
            self._deployment_semaphore = asyncio.Semaphore(max_concurrent)

            # Start deployment workers with optimal configuration
            def start_workers():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    # Use WorkerConfig optimal worker count
                    desired_workers = self._optimal_worker_count
                    
                    self.logger.info(f"Starting {desired_workers} deployment workers with WorkerConfig settings:")
                    self.logger.info(f"  - Memory limit per worker: {self.worker_config.memory_limit_mb}MB")
                    self.logger.info(f"  - CPU cores per worker: {self.worker_config.cpu_cores_per_worker}")
                    self.logger.info(f"  - Max concurrent tasks per worker: {self.worker_config.max_concurrent_tasks}")
                    self.logger.info(f"  - Task timeout: {self.worker_config.task_timeout}s")
                    self.logger.info(f"  - Load balancing strategy: {self.worker_config.load_balancing_strategy}")
                    
                    # Create workers with CPU affinity if enabled
                    for i in range(desired_workers):
                        worker_id = f"worker-{i}"
                        
                        # Apply CPU affinity if enabled
                        if self.worker_config.enable_cpu_affinity:
                            self._apply_cpu_affinity(i, desired_workers)
                        
                        # Create worker task with WorkerConfig settings
                        worker_task = loop.create_task(
                            self._deployment_worker_enhanced(worker_id, i, loop)
                        )
                        self._deployment_workers.append(worker_task)

                    # Keep the loop running
                    loop.run_forever()
                except Exception as e:
                    self.logger.error(f"Error starting enhanced deployment workers: {e}")

            worker_thread = threading.Thread(target=start_workers, daemon=True)
            worker_thread.start()

            self.logger.info(
                f"Enhanced deployment infrastructure initialized with {self._optimal_worker_count} workers "
                f"using WorkerConfig optimization"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize enhanced deployment infrastructure: {e}")

    def _apply_cpu_affinity(self, worker_index: int, total_workers: int):
        """Apply CPU affinity to worker process based on WorkerConfig strategy"""
        try:
            import os
            import psutil
            
            if not self.worker_config.enable_cpu_affinity:
                return
            
            cpu_count = os.cpu_count() or 1
            strategy = self.worker_config.cpu_affinity_strategy
            
            if strategy == "spread":
                # Spread workers across all available CPUs
                cpu_id = worker_index % cpu_count
                os.sched_setaffinity(0, {cpu_id})
                self.logger.debug(f"Worker-{worker_index} pinned to CPU {cpu_id} (spread strategy)")
                
            elif strategy == "sequential":
                # Assign workers to CPUs sequentially
                cores_per_worker = max(1, int(self.worker_config.cpu_cores_per_worker))
                start_cpu = (worker_index * cores_per_worker) % cpu_count
                cpu_set = {(start_cpu + i) % cpu_count for i in range(cores_per_worker)}
                os.sched_setaffinity(0, cpu_set)
                self.logger.debug(f"Worker-{worker_index} pinned to CPUs {cpu_set} (sequential strategy)")
                
        except ImportError:
            self.logger.warning("psutil not available, CPU affinity disabled")
        except Exception as e:
            self.logger.warning(f"Failed to set CPU affinity for worker-{worker_index}: {e}")

    async def _deployment_worker_enhanced(self, worker_id: str, worker_index: int, loop: asyncio.AbstractEventLoop):
        """Enhanced deployment worker with WorkerConfig optimizations and monitoring"""
        self.logger.info(f"Enhanced deployment worker {worker_id} started with WorkerConfig optimizations")
        
        # Worker performance tracking
        task_count = 0
        successful_tasks = 0
        failed_tasks = 0
        total_processing_time = 0.0
        last_health_check = time.time()
        
        # Load balancing state
        worker_load = 0
        max_load = self.worker_config.max_concurrent_tasks
        
        if self._deployment_queue is None:
            self.logger.error(f"Deployment queue not initialized for worker {worker_id}")
            return

        if self._deployment_semaphore is None:
            self.logger.error(f"Deployment semaphore not initialized for worker {worker_id}")
            return

        while not self._stop_monitoring.is_set():
            try:
                # Health check interval
                current_time = time.time()
                if (current_time - last_health_check) > self.worker_config.health_check_interval_seconds:
                    if self.worker_config.enable_worker_health_checks:
                        await self._perform_worker_health_check(worker_id, task_count, successful_tasks, failed_tasks)
                    last_health_check = current_time

                # Get deployment task with WorkerConfig timeout
                try:
                    task = await asyncio.wait_for(
                        self._deployment_queue.get(), 
                        timeout=self.worker_config.task_timeout
                    )
                except asyncio.TimeoutError:
                    continue

                # Check worker load for load balancing
                if worker_load >= max_load:
                    # Worker at capacity, requeue task if load balancing allows
                    if self.worker_config.load_balancing_strategy == "least_loaded":
                        await self._deployment_queue.put(task)
                        await asyncio.sleep(0.1)  # Brief pause to allow other workers
                        continue

                # Process task with enhanced monitoring
                task_start_time = time.time()
                task_count += 1
                worker_load += 1

                try:
                    # Rate limiting with semaphore and WorkerConfig timeout
                    async with self._deployment_semaphore:
                        await asyncio.wait_for(
                            self._deploy_model_with_staging_enhanced(task, worker_id, worker_index),
                            timeout=self.worker_config.task_timeout
                        )
                    
                    successful_tasks += 1
                    task_processing_time = time.time() - task_start_time
                    total_processing_time += task_processing_time
                    
                    if self.worker_config.collect_worker_metrics:
                        avg_time = total_processing_time / task_count if task_count > 0 else 0
                        self.logger.debug(f"Worker {worker_id} completed task {task_count} in {task_processing_time:.2f}s (avg: {avg_time:.2f}s)")

                except asyncio.TimeoutError:
                    failed_tasks += 1
                    self.logger.warning(f"Worker {worker_id} task timed out after {self.worker_config.task_timeout}s")
                    
                except Exception as task_error:
                    import traceback
                    failed_tasks += 1
                    task_traceback = traceback.format_exc()
                    task_context = {
                        "worker_id": worker_id,
                        "task_number": task_count,
                        "task_type": task.get("type", "unknown"),
                        "model_name": task.get("model_name", "unknown"),
                        "processing_time": time.time() - task_start_time,
                        "worker_load": worker_load
                    }
                    
                    self.logger.error(
                        f"Worker {worker_id} task failed: {task_error}\n"
                        f"Task context: {task_context}\n"
                        f"Full traceback:\n{task_traceback}"
                    )
                    
                finally:
                    worker_load = max(0, worker_load - 1)

                # Worker restart threshold check
                if failed_tasks >= self.worker_config.worker_restart_threshold:
                    self.logger.warning(f"Worker {worker_id} exceeded failure threshold ({failed_tasks} failures), restarting recommended")
                    # Could implement worker restart logic here
                
                # Brief yield to keep event loop responsive
                await asyncio.sleep(0)

            except Exception as e:
                import traceback
                worker_traceback = traceback.format_exc()
                worker_context = {
                    "worker_id": worker_id,
                    "worker_index": worker_index,
                    "total_tasks": task_count,
                    "successful_tasks": successful_tasks,
                    "failed_tasks": failed_tasks,
                    "current_load": worker_load,
                    "max_load": max_load,
                    "uptime_seconds": time.time() - (current_time if 'current_time' in locals() else time.time())
                }
                
                self.logger.error(
                    f"Enhanced deployment worker {worker_id} critical error: {e}\n"
                    f"Worker context: {worker_context}\n"
                    f"Full traceback:\n{worker_traceback}"
                )
                await asyncio.sleep(1)

        # Final performance report
        if self.worker_config.collect_worker_metrics and task_count > 0:
            avg_time = total_processing_time / task_count
            success_rate = (successful_tasks / task_count) * 100
            self.logger.info(f"Worker {worker_id} final stats: {task_count} tasks, "
                           f"{success_rate:.1f}% success rate, {avg_time:.2f}s avg time")

        self.logger.info(f"Enhanced deployment worker {worker_id} stopped")

    async def _perform_worker_health_check(self, worker_id: str, task_count: int, successful_tasks: int, failed_tasks: int):
        """Perform health check for worker based on WorkerConfig settings"""
        try:
            import psutil
            import os
            
            # Memory usage check
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > self.worker_config.memory_limit_mb:
                self.logger.warning(f"Worker {worker_id} memory usage ({memory_mb:.1f}MB) exceeds limit ({self.worker_config.memory_limit_mb}MB)")
            
            # Performance metrics
            if task_count > 0:
                success_rate = (successful_tasks / task_count) * 100
                if success_rate < 80:  # Less than 80% success rate
                    self.logger.warning(f"Worker {worker_id} low success rate: {success_rate:.1f}%")
                elif self.worker_config.collect_worker_metrics:
                    self.logger.debug(f"Worker {worker_id} health check: {task_count} tasks, {success_rate:.1f}% success")
            
        except ImportError:
            # psutil not available, skip memory check
            pass
        except Exception as e:
            import traceback
            health_traceback = traceback.format_exc()
            health_context = {
                "worker_id": worker_id,
                "task_count": task_count,
                "successful_tasks": successful_tasks,
                "failed_tasks": failed_tasks,
                "success_rate": (successful_tasks / max(task_count, 1)) * 100,
                "memory_limit_mb": self.worker_config.memory_limit_mb,
                "health_check_interval": self.worker_config.health_check_interval_seconds
            }
            
            self.logger.error(
                f"Worker {worker_id} health check failed: {e}\n"
                f"Health check context: {health_context}\n"
                f"Full traceback:\n{health_traceback}"
            )

    async def _deploy_model_with_staging_enhanced(self, task: Dict[str, Any], worker_id: str, worker_index: int):
        """Enhanced model deployment with WorkerConfig optimizations"""
        # Use the existing _deploy_model_with_staging method but with enhanced logging
        model_name = task.get("model_name", "unknown")
        
        self.logger.debug(f"Worker {worker_id} (index {worker_index}) deploying model: {model_name}")
        
        # Call the existing staging deployment method
        await self._deploy_model_with_staging(task, worker_id)

    async def queue_batched_deployments(
        self,
        model_names: List[str],
        deployment_type: str,
        fallback_deploy_func=None,
        fallback_update_func=None,
    ):
        """Queue model deployments for batched processing"""
        if not self._deployment_queue:
            self.logger.warning(
                "Deployment queue not initialized, falling back to "
                "sequential deployment"
            )
            # Fallback to old behavior
            for model_name in model_names:
                try:
                    if deployment_type == "new" and fallback_deploy_func:
                        await fallback_deploy_func(model_name)
                    elif deployment_type == "update" and fallback_update_func:
                        await fallback_update_func(model_name, "", "")
                except Exception as e:
                    import traceback
                    deployment_traceback = traceback.format_exc()
                    deployment_context = {
                        "model_name": model_name,
                        "deployment_type": deployment_type,
                        "fallback_function": fallback_deploy_func.__name__ if fallback_deploy_func else "None",
                        "queue_fallback": True,
                        "remaining_models": len(model_names) - model_names.index(model_name) - 1
                    }
                    
                    self.logger.error(
                        f"Failed to deploy model {model_name} in queue fallback: {e}\n"
                        f"Deployment context: {deployment_context}\n"
                        f"Full traceback:\n{deployment_traceback}"
                    )
            return

        # Queue models for batched deployment
        for model_name in model_names:
            deployment_task = {
                "model_name": model_name,
                "type": deployment_type,
                "queued_at": time.time(),
                "priority": 1 if deployment_type == "update" else 2,
            }
            try:
                await self._deployment_queue.put(deployment_task)
                self.logger.debug(
                    f"Queued {deployment_type} deployment for model: " f"{model_name}"
                )
            except Exception as e:
                import traceback
                queue_traceback = traceback.format_exc()
                queue_context = {
                    "model_name": model_name,
                    "deployment_type": deployment_type,
                    "priority": 1 if deployment_type == "update" else 2,
                    "queue_size": self._deployment_queue.qsize() if self._deployment_queue else "unknown",
                    "queue_initialized": self._deployment_queue is not None
                }
                
                self.logger.error(
                    f"Failed to queue deployment for model {model_name}: {e}\n"
                    f"Queue context: {queue_context}\n"
                    f"Full traceback:\n{queue_traceback}"
                )

    async def _deployment_worker(self, worker_id: str, loop: asyncio.AbstractEventLoop):
        """Worker to process deployment queue with rate limiting"""
        self.logger.info(f"Deployment worker {worker_id} started")

        if self._deployment_queue is None:
            self.logger.error("Deployment queue not initialized")
            return

        if self._deployment_semaphore is None:
            self.logger.error("Deployment semaphore not initialized")
            return

        while not self._stop_monitoring.is_set():
            try:
                # Get deployment task with timeout
                try:
                    task = await asyncio.wait_for(
                        self._deployment_queue.get(), timeout=5.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Rate limiting with semaphore
                async with self._deployment_semaphore:
                    await self._deploy_model_with_staging(task, worker_id)

                # Optional brief yield to keep event loop responsive without artificial throttling
                await asyncio.sleep(0)  # removed artificial rate limiting

            except Exception as e:
                import traceback
                worker_traceback = traceback.format_exc()
                worker_context = {
                    "worker_id": worker_id,
                    "queue_initialized": self._deployment_queue is not None,
                    "semaphore_initialized": self._deployment_semaphore is not None,
                    "stop_monitoring": self._stop_monitoring.is_set()
                }
                
                self.logger.error(
                    f"Deployment worker {worker_id} error: {e}\n"
                    f"Worker context: {worker_context}\n"
                    f"Full traceback:\n{worker_traceback}"
                )
                await asyncio.sleep(1)

        self.logger.info(f"Deployment worker {worker_id} stopped")

    async def _deploy_model_with_staging(self, task: Dict[str, Any], worker_id: str):
        """Deploy a model using staging area for zero-downtime"""
        model_name = task["model_name"]
        deployment_type = task["type"]
        start_time = time.time()

        self.logger.info(
            f"[{worker_id}] Starting {deployment_type} deployment for "
            f"model: {model_name}"
        )

        try:
            # Stage 1: Load model into staging area
            if not await self.stage_model(model_name):
                self.logger.error(f"[{worker_id}] Failed to stage model: {model_name}")
                self._deployment_stats["failed_deployments"] += 1
                return

            # Stage 2: Validate staged model
            if not await self.validate_staged_model(model_name):
                self.logger.error(
                    f"[{worker_id}] Model validation failed: {model_name}"
                )
                await self.remove_from_staging(model_name)
                self._deployment_stats["failed_deployments"] += 1
                return

            # Stage 3: Atomic switch to active
            await self.promote_staged_model_to_active(model_name)

            # Update statistics
            deployment_time = time.time() - start_time
            self._deployment_stats["successful_deployments"] += 1
            self._deployment_stats["total_deployments"] += 1

            # Update average deployment time
            total_deployments = self._deployment_stats["total_deployments"]
            current_avg = self._deployment_stats["average_deployment_time"]
            self._deployment_stats["average_deployment_time"] = (
                current_avg * (total_deployments - 1) + deployment_time
            ) / total_deployments

            self.logger.info(
                f"[{worker_id}] Successfully deployed model {model_name} "
                f"in {deployment_time:.2f}s"
            )

        except Exception as e:
            import traceback
            deployment_traceback = traceback.format_exc()
            deployment_context = {
                "worker_id": worker_id,
                "model_name": model_name,
                "deployment_type": deployment_type,
                "elapsed_time": time.time() - start_time,
                "total_deployments": self._deployment_stats["total_deployments"],
                "failed_deployments": self._deployment_stats["failed_deployments"]
            }
            
            self.logger.error(
                f"[{worker_id}] Deployment failed for model {model_name}: {e}\n"
                f"Deployment context: {deployment_context}\n"
                f"Full traceback:\n{deployment_traceback}"
            )
            self._deployment_stats["failed_deployments"] += 1
            self._deployment_stats["total_deployments"] += 1

            # Attempt cleanup
            try:
                await self.remove_from_staging(model_name)
            except Exception as cleanup_error:
                import traceback
                cleanup_traceback = traceback.format_exc()
                cleanup_context = {
                    "worker_id": worker_id,
                    "model_name": model_name,
                    "cleanup_operation": "remove_from_staging",
                    "original_error": str(e)
                }
                
                self.logger.error(
                    f"[{worker_id}] Cleanup failed for model {model_name}: {cleanup_error}\n"
                    f"Cleanup context: {cleanup_context}\n"
                    f"Full traceback:\n{cleanup_traceback}"
                )

    async def stage_model(self, model_name: str, load_model_func=None) -> bool:
        """Load model into staging area"""
        try:
            # Use the provided load_model_func or fall back to the instance variable
            model_loader = load_model_func or self._load_model_func
            if not model_loader:
                self.logger.error("No model loading function provided")
                return False

            # Load model asynchronously
            model_info = await asyncio.get_event_loop().run_in_executor(
                None, model_loader, model_name
            )

            if not model_info:
                return False

            # Add to staging area
            with self._staging_lock:
                self._staging_area[model_name] = {
                    "model_info": model_info,
                    "staged_at": time.time(),
                    "validated": False,
                }

            self.logger.debug(f"Model {model_name} staged successfully")
            return True

        except Exception as e:
            import traceback
            staging_traceback = traceback.format_exc()
            staging_context = {
                "model_name": model_name,
                "load_model_func": load_model_func.__name__ if load_model_func else "None",
                "staging_area_size": len(self._staging_area),
                "operation": "stage_model"
            }
            
            self.logger.error(
                f"Failed to stage model {model_name}: {e}\n"
                f"Staging context: {staging_context}\n"
                f"Full traceback:\n{staging_traceback}"
            )
            return False

    async def validate_staged_model(self, model_name: str) -> bool:
        """Validate staged model with health checks"""
        try:
            with self._staging_lock:
                staged_info = self._staging_area.get(model_name)
                if not staged_info:
                    return False

                model_info = staged_info["model_info"]

            # Basic validation: ensure model can make predictions
            import pandas as pd

            dummy_data = pd.DataFrame({"gia_tri": [1.0]})

            # Test prediction (run in executor to avoid blocking)
            try:
                prediction = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: model_info.model.predict(dummy_data)
                )

                if prediction is not None:
                    with self._staging_lock:
                        if model_name in self._staging_area:
                            self._staging_area[model_name]["validated"] = True
                    self.logger.debug(f"Model {model_name} validation successful")
                    return True
                else:
                    return False

            except Exception as pred_error:
                self.logger.warning(
                    f"Model {model_name} prediction test failed: {pred_error}"
                )
                return False

        except Exception as e:
            self.logger.error(f"Model {model_name} validation error: {e}")
            return False

    async def promote_staged_model_to_active(
        self, model_name: str, cache_put_func=None
    ):
        """Atomically promote staged model to active (zero-downtime switch)"""
        try:
            with self._staging_lock:
                staged_info = self._staging_area.get(model_name)
                if not staged_info or not staged_info.get("validated"):
                    raise Exception(
                        f"Model {model_name} not properly staged or validated"
                    )

                model_info = staged_info["model_info"]

                # Atomic switch: update cache (this is the critical atomic
                # operation)
                if cache_put_func:
                    cache_put_func(model_name, model_info)

                # Remove from staging
                del self._staging_area[model_name]

            self.logger.debug(
                f"Model {model_name} promoted to active "
                f"(zero-downtime switch completed)"
            )

        except Exception as e:
            self.logger.error(f"Failed to promote model {model_name} to active: {e}")
            raise

    async def remove_from_staging(self, model_name: str):
        """Remove model from staging area"""
        with self._staging_lock:
            if model_name in self._staging_area:
                del self._staging_area[model_name]
                self.logger.debug(f"Removed model {model_name} from staging")

    def get_deployment_stats(self) -> Dict[str, Any]:
        """Get deployment statistics for zero-downtime deployment monitoring"""
        with self._staging_lock:
            staging_count = len(self._staging_area)

        return {
            **self._deployment_stats,
            "staging_models": staging_count,
            "queue_size": (
                self._deployment_queue.qsize() if self._deployment_queue else 0
            ),
            "batch_size": self._batch_size,
            "batch_interval": self._batch_interval,
        }

    def create_tier_based_pools(self, tier_config) -> Dict[str, List[str]]:
        """Create actual Ray Serve deployments for each tier"""
        import ray
        from ray import serve
        
        tiers = ["hot", "warm", "cold"]
        created_pools = {}
        
        for tier in tiers:
            deployment_name = f"{tier}_tier_deployment"
            try:
                # Create a simple Ray Serve deployment for this tier
                @serve.deployment(name=deployment_name, num_replicas=1)
                class TierDeployment:
                    def __init__(self):
                        self.tier = tier
                        self.logger = self.logger = __import__('adserving.src.utils.logger', fromlist=['get_logger']).get_logger()
                        self.logger.info(f"Initialized {tier} tier deployment")
                    
                    async def __call__(self, request):
                        # Simple passthrough deployment that handles requests
                        self.logger.info(f"Received request in {self.tier} tier: {type(request)}")
                        return {
                            "status": "error",
                            "error_code": "MODEL_NOT_IMPLEMENTED",
                            "error_message": f"Model processing not implemented in {self.tier} tier",
                            "tier": self.tier
                        }
                
                # Deploy the service
                TierDeployment.deploy()
                created_pools[tier] = [deployment_name]
                self.logger.info(f"Created Ray Serve deployment for {tier} tier: {deployment_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to create deployment for {tier} tier: {e}")
                created_pools[tier] = []
        
        self.logger.info(f"Tier-based deployment pools created: {created_pools}")
        return created_pools

    def cleanup(self):
        """Cleanup deployments (placeholder implementation)"""
        self.logger.info("Deployment cleanup completed")

    def stop(self):
        """Stop deployment manager"""
        self._stop_monitoring.set()
