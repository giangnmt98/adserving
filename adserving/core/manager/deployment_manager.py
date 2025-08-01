"""
Zero-Downtime Deployment Manager

This module handles zero-downtime model deployments with staging and validation.
"""

import asyncio
import logging
import threading
import time
from typing import Any, Dict, List, Optional


class DeploymentManager:
    """Manages zero-downtime model deployments"""

    def __init__(self, batch_size: int = 5, batch_interval: float = 2.0, load_model_func=None):
        self.logger = logging.getLogger(__name__)
        self._batch_size = batch_size
        self._batch_interval = batch_interval
        self._load_model_func = load_model_func

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
        """Initialize deployment queue and workers for zero-downtime deployment"""
        try:
            # Initialize deployment queue and semaphore
            self._deployment_queue = asyncio.Queue()
            self._deployment_semaphore = asyncio.Semaphore(self._batch_size)

            # Start deployment workers in the background
            def start_workers():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    # Start deployment workers
                    for i in range(min(3, self._batch_size)):
                        worker_task = loop.create_task(
                            self._deployment_worker(f"worker-{i}", loop)
                        )
                        self._deployment_workers.append(worker_task)

                    # Keep the loop running
                    loop.run_forever()
                except Exception as e:
                    self.logger.error(f"Error starting deployment workers: {e}")

            worker_thread = threading.Thread(target=start_workers, daemon=True)
            worker_thread.start()

            self.logger.info(
                f"Deployment infrastructure "
                f"initialized with {min(3, self._batch_size)} workers"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize deployment infrastructure: {e}")

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
                    self.logger.error(f"Failed to deploy model {model_name}: {e}")
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
                self.logger.error(
                    f"Failed to queue deployment for model {model_name}: {e}"
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

                # Rate limiting between deployments
                await asyncio.sleep(self._batch_interval / self._batch_size)

            except Exception as e:
                self.logger.error(f"Deployment worker {worker_id} error: {e}")
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
            self.logger.error(
                f"[{worker_id}] Deployment failed for model {model_name}: {e}"
            )
            self._deployment_stats["failed_deployments"] += 1
            self._deployment_stats["total_deployments"] += 1

            # Attempt cleanup
            try:
                await self.remove_from_staging(model_name)
            except Exception as cleanup_error:
                self.logger.error(
                    f"[{worker_id}] Cleanup failed for model "
                    f"{model_name}: {cleanup_error}"
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
            self.logger.error(f"Failed to stage model {model_name}: {e}")
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

            self.logger.info(
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

    def stop(self):
        """Stop deployment manager"""
        self._stop_monitoring.set()
