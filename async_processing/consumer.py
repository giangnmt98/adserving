"""
Ray-based async consumer/worker for processing Redis Stream messages.
Handles distributed processing of training data and inference results.
"""

import asyncio
import logging
import traceback
from typing import Any, Dict, List, Optional, Tuple

import ray
import redis
from redis import Redis
from redis.exceptions import RedisError

from .db import DatabaseConfig, DatabaseService
from .producer import RedisConfig

logger = logging.getLogger(__name__)


class ConsumerConfig:
    """Consumer configuration class"""

    def __init__(
        self,
        consumer_name: str = "async_worker",
        group_name: str = "processing_workers",
        batch_size: int = 10,
        block_time_ms: int = 1000,
        max_retries: int = 3,
        retry_delay_seconds: int = 5,
        processing_timeout_seconds: int = 300,
        health_check_interval: int = 30,
        max_pending_messages: int = 1000,
    ):
        self.consumer_name = consumer_name
        self.group_name = group_name
        self.batch_size = batch_size
        self.block_time_ms = block_time_ms
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.processing_timeout_seconds = processing_timeout_seconds
        self.health_check_interval = health_check_interval
        self.max_pending_messages = max_pending_messages


@ray.remote
class AsyncWorker:
    """Ray-based async worker for processing stream messages"""

    def __init__(
        self,
        redis_config: RedisConfig,
        db_config: DatabaseConfig,
        consumer_config: ConsumerConfig,
    ):
        self.redis_config = redis_config
        self.db_config = db_config
        self.consumer_config = consumer_config
        self.redis_client: Optional[Redis] = None
        self.db_service: Optional[DatabaseService] = None
        self._initialized = False
        self._running = False

    async def initialize(self) -> None:
        """Initialize worker components"""
        if self._initialized:
            return

        try:
            # Initialize Redis client
            self.redis_client = Redis(
                host=self.redis_config.host,
                port=self.redis_config.port,
                db=self.redis_config.db,
                password=self.redis_config.password,
                socket_timeout=self.redis_config.socket_timeout,
                decode_responses=False,  # Keep bytes for stream processing
            )

            # Test Redis connection
            self.redis_client.ping()

            # Initialize database service
            self.db_service = DatabaseService(self.db_config)
            await self.db_service.initialize()

            self._initialized = True
            logger.info(f"Worker {self.consumer_config.consumer_name} initialized")

        except Exception as e:
            logger.error(f"Failed to initialize worker: {e}")
            raise

    async def close(self) -> None:
        """Close worker connections"""
        self._running = False

        if self.redis_client:
            self.redis_client.close()

        if self.db_service:
            await self.db_service.close()

        logger.info(f"Worker {self.consumer_config.consumer_name} closed")

    async def process_training_data_message(self, message_data: Dict[str, Any]) -> bool:
        """Process training data message"""
        try:
            # Extract message fields
            ma_don_vi = message_data.get("ma_don_vi")
            ma_bao_cao = message_data.get("ma_bao_cao")
            ky_du_lieu = message_data.get("ky_du_lieu")
            ma_tieu_chi = message_data.get("ma_tieu_chi")
            fld_code = message_data.get("fld_code")
            gia_tri = message_data.get("gia_tri")

            if not all(
                [ma_don_vi, ma_bao_cao, ky_du_lieu, ma_tieu_chi, fld_code, gia_tri]
            ):
                logger.error("Missing required fields in training data message")
                return False

            # Convert gia_tri to float if needed
            if isinstance(gia_tri, str):
                gia_tri = float(gia_tri)

            # Save to database
            record_id = await self.db_service.training_data.insert_training_data(
                ma_don_vi=ma_don_vi,
                ma_bao_cao=ma_bao_cao,
                ky_du_lieu=ky_du_lieu,
                ma_tieu_chi=ma_tieu_chi,
                fld_code=fld_code,
                gia_tri=gia_tri,
            )

            logger.debug(f"Training data saved with ID: {record_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to process training data message: {e}")
            logger.error(traceback.format_exc())
            return False

    async def process_inference_result_message(
        self, message_data: Dict[str, Any]
    ) -> bool:
        """Process inference result message"""
        try:
            # Extract message fields
            request_id = message_data.get("request_id")
            model_name = message_data.get("model_name")
            model_version = message_data.get("model_version")
            ma_tieu_chi = message_data.get("ma_tieu_chi")
            fld_code = message_data.get("fld_code")
            is_anomaly_str = message_data.get("is_anomaly", "false")
            anomaly_score = message_data.get("anomaly_score")
            anomaly_threshold = message_data.get("anomaly_threshold")
            processing_time = message_data.get("processing_time")
            error_message = message_data.get("error_message")
            status = message_data.get("status", "success")

            if not all([model_name, request_id]):
                logger.error("Missing required fields in inference result message")
                return False

            # Convert types
            is_anomaly = is_anomaly_str.lower() == "true"

            if anomaly_score is not None:
                anomaly_score = float(anomaly_score)
            if anomaly_threshold is not None:
                threshold = float(anomaly_threshold)
            if processing_time is not None:
                processing_time = int(processing_time)

            # Save to database
            record_id = await self.db_service.inference_results.insert_inference_result(
                model_name=model_name,
                request_id=request_id,
                ma_tieu_chi=ma_tieu_chi,
                fld_code=fld_code,
                is_anomaly=is_anomaly,
                anomaly_score=anomaly_score,
                anomaly_threshold=anomaly_threshold,
                processing_time=processing_time,
                model_version=model_version,
                error_message=error_message,
                status=status,
            )

            logger.debug(f"Inference result saved with ID: {record_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to process inference result message: {e}")
            logger.error(traceback.format_exc())
            return False

    async def process_message(
        self, stream_name: str, message_id: str, message_data: Dict[str, Any]
    ) -> bool:
        """Process a single message based on data type"""
        try:
            # Track message processing in database
            await self.db_service.queue_status.insert_queue_status(
                queue_name=stream_name,
                message_id=message_id,
                status="processing",
            )

            # Determine message type and process accordingly
            data_type = message_data.get("data_type")
            success = False

            if data_type == "training_data":
                success = await self.process_training_data_message(message_data)
            elif data_type == "inference_result":
                success = await self.process_inference_result_message(message_data)
            else:
                logger.error(f"Unknown data type: {data_type}")
                success = False

            # Update processing status
            status = "completed" if success else "failed"
            await self.db_service.queue_status.update_queue_status(
                message_id=message_id,
                status=status,
                error_message=None if success else "Processing failed",
            )

            return success

        except Exception as e:
            logger.error(f"Failed to process message {message_id}: {e}")
            await self.db_service.queue_status.update_queue_status(
                message_id=message_id,
                status="failed",
                error_message=str(e),
            )
            return False

    async def consume_stream(
        self, stream_name: str, last_id: str = ">"
    ) -> List[Tuple[str, bool]]:
        """Consume messages from a Redis Stream"""
        if not self._initialized:
            await self.initialize()

        try:
            # Read messages from stream
            messages = self.redis_client.xreadgroup(
                self.consumer_config.group_name,
                self.consumer_config.consumer_name,
                {stream_name: last_id},
                count=self.consumer_config.batch_size,
                block=self.consumer_config.block_time_ms,
            )

            results = []

            for stream, stream_messages in messages:
                for message_id, fields in stream_messages:
                    # Convert bytes to strings
                    message_id = (
                        message_id.decode()
                        if isinstance(message_id, bytes)
                        else message_id
                    )
                    message_data = {
                        k.decode() if isinstance(k, bytes) else k: (
                            v.decode() if isinstance(v, bytes) else v
                        )
                        for k, v in fields.items()
                    }

                    # Process message
                    success = await self.process_message(
                        stream_name, message_id, message_data
                    )
                    results.append((message_id, success))

                    # Acknowledge message if processed successfully
                    if success:
                        self.redis_client.xack(
                            stream_name, self.consumer_config.group_name, message_id
                        )

            return results

        except RedisError as e:
            logger.error(f"Redis error while consuming stream {stream_name}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error while consuming stream {stream_name}: {e}")
            return []

    async def run_worker(self, streams: List[str]) -> None:
        """Run worker to continuously process streams"""
        if not self._initialized:
            await self.initialize()

        self._running = True
        logger.info(
            f"Worker {self.consumer_config.consumer_name} started for streams: {streams}"
        )

        while self._running:
            try:
                for stream_name in streams:
                    results = await self.consume_stream(stream_name)

                    if results:
                        successful = sum(1 for _, success in results if success)
                        total = len(results)
                        logger.info(
                            f"Processed {successful}/{total} messages from {stream_name}"
                        )

                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                await asyncio.sleep(self.consumer_config.retry_delay_seconds)

    async def health_check(self) -> Dict[str, Any]:
        """Check worker health"""
        try:
            redis_health = self.redis_client.ping() if self.redis_client else False
            db_health = (
                await self.db_service.health_check() if self.db_service else False
            )

            return {
                "worker_name": self.consumer_config.consumer_name,
                "initialized": self._initialized,
                "running": self._running,
                "redis_health": redis_health,
                "database_health": db_health,
                "overall_health": redis_health and db_health and self._initialized,
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "worker_name": self.consumer_config.consumer_name,
                "initialized": False,
                "running": False,
                "redis_health": False,
                "database_health": False,
                "overall_health": False,
                "error": str(e),
            }


class AsyncConsumerManager:
    """Manager for multiple async workers"""

    def __init__(
        self,
        redis_config: RedisConfig,
        db_config: DatabaseConfig,
        num_workers: int = 4,
    ):
        self.redis_config = redis_config
        self.db_config = db_config
        self.num_workers = num_workers
        self.workers: List[ray.ObjectRef] = []
        self.worker_configs: List[ConsumerConfig] = []

    def create_workers(self) -> None:
        """Create Ray workers"""
        for i in range(self.num_workers):
            consumer_config = ConsumerConfig(
                consumer_name=f"worker_{i}",
                group_name="processing_workers",
            )
            self.worker_configs.append(consumer_config)

            worker = AsyncWorker.remote(
                self.redis_config,
                self.db_config,
                consumer_config,
            )
            self.workers.append(worker)

        logger.info(f"Created {self.num_workers} async workers")

    async def start_workers(self, streams: List[str] = None) -> None:
        """Start all workers"""
        if streams is None:
            streams = ["training_data_stream", "inference_results_stream"]

        if not self.workers:
            self.create_workers()

        # Start workers
        tasks = []
        for worker in self.workers:
            task = worker.run_worker.remote(streams)
            tasks.append(task)

        logger.info(f"Started {len(self.workers)} workers for streams: {streams}")

        # Wait for all workers (this will run indefinitely)
        try:
            await asyncio.gather(*[ray.get(task) for task in tasks])
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, stopping workers...")
            await self.stop_workers()

    async def stop_workers(self) -> None:
        """Stop all workers"""
        for worker in self.workers:
            try:
                ray.get(worker.close.remote())
            except Exception as e:
                logger.error(f"Error stopping worker: {e}")

        self.workers.clear()
        logger.info("All workers stopped")

    async def get_workers_health(self) -> List[Dict[str, Any]]:
        """Get health status of all workers"""
        health_checks = []
        for worker in self.workers:
            try:
                health = ray.get(worker.health_check.remote())
                health_checks.append(health)
            except Exception as e:
                health_checks.append(
                    {
                        "worker_name": "unknown",
                        "overall_health": False,
                        "error": str(e),
                    }
                )

        return health_checks


# Global consumer manager
_consumer_manager: Optional[AsyncConsumerManager] = None


def get_consumer_manager() -> AsyncConsumerManager:
    """Get global consumer manager instance"""
    global _consumer_manager
    if _consumer_manager is None:
        raise RuntimeError("Consumer manager not initialized")
    return _consumer_manager


def initialize_consumer_manager(
    redis_config: RedisConfig,
    db_config: DatabaseConfig,
    num_workers: int = 4,
) -> AsyncConsumerManager:
    """Initialize global consumer manager"""
    global _consumer_manager

    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init()

    _consumer_manager = AsyncConsumerManager(redis_config, db_config, num_workers)
    return _consumer_manager
