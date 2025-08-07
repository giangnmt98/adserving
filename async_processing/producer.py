"""
Redis producer for async data processing system.
Handles sending API request data to Redis Streams for async processing.
"""

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Union

import redis
from redis import Redis
from redis.exceptions import ConnectionError, RedisError

logger = logging.getLogger(__name__)


class RedisConfig:
    """Redis configuration class"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        socket_timeout: int = 30,
        socket_connect_timeout: int = 30,
        socket_keepalive: bool = True,
        socket_keepalive_options: Optional[Dict[str, int]] = None,
        connection_pool_max_connections: int = 50,
        retry_on_timeout: bool = True,
        health_check_interval: int = 30,
    ):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.socket_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout
        self.socket_keepalive = socket_keepalive
        self.socket_keepalive_options = socket_keepalive_options or {}
        self.connection_pool_max_connections = connection_pool_max_connections
        self.retry_on_timeout = retry_on_timeout
        self.health_check_interval = health_check_interval


class RedisStreamProducer:
    """Redis Stream producer for async data processing"""

    def __init__(self, config: RedisConfig):
        self.config = config
        self.redis_client: Optional[Redis] = None
        self._connection_pool = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize Redis connection"""
        if self._initialized:
            return

        try:
            # Create connection pool
            self._connection_pool = redis.ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                socket_keepalive=self.config.socket_keepalive,
                socket_keepalive_options=self.config.socket_keepalive_options,
                max_connections=self.config.connection_pool_max_connections,
                retry_on_timeout=self.config.retry_on_timeout,
                health_check_interval=self.config.health_check_interval,
            )

            # Create Redis client
            self.redis_client = Redis(connection_pool=self._connection_pool)

            # Test connection
            self.redis_client.ping()
            self._initialized = True

            logger.info(
                f"Redis producer initialized: {self.config.host}:"
                f"{self.config.port}/{self.config.db}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Redis producer: {e}")
            raise

    def close(self) -> None:
        """Close Redis connection"""
        if self._connection_pool:
            self._connection_pool.disconnect()
            self._initialized = False
            logger.info("Redis producer connection closed")

    def _ensure_initialized(self) -> None:
        """Ensure Redis client is initialized"""
        if not self._initialized:
            self.initialize()

    def send_training_data(
        self,
        ma_don_vi: str,
        ma_bao_cao: str,
        ky_du_lieu: str,
        ma_tieu_chi: str,
        fld_code: str,
        gia_tri: float,
        stream_name: str = "training_data_stream",
    ) -> str:
        """Send training data to Redis Stream"""
        self._ensure_initialized()

        message_data = {
            "message_id": str(uuid.uuid4()),
            "timestamp": int(time.time() * 1000),  # milliseconds
            "ma_don_vi": ma_don_vi,
            "ma_bao_cao": ma_bao_cao,
            "ky_du_lieu": ky_du_lieu,
            "ma_tieu_chi": ma_tieu_chi,
            "fld_code": fld_code,
            "gia_tri": gia_tri,
            "data_type": "training_data",
        }

        # Remove None values
        message_data = {k: v for k, v in message_data.items() if v is not None}

        try:
            message_id = self.redis_client.xadd(stream_name, message_data)
            logger.debug(
                f"Training data sent to stream {stream_name}: "
                f"{message_data['message_id']}"
            )
            return message_id.decode() if isinstance(message_id, bytes) else message_id
        except RedisError as e:
            logger.error(f"Failed to send training data to Redis: {e}")
            raise

    def send_inference_request(
        self,
        model_name: str,
        request_id: str,
        ma_don_vi: Optional[str] = None,
        ma_bao_cao: Optional[str] = None,
        ma_tieu_chi: Optional[str] = None,
        ky_du_lieu: Optional[str] = None,
        fld_code: Optional[str] = None,
        is_anomaly: bool = False,
        anomaly_score: Optional[float] = None,
        threshold: Optional[float] = None,
        processing_time: Optional[int] = None,
        model_version: Optional[str] = None,
        error_message: Optional[str] = None,
        status: str = "success",
        stream_name: str = "inference_results_stream",
    ) -> str:
        """Send inference result to Redis Stream"""
        self._ensure_initialized()

        message_data = {
            "message_id": str(uuid.uuid4()),
            "timestamp": int(time.time() * 1000),  # milliseconds
            "model_name": model_name,
            "request_id": request_id,
            "ma_don_vi": ma_don_vi,
            "ma_bao_cao": ma_bao_cao,
            "ma_tieu_chi": ma_tieu_chi,
            "ky_du_lieu": ky_du_lieu,
            "fld_code": fld_code,
            "is_anomaly": str(is_anomaly).lower(),
            "anomaly_score": anomaly_score,
            "threshold": threshold,
            "processing_time": processing_time,
            "model_version": model_version,
            "status": status,
            "error_message": error_message,
            "data_type": "inference_result",
        }

        # Remove None values
        message_data = {k: v for k, v in message_data.items() if v is not None}

        try:
            message_id = self.redis_client.xadd(stream_name, message_data)
            logger.debug(
                f"Inference result sent to stream {stream_name}: "
                f"{message_data['message_id']}"
            )
            return message_id.decode() if isinstance(message_id, bytes) else message_id
        except RedisError as e:
            logger.error(f"Failed to send inference result to Redis: {e}")
            raise

    def send_batch_data(
        self,
        messages: List[Dict[str, Any]],
        stream_name: str = "batch_processing_stream",
    ) -> List[str]:
        """Send batch of messages to Redis Stream"""
        self._ensure_initialized()

        message_ids = []
        pipeline = self.redis_client.pipeline()

        try:
            for message in messages:
                message_data = {
                    "message_id": str(uuid.uuid4()),
                    "timestamp": int(time.time() * 1000),
                    **message,
                }
                pipeline.xadd(stream_name, message_data)

            results = pipeline.execute()
            message_ids = [
                result.decode() if isinstance(result, bytes) else result
                for result in results
            ]

            logger.info(f"Batch of {len(messages)} messages sent to {stream_name}")
            return message_ids
        except RedisError as e:
            logger.error(f"Failed to send batch data to Redis: {e}")
            raise

    def get_stream_info(self, stream_name: str) -> Dict[str, Any]:
        """Get information about a Redis Stream"""
        self._ensure_initialized()

        try:
            info = self.redis_client.xinfo_stream(stream_name)
            return {
                "length": info.get(b"length", 0),
                "radix_tree_keys": info.get(b"radix-tree-keys", 0),
                "radix_tree_nodes": info.get(b"radix-tree-nodes", 0),
                "groups": info.get(b"groups", 0),
                "last_generated_id": info.get(b"last-generated-id", b"").decode(),
                "first_entry": info.get(b"first-entry"),
                "last_entry": info.get(b"last-entry"),
            }
        except RedisError as e:
            logger.error(f"Failed to get stream info for {stream_name}: {e}")
            return {}

    def create_consumer_group(
        self,
        stream_name: str,
        group_name: str,
        consumer_id: str = "0",
        mkstream: bool = True,
    ) -> bool:
        """Create consumer group for stream processing"""
        self._ensure_initialized()

        try:
            self.redis_client.xgroup_create(
                stream_name, group_name, consumer_id, mkstream=mkstream
            )
            logger.info(
                f"Consumer group '{group_name}' created for stream '{stream_name}'"
            )
            return True
        except redis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.info(f"Consumer group '{group_name}' already exists")
                return True
            else:
                logger.error(f"Failed to create consumer group: {e}")
                return False
        except RedisError as e:
            logger.error(f"Failed to create consumer group: {e}")
            return False

    def health_check(self) -> bool:
        """Check Redis connection health"""
        try:
            self._ensure_initialized()
            self.redis_client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False

    def get_stream_length(self, stream_name: str) -> int:
        """Get the length of a Redis Stream"""
        self._ensure_initialized()

        try:
            return self.redis_client.xlen(stream_name)
        except RedisError as e:
            logger.error(f"Failed to get stream length for {stream_name}: {e}")
            return 0

    def trim_stream(
        self,
        stream_name: str,
        max_length: int,
        approximate: bool = True,
    ) -> int:
        """Trim Redis Stream to maximum length"""
        self._ensure_initialized()

        try:
            return self.redis_client.xtrim(
                stream_name, maxlen=max_length, approximate=approximate
            )
        except RedisError as e:
            logger.error(f"Failed to trim stream {stream_name}: {e}")
            return 0


class AsyncDataProducer:
    """High-level async data producer combining different data types"""

    def __init__(self, config: RedisConfig):
        self.producer = RedisStreamProducer(config)
        self.training_stream = "training_data_stream"
        self.inference_stream = "inference_results_stream"
        self.batch_stream = "batch_processing_stream"

    def initialize(self) -> None:
        """Initialize producer and create consumer groups"""
        self.producer.initialize()

        # Create consumer groups for different streams
        self.producer.create_consumer_group(self.training_stream, "training_workers")
        self.producer.create_consumer_group(self.inference_stream, "inference_workers")
        self.producer.create_consumer_group(self.batch_stream, "batch_workers")

    def close(self) -> None:
        """Close producer connection"""
        self.producer.close()

    def send_api_request_data(
        self,
        model_name: str,
        request_id: str,
        input_data: Dict[str, Any],
        prediction: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, str]:
        """Send API request data for async processing"""
        message_ids = {}

        if not model_name or not request_id:
            raise ValueError("model_name and request_id are required")

        if not input_data or not isinstance(input_data, dict):
            raise ValueError("input_data must be a non-empty dictionary")

        if prediction is not None and not isinstance(prediction, dict):
            raise ValueError("prediction must be a dictionary if provided")

        try:
            # Send training data
            training_id = self.producer.send_training_data(
                ma_don_vi=input_data.get("ma_don_vi"),
                ma_bao_cao=input_data.get("ma_bao_cao"),
                ky_du_lieu=input_data.get("ky_du_lieu"),
                ma_tieu_chi=input_data.get("ma_tieu_chi"),
                fld_code=input_data.get("fld_code"),
                gia_tri=input_data.get("gia_tri"),
                stream_name=self.training_stream,
            )
            message_ids["training_data"] = training_id

            # Send inference result if provided
            if prediction:
                inference_id = self.producer.send_inference_request(
                    model_name=model_name,
                    request_id=request_id,
                    ma_don_vi=prediction.get("ma_don_vi"),
                    ma_bao_cao=prediction.get("ma_bao_cao"),
                    ma_tieu_chi=prediction.get("ma_tieu_chi"),
                    ky_du_lieu=prediction.get("ky_du_lieu"),
                    fld_code=prediction.get("fld_code"),
                    is_anomaly=prediction.get("is_anomaly", False),
                    anomaly_score=prediction.get("anomaly_score"),
                    threshold=prediction.get("threshold"),
                    processing_time=prediction.get("processing_time"),
                    model_version=prediction.get("model_version"),
                    error_message=prediction.get("error_message"),
                    status=prediction.get("status", "success"),
                    stream_name=self.inference_stream,
                )
                message_ids["inference_result"] = inference_id

            return message_ids

        except Exception as e:
            logger.error(f"Failed to send API request data: {str(e)}")
            raise

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status for all streams"""
        return {
            "training_stream": self.producer.get_stream_info(self.training_stream),
            "inference_stream": self.producer.get_stream_info(self.inference_stream),
            "batch_stream": self.producer.get_stream_info(self.batch_stream),
            "health": self.producer.health_check(),
        }


# Global producer instance
_producer: Optional[AsyncDataProducer] = None


def get_producer() -> AsyncDataProducer:
    """Get global producer instance"""
    global _producer
    if _producer is None:
        raise RuntimeError("Producer not initialized")
    return _producer


def initialize_producer(config: RedisConfig) -> AsyncDataProducer:
    """Initialize global producer"""
    global _producer
    _producer = AsyncDataProducer(config)
    _producer.initialize()
    return _producer
