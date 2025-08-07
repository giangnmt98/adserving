"""
Async Processing Package for Adserving System

This package provides asynchronous data processing capabilities using:
- Redis Streams for message queuing
- Ray Tasks for distributed processing
- PostgreSQL with TimescaleDB for data storage

Components:
- producer: Redis producer for sending data to streams
- consumer: Ray-based workers for processing stream messages
- db: Database connection and data access layer
- config: Configuration management
- models.sql: Database schema

Usage:
    from adserving.async_processing import AsyncProcessingSystem

    # Initialize system
    system = AsyncProcessingSystem()
    await system.initialize()

    # Send data for async processing
    await system.send_training_data(
        model_name="my_model",
        request_id="req_123",
        input_data={"feature1": 1.0, "feature2": 2.0}
    )

    # Start workers
    await system.start_workers()
"""

from .config import (
    AsyncProcessingConfig,
    get_async_config,
    load_async_config_from_env,
    set_async_config,
)
from .consumer import (
    AsyncConsumerManager,
    ConsumerConfig,
    get_consumer_manager,
    initialize_consumer_manager,
)
from .db import (
    DatabaseConfig,
    DatabaseService,
    get_database_service,
    initialize_database_service,
)
from .producer import AsyncDataProducer, RedisConfig, get_producer, initialize_producer

__version__ = "1.0.0"
__author__ = "Adserving Team"

__all__ = [
    # Configuration
    "AsyncProcessingConfig",
    "get_async_config",
    "set_async_config",
    "load_async_config_from_env",
    # Database
    "DatabaseConfig",
    "DatabaseService",
    "get_database_service",
    "initialize_database_service",
    # Producer
    "RedisConfig",
    "AsyncDataProducer",
    "get_producer",
    "initialize_producer",
    # Consumer
    "ConsumerConfig",
    "AsyncConsumerManager",
    "get_consumer_manager",
    "initialize_consumer_manager",
    # Main system
    "AsyncProcessingSystem",
]


class AsyncProcessingSystem:
    """
    Main system class that integrates all async processing components
    """

    def __init__(self, config: AsyncProcessingConfig = None):
        """Initialize async processing system"""
        self.config = config or get_async_config()
        self.producer = None
        self.consumer_manager = None
        self.database_service = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all system components"""
        if self._initialized:
            return

        # Initialize database service
        self.database_service = initialize_database_service(
            self.config.get_database_config()
        )
        await self.database_service.initialize()

        # Initialize producer
        self.producer = initialize_producer(self.config.get_redis_config())

        # Initialize consumer manager
        self.consumer_manager = initialize_consumer_manager(
            self.config.get_redis_config(),
            self.config.get_database_config(),
            self.config.num_workers,
        )

        self._initialized = True

    async def close(self) -> None:
        """Close all system components"""
        if self.consumer_manager:
            await self.consumer_manager.stop_workers()

        if self.producer:
            self.producer.close()

        if self.database_service:
            await self.database_service.close()

        self._initialized = False

    async def setup_database(self, schema_path: str = None) -> None:
        """Setup database schema"""
        if not self._initialized:
            await self.initialize()

        if schema_path is None:
            import os

            schema_path = os.path.join(os.path.dirname(__file__), "models.sql")

        await self.database_service.setup_schema(schema_path)

    async def send_training_data(
        self, model_name: str, request_id: str, input_data: dict, **kwargs
    ) -> str:
        """Send training data for async processing"""
        if not self._initialized:
            await self.initialize()

        return self.producer.producer.send_training_data(
            model_name=model_name,
            request_id=request_id,
            input_data=input_data,
            stream_name=self.config.get_stream_name("training_data"),
            **kwargs
        )

    async def send_inference_result(
        self, model_name: str, request_id: str, prediction: dict, **kwargs
    ) -> str:
        """Send inference result for async processing"""
        if not self._initialized:
            await self.initialize()

        return self.producer.producer.send_inference_request(
            model_name=model_name,
            request_id=request_id,
            prediction=prediction,
            stream_name=self.config.get_stream_name("inference_results"),
            **kwargs
        )

    async def send_api_request_data(
        self,
        model_name: str,
        request_id: str,
        input_data: dict,
        prediction: dict = None,
        **kwargs
    ) -> dict:
        """Send complete API request data for async processing"""
        if not self._initialized:
            await self.initialize()

        return self.producer.send_api_request_data(
            model_name=model_name,
            request_id=request_id,
            input_data=input_data,
            prediction=prediction,
            **kwargs
        )

    async def start_workers(self, streams: list = None) -> None:
        """Start async workers"""
        if not self._initialized:
            await self.initialize()

        if streams is None:
            streams = [
                self.config.get_stream_name("training_data"),
                self.config.get_stream_name("inference_results"),
            ]

        await self.consumer_manager.start_workers(streams)

    async def stop_workers(self) -> None:
        """Stop async workers"""
        if self.consumer_manager:
            await self.consumer_manager.stop_workers()

    async def get_system_status(self) -> dict:
        """Get comprehensive system status"""
        if not self._initialized:
            return {"initialized": False}

        status = {
            "initialized": True,
            "config": {
                "async_processing_enabled": self.config.is_enabled(),
                "num_workers": self.config.num_workers,
                "stream_names": self.config.stream_names,
            },
        }

        # Producer status
        if self.producer:
            status["producer"] = self.producer.get_system_status()

        # Database status
        if self.database_service:
            status["database"] = {"health": await self.database_service.health_check()}

        # Workers status
        if self.consumer_manager:
            status["workers"] = await self.consumer_manager.get_workers_health()

        return status

    async def get_model_performance_stats(
        self, model_name: str, hours: int = 24
    ) -> dict:
        """Get model performance statistics"""
        if not self._initialized:
            await self.initialize()

        return (
            await self.database_service.inference_results.get_model_performance_stats(
                model_name, hours
            )
        )

    def is_enabled(self) -> bool:
        """Check if async processing is enabled"""
        return self.config.is_enabled()

    def is_initialized(self) -> bool:
        """Check if system is initialized"""
        return self._initialized
