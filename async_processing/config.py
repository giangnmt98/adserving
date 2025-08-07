"""
Configuration classes for async processing system.
Extends the existing adserving configuration system.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from .consumer import ConsumerConfig
from .db import DatabaseConfig
from .producer import RedisConfig


@dataclass
class AsyncProcessingConfig:
    """Configuration for async processing system"""

    # Redis configuration
    redis: RedisConfig = field(default_factory=lambda: RedisConfig())

    # Database configuration
    database: DatabaseConfig = field(default_factory=lambda: DatabaseConfig())

    # Consumer configuration
    consumer: ConsumerConfig = field(default_factory=lambda: ConsumerConfig())

    # System settings
    enable_async_processing: bool = True
    num_workers: int = 4
    stream_names: Dict[str, str] = field(
        default_factory=lambda: {
            "training_data": "training_data_stream",
            "inference_results": "inference_results_stream",
            "batch_processing": "batch_processing_stream",
        }
    )

    # Performance settings
    max_stream_length: int = 100000
    stream_trim_interval_seconds: int = 3600  # 1 hour
    enable_stream_monitoring: bool = True
    monitoring_interval_seconds: int = 60

    # Retry settings
    max_retry_attempts: int = 3
    retry_backoff_seconds: int = 5
    dead_letter_queue_enabled: bool = True

    # Health check settings
    health_check_interval_seconds: int = 30
    health_check_timeout_seconds: int = 10

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "redis": {
                "host": self.redis.host,
                "port": self.redis.port,
                "db": self.redis.db,
                "password": self.redis.password,
                "socket_timeout": self.redis.socket_timeout,
                "socket_connect_timeout": self.redis.socket_connect_timeout,
                "socket_keepalive": self.redis.socket_keepalive,
                "socket_keepalive_options": self.redis.socket_keepalive_options,
                "connection_pool_max_connections": self.redis.connection_pool_max_connections,
                "retry_on_timeout": self.redis.retry_on_timeout,
                "health_check_interval": self.redis.health_check_interval,
            },
            "database": {
                "host": self.database.host,
                "port": self.database.port,
                "database": self.database.database,
                "username": self.database.username,
                "password": self.database.password,
                "min_connections": self.database.min_connections,
                "max_connections": self.database.max_connections,
                "command_timeout": self.database.command_timeout,
                "server_settings": self.database.server_settings,
            },
            "consumer": {
                "consumer_name": self.consumer.consumer_name,
                "group_name": self.consumer.group_name,
                "batch_size": self.consumer.batch_size,
                "block_time_ms": self.consumer.block_time_ms,
                "max_retries": self.consumer.max_retries,
                "retry_delay_seconds": self.consumer.retry_delay_seconds,
                "processing_timeout_seconds": self.consumer.processing_timeout_seconds,
                "health_check_interval": self.consumer.health_check_interval,
                "max_pending_messages": self.consumer.max_pending_messages,
            },
            "system": {
                "enable_async_processing": self.enable_async_processing,
                "num_workers": self.num_workers,
                "stream_names": self.stream_names,
                "max_stream_length": self.max_stream_length,
                "stream_trim_interval_seconds": self.stream_trim_interval_seconds,
                "enable_stream_monitoring": self.enable_stream_monitoring,
                "monitoring_interval_seconds": self.monitoring_interval_seconds,
                "max_retry_attempts": self.max_retry_attempts,
                "retry_backoff_seconds": self.retry_backoff_seconds,
                "dead_letter_queue_enabled": self.dead_letter_queue_enabled,
                "health_check_interval_seconds": self.health_check_interval_seconds,
                "health_check_timeout_seconds": self.health_check_timeout_seconds,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "AsyncProcessingConfig":
        """Create from dictionary"""
        redis_data = data.get("redis", {})
        database_data = data.get("database", {})
        consumer_data = data.get("consumer", {})
        system_data = data.get("system", {})

        return cls(
            redis=RedisConfig(
                host=redis_data.get("host", "localhost"),
                port=redis_data.get("port", 6379),
                db=redis_data.get("db", 0),
                password=redis_data.get("password"),
                socket_timeout=redis_data.get("socket_timeout", 30),
                socket_connect_timeout=redis_data.get("socket_connect_timeout", 30),
                socket_keepalive=redis_data.get("socket_keepalive", True),
                socket_keepalive_options=redis_data.get("socket_keepalive_options", {}),
                connection_pool_max_connections=redis_data.get(
                    "connection_pool_max_connections", 50
                ),
                retry_on_timeout=redis_data.get("retry_on_timeout", True),
                health_check_interval=redis_data.get("health_check_interval", 30),
            ),
            database=DatabaseConfig(
                host=database_data.get("host", "localhost"),
                port=database_data.get("port", 5432),
                database=database_data.get("database", "adserving"),
                username=database_data.get("username", "postgres"),
                password=database_data.get("password", "password"),
                min_connections=database_data.get("min_connections", 5),
                max_connections=database_data.get("max_connections", 20),
                command_timeout=database_data.get("command_timeout", 60),
                server_settings=database_data.get("server_settings", {}),
            ),
            consumer=ConsumerConfig(
                consumer_name=consumer_data.get("consumer_name", "async_worker"),
                group_name=consumer_data.get("group_name", "processing_workers"),
                batch_size=consumer_data.get("batch_size", 10),
                block_time_ms=consumer_data.get("block_time_ms", 1000),
                max_retries=consumer_data.get("max_retries", 3),
                retry_delay_seconds=consumer_data.get("retry_delay_seconds", 5),
                processing_timeout_seconds=consumer_data.get(
                    "processing_timeout_seconds", 300
                ),
                health_check_interval=consumer_data.get("health_check_interval", 30),
                max_pending_messages=consumer_data.get("max_pending_messages", 1000),
            ),
            enable_async_processing=system_data.get("enable_async_processing", True),
            num_workers=system_data.get("num_workers", 4),
            stream_names=system_data.get(
                "stream_names",
                {
                    "training_data": "training_data_stream",
                    "inference_results": "inference_results_stream",
                    "batch_processing": "batch_processing_stream",
                },
            ),
            max_stream_length=system_data.get("max_stream_length", 100000),
            stream_trim_interval_seconds=system_data.get(
                "stream_trim_interval_seconds", 3600
            ),
            enable_stream_monitoring=system_data.get("enable_stream_monitoring", True),
            monitoring_interval_seconds=system_data.get(
                "monitoring_interval_seconds", 60
            ),
            max_retry_attempts=system_data.get("max_retry_attempts", 3),
            retry_backoff_seconds=system_data.get("retry_backoff_seconds", 5),
            dead_letter_queue_enabled=system_data.get(
                "dead_letter_queue_enabled", True
            ),
            health_check_interval_seconds=system_data.get(
                "health_check_interval_seconds", 30
            ),
            health_check_timeout_seconds=system_data.get(
                "health_check_timeout_seconds", 10
            ),
        )

    def get_redis_config(self) -> RedisConfig:
        """Get Redis configuration"""
        return self.redis

    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration"""
        return self.database

    def get_consumer_config(self) -> ConsumerConfig:
        """Get consumer configuration"""
        return self.consumer

    def is_enabled(self) -> bool:
        """Check if async processing is enabled"""
        return self.enable_async_processing

    def get_stream_name(self, stream_type: str) -> str:
        """Get stream name by type"""
        return self.stream_names.get(stream_type, f"{stream_type}_stream")


@dataclass
class AsyncProcessingEnvironmentConfig:
    """Environment-based configuration for async processing"""

    # Redis environment variables
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    REDIS_SOCKET_TIMEOUT: int = 30
    REDIS_MAX_CONNECTIONS: int = 50

    # Database environment variables
    DATABASE_HOST: str = "localhost"
    DATABASE_PORT: int = 5432
    DATABASE_NAME: str = "adserving"
    DATABASE_USER: str = "postgres"
    DATABASE_PASSWORD: str = "password"
    DATABASE_MIN_CONNECTIONS: int = 5
    DATABASE_MAX_CONNECTIONS: int = 20
    DATABASE_COMMAND_TIMEOUT: int = 60

    # Consumer environment variables
    CONSUMER_BATCH_SIZE: int = 10
    CONSUMER_BLOCK_TIME_MS: int = 1000
    CONSUMER_MAX_RETRIES: int = 3
    CONSUMER_RETRY_DELAY: int = 5
    CONSUMER_TIMEOUT: int = 300

    # System environment variables
    ASYNC_PROCESSING_ENABLED: bool = True
    ASYNC_PROCESSING_NUM_WORKERS: int = 4
    ASYNC_PROCESSING_MAX_STREAM_LENGTH: int = 100000
    ASYNC_PROCESSING_MONITORING_ENABLED: bool = True
    ASYNC_PROCESSING_MONITORING_INTERVAL: int = 60

    def to_async_processing_config(self) -> AsyncProcessingConfig:
        """Convert to AsyncProcessingConfig"""
        return AsyncProcessingConfig(
            redis=RedisConfig(
                host=self.REDIS_HOST,
                port=self.REDIS_PORT,
                db=self.REDIS_DB,
                password=self.REDIS_PASSWORD,
                socket_timeout=self.REDIS_SOCKET_TIMEOUT,
                connection_pool_max_connections=self.REDIS_MAX_CONNECTIONS,
            ),
            database=DatabaseConfig(
                host=self.DATABASE_HOST,
                port=self.DATABASE_PORT,
                database=self.DATABASE_NAME,
                username=self.DATABASE_USER,
                password=self.DATABASE_PASSWORD,
                min_connections=self.DATABASE_MIN_CONNECTIONS,
                max_connections=self.DATABASE_MAX_CONNECTIONS,
                command_timeout=self.DATABASE_COMMAND_TIMEOUT,
            ),
            consumer=ConsumerConfig(
                batch_size=self.CONSUMER_BATCH_SIZE,
                block_time_ms=self.CONSUMER_BLOCK_TIME_MS,
                max_retries=self.CONSUMER_MAX_RETRIES,
                retry_delay_seconds=self.CONSUMER_RETRY_DELAY,
                processing_timeout_seconds=self.CONSUMER_TIMEOUT,
            ),
            enable_async_processing=self.ASYNC_PROCESSING_ENABLED,
            num_workers=self.ASYNC_PROCESSING_NUM_WORKERS,
            max_stream_length=self.ASYNC_PROCESSING_MAX_STREAM_LENGTH,
            enable_stream_monitoring=self.ASYNC_PROCESSING_MONITORING_ENABLED,
            monitoring_interval_seconds=self.ASYNC_PROCESSING_MONITORING_INTERVAL,
        )


def load_async_config_from_env() -> AsyncProcessingConfig:
    """Load async processing config from environment variables"""
    import os

    env_config = AsyncProcessingEnvironmentConfig(
        # Redis
        REDIS_HOST=os.getenv("REDIS_HOST", "localhost"),
        REDIS_PORT=int(os.getenv("REDIS_PORT", "6379")),
        REDIS_DB=int(os.getenv("REDIS_DB", "0")),
        REDIS_PASSWORD=os.getenv("REDIS_PASSWORD"),
        REDIS_SOCKET_TIMEOUT=int(os.getenv("REDIS_SOCKET_TIMEOUT", "30")),
        REDIS_MAX_CONNECTIONS=int(os.getenv("REDIS_MAX_CONNECTIONS", "50")),
        # Database
        DATABASE_HOST=os.getenv("DATABASE_HOST", "localhost"),
        DATABASE_PORT=int(os.getenv("DATABASE_PORT", "5432")),
        DATABASE_NAME=os.getenv("DATABASE_NAME", "adserving"),
        DATABASE_USER=os.getenv("DATABASE_USER", "postgres"),
        DATABASE_PASSWORD=os.getenv("DATABASE_PASSWORD", "password"),
        DATABASE_MIN_CONNECTIONS=int(os.getenv("DATABASE_MIN_CONNECTIONS", "5")),
        DATABASE_MAX_CONNECTIONS=int(os.getenv("DATABASE_MAX_CONNECTIONS", "20")),
        DATABASE_COMMAND_TIMEOUT=int(os.getenv("DATABASE_COMMAND_TIMEOUT", "60")),
        # Consumer
        CONSUMER_BATCH_SIZE=int(os.getenv("CONSUMER_BATCH_SIZE", "10")),
        CONSUMER_BLOCK_TIME_MS=int(os.getenv("CONSUMER_BLOCK_TIME_MS", "1000")),
        CONSUMER_MAX_RETRIES=int(os.getenv("CONSUMER_MAX_RETRIES", "3")),
        CONSUMER_RETRY_DELAY=int(os.getenv("CONSUMER_RETRY_DELAY", "5")),
        CONSUMER_TIMEOUT=int(os.getenv("CONSUMER_TIMEOUT", "300")),
        # System
        ASYNC_PROCESSING_ENABLED=os.getenv("ASYNC_PROCESSING_ENABLED", "true").lower()
        == "true",
        ASYNC_PROCESSING_NUM_WORKERS=int(
            os.getenv("ASYNC_PROCESSING_NUM_WORKERS", "4")
        ),
        ASYNC_PROCESSING_MAX_STREAM_LENGTH=int(
            os.getenv("ASYNC_PROCESSING_MAX_STREAM_LENGTH", "100000")
        ),
        ASYNC_PROCESSING_MONITORING_ENABLED=os.getenv(
            "ASYNC_PROCESSING_MONITORING_ENABLED", "true"
        ).lower()
        == "true",
        ASYNC_PROCESSING_MONITORING_INTERVAL=int(
            os.getenv("ASYNC_PROCESSING_MONITORING_INTERVAL", "60")
        ),
    )

    return env_config.to_async_processing_config()


# Global async processing config
_async_config: Optional[AsyncProcessingConfig] = None


def get_async_config() -> AsyncProcessingConfig:
    """Get global async processing configuration"""
    global _async_config
    if _async_config is None:
        _async_config = load_async_config_from_env()
    return _async_config


def set_async_config(config: AsyncProcessingConfig) -> None:
    """Set global async processing configuration"""
    global _async_config
    _async_config = config
