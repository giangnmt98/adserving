"""
Database connection and data access layer for async processing system.
Supports PostgreSQL with connection pooling and TimescaleDB integration.
"""

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import asyncpg
from asyncpg import Connection, Pool

from adserving.src.utils.logger import get_logger

logger = get_logger()


class DatabaseConfig:
    """Database configuration class"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "adserving",
        username: str = "postgres",
        password: str = "password",
        min_connections: int = 5,
        max_connections: int = 20,
        command_timeout: int = 60,
        server_settings: Optional[Dict[str, str]] = None,
    ):
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.command_timeout = command_timeout
        self.server_settings = server_settings or {}

    @property
    def dsn(self) -> str:
        """Get database connection string"""
        return (
            f"postgresql://{self.username}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
        )


class AsyncDatabaseManager:
    """Async database manager with connection pooling"""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.pool: Optional[Pool] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize database connection pool"""
        if self._initialized:
            return

        try:
            self.pool = await asyncpg.create_pool(
                dsn=self.config.dsn,
                min_size=self.config.min_connections,
                max_size=self.config.max_connections,
                command_timeout=self.config.command_timeout,
                server_settings=self.config.server_settings,
            )
            self._initialized = True
            logger.info(
                f"Database pool initialized: {self.config.host}:"
                f"{self.config.port}/{self.config.database}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise

    async def close(self) -> None:
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            self._initialized = False
            logger.info("Database pool closed")

    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool"""
        if not self._initialized:
            await self.initialize()

        async with self.pool.acquire() as connection:
            yield connection

    async def execute_script(self, script_path: str) -> None:
        """Execute SQL script file"""
        try:
            with open(script_path, "r", encoding="utf-8") as f:
                script = f.read()

            async with self.get_connection() as conn:
                await conn.execute(script)
            logger.info(f"Successfully executed script: {script_path}")
        except Exception as e:
            logger.error(f"Failed to execute script {script_path}: {e}")
            raise


class TrainingDataRepository:
    """Repository for training data operations"""

    def __init__(self, db_manager: AsyncDatabaseManager):
        self.db_manager = db_manager

    async def insert_training_data(
        self,
        ma_don_vi: str,
        ma_bao_cao: str,
        ky_du_lieu: str,
        ma_tieu_chi: str,
        fld_code: str,
        gia_tri: float,
    ) -> int:
        """Insert training data record"""
        query = """
                INSERT INTO training_data (ma_don_vi, ma_bao_cao, ky_du_lieu, ma_tieu_chi,
                                           fld_code, gia_tri)
                VALUES ($1, $2, $3, $4, $5, $6) RETURNING id \
                """

        async with self.db_manager.get_connection() as conn:
            row = await conn.fetchrow(
                query, ma_don_vi, ma_bao_cao, ky_du_lieu, ma_tieu_chi, fld_code, gia_tri
            )
            return row["id"]


class InferenceResultsRepository:
    """Repository for inference results operations"""

    def __init__(self, db_manager: AsyncDatabaseManager):
        self.db_manager = db_manager

    async def insert_inference_result(
        self,
        model_name: str,
        request_id: str,
        ma_tieu_chi: Optional[str] = None,
        fld_code: Optional[str] = None,
        is_anomaly: bool = False,
        anomaly_score: Optional[float] = None,
        anomaly_threshold: Optional[float] = None,
        processing_time: Optional[int] = None,
        model_version: Optional[str] = None,
        error_message: Optional[str] = None,
        status: str = "success",
    ) -> int:
        """Insert inference result record"""
        query = """
                INSERT INTO inference_results (model_name, request_id, ma_tieu_chi,
                                               fld_code, is_anomaly, anomaly_score, anomaly_threshold,
                                               processing_time, model_version, error_message, status)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11) RETURNING id \
                """

        async with self.db_manager.get_connection() as conn:
            row = await conn.fetchrow(
                query,
                model_name,
                request_id,
                ma_tieu_chi,
                fld_code,
                is_anomaly,
                anomaly_score,
                anomaly_threshold,
                processing_time,
                model_version,
                error_message,
                status,
            )
            return row["id"]


class QueueStatusRepository:
    """Repository for queue status operations"""

    def __init__(self, db_manager: AsyncDatabaseManager):
        self.db_manager = db_manager

    async def insert_queue_status(
        self,
        queue_name: str,
        message_id: str,
        status: str = "pending",
    ) -> int:
        """Insert queue status record"""
        query = """
        INSERT INTO queue_status (queue_name, message_id, status)
        VALUES ($1, $2, $3)
        RETURNING id
        """

        async with self.db_manager.get_connection() as conn:
            row = await conn.fetchrow(query, queue_name, message_id, status)
            return row["id"]

    async def update_queue_status(
        self,
        message_id: str,
        status: str,
        error_message: Optional[str] = None,
        retry_count: Optional[int] = None,
    ) -> None:
        """Update queue status"""
        query = """
        UPDATE queue_status
        SET status = $2, error_message = $3, retry_count = COALESCE($4, retry_count),
            processing_started_at = CASE WHEN $2 = 'processing'
                                   THEN NOW() ELSE processing_started_at END,
            processing_completed_at = CASE WHEN $2 IN ('completed', 'failed')
                                     THEN NOW() ELSE processing_completed_at END
        WHERE message_id = $1
        """

        async with self.db_manager.get_connection() as conn:
            await conn.execute(query, message_id, status, error_message, retry_count)

    async def get_failed_messages(
        self,
        queue_name: str,
        max_retry_count: int = 3,
    ) -> List[Dict[str, Any]]:
        """Get failed messages for retry"""
        query = """
        SELECT * FROM queue_status
        WHERE queue_name = $1
        AND status = 'failed'
        AND retry_count < $2
        ORDER BY created_at ASC
        """

        async with self.db_manager.get_connection() as conn:
            rows = await conn.fetch(query, queue_name, max_retry_count)
            return [dict(row) for row in rows]


class DatabaseService:
    """Main database service combining all repositories"""

    def __init__(self, config: DatabaseConfig):
        self.db_manager = AsyncDatabaseManager(config)
        self.training_data = TrainingDataRepository(self.db_manager)
        self.inference_results = InferenceResultsRepository(self.db_manager)
        self.queue_status = QueueStatusRepository(self.db_manager)

    async def initialize(self) -> None:
        """Initialize database service"""
        await self.db_manager.initialize()

    async def close(self) -> None:
        """Close database service"""
        await self.db_manager.close()

    async def setup_schema(self, schema_path: str = "models.sql") -> None:
        """Setup database schema"""
        await self.db_manager.execute_script(schema_path)

    async def health_check(self) -> bool:
        """Check database health"""
        try:
            async with self.db_manager.get_connection() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False


# Global database service instance
_db_service: Optional[DatabaseService] = None


def get_database_service() -> DatabaseService:
    """Get global database service instance"""
    global _db_service
    if _db_service is None:
        raise RuntimeError("Database service not initialized")
    return _db_service


def initialize_database_service(config: DatabaseConfig) -> DatabaseService:
    """Initialize global database service"""
    global _db_service
    _db_service = DatabaseService(config)
    return _db_service
