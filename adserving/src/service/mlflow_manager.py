"""
MLflow connection management
Handles MLflow server connection verification with retry logic
"""

import time

from ..config.config import Config
from adserving.src.utils.logger import get_logger

logger = get_logger()


class MLflowManager:
    """Manages MLflow server connections"""

    def verify_connection(
        self, config: Config, max_retries: int = 3, retry_delay: float = 2.0
    ) -> bool:
        """Verify MLflow connection with retry mechanism."""
        mlflow_uri = config.mlflow.tracking_uri
        logger.info(f"Verifying MLflow connection to: {mlflow_uri}")

        for attempt in range(1, max_retries + 1):
            if self._attempt_connection(mlflow_uri, attempt, max_retries):
                return True

            if attempt < max_retries:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

        self._log_connection_failure(mlflow_uri, max_retries)
        return False

    def _attempt_connection(
        self, mlflow_uri: str, attempt: int, max_retries: int
    ) -> bool:
        """Attempt a single MLflow connection."""
        try:
            logger.info(f"MLflow connection attempt {attempt}/{max_retries}...")

            # Import here to avoid circular imports
            from mlflow.tracking import MlflowClient

            client = MlflowClient(tracking_uri=mlflow_uri)

            # Test connection with lightweight operation
            client.search_experiments(max_results=1)

            logger.info("MLflow connection successful!")
            return True

        except Exception as e:
            logger.warning(f"Attempt {attempt}/{max_retries} failed: {e}")
            return False

    def _log_connection_failure(self, mlflow_uri: str, max_retries: int) -> None:
        """Log detailed connection failure information."""
        logger.error("=" * 70)
        logger.error("CRITICAL: MLflow Connection Failed!")
        logger.error("=" * 70)
        logger.error(f"MLflow server: {mlflow_uri}")
        logger.error(f"Attempts made: {max_retries}")
        logger.error("=" * 70)
