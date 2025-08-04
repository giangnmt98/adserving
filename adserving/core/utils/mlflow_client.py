"""
MLflow Client with Connection Pooling

This module provides an optimized MLflow client with connection pooling.
"""

import logging
from typing import Dict, List

from mlflow.tracking import MlflowClient
from urllib3.util.retry import Retry


class MLflowClient:
    """MLflow client wrapper with connection pooling optimization"""

    def __init__(self, tracking_uri: str):
        self.tracking_uri = tracking_uri
        self.logger = logging.getLogger(__name__)
        self.client = self._create_optimized_client(tracking_uri)

    def _create_optimized_client(self, tracking_uri: str) -> MlflowClient:
        """Create MLflow client with connection pooling optimization"""
        try:
            # Create MLflow client
            client = MlflowClient(tracking_uri=tracking_uri)

            # Configure connection pooling for the underlying HTTP session
            if hasattr(client, "_tracking_client") and hasattr(
                client._tracking_client, "_session"
            ):
                session = client._tracking_client._session

                # Configure retry strategy
                retry_strategy = Retry(
                    total=3,
                    backoff_factor=0.1,
                    status_forcelist=[429, 500, 502, 503, 504],
                )

                # Configure connection pooling
                from requests.adapters import HTTPAdapter

                class PooledHTTPAdapter(HTTPAdapter):
                    def __init__(self, *args, **kwargs):
                        self.pool_connections = kwargs.pop("pool_connections", 10)
                        self.pool_maxsize = kwargs.pop("pool_maxsize", 20)
                        super().__init__(*args, **kwargs)

                    def init_poolmanager(self, *args, **kwargs):
                        kwargs["maxsize"] = self.pool_maxsize
                        kwargs["block"] = False
                        return super().init_poolmanager(*args, **kwargs)

                # Mount adapters with connection pooling
                pooled_adapter = PooledHTTPAdapter(
                    pool_connections=10, pool_maxsize=20, max_retries=retry_strategy
                )

                session.mount("http://", pooled_adapter)
                session.mount("https://", pooled_adapter)

                self.logger.info(
                    "MLflow client configured with connection pooling "
                    "(10 pools, 20 max connections)"
                )

            return client

        except Exception as e:
            self.logger.warning(
                f"Failed to configure connection pooling for MLflow client: {e}"
            )
            # Fallback to standard client
            return MlflowClient(tracking_uri=tracking_uri)

    def get_production_models_with_versions(self) -> Dict[str, str]:
        """Get production models with their versions"""
        try:
            models_with_versions = {}
            for rm in self.client.search_registered_models():
                for mv in rm.latest_versions:
                    if mv.current_stage.lower() == "production":
                        models_with_versions[rm.name] = mv.version
                        break
            return models_with_versions
        except Exception as e:
            self.logger.error(f"Error getting production models with versions: {e}")
            return {}

    def get_production_models(self) -> List[str]:
        """Get list of production models from MLflow"""
        try:
            models = []
            for rm in self.client.search_registered_models():
                for mv in rm.latest_versions:
                    if mv.current_stage.lower() == "production":
                        models.append(rm.name)
                        break
            return models
        except Exception as e:
            self.logger.error(f"Error getting production models: {e}")
            return []

    def get_model_version_by_alias(self, model_name: str, alias: str):
        """Get model version by alias"""
        return self.client.get_model_version_by_alias(model_name, alias)

    def search_model_versions(self, **kwargs):
        """Search model versions"""
        return self.client.search_model_versions(**kwargs)

    def search_registered_models(self):
        """Search registered models"""
        return self.client.search_registered_models()
