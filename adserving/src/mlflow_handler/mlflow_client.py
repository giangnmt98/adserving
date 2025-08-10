"""
MLflow Client with Connection Pooling

This module provides an optimized MLflow client with connection pooling.
"""

from typing import Any, Dict, List, Optional

from mlflow.tracking import MlflowClient
from urllib3.util.retry import Retry

from adserving.src.utils.logger import get_logger


class MLflowClient:
    """MLflow client wrapper with connection pooling optimization"""

    def __init__(self, tracking_uri: str):
        self.tracking_uri = tracking_uri
        self.logger = get_logger()
        self.client = self._create_client(tracking_uri)

    def _create_client(self, tracking_uri: str) -> MlflowClient:
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

    def _paginate_registered_models(self, max_per_page: int = 100) -> List[Any]:
        """Fetch all registered models with pagination."""
        all_models: List[Any] = []
        page_token: Optional[str] = None
        while True:
            # MlflowClient.search_registered_models hỗ trợ page_token và max_results
            result = self.client.search_registered_models(
                max_results=max_per_page, page_token=page_token
            )
            # result trả về list + thuộc tính token (tuỳ phiên bản MLflow)
            if not result:
                break
            all_models.extend(result)
            # Lấy page_token tiếp theo (tuỳ SDK; nếu không có, dừng)
            try:
                page_token = result.token  # type: ignore[attr-defined]
            except Exception:
                page_token = None
            if not page_token:
                break
        return all_models

    def get_production_models_with_versions(self) -> Dict[str, str]:
        """Get production models with their versions (paginate all)."""
        try:
            models_with_versions: Dict[str, str] = {}
            # Lấy hết registered models qua pagination
            for rm in self._paginate_registered_models(max_per_page=100):
                # Lặp các latest_versions, lọc Production
                for mv in getattr(rm, "latest_versions", []) or []:
                    try:
                        if str(mv.current_stage).lower() == "production":
                            models_with_versions[rm.name] = mv.version
                            break
                    except Exception:
                        continue
            self.logger.info(
                f"Discovered {len(models_with_versions)} Production models from MLflow."
            )
            return models_with_versions
        except Exception as e:
            self.logger.error(f"Error getting production models with versions: {e}")
            return {}

    def get_production_models(self) -> List[str]:
        """Get list of production models from MLflow (paginate all)."""
        try:
            models: List[str] = []
            for rm in self._paginate_registered_models(max_per_page=100):
                for mv in getattr(rm, "latest_versions", []) or []:
                    try:
                        if str(mv.current_stage).lower() == "production":
                            models.append(rm.name)
                            break
                    except Exception:
                        continue
            self.logger.info(f"Discovered {len(models)} Production model names.")
            return models
        except Exception as e:
            self.logger.error(f"Error getting production models: {e}")
            return []

    def get_model_version_by_alias(self, model_name: str, alias: str):
        """Get a model version by alias"""
        return self.client.get_model_version_by_alias(model_name, alias)

    def search_model_versions(self, **kwargs):
        """Search model versions"""
        return self.client.search_model_versions(**kwargs)

    def search_registered_models(self):
        """Search registered models"""
        return self.client.search_registered_models()

    def get_model_parameters_by_stage(
        self, model_name: str, stages: str = "Production"
    ) -> Dict[str, Any]:
        """
        Get parameters from a model in the Production stage

        Args:
            model_name: Name of the registered model
            stages: Stage of the model version (default: Production)
        Returns:
            Dict containing model parameters from Production stage
            Returns empty dict if model not found or no parameters
        """
        try:
            production_version = self._get_model_version(model_name, stages)
            if not production_version:
                return {}

            parameters = {}

            # Get parameters in priority order
            parameters.update(self._get_version_parameters(production_version))
            parameters.update(self._get_run_parameters(production_version.run_id))
            parameters.update(self._get_model_parameters(model_name, parameters))

            self.logger.debug(
                f"Loaded {len(parameters)} parameters for Production model {model_name}"
            )
            return parameters

        except Exception as e:
            self.logger.error(
                f"Error getting Production model parameters for {model_name}: {e}"
            )
            return {}

    def _get_model_version(self, model_name: str, stages: str):
        production_versions = self.client.get_latest_versions(
            name=model_name, stages=[stages]
        )

        if not production_versions:
            self.logger.warning(f"No Production version found for model {model_name}")
            return None

        production_version = production_versions[0]
        version_number = production_version.version

        self.logger.debug(
            f"Found Production model {model_name} version {version_number}"
        )
        return production_version

    def _get_version_parameters(self, production_version) -> Dict:
        parameters = {}
        if hasattr(production_version, "tags") and production_version.tags:
            parameters.update(production_version.tags)
            self.logger.debug(
                f"Loaded {len(production_version.tags)}"
                f" parameters from model version tags"
            )
        return parameters

    def _get_run_parameters(self, run_id: str) -> Dict:
        parameters = {}
        try:
            run = self.client.get_run(run_id)

            if hasattr(run.data, "params") and run.data.params:
                parameters.update(run.data.params)
                self.logger.debug(
                    f"Added {len(run.data.params)} parameters from run params"
                )

            if hasattr(run.data, "tags") and run.data.tags:
                parameters.update(run.data.tags)
                self.logger.debug(
                    f"Added {len(run.data.tags)} parameters from run tags"
                )

        except Exception as e:
            self.logger.warning(f"Could not fetch run data: {e}")

        return parameters

    def _get_model_parameters(self, model_name: str, existing_params: Dict) -> Dict:
        parameters = {}
        try:
            registered_model = self.client.get_registered_model(model_name)
            if hasattr(registered_model, "tags") and registered_model.tags:
                for key, value in registered_model.tags.items():
                    if key not in existing_params:
                        parameters[key] = value
                self.logger.debug(
                    "Added fallback parameters from registered model tags"
                )

        except Exception as e:
            self.logger.warning(
                f"Could not fetch registered model tags for {model_name}: {e}"
            )

        return parameters

    def get_threshold_parameters(self, model_name: str) -> float:
        """
        Get threshold-related parameters from a Production model

        Args:
            model_name: Name of the model

        Returns:
            Dict with threshold parameters
        """
        all_params = self.get_model_parameters_by_stage(model_name)

        # Extract threshold-related parameters
        threshold_params = {}
        threshold_keys = [
            "threshold",
            "probability_threshold",
            "anomaly_threshold",
            "confidence_threshold",
            "decision_threshold",
            "default_probability_threshold",
            "default_anomaly_threshold",
        ]
        threshold_value = 0.0
        for key in threshold_keys:
            if key in all_params:
                try:
                    threshold_params[key] = float(all_params[key])
                    threshold_value = threshold_params[key]
                except (ValueError, TypeError):
                    self.logger.warning(
                        f"Could not convert {key}={all_params[key]}"
                        f" to float for model {model_name}"
                    )

        return threshold_value
