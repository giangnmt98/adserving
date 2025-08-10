# Python
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import mlflow.pyfunc
from ray import serve

from adserving.src.mlflow_utils.mlflow_client import MLflowClient as WrappedMLflowClient
from adserving.src.mlflow_utils.mlflow_parameter_updater import MLflowParameterUpdater
from adserving.src.utils.logger import get_logger


def _build_model_uri(name: str, version: str | int) -> str:
    return f"models:/{name}/{version}"


def _parse_model_name(model_name: str) -> Tuple[str, str]:
    parts = model_name.split("_")
    if len(parts) < 4:
        return "", ""
    fld_code = parts[-1]
    ma_tieu_chi = "_".join(parts[2:-1]) if len(parts) > 3 else ""
    return ma_tieu_chi, fld_code


@serve.deployment(
    name="preloaded_model_server",
    ray_actor_options={"num_cpus": 1},
    autoscaling_config={
        "min_replicas": 2,
        "max_replicas": 8,
        "target_num_ongoing_requests_per_replica": 12,
    },
)
class PreloadedModelServer:
    """
    Một deployment duy nhất:
    - Preload tất cả Production models từ MLflow.
    - Duy trì zero-downtime bằng watcher phát hiện bản mới, load vào staging và atomic swap sang active.
    """

    def __init__(
        self,
        tracking_uri: str,
        max_workers: int = 8,
        max_load_concurrency: int = 8,
        watcher_interval_seconds: int = 60,
        sanity_check_enabled: bool = False,
        sanity_inputs: Optional[List[float]] = None,
    ) -> None:
        self.logger = get_logger()
        self.is_ready: bool = False

        # Bản đồ active/staging
        self.models_active: Dict[str, Any] = {}
        self.model_versions: Dict[str, str] = {}
        self.thresholds: Dict[str, Optional[float]] = {}

        self.models_staging: Dict[str, Any] = {}
        self.thresholds_staging: Dict[str, Optional[float]] = {}
        self.model_versions_staging: Dict[str, str] = {}

        self._lock = asyncio.Lock()

        self._client = WrappedMLflowClient(tracking_uri=tracking_uri)
        self._updater = MLflowParameterUpdater(self._client)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        self._watcher_interval = max(5, int(watcher_interval_seconds))
        self._sanity_enabled = bool(sanity_check_enabled)
        self._sanity_inputs = sanity_inputs or []

        # Preload đồng bộ khi khởi động
        self.logger.info("Preloading Production models from MLflow...")
        self._preload_all_sync(max_load_concurrency)
        self.is_ready = True
        self.logger.info(f"Preload completed. Total loaded: {len(self.models_active)} models.")

        # Khởi động watcher nền
        loop = asyncio.get_event_loop()
        loop.create_task(self._watch_production_models())

    # -------------------------- utils --------------------------

    def _list_production(self) -> Dict[str, str]:
        try:
            return self._client.get_production_models_with_versions()
        except Exception as e:
            self.logger.error(f"List production models failed: {e}")
            return {}

    def _fetch_threshold(self, model_name: str) -> Optional[float]:
        try:
            params = self._updater.get_current_parameters(model_name)
            if isinstance(params, dict):
                th = params.get("anomaly_threshold")
                if th is None:
                    th = params.get("threshold")
                if th is not None:
                    return float(th)
        except Exception as e:
            self.logger.debug(f"Cannot fetch threshold for {model_name}: {e}")
        return None

    def _load_model_version(self, name: str, ver: str | int) -> Tuple[Optional[Any], Optional[float], Optional[str]]:
        """Load 1 model version, trả model, threshold, error."""
        try:
            uri = _build_model_uri(name, ver)
            model = mlflow.pyfunc.load_model(uri)
            th = self._fetch_threshold(name)
            return model, th, None
        except Exception as e:
            return None, None, str(e)

    async def _sanity_check(self, model: Any) -> bool:
        """Sanity check đơn giản, không bắt buộc."""
        if not self._sanity_enabled or not self._sanity_inputs:
            return True
        try:
            for v in self._sanity_inputs:
                _ = float(model.predict(float(v)))
            return True
        except Exception as e:
            self.logger.error(f"Sanity check failed: {e}")
            return False

    # ------------------------ preload/init ----------------------

    def _preload_all_sync(self, max_concurrency: int) -> None:
        prod_map = self._list_production()
        if not prod_map:
            self.logger.warning("No Production models found in MLflow.")
            return

        with ThreadPoolExecutor(max_workers=max(1, int(max_concurrency))) as ex:
            futures = {}
            for name, ver in prod_map.items():
                futures[ex.submit(self._load_model_version, name, ver)] = (name, ver)

            ok = 0
            for fut in as_completed(futures):
                name, ver = futures[fut]
                model, th, err = fut.result()
                if err or model is None:
                    self.logger.error(f"Load model {name} v{ver} failed: {err}")
                    continue
                self.models_active[name] = model
                self.model_versions[name] = str(ver)
                self.thresholds[name] = th
                ok += 1
        self.logger.info(f"Loaded {ok}/{len(prod_map)} production models.")

    # ------------------------- watcher --------------------------

    async def _watch_production_models(self) -> None:
        self.logger.info(f"Watcher started (interval={self._watcher_interval}s).")
        while True:
            try:
                await asyncio.sleep(self._watcher_interval)
                prod_map = self._list_production()
                if not prod_map:
                    continue

                # Phát hiện model mới/cập nhật
                updates: List[Tuple[str, str]] = []
                for name, ver in prod_map.items():
                    active_ver = self.model_versions.get(name)
                    if active_ver is None or str(active_ver) != str(ver):
                        updates.append((name, str(ver)))

                if not updates:
                    continue

                self.logger.info(f"Detected {len(updates)} production updates: {updates}")

                # Load vào staging song song
                loop = asyncio.get_event_loop()
                load_tasks = []
                for name, ver in updates:
                    load_tasks.append(loop.run_in_executor(self._executor, self._load_model_version, name, ver))

                results = await asyncio.gather(*load_tasks, return_exceptions=False)

                # Ghi staging + sanity + swap
                for (name, ver), (model, th, err) in zip(updates, results):
                    if err or model is None:
                        self.logger.error(f"Staging load failed for {name} v{ver}: {err}")
                        continue

                    if not await self._sanity_check(model):
                        self.logger.error(f"Sanity check failed for {name} v{ver}, skip swap.")
                        continue

                    async with self._lock:
                        # Put into staging
                        self.models_staging[name] = model
                        self.model_versions_staging[name] = ver
                        self.thresholds_staging[name] = th

                        # Atomic swap
                        self.models_active[name] = self.models_staging.pop(name)
                        self.model_versions[name] = self.model_versions_staging.pop(name)
                        self.thresholds[name] = self.thresholds_staging.pop(name)
                        self.logger.info(f"Swapped {name} to version {ver} (zero-downtime).")

            except Exception as e:
                self.logger.error(f"Watcher loop error: {e}")

    # ------------------------ predict APIs ----------------------

    def ready(self) -> bool:
        return self.is_ready

    def list_models(self) -> List[str]:
        return sorted(self.models_active.keys()) if self.is_ready else []

    def _threshold_of(self, model_name: str) -> Optional[float]:
        return self.thresholds.get(model_name)

    def _predict_one(self, model_name: str, value: float) -> Dict[str, Any]:
        model = self.models_active.get(model_name)
        ma_tieu_chi, fld_code = _parse_model_name(model_name)

        if model is None:
            return {
                "model_name": model_name,
                "ma_tieu_chi": ma_tieu_chi,
                "fld_code": fld_code,
                "is_anomaly": False,
                "anomaly_score": None,
                "anomaly_threshold": self._threshold_of(model_name),
                "processing_time": 0.0,
                "model_version": self.model_versions.get(model_name),
                "error_message": f"Unknown model: {model_name}",
                "status": "error",
            }

        t0 = time.time()
        try:
            score = float(model.predict(float(value)))
            dt = time.time() - t0
        except Exception as e:
            dt = time.time() - t0
            return {
                "model_name": model_name,
                "ma_tieu_chi": ma_tieu_chi,
                "fld_code": fld_code,
                "is_anomaly": False,
                "anomaly_score": None,
                "anomaly_threshold": self._threshold_of(model_name),
                "processing_time": dt,
                "model_version": self.model_versions.get(model_name),
                "error_message": str(e),
                "status": "error",
            }

        threshold = self._threshold_of(model_name)
        is_anomaly = False
        if threshold is not None:
            is_anomaly = bool(score < threshold if threshold <= 0 else score > threshold)

        return {
            "model_name": model_name,
            "ma_tieu_chi": ma_tieu_chi,
            "fld_code": fld_code,
            "is_anomaly": is_anomaly,
            "anomaly_score": score,
            "anomaly_threshold": threshold,
            "processing_time": dt,
            "model_version": self.model_versions.get(model_name),
            "error_message": None,
            "status": "success",
        }

    async def predict_multi(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.is_ready:
            raise RuntimeError("Server not ready.")
        loop = asyncio.get_running_loop()
        max_conc = getattr(self._executor, "_max_workers", 8)
        sem = asyncio.Semaphore(max(1, int(max_conc)))

        async def _wrap(idx: int, m: str, v: float):
            async with sem:
                res = await loop.run_in_executor(self._executor, self._predict_one, m, v)
                return idx, res

        coros = [_wrap(i, str(t["model_name"]), float(t["value"])) for i, t in enumerate(tasks)]
        outs = await asyncio.gather(*coros)
        outs.sort(key=lambda x: x[0])
        return [o[1] for o in outs]

    async def validate_and_predict(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.is_ready:
            raise RuntimeError("Server not ready.")

        valid: List[Tuple[int, str, float]] = []
        failed: List[Dict[str, Any]] = []

        for i, t in enumerate(tasks):
            try:
                m = str(t["model_name"])
                v = float(t["value"])
            except Exception:
                failed.append(
                    {
                        "element_index": i,
                        "model_name": t.get("model_name"),
                        "ma_tieu_chi": "",
                        "fld_code": "",
                        "error": "invalid_task",
                        "error_details": "Thiếu hoặc sai định dạng model_name / value",
                    }
                )
                continue

            if m not in self.models_active:
                mtc, fld = _parse_model_name(m)
                failed.append(
                    {
                        "element_index": i,
                        "model_name": m,
                        "ma_tieu_chi": mtc,
                        "fld_code": fld,
                        "error": "model_not_found",
                        "error_details": "Model không tồn tại trong preload.",
                    }
                )
                continue
            valid.append((i, m, v))

        if not valid:
            return {"results": [], "failed_elements": failed}

        loop = asyncio.get_running_loop()
        max_conc = getattr(self._executor, "_max_workers", 8)
        sem = asyncio.Semaphore(max(1, int(max_conc)))

        async def _one(idx: int, m: str, v: float):
            async with sem:
                res = await loop.run_in_executor(self._executor, self._predict_one, m, v)
                return idx, res

        outs = await asyncio.gather(*[_one(i, m, v) for (i, m, v) in valid])
        outs.sort(key=lambda x: x[0])
        results = [o[1] for o in outs]
        return {"results": results, "failed_elements": failed}

    # Convenience admin APIs
    def get_model_details(self, model_name: str) -> Dict[str, Any]:
        loaded = model_name in self.models_active
        return {
            "model_name": model_name,
            "loaded": loaded,
            "model_version": self.model_versions.get(model_name),
            "anomaly_threshold": self.thresholds.get(model_name),
        }