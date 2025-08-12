import json
import os
import random
import string
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from pyod.models.deepsvdd import DeepSVDD
from sklearn.preprocessing import StandardScaler

"""
Script: anomaly_clone_float_models.py

- Trains DeepSVDD on 1D feature 'gia_tri'.
- Inference wrapper accepts ONLY a single float.
- Auto-computes anomaly_threshold via training-score quantile by
  contamination and logs all stats to MLflow params.
- Clones: trains on original combination but registers under new names,
  runs in parallel, and outputs a JSON of registered model names.

Prereqs:
- MLflow server reachable at MLFLOW_TRACKING_URI or http://localhost:5000
- CSV data: bao_cao_dulieu_not_none.csv with columns:
    ma_don_vi, ma_bao_cao, ma_tieu_chi, fld_code,
    ky_du_lieu, gia_tri
"""

# MLflow setup
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("Anomaly_Detection_Models")


@dataclass
class TrainConfig:
    contamination: float = 0.1
    epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-3
    random_state: int = 42


def _as_2d(values: pd.Series) -> np.ndarray:
    return values.astype(float).to_numpy().reshape(-1, 1)


def _auto_threshold(scores: np.ndarray, contamination: float) -> float:
    # Lower decision_function score => more anomalous
    return float(np.quantile(scores, contamination))


class AnomalyDetectionWrapperFloat:
    """
    Inference wrapper stored in MLflow that enforces float input.

    Methods:
      - predict(x: float) -> float decision score
      - predict_label(x: float, threshold: Optional[float]) -> int
        Returns -1 for anomaly, 1 for normal.
    """

    def __init__(
        self,
        model: DeepSVDD,
        scaler: StandardScaler,
        anomaly_threshold: float,
    ):
        self.model = model
        self.scaler = scaler
        self.anomaly_threshold = float(anomaly_threshold)

    def predict(self, x: float) -> float:
        if not isinstance(x, (float, int)):
            raise TypeError("Input must be a single float.")
        arr = np.array([[float(x)]], dtype=float)
        arr_scaled = self.scaler.transform(arr)
        score = self.model.decision_function(arr_scaled)[0]
        return float(score)

    def predict_label(self, x: float, threshold: Optional[float] = None) -> int:
        thr = self.anomaly_threshold if threshold is None else float(threshold)
        score = self.predict(x)
        return -1 if score < thr else 1


class SimpleAnomalyDetectionModel:
    """
    Train DeepSVDD on univariate 'gia_tri' only.
    Supports cloning by registering new names for the same trained spec.
    """

    def __init__(self, cfg: Optional[TrainConfig] = None):
        self.cfg = cfg or TrainConfig()

    def _prepare(self, data: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
        x = _as_2d(data["gia_tri"])
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        return x_scaled, scaler

    def train_model(
        self, data: pd.DataFrame
    ) -> Tuple[DeepSVDD, StandardScaler, float, Dict[str, float]]:
        x_scaled, scaler = self._prepare(data)
        model = DeepSVDD(
            contamination=self.cfg.contamination,
            epochs=self.cfg.epochs,
            batch_size=self.cfg.batch_size,
            random_state=self.cfg.random_state,
            n_features=features.shape[1],
        )
        model.fit(x_scaled)

        # Compute training scores and choose threshold by contamination quantile
        scores = model.decision_function(x_scaled)
        thr = _auto_threshold(scores, self.cfg.contamination)
        stats = {
            "train_score_min": float(np.min(scores)),
            "train_score_p10": float(np.quantile(scores, 0.10)),
            "train_score_p50": float(np.quantile(scores, 0.50)),
            "train_score_p90": float(np.quantile(scores, 0.90)),
            "train_score_max": float(np.max(scores)),
            "actual_anomaly_rate": float(np.mean(scores < thr)),
        }
        return model, scaler, thr, stats

    def save_to_mlflow(
        self,
        model: DeepSVDD,
        scaler: StandardScaler,
        anomaly_threshold: float,
        model_name: str,
        num_samples: int,
        date_range: str,
        stats: Dict[str, float],
    ) -> Dict[str, str]:
        wrapped = AnomalyDetectionWrapperFloat(
            model=model, scaler=scaler, anomaly_threshold=anomaly_threshold
        )
        with mlflow.start_run(run_name=f"register_{model_name}") as run:
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("training_samples", int(num_samples))
            mlflow.log_param("input_type", "float")
            mlflow.log_param("algorithm", "deepsvdd")
            mlflow.log_param("contamination", self.cfg.contamination)
            mlflow.log_param("epochs", self.cfg.epochs)
            mlflow.log_param("batch_size", self.cfg.batch_size)
            mlflow.log_param("lr", self.cfg.lr)
            mlflow.log_param("random_state", self.cfg.random_state)
            mlflow.log_param("threshold_method", "quantile_by_contamination")
            mlflow.log_param("anomaly_threshold", anomaly_threshold)
            mlflow.log_param("data_date_range", date_range)
            for k, v in stats.items():
                mlflow.log_param(k, v)
            mlflow.set_tag("model_type", "anomaly_detection")
            mlflow.set_tag("input_type", "float")
            mlflow.set_tag("version", "2.0")

            mlflow.sklearn.log_model(
                sk_model=wrapped,
                artifact_path="model",
                registered_model_name=model_name,
            )
            run_id = run.info.run_id

        time.sleep(0.5)
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(model_name, stages=["None"])
        if versions:
            version = versions[0].version
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production",
            )
        return {
            "model_name": model_name,
            "mlflow_run_id": run_id,
            "production_version": str(versions[0].version) if versions else "n/a",
        }


def _suffix(k: int = 6) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(random.choices(alphabet, k=k))


def train_clone_task(
    sub_data: pd.DataFrame,
    row: pd.Series,
    cfg: TrainConfig,
    clone_index: int,
) -> Dict[str, str]:
    detector = SimpleAnomalyDetectionModel(cfg)
    model, scaler, thr, stats = detector.train_model(sub_data)
    date_range = f"{sub_data['ky_du_lieu'].min()} to {sub_data['ky_du_lieu'].max()}"
    new_chi_tieu = f"{row['ma_tieu_chi']}_R{_suffix(6)}"
    model_name = (
        f"{row['ma_don_vi']}_{row['ma_bao_cao']}" f"_{new_chi_tieu}_{row['fld_code']}"
    )
    return detector.save_to_mlflow(
        model=model,
        scaler=scaler,
        anomaly_threshold=thr,
        model_name=model_name,
        num_samples=len(sub_data),
        date_range=date_range,
        stats=stats,
    )


def main() -> None:
    data_path = os.getenv("DATA_PATH", "bao_cao_dulieu_not_none.csv")
    clones_per_combo = int(os.getenv("CLONES_PER_COMBO", "55"))
    max_workers = int(os.getenv("MAX_WORKERS", "16"))

    if not os.path.exists(data_path):
        print(f"Data not found: {data_path}")
        return

    data = pd.read_csv(data_path)
    data["ky_du_lieu"] = pd.to_datetime(data["ky_du_lieu"])

    combos = data.drop_duplicates(
        subset=["ma_don_vi", "ma_bao_cao", "ma_tieu_chi", "fld_code"]
    )

    tasks = []
    for _, row in combos.iterrows():
        sub = data[
            (data["ma_don_vi"] == row["ma_don_vi"])
            & (data["ma_bao_cao"] == row["ma_bao_cao"])
            & (data["ma_tieu_chi"] == row["ma_tieu_chi"])
            & (data["fld_code"] == row["fld_code"])
        ]
        if len(sub) < 5:
            continue
        for idx in range(clones_per_combo):
            tasks.append((sub.copy(), row.copy(), TrainConfig(), idx + 1))

    print(f"Total clone tasks: {len(tasks)}")
    results = []
    failures = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(train_clone_task, *task) for task in tasks]
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as exc:
                failures += 1
                print(f"Task failed: {exc}")

    list_model = os.listdir("./mlruns/models")
    df = pd.DataFrame(list_model, columns=["model_name"])
    df.to_csv("list_model.csv", index=False)
    print(f"Saved {len(results)} models. Failures: {failures}.")


if __name__ == "__main__":
    main()
