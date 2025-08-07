# train_clone_models.py (parallel version)
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
from datetime import datetime
import time
import json
import os
import random
import string
from concurrent.futures import ThreadPoolExecutor, as_completed

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Anomaly_Detection_Models")


class SimpleAnomalyDetectionModel:
    def __init__(self):
        self.model = None
        self.scaler = None

    def prepare_features(self, data):
        data_sorted = data.sort_values('ky_du_lieu')
        features = pd.DataFrame()
        features['gia_tri'] = data_sorted['gia_tri']
        features['rolling_mean_3'] = data_sorted['gia_tri'].rolling(3, min_periods=1).mean()
        features['rolling_std_3'] = data_sorted['gia_tri'].rolling(3, min_periods=1).std().fillna(0)
        features['rolling_mean_7'] = data_sorted['gia_tri'].rolling(7, min_periods=1).mean()
        features['rolling_std_7'] = data_sorted['gia_tri'].rolling(7, min_periods=1).std().fillna(0)
        mean_val = data_sorted['gia_tri'].mean()
        std_val = data_sorted['gia_tri'].std()
        features['deviation_from_mean'] = np.abs(data_sorted['gia_tri'] - mean_val)
        features['pct_change'] = data_sorted['gia_tri'].pct_change().fillna(0)
        features['z_score'] = np.abs((data_sorted['gia_tri'] - mean_val) / std_val)
        return features.fillna(0)

    def train_model(self, data):
        features = self.prepare_features(data)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        model = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
        model.fit(features_scaled)
        return model, scaler

    def save_to_mlflow(self, model, scaler, model_name, num_samples):
        class AnomalyDetectionWrapper:
            def __init__(self, model, scaler):
                self.model = model
                self.scaler = scaler

            def predict(self, X):
                if isinstance(X, pd.DataFrame) and 'gia_tri' in X.columns:
                    features = self._prepare_features(X)
                    features_scaled = self.scaler.transform(features)
                    preds = self.model.predict(features_scaled)
                    return preds == -1
                else:
                    raise ValueError("Input phải là DataFrame với cột 'gia_tri'")

            def _prepare_features(self, data):
                features = pd.DataFrame()
                features['gia_tri'] = data['gia_tri']
                features['rolling_mean_3'] = data['gia_tri'].rolling(3, min_periods=1).mean()
                features['rolling_std_3'] = data['gia_tri'].rolling(3, min_periods=1).std().fillna(0)
                features['rolling_mean_7'] = data['gia_tri'].rolling(7, min_periods=1).mean()
                features['rolling_std_7'] = data['gia_tri'].rolling(7, min_periods=1).std().fillna(0)
                mean_val = data['gia_tri'].mean()
                std_val = data['gia_tri'].std()
                features['deviation_from_mean'] = np.abs(data['gia_tri'] - mean_val)
                features['pct_change'] = data['gia_tri'].pct_change().fillna(0)
                features['z_score'] = np.abs((data['gia_tri'] - mean_val) / std_val)
                return features.fillna(0)

        wrapped_model = AnomalyDetectionWrapper(model, scaler)

        with mlflow.start_run(run_name=f"train_{model_name}"):
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("training_samples", num_samples)
            mlflow.log_param("anomaly_threshold", 0.0000001)
            mlflow.sklearn.log_model(
                sk_model=wrapped_model,
                artifact_path="model",
                registered_model_name=model_name
            )
            run_id = mlflow.active_run().info.run_id

        time.sleep(1)
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(model_name, stages=["None"])
        if versions:
            version = versions[0].version
            client.transition_model_version_stage(model_name, version, stage="Production")
            print(f"{model_name} v{version} chuyển Production")
        return {
            "model_name": model_name,
            "ma_don_vi": model_name.split("_")[0],
            "ma_bao_cao": model_name.split("_")[1],
            "ma_tieu_chi": model_name.split("_")[2],
            "fld_code": model_name.split("_")[-1]
        }


def train_and_register_single_task(sub_data, row, i):
    detector = SimpleAnomalyDetectionModel()
    random_suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    new_chi_tieu = f"{row['ma_tieu_chi']}_R{random_suffix}"
    model_name = f"{row['ma_don_vi']}_{row['ma_bao_cao']}_{new_chi_tieu}_{row['fld_code']}"
    model, scaler = detector.train_model(sub_data)
    return detector.save_to_mlflow(model, scaler, model_name, len(sub_data))


if __name__ == "__main__":
    N = 60
    MAX_WORKERS = 20
    data_path = "bao_cao_dulieu_not_none.csv"
    data = pd.read_csv(data_path)
    data['ky_du_lieu'] = pd.to_datetime(data['ky_du_lieu'])

    combos = data.drop_duplicates(subset=['ma_don_vi', 'ma_bao_cao', 'ma_tieu_chi', 'fld_code'])
    tasks = []

    for _, row in combos.iterrows():
        sub_data = data[(data['ma_don_vi'] == row['ma_don_vi']) &
                        (data['ma_bao_cao'] == row['ma_bao_cao']) &
                        (data['ma_tieu_chi'] == row['ma_tieu_chi']) &
                        (data['fld_code'] == row['fld_code'])]
        if len(sub_data) < 5:
            continue
        for i in range(1, N + 1):
            tasks.append((sub_data.copy(), row, i))

    results = []
    print(f"Running {len(tasks)} model clones in parallel...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_cfg = {
            executor.submit(train_and_register_single_task, *task): task for task in tasks
        }
        for future in as_completed(future_to_cfg):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Task failed: {str(e)}")

    output_path = "generated_model_names.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Đã lưu {len(results)} model clone vào {output_path}")
