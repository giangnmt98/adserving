import time
import warnings
import random
import string
from dataclasses import dataclass
from typing import Optional

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Configure MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Anomaly_Detection_Models")


@dataclass
class TrainConfig:
    contamination: float = 0.1
    n_estimators: int = 100
    random_state: int = 42


class SimpleAnomalyDetectionModel:
    """
    Train IsolationForest on univariate values (gia_tri).
    Prediction strictly accepts a single float and returns an anomaly score.
    """

    def __init__(self, cfg: Optional[TrainConfig] = None):
        self.cfg = cfg or TrainConfig()
        self.model: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None

    @staticmethod
    def _values_to_2d(values: pd.Series) -> np.ndarray:
        """Convert a 1D series of numbers to a 2D array [[x1], [x2], ...]."""
        return values.astype(float).to_numpy().reshape(-1, 1)

    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for training: use only 'gia_tri' as 1D input.
        """
        if "gia_tri" not in data.columns:
            raise ValueError("Data must contain column 'gia_tri'.")
        values_2d = self._values_to_2d(data["gia_tri"])
        if self.scaler is None:
            self.scaler = StandardScaler()
            return self.scaler.fit_transform(values_2d)
        return self.scaler.transform(values_2d)

    def train_and_save_model(
            self,
            data: pd.DataFrame,
            ma_don_vi: str,
            ma_bao_cao: str,
            ma_tieu_chi: str,
            fld_code: str,
            is_clone: bool = False,
            original_ma_tieu_chi: Optional[str] = None,
    ) -> bool:
        """
        Train model and register to MLflow. The model expects a single float
        for prediction input.
        """
        model_name = f"{ma_don_vi}_{ma_bao_cao}_{ma_tieu_chi}_{fld_code}"
        filter_ma_tieu_chi = original_ma_tieu_chi if is_clone else ma_tieu_chi

        filtered = data[
            (data["ma_don_vi"] == ma_don_vi)
            & (data["ma_bao_cao"] == ma_bao_cao)
            & (data["ma_tieu_chi"] == filter_ma_tieu_chi)
            & (data["fld_code"] == fld_code)
            ].copy()

        if len(filtered) < 5:
            print(
                "Không đủ dữ liệu để train model "
                f"{model_name} (cần ít nhất 5 điểm dữ liệu)"
            )
            return False

        print(f"Training model {model_name} với {len(filtered)} mẫu...")

        features = self._prepare_features(filtered)

        iso_forest = IsolationForest(
            contamination=self.cfg.contamination,
            random_state=self.cfg.random_state,
            n_estimators=self.cfg.n_estimators,
        )
        iso_forest.fit(features)

        # Compute optimal anomaly threshold from training scores.
        # Lower decision scores are more anomalous.
        train_scores = iso_forest.decision_function(features)
        # Choose threshold so that approximately 'contamination' fraction are anomalies.
        anomaly_threshold = float(np.quantile(train_scores, self.cfg.contamination))

        class FloatInputAnomalyWrapper:
            """
            Wrapper exposes predict(x: float) -> float anomaly score.
            Negative score indicates more anomalous.
            """

            def __init__(self, model: IsolationForest, scaler: StandardScaler, threshold: float):
                self.model = model
                self.scaler = scaler
                self.anomaly_threshold = float(threshold)

            def predict(self, x: float) -> float:
                if not isinstance(x, (float, int)):
                    raise TypeError("Input must be a single float.")
                arr = np.array([[float(x)]])
                arr_scaled = self.scaler.transform(arr)
                score = self.model.decision_function(arr_scaled)[0]
                return float(score)

            def predict_label(self, x: float, threshold: float | None = None) -> int:
                """
                Return 1 if normal, -1 if anomaly, based on decision score.
                If 'threshold' is provided, it overrides the stored threshold.
                """
                score = self.predict(x)
                thr = self.anomaly_threshold if threshold is None else float(threshold)
                return -1 if score < thr else 1

        wrapped_model = FloatInputAnomalyWrapper(iso_forest, self.scaler, anomaly_threshold)

        run_name = f"clone_{model_name}" if is_clone else f"train_{model_name}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("ma_don_vi", ma_don_vi)
            mlflow.log_param("ma_bao_cao", ma_bao_cao)
            mlflow.log_param("ma_tieu_chi", ma_tieu_chi)
            mlflow.log_param("fld_code", fld_code)
            mlflow.log_param("input_type", "float")
            mlflow.log_param("algorithm", "isolation_forest")
            mlflow.log_param("anomaly_threshold", anomaly_threshold)
            mlflow.log_param("threshold_method", "quantile_by_contamination")
            mlflow.log_param("contamination", self.cfg.contamination)
            mlflow.log_param("n_estimators", self.cfg.n_estimators)
            mlflow.log_param("random_state", self.cfg.random_state)
            mlflow.log_param("train_score_p50", float(np.median(train_scores)))
            mlflow.log_param("train_score_p10", float(np.quantile(train_scores, 0.10)))
            mlflow.log_param("train_score_p90", float(np.quantile(train_scores, 0.90)))

            if is_clone:
                mlflow.log_param("is_clone", True)
                mlflow.log_param("original_ma_tieu_chi", original_ma_tieu_chi)
            else:
                mlflow.log_param("is_clone", False)

            date_range = (
                f"{filtered['ky_du_lieu'].min()} to {filtered['ky_du_lieu'].max()}"
            )
            mlflow.log_param("data_date_range", date_range)

            model_metadata = {
                "algorithm": "isolation_forest",
                "input_type": "float",
                "training_samples": int(len(filtered)),
                "feature_count": 1,
                "is_clone": is_clone,
            }
            if is_clone:
                model_metadata["original_ma_tieu_chi"] = original_ma_tieu_chi

            mlflow.set_tag("model_type", "anomaly_detection")
            mlflow.set_tag("algorithm", "isolation_forest")
            mlflow.set_tag("version", "2.0")
            mlflow.set_tag("created_by", "SimpleAnomalyDetectionModel")
            if is_clone:
                mlflow.set_tag("is_clone", "true")
                mlflow.set_tag("original_ma_tieu_chi", original_ma_tieu_chi)

            mlflow.sklearn.log_model(
                sk_model=wrapped_model,
                name="model",
                registered_model_name=model_name,
                metadata=model_metadata,
            )
            run_id = mlflow.active_run().info.run_id
            print(f"MLflow run ID: {run_id}")

        time.sleep(2)

        try:
            client = mlflow.tracking.MlflowClient()
            latest_versions = client.get_latest_versions(model_name, stages=["None"])

            if latest_versions:
                new_version = latest_versions[0].version

                all_versions = client.search_model_versions(f"name='{model_name}'")
                for v in all_versions:
                    if v.version != new_version and v.current_stage != "Archived":
                        client.transition_model_version_stage(
                            name=model_name, version=v.version, stage="Archived"
                        )
                        print(f"Archived old version v{v.version}")

                client.transition_model_version_stage(
                    name=model_name, version=new_version, stage="Production"
                )
                print(
                    f"Model {model_name} v{new_version} đã được chuyển sang Production"
                )

                if is_clone:
                    print(f"Clone từ ma_tieu_chi gốc: {original_ma_tieu_chi}")
            else:
                print(f"Không tìm thấy version cho model {model_name}")

        except Exception as exc:
            print(f"Lỗi khi chuyển model sang Production: {exc}")

        print(f"Đã train và lưu model {model_name}")
        return True


def generate_random_suffix(length: int = 3) -> str:
    chars = string.ascii_uppercase + string.digits
    return "".join(random.choice(chars) for _ in range(length))


def clone_single_combination(data: pd.DataFrame, combination_row: pd.Series) -> None:
    print(
        "Nhân bản: "
        f"{combination_row['ma_don_vi']}_{combination_row['ma_bao_cao']}"
        f"_{combination_row['ma_tieu_chi']}_{combination_row['fld_code']}"
    )
    try:
        num_clones = int(input("Nhập số lượng bản sao muốn tạo: "))
        if num_clones <= 0:
            print("Số lượng phải > 0")
            return
    except ValueError:
        print("Vui lòng nhập số")
        return

    create_clones_for_combination(data, combination_row, num_clones)


def create_clones_for_combination(
        data: pd.DataFrame,
        combination_row: pd.Series,
        num_clones: int,
        show_individual_progress: bool = True,
) -> int:
    detector = SimpleAnomalyDetectionModel()
    successful_count = 0

    original_ma_tieu_chi = combination_row["ma_tieu_chi"]

    for i in range(num_clones):
        try:
            suffix = generate_random_suffix(3)
            new_ma_tieu_chi = f"{original_ma_tieu_chi}-{suffix}"

            if show_individual_progress:
                print(f"Tạo clone {i + 1}/{num_clones}: {new_ma_tieu_chi}")

            ok = detector.train_and_save_model(
                data=data,
                ma_don_vi=combination_row["ma_don_vi"],
                ma_bao_cao=combination_row["ma_bao_cao"],
                ma_tieu_chi=new_ma_tieu_chi,
                fld_code=combination_row["fld_code"],
                is_clone=True,
                original_ma_tieu_chi=original_ma_tieu_chi,
            )
            if ok:
                successful_count += 1
                if show_individual_progress:
                    print(f"Clone {i + 1} thành công")
            else:
                if show_individual_progress:
                    print(f"Clone {i + 1} thất bại")
        except Exception as exc:
            if show_individual_progress:
                print(f"Clone {i + 1} lỗi: {exc}")

    return successful_count


def clone_multiple_combinations(
        data: pd.DataFrame, combinations_df: pd.DataFrame
) -> None:
    try:
        num_per_combo = int(
            input("Nhập số lượng bản sao cho MỖI combination: ")
        )
        if num_per_combo <= 0:
            print("Số lượng phải > 0")
            return
    except ValueError:
        print("Vui lòng nhập số")
        return

    print(
        f"Bắt đầu nhân bản {len(combinations_df)} combinations, "
        f"mỗi cái {num_per_combo} bản sao..."
    )

    total_clones = 0
    successful = 0
    failed = 0
    start = time.time()

    for i, (_, row) in enumerate(combinations_df.iterrows(), 1):
        print(
            f"Nhân bản {i}/{len(combinations_df)}: "
            f"{row['ma_don_vi']}_{row['ma_bao_cao']}"
            f"_{row['ma_tieu_chi']}_{row['fld_code']}"
        )
        try:
            ok_count = create_clones_for_combination(
                data, row, num_per_combo, show_individual_progress=False
            )
            successful += ok_count
            failed += (num_per_combo - ok_count)
            total_clones += num_per_combo
            print(f"Hoàn thành: {ok_count}/{num_per_combo} clones thành công")
        except Exception as exc:
            print(f"Lỗi khi nhân bản: {exc}")
            failed += num_per_combo
            total_clones += num_per_combo

        if i % 5 == 0:
            elapsed = time.time() - start
            avg = elapsed / i
            remaining = (len(combinations_df) - i) * avg
            print(
                f"Tiến độ: {i}/{len(combinations_df)} "
                f"({i / len(combinations_df) * 100:.1f}%)"
            )
            print(f"Thời gian còn lại ước tính: {remaining / 60:.1f} phút")

    total_time = time.time() - start
    print("KẾT QUẢ NHÂN BẢN:")
    print(f"Thành công: {successful} clones")
    print(f"Thất bại: {failed} clones")
    if total_clones > 0:
        rate = successful / total_clones * 100
        print(f"Tỷ lệ thành công: {rate:.1f}%")
        print(f"Tổng thời gian: {total_time / 60:.1f} phút")
        print(f"Trung bình: {total_time / total_clones:.1f} giây/clone")


def train_and_save_all_models() -> None:
    print("TRAIN VÀ SAVE TẤT CẢ MODELS VÀO MLFLOW")
    try:
        data = pd.read_csv("bao_cao_dulieu_not_none.csv")
        data["ky_du_lieu"] = pd.to_datetime(data["ky_du_lieu"])
        print(f"Đã đọc {len(data)} dòng dữ liệu")
        print(
            f"Khoảng thời gian: {data['ky_du_lieu'].min()} "
            f"đến {data['ky_du_lieu'].max()}"
        )
    except FileNotFoundError:
        print("Không tìm thấy file 'bao_cao_dulieu_not_none.csv'")
        return
    except Exception as exc:
        print(f"Lỗi đọc dữ liệu: {exc}")
        return

    detector = SimpleAnomalyDetectionModel()
    combinations = (
        data.groupby(["ma_don_vi", "ma_bao_cao", "ma_tieu_chi", "fld_code"])
        .size()
        .reset_index(name="count")
    )
    print(f"Tìm thấy {len(combinations)} combinations để train")

    successful_models = 0
    failed_models = 0
    start = time.time()

    for i, row in combinations.iterrows():
        model_id = (
            f"{row['ma_don_vi']}_{row['ma_bao_cao']}"
            f"_{row['ma_tieu_chi']}_{row['fld_code']}"
        )
        print(f"Training {i + 1}/{len(combinations)}: {model_id}")
        print(f"Số mẫu dữ liệu: {row['count']}")

        try:
            ok = detector.train_and_save_model(
                data,
                row["ma_don_vi"],
                row["ma_bao_cao"],
                row["ma_tieu_chi"],
                row["fld_code"],
            )
            if ok:
                successful_models += 1
                print("Thành công")
            else:
                failed_models += 1
                print("Thất bại - không đủ dữ liệu")
        except Exception as exc:
            failed_models += 1
            print(f"Thất bại - Lỗi: {exc}")

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start
            avg = elapsed / (i + 1)
            remaining = (len(combinations) - i - 1) * avg
            pct = (i + 1) / len(combinations) * 100
            print(f"Tiến độ: {i + 1}/{len(combinations)} ({pct:.1f}%)")
            print(f"Thời gian còn lại ước tính: {remaining / 60:.1f} phút")

    total = successful_models + failed_models
    total_time = time.time() - start
    print("KẾT QUẢ TRAINING:")
    print(f"Thành công: {successful_models} models")
    print(f"Thất bại: {failed_models} models")
    if total > 0:
        rate = successful_models / total * 100
        print(f"Tỷ lệ thành công: {rate:.1f}%")
        print(f"Tổng thời gian: {total_time / 60:.1f} phút")
        print(f"Trung bình: {total_time / total:.1f} giây/model")
    print("Models đã được lưu vào MLflow")


def test_model_loading_and_prediction() -> None:
    """Load a few Production models and test float-only prediction."""
    print("TEST LOAD MODEL VÀ DỰ ĐOÁN VỚI FLOAT")
    client = mlflow.tracking.MlflowClient()

    try:
        registered = client.search_registered_models()
        if not registered:
            print("Không có model nào trong MLflow Registry")
            return

        print(f"Tìm thấy {len(registered)} models, kiểm tra tối đa 5 mẫu.")
        for i, model in enumerate(registered[:5], 1):
            name = model.name
            try:
                prod = client.get_latest_versions(name, stages=["Production"])
                if not prod:
                    print(f"{i}. {name} (không có Production version)")
                    continue

                version = prod[0].version
                print(f"{i}. Testing: {name} (Production v{version})")

                model_uri = f"models:/{name}/Production"
                loaded = mlflow.sklearn.load_model(model_uri)

                test_values = [10.0, 50.0, 100.0, 500.0]
                scores = []
                labels = []
                for v in test_values:
                    s = loaded.predict(v)  # float -> score
                    scores.append(s)
                    # If wrapper has predict_label, use it; otherwise infer by s<0
                    if hasattr(loaded, "predict_label"):
                        labels.append(loaded.predict_label(v))
                    else:
                        labels.append(-1 if s < 0 else 1)

                print("Scores:", [round(float(x), 4) for x in scores])
                print("Labels:", labels)

            except Exception as exc:
                print(f"Lỗi khi test model {name}: {exc}")

    except Exception as exc:
        print(f"Lỗi khi truy cập MLflow Registry: {exc}")


def show_registry_stats() -> None:
    print("THỐNG KÊ MLFLOW REGISTRY")
    try:
        client = mlflow.tracking.MlflowClient()
        registered = client.search_registered_models()

        if not registered:
            print("Không có model nào trong MLflow Registry")
            return

        print(f"Tổng số models: {len(registered)}")
        stage_counts = {"Production": 0, "Staging": 0, "Archived": 0, "None": 0}

        for model in registered:
            try:
                versions = client.get_latest_versions(model.name)
                for v in versions:
                    stage = v.current_stage
                    stage_counts[stage] = stage_counts.get(stage, 0) + 1
            except Exception as exc:
                print(f"Lỗi khi xử lý model {model.name}: {exc}")

        print("Phân phối theo Stage:")
        for stage, count in stage_counts.items():
            if count > 0:
                print(f"  {stage}: {count}")

    except Exception as exc:
        print(f"Lỗi khi truy cập MLflow Registry: {exc}")


def main() -> None:
    while True:
        print("ANOMALY DETECTION - FLOAT INPUT VERSION")
        print("1. TRAIN VÀ SAVE TẤT CẢ MODELS")
        print("2. TEST LOAD MODELS VÀ PREDICT FLOAT")
        print("3. THỐNG KÊ MLFLOW REGISTRY")
        print("4. NHÂN BẢN MODELS (CLONE)")
        print("0. THOÁT")

        choice = input("Chọn chức năng (0-4): ").strip()

        if choice == "1":
            train_and_save_all_models()
        elif choice == "2":
            test_model_loading_and_prediction()
        elif choice == "3":
            show_registry_stats()
        elif choice == "4":
            # Load data once here for cloning UI flow
            try:
                data = pd.read_csv("bao_cao_dulieu_not_none.csv")
                data["ky_du_lieu"] = pd.to_datetime(data["ky_du_lieu"])
            except FileNotFoundError:
                print("Không tìm thấy file 'bao_cao_dulieu_not_none.csv'")
                continue
            except Exception as exc:
                print(f"Lỗi đọc dữ liệu: {exc}")
                continue

            combinations = (
                data.groupby(
                    ["ma_don_vi", "ma_bao_cao", "ma_tieu_chi", "fld_code"]
                )
                .size()
                .reset_index(name="count")
            )
            valid = combinations[combinations["count"] >= 5].reset_index(drop=True)
            if len(valid) == 0:
                print("Không có combination nào đủ dữ liệu để nhân bản")
                continue

            for i, row in valid.head(10).iterrows():
                print(
                    f"{i + 1:2d}. {row['ma_don_vi']}"
                    f"_{row['ma_bao_cao']}"
                    f"_{row['ma_tieu_chi']}"
                    f"_{row['fld_code']} ({row['count']} samples)"
                )
            if len(valid) > 10:
                print(f"... và {len(valid) - 10} combinations khác")

            print("Chọn chế độ:")
            print("1. Nhân bản combination cụ thể")
            print("2. Nhân bản ngẫu nhiên")
            print("3. Nhân bản tất cả")
            print("0. Quay lại menu")
            sub = input("Chọn (0-3): ").strip()

            if sub == "0":
                continue
            elif sub == "1":
                try:
                    idx = int(
                        input(
                            f"Nhập số thứ tự (1-{len(valid)}): "
                        )
                    ) - 1
                    if 0 <= idx < len(valid):
                        sel = valid.iloc[idx]
                        clone_single_combination(data, sel)
                    else:
                        print("Số thứ tự không hợp lệ")
                except ValueError:
                    print("Vui lòng nhập số")
            elif sub == "2":
                try:
                    k = int(input("Số lượng combinations ngẫu nhiên: "))
                    if k > 0:
                        sel_df = valid.sample(min(k, len(valid)))
                        clone_multiple_combinations(data, sel_df)
                    else:
                        print("Số lượng phải > 0")
                except ValueError:
                    print("Vui lòng nhập số")
            elif sub == "3":
                confirm = input(
                    "Nhân bản TẤT CẢ combinations? (y/N): "
                ).strip().lower()
                if confirm == "y":
                    clone_multiple_combinations(data, valid)
                else:
                    print("Đã hủy")
            else:
                print("Lựa chọn không hợp lệ")
        elif choice == "0":
            print("Thoát.")
            break
        else:
            print("Lựa chọn không hợp lệ.")


if __name__ == "__main__":
    main()
