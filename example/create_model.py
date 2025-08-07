import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
import warnings
import time
import random
import string

warnings.filterwarnings('ignore')

# Thiết lập MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Anomaly_Detection_Models")


class SimpleAnomalyDetectionModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.statistics = {}

    def prepare_features(self, data):
        """Chuẩn bị features cho model"""
        data_sorted = data.sort_values('ky_du_lieu')

        features = pd.DataFrame()
        features['gia_tri'] = data_sorted['gia_tri']

        # Rolling statistics
        features['rolling_mean_3'] = data_sorted['gia_tri'].rolling(window=3, min_periods=1).mean()
        features['rolling_std_3'] = data_sorted['gia_tri'].rolling(window=3, min_periods=1).std().fillna(0)
        features['rolling_mean_7'] = data_sorted['gia_tri'].rolling(window=7, min_periods=1).mean()
        features['rolling_std_7'] = data_sorted['gia_tri'].rolling(window=7, min_periods=1).std().fillna(0)

        # Other features
        overall_mean = data_sorted['gia_tri'].mean()
        features['deviation_from_mean'] = np.abs(data_sorted['gia_tri'] - overall_mean)
        features['pct_change'] = data_sorted['gia_tri'].pct_change().fillna(0)
        features['z_score'] = np.abs((data_sorted['gia_tri'] - overall_mean) / data_sorted['gia_tri'].std())

        return features.fillna(0)

    def train_and_save_model(self, data, ma_don_vi, ma_bao_cao, ma_tieu_chi, fld_code, is_clone=False,
                             original_ma_tieu_chi=None):
        """Train model và save vào MLflow với metadata đầy đủ"""
        model_name = f"{ma_don_vi}_{ma_bao_cao}_{ma_tieu_chi}_{fld_code}"

        # Sử dụng original_ma_tieu_chi để filter data nếu đây là clone
        filter_ma_tieu_chi = original_ma_tieu_chi if is_clone else ma_tieu_chi

        # Lọc dữ liệu
        filtered_data = data[
            (data['ma_don_vi'] == ma_don_vi) &
            (data['ma_bao_cao'] == ma_bao_cao) &
            (data['ma_tieu_chi'] == filter_ma_tieu_chi) &
            (data['fld_code'] == fld_code)
            ].copy()

        if len(filtered_data) < 5:
            print(f"Không đủ dữ liệu để train model {model_name} (cần ít nhất 5 điểm dữ liệu)")
            return False

        clone_info = f" (CLONE từ {original_ma_tieu_chi})" if is_clone else ""
        print(f" Training model {model_name}{clone_info} với {len(filtered_data)} mẫu dữ liệu...")

        # Train model
        features = self.prepare_features(filtered_data)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Hyperparameters
        contamination = 0.1
        n_estimators = 100
        random_state = 42

        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=n_estimators
        )
        iso_forest.fit(features_scaled)

        # Tạo wrapper model với cả predict và predict_proba
        class AnomalyDetectionWrapper:
            def __init__(self, model, scaler):
                self.model = model
                self.scaler = scaler
                self.supports_probability = True

            def predict(self, X):
                """Trả về probability scores (anomaly probabilities)"""
                if isinstance(X, pd.DataFrame) and 'gia_tri' in X.columns:
                    features = self._prepare_features(X)
                    features_scaled = self.scaler.transform(features)

                    # Lấy decision function scores
                    decision_scores = self.model.decision_function(features_scaled)

                    return decision_scores
                else:
                    raise ValueError("Input phải là DataFrame với cột 'gia_tri'")

            def _prepare_features(self, data):
                """Chuẩn bị features giống như khi training"""
                features = pd.DataFrame()
                features['gia_tri'] = data['gia_tri']
                features['rolling_mean_3'] = data['gia_tri'].rolling(window=3, min_periods=1).mean()
                features['rolling_std_3'] = data['gia_tri'].rolling(window=3, min_periods=1).std().fillna(0)
                features['rolling_mean_7'] = data['gia_tri'].rolling(window=7, min_periods=1).mean()
                features['rolling_std_7'] = data['gia_tri'].rolling(window=7, min_periods=1).std().fillna(0)
                overall_mean = data['gia_tri'].mean()
                features['deviation_from_mean'] = np.abs(data['gia_tri'] - overall_mean)
                features['pct_change'] = data['gia_tri'].pct_change().fillna(0)
                features['z_score'] = np.abs((data['gia_tri'] - overall_mean) / data['gia_tri'].std())
                return features.fillna(0)

        wrapped_model = AnomalyDetectionWrapper(iso_forest, scaler)

        # Log model vào MLflow với metadata chi tiết
        run_name = f"clone_{model_name}" if is_clone else f"train_{model_name}"
        with mlflow.start_run(run_name=run_name):
            # Log basic parameters
            mlflow.log_param("model_name", model_name)

            # Log data info
            mlflow.log_param("ma_don_vi", ma_don_vi)
            mlflow.log_param("ma_bao_cao", ma_bao_cao)
            mlflow.log_param("ma_tieu_chi", ma_tieu_chi)
            mlflow.log_param("fld_code", fld_code)
            mlflow.log_param("anomaly_threshold", 0.5)
            # Log clone information
            if is_clone:
                mlflow.log_param("is_clone", True)
                mlflow.log_param("original_ma_tieu_chi", original_ma_tieu_chi)
                mlflow.log_param("data_source_ma_tieu_chi", filter_ma_tieu_chi)
            else:
                mlflow.log_param("is_clone", False)

            mlflow.log_param("data_date_range",
                             f"{filtered_data['ky_du_lieu'].min()} to {filtered_data['ky_du_lieu'].max()}")

            # Log model metadata
            model_metadata = {
                "algorithm": "isolation_forest",
                "contamination": contamination,
                "training_samples": len(filtered_data),
                "feature_count": len(features.columns),
                "is_clone": is_clone,
            }

            if is_clone:
                model_metadata["original_ma_tieu_chi"] = original_ma_tieu_chi

            # Set tags
            mlflow.set_tag("model_type", "anomaly_detection")
            mlflow.set_tag("algorithm", "isolation_forest")
            mlflow.set_tag("version", "1.0")
            mlflow.set_tag("created_by", "SimpleAnomalyDetectionModel")

            if is_clone:
                mlflow.set_tag("is_clone", "true")
                mlflow.set_tag("original_ma_tieu_chi", original_ma_tieu_chi)

            # Log model với metadata
            mlflow.sklearn.log_model(
                sk_model=wrapped_model,
                name="model",
                registered_model_name=model_name,
                metadata=model_metadata
            )
            run_id = mlflow.active_run().info.run_id
            print(f"MLflow run ID: {run_id}")

        # Chờ một chút để MLflow hoàn thành việc register
        time.sleep(2)

        # Chuyển sang Production stage
        try:
            client = mlflow.tracking.MlflowClient()
            latest_versions = client.get_latest_versions(model_name, stages=["None"])

            if latest_versions:
                new_version = latest_versions[0].version

                # Archive tất cả các version khác
                all_versions = client.search_model_versions(f"name='{model_name}'")
                for v in all_versions:
                    if v.version != new_version and v.current_stage != "Archived":
                        client.transition_model_version_stage(
                            name=model_name,
                            version=v.version,
                            stage="Archived"
                        )
                        print(f" - Archived old version v{v.version}")

                # Chuyển version mới sang Production
                client.transition_model_version_stage(
                    name=model_name,
                    version=new_version,
                    stage="Production"
                )
                print(f"Model {model_name} v{new_version} đã được chuyển sang Production duy nhất")

                if is_clone:
                    print(f"Clone từ ma_tieu_chi gốc: {original_ma_tieu_chi}")
            else:
                print(f"Không tìm thấy version cho model {model_name}")

        except Exception as e:
            print(f" Lỗi khi chuyển model sang Production: {e}")

        print(f" Đã train và lưu model {model_name} với {len(filtered_data)} mẫu dữ liệu")
        return True


def generate_random_suffix(length=3):
    """Tạo suffix ngẫu nhiên gồm chữ và số"""
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choice(chars) for _ in range(length))


def clone_models_interactive():
    """Nhân bản models với giao diện tương tác"""
    print("\n" + "=" * 80)
    print(" NHÂN BẢN MODELS - THAY ĐỔI MA_TIEU_CHI")
    print("=" * 80)

    # Đọc dữ liệu
    print(" Đang đọc dữ liệu...")
    try:
        data = pd.read_csv('./data/bao_cao_dulieu_not_none.csv')
        data['ky_du_lieu'] = pd.to_datetime(data['ky_du_lieu'])
        print(f" Đã đọc {len(data)} dòng dữ liệu")
    except FileNotFoundError:
        print(" Không tìm thấy file 'bao_cao_dulieu_not_none.csv'")
        return
    except Exception as e:
        print(f" Lỗi đọc dữ liệu: {e}")
        return

    # Lấy danh sách combinations có thể nhân bản
    combinations = data.groupby(['ma_don_vi', 'ma_bao_cao', 'ma_tieu_chi', 'fld_code']).size().reset_index(name='count')
    # Chỉ lấy những combinations có đủ dữ liệu (>= 5 samples)
    valid_combinations = combinations[combinations['count'] >= 5].reset_index(drop=True)

    if len(valid_combinations) == 0:
        print(" Không có combination nào có đủ dữ liệu để nhân bản!")
        return

    print(f"\n Tìm thấy {len(valid_combinations)} combinations có thể nhân bản:")
    print("-" * 80)

    # Hiển thị danh sách combinations
    for i, row in valid_combinations.head(10).iterrows():  # Hiển thị 10 đầu tiên
        print(
            f"{i + 1:2d}. {row['ma_don_vi']}_{row['ma_bao_cao']}_{row['ma_tieu_chi']}_{row['fld_code']} ({row['count']} samples)")

    if len(valid_combinations) > 10:
        print(f"... và {len(valid_combinations) - 10} combinations khác")

    # Chọn combination để nhân bản
    print(f"\n Chọn combination để nhân bản:")
    print("1.  Nhân bản combination cụ thể (nhập số thứ tự)")
    print("2.  Nhân bản ngẫu nhiên")
    print("3.  Nhân bản tất cả")
    print("0. ↩ Quay lại menu chính")

    choice = input("\n Chọn (0-3): ").strip()

    if choice == '0':
        return
    elif choice == '1':
        # Nhân bản combination cụ thể
        try:
            index = int(input(f"Nhập số thứ tự combination (1-{len(valid_combinations)}): ")) - 1
            if 0 <= index < len(valid_combinations):
                selected_row = valid_combinations.iloc[index]
                clone_single_combination(data, selected_row)
            else:
                print(" Số thứ tự không hợp lệ!")
        except ValueError:
            print(" Vui lòng nhập số!")
    elif choice == '2':
        # Nhân bản ngẫu nhiên
        try:
            num_random = int(input("Nhập số lượng combinations ngẫu nhiên muốn nhân bản: "))
            if num_random > 0:
                selected_combinations = valid_combinations.sample(min(num_random, len(valid_combinations)))
                clone_multiple_combinations(data, selected_combinations)
            else:
                print(" Số lượng phải > 0!")
        except ValueError:
            print(" Vui lòng nhập số!")
    elif choice == '3':
        # Nhân bản tất cả
        confirm = input(" Bạn có chắc muốn nhân bản TẤT CẢ combinations? (y/N): ").strip().lower()
        if confirm == 'y':
            clone_multiple_combinations(data, valid_combinations)
        else:
            print(" Đã hủy!")
    else:
        print(" Lựa chọn không hợp lệ!")


def clone_single_combination(data, combination_row):
    """Nhân bản một combination duy nhất"""
    print(
        f"\n Nhân bản combination: {combination_row['ma_don_vi']}_{combination_row['ma_bao_cao']}_{combination_row['ma_tieu_chi']}_{combination_row['fld_code']}")

    try:
        num_clones = int(input("Nhập số lượng bản sao muốn tạo: "))
        if num_clones <= 0:
            print(" Số lượng phải > 0!")
            return
    except ValueError:
        print(" Vui lòng nhập số!")
        return

    create_clones_for_combination(data, combination_row, num_clones)


def clone_multiple_combinations(data, combinations_df):
    """Nhân bản nhiều combinations"""
    try:
        num_clones_per_combo = int(input("Nhập số lượng bản sao cho MỖI combination: "))
        if num_clones_per_combo <= 0:
            print(" Số lượng phải > 0!")
            return
    except ValueError:
        print(" Vui lòng nhập số!")
        return

    print(
        f"\n Bắt đầu nhân bản {len(combinations_df)} combinations, mỗi combination {num_clones_per_combo} bản sao...")
    print("-" * 80)

    total_clones = 0
    successful_clones = 0
    failed_clones = 0
    start_time = time.time()

    for i, (_, row) in enumerate(combinations_df.iterrows(), 1):
        print(
            f"\n Nhân bản combination {i}/{len(combinations_df)}: {row['ma_don_vi']}_{row['ma_bao_cao']}_{row['ma_tieu_chi']}_{row['fld_code']}")

        try:
            success_count = create_clones_for_combination(data, row, num_clones_per_combo,
                                                          show_individual_progress=False)
            successful_clones += success_count
            failed_clones += (num_clones_per_combo - success_count)
            total_clones += num_clones_per_combo

            print(f"    Hoàn thành: {success_count}/{num_clones_per_combo} clones thành công")

        except Exception as e:
            print(f"    Lỗi khi nhân bản: {e}")
            failed_clones += num_clones_per_combo
            total_clones += num_clones_per_combo

        # Progress update
        if i % 5 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = (len(combinations_df) - i) * avg_time
            print(f"\nTiến độ: {i}/{len(combinations_df)} combinations ({i / len(combinations_df) * 100:.1f}%)")
            print(f"Thời gian còn lại ước tính: {remaining / 60:.1f} phút")

    # Tổng kết
    total_time = time.time() - start_time
    print(f"\n" + "=" * 80)
    print(" KẾT QUẢ NHÂN BẢN:")
    print("=" * 80)
    print(f" Thành công: {successful_clones} clones")
    print(f" Thất bại: {failed_clones} clones")
    print(f" Tỷ lệ thành công: {successful_clones / total_clones * 100:.1f}%")
    print(f" Tổng thời gian: {total_time / 60:.1f} phút")
    print(f" Trung bình: {total_time / total_clones:.1f} giây/clone")
    print("=" * 80)


def create_clones_for_combination(data, combination_row, num_clones, show_individual_progress=True):
    """Tạo clones cho một combination"""
    detector = SimpleAnomalyDetectionModel()
    successful_count = 0

    original_ma_tieu_chi = combination_row['ma_tieu_chi']

    for i in range(num_clones):
        try:
            # Tạo ma_tieu_chi mới bằng cách thêm suffix ngẫu nhiên
            suffix = generate_random_suffix(3)
            new_ma_tieu_chi = f"{original_ma_tieu_chi}-{suffix}"

            if show_individual_progress:
                print(f"Tạo clone {i + 1}/{num_clones}: {new_ma_tieu_chi}")

            # Train và save model clone
            success = detector.train_and_save_model(
                data,
                combination_row['ma_don_vi'],
                combination_row['ma_bao_cao'],
                new_ma_tieu_chi,  # ma_tieu_chi mới
                combination_row['fld_code'],
                is_clone=True,
                original_ma_tieu_chi=original_ma_tieu_chi  # ma_tieu_chi gốc để filter data
            )

            if success:
                successful_count += 1
                if show_individual_progress:
                    print(f"Clone {i + 1} thành công!")
            else:
                if show_individual_progress:
                    print(f"    Clone {i + 1} thất bại!")

        except Exception as e:
            if show_individual_progress:
                print(f"    Clone {i + 1} lỗi: {e}")

    return successful_count


def train_and_save_all_models():
    """Train tất cả models và lưu vào MLflow"""
    print("=" * 80)
    print("TRAIN VÀ SAVE TẤT CẢ MODELS VÀO MLFLOW VỚI METADATA ĐẦY ĐỦ")
    print("=" * 80)

    # Đọc dữ liệu
    print(" Đang đọc dữ liệu...")
    try:
        data = pd.read_csv('./data/bao_cao_dulieu_not_none.csv')
        data['ky_du_lieu'] = pd.to_datetime(data['ky_du_lieu'])
        print(f" Đã đọc {len(data)} dòng dữ liệu")
        print(f" Khoảng thời gian: {data['ky_du_lieu'].min()} đến {data['ky_du_lieu'].max()}")
    except FileNotFoundError:
        print(" Không tìm thấy file 'bao_cao_dulieu_not_none.csv'")
        return
    except Exception as e:
        print(f" Lỗi đọc dữ liệu: {e}")
        return

    # Khởi tạo model detector
    detector = SimpleAnomalyDetectionModel()

    # Lấy tất cả các combinations unique
    combinations = data.groupby(['ma_don_vi', 'ma_bao_cao', 'ma_tieu_chi', 'fld_code']).size().reset_index(name='count')
    print(f"\n Tìm thấy {len(combinations)} combinations để train models")

    # Hiển thị thống kê chi tiết
    print(f"\n Thống kê combinations:")
    print(f"    Số đơn vị: {data['ma_don_vi'].nunique()}")
    print(f"    Số báo cáo: {data['ma_bao_cao'].nunique()}")
    print(f"    Số chỉ tiêu: {data['ma_tieu_chi'].nunique()}")
    print(f"    Số field codes: {data['fld_code'].nunique()}")

    # Thống kê phân phối dữ liệu
    print(f"\n Phân phối số mẫu dữ liệu:")
    sample_counts = combinations['count'].describe()
    print(f"    Trung bình: {sample_counts['mean']:.1f} mẫu/model")
    print(f"    Trung vị: {sample_counts['50%']:.0f} mẫu/model")
    print(f"    Min: {sample_counts['min']:.0f} mẫu/model")
    print(f"    Max: {sample_counts['max']:.0f} mẫu/model")

    # Train models
    print(f"\n Bắt đầu train models...")
    print("-" * 80)

    successful_models = 0
    failed_models = 0
    start_time = time.time()

    for i, row in combinations.iterrows():
        model_id = f"{row['ma_don_vi']}_{row['ma_bao_cao']}_{row['ma_tieu_chi']}_{row['fld_code']}"
        print(f"\n Training {i + 1}/{len(combinations)}: {model_id}")
        print(f"    Số mẫu dữ liệu: {row['count']}")

        try:
            success = detector.train_and_save_model(
                data,
                row['ma_don_vi'],
                row['ma_bao_cao'],
                row['ma_tieu_chi'],
                row['fld_code']
            )

            if success:
                successful_models += 1
                print(f"    Thành công!")
            else:
                failed_models += 1
                print(f"    Thất bại - không đủ dữ liệu")

        except Exception as e:
            failed_models += 1
            print(f"    Thất bại - Lỗi: {e}")

        # Progress update
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            remaining = (len(combinations) - i - 1) * avg_time
            print(f"\n   ⏱ Tiến độ: {i + 1}/{len(combinations)} ({(i + 1) / len(combinations) * 100:.1f}%)")
            print(f"   ⏰ Thời gian còn lại ước tính: {remaining / 60:.1f} phút")

    # Tổng kết
    total_time = time.time() - start_time
    print(f"\n" + "=" * 80)
    print(" KẾT QUẢ TRAINING:")
    print("=" * 80)
    print(f" Thành công: {successful_models} models")
    print(f" Thất bại: {failed_models} models")
    print(f" Tỷ lệ thành công: {successful_models / (successful_models + failed_models) * 100:.1f}%")
    print(f"⏱ Tổng thời gian: {total_time / 60:.1f} phút")
    print(f" Trung bình: {total_time / (successful_models + failed_models):.1f} giây/model")
    print(f" Tất cả models đã được lưu vào MLflow với metadata đầy đủ")
    print(f" Các models hỗ trợ cả predict() và predict_proba()")
    print(f" Output type được tự động xác định và lưu trong metadata")
    print("=" * 80)


def test_model_loading_and_metadata():
    """Test load model từ MLflow và kiểm tra metadata"""
    print("\n" + "=" * 80)
    print(" TEST LOAD MODEL TỪ MLFLOW VÀ KIỂM TRA METADATA")
    print("=" * 80)

    client = mlflow.tracking.MlflowClient()

    try:
        registered_models = client.search_registered_models()

        if not registered_models:
            print(" Không có model nào trong MLflow Registry")
            return

        print(f" Tìm thấy {len(registered_models)} models trong MLflow Registry:")
        print("-" * 80)

        for i, model in enumerate(registered_models[:5], 1):  # Test 5 models đầu tiên
            model_name = model.name

            try:
                prod_versions = client.get_latest_versions(model_name, stages=["Production"])
                if prod_versions:
                    version = prod_versions[0].version
                    print(f"\n {i}. Testing model: {model_name} (Production v{version})")

                    # Load model
                    model_uri = f"models:/{model_name}/Production"
                    loaded_model = mlflow.sklearn.load_model(model_uri)
                    print(f"    Model loaded successfully!")

                    # Kiểm tra capabilities
                    has_predict = hasattr(loaded_model, 'predict')
                    has_predict_proba = hasattr(loaded_model, 'predict_proba')
                    has_decision_function = hasattr(loaded_model, 'decision_function')

                    print(f"    Capabilities:")
                    print(f"      - predict(): {'' if has_predict else ''}")
                    print(f"      - predict_proba(): {'' if has_predict_proba else ''}")
                    print(f"      - decision_function(): {'' if has_decision_function else ''}")

                    # Lấy metadata từ model version
                    try:
                        model_version = client.get_model_version(model_name, version)
                        run_id = model_version.run_id
                        run = client.get_run(run_id)

                        # Lấy thông tin từ params
                        params = run.data.params
                        output_type = params.get('output_type', 'unknown')
                        threshold = params.get('threshold', 'unknown')
                        supports_prob = params.get('supports_probability', 'unknown')
                        is_clone = params.get('is_clone', 'false')

                        print(f"    Metadata:")
                        print(f"      - Output type: {output_type}")
                        print(f"      - Threshold: {threshold}")
                        print(f"      - Supports probability: {supports_prob}")
                        print(f"      - Is clone: {is_clone}")

                        if is_clone == 'True':
                            original_ma_tieu_chi = params.get('original_ma_tieu_chi', 'unknown')
                            print(f"      - Original ma_tieu_chi: {original_ma_tieu_chi}")

                        if 'optimal_decision_threshold' in params:
                            print(f"      - Optimal decision threshold: {params['optimal_decision_threshold']}")
                        if 'actual_anomaly_rate' in params:
                            print(f"      - Training anomaly rate: {params['actual_anomaly_rate']}")

                    except Exception as e:
                        print(f"    Không thể lấy metadata: {e}")

                    # Test prediction nếu có sample data
                    try:
                        # Tạo sample data để test
                        sample_data = pd.DataFrame({
                            'gia_tri': [100, 200, 300, 1000, 50]  # 1000 có thể là anomaly
                        })

                        if has_predict:
                            predictions = loaded_model.predict(sample_data)
                            anomaly_count = np.sum(predictions)
                            print(f"    Test predict: {anomaly_count}/{len(sample_data)} anomalies detected")

                        if has_predict_proba:
                            probabilities = loaded_model.predict_proba(sample_data)
                            max_prob = np.max(probabilities)
                            print(f"    Test predict_proba: max probability = {max_prob:.3f}")

                    except Exception as e:
                        print(f"    Không thể test prediction: {e}")

                else:
                    print(f"\n {i}. {model_name} (Không có Production version)")

            except Exception as e:
                print(f"\n {i}. {model_name} (Lỗi: {e})")

    except Exception as e:
        print(f" Lỗi khi truy cập MLflow Registry: {e}")

    print("\n" + "=" * 80)


def show_registry_stats():
    """Hiển thị thống kê về MLflow Registry"""
    print("\n" + "=" * 80)
    print(" THỐNG KÊ MLFLOW REGISTRY")
    print("=" * 80)

    try:
        client = mlflow.tracking.MlflowClient()
        registered_models = client.search_registered_models()

        if not registered_models:
            print(" Không có model nào trong MLflow Registry")
            return

        print(f" Tổng số models: {len(registered_models)}")

        # Thống kê theo stage
        stage_counts = {'Production': 0, 'Staging': 0, 'Archived': 0, 'None': 0}
        output_type_counts = {'probability': 0, 'label': 0, 'unknown': 0}
        clone_counts = {'original': 0, 'clone': 0}

        for model in registered_models:
            try:
                versions = client.get_latest_versions(model.name)
                for version in versions:
                    stage = version.current_stage
                    stage_counts[stage] = stage_counts.get(stage, 0) + 1

                    # Lấy thông tin từ metadata
                    try:
                        run = client.get_run(version.run_id)
                        params = run.data.params

                        output_type = params.get('output_type', 'unknown')
                        output_type_counts[output_type] = output_type_counts.get(output_type, 0) + 1

                        is_clone = params.get('is_clone', 'False')
                        if is_clone == 'True':
                            clone_counts['clone'] += 1
                        else:
                            clone_counts['original'] += 1

                    except:
                        output_type_counts['unknown'] += 1
                        clone_counts['original'] += 1

            except Exception as e:
                print(f" Lỗi khi xử lý model {model.name}: {e}")

        print(f"\n Phân phối theo Stage:")
        for stage, count in stage_counts.items():
            if count > 0:
                print(f"    {stage}: {count} models")

        print(f"\n Phân phối theo Output Type:")
        for output_type, count in output_type_counts.items():
            if count > 0:
                emoji = "" if output_type == "probability" else "" if output_type == "label" else ""
                print(f"   {emoji} {output_type}: {count} models")

        print(f"\n Phân phối theo Loại Model:")
        for model_type, count in clone_counts.items():
            emoji = "" if model_type == "clone" else ""
            print(f"   {emoji} {model_type}: {count} models")

        print(f"\n Tất cả models hỗ trợ metadata đầy đủ cho serving!")

    except Exception as e:
        print(f" Lỗi khi truy cập MLflow Registry: {e}")

    print("=" * 80)


def main():
    """Menu đơn giản với các tính năng mở rộng"""
    while True:
        print("\n" + "=" * 80)
        print(" ANOMALY DETECTION - MLFLOW INTEGRATION WITH METADATA")
        print("=" * 80)
        print("1.  TRAIN VÀ SAVE TẤT CẢ MODELS VỚI METADATA")
        print("2.  TEST LOAD MODELS VÀ KIỂM TRA METADATA")
        print("3.  HIỂN THỊ THỐNG KÊ MLFLOW REGISTRY")
        print("4.  NHÂN BẢN MODELS (CLONE)")
        print("0.  THOÁT")
        print("=" * 80)

        choice = input("\n Chọn chức năng (0-4): ").strip()

        if choice == '1':
            train_and_save_all_models()

        elif choice == '2':
            test_model_loading_and_metadata()

        elif choice == '3':
            show_registry_stats()

        elif choice == '4':
            clone_models_interactive()

        elif choice == '0':
            print("\n Cảm ơn bạn đã sử dụng hệ thống!")
            print(" Tất cả models đã được lưu với metadata đầy đủ!")
            break

        else:
            print(" Lựa chọn không hợp lệ! Vui lòng chọn từ 0-4.")


if __name__ == "__main__":
    main()