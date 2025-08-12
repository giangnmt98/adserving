# AdServing — Hệ thống phục vụ mô hình ML (FastAPI + Ray Serve + MLflow)

AdServing là hệ thống phục vụ suy luận bất thường (anomaly detection) ở quy mô lớn, tích hợp:
- FastAPI cho REST API
- Ray Serve để nạp sẵn (preload) và triển khai hàng trăm mô hình đồng thời
- MLflow (Model Registry) để quản lý phiên bản mô hình và tham số
- Pipeline audit (tùy chọn) qua Redis Streams và PostgreSQL để lưu mẫu training và kết quả suy luận phục vụ giám sát/huấn luyện lại

Tài liệu này mô tả cấu trúc repo, cách cài đặt/chạy, cấu hình, và cách sử dụng API.

Liên quan: xem thêm docs/ARCHITECTURE.md để hiểu kiến trúc tổng thể.

---

## 1) Cấu trúc thư mục

- app.py — điểm vào ứng dụng FastAPI; khởi tạo Ray Serve và preload mô hình, mount router API
- config.yaml — cấu hình runtime (mlflow, ray serve, autoscaling, preload, watcher, logging, monitoring, audit, redis, database, api...)
- requirements.txt — thư viện Python cần thiết (runtime + dev/test)
- Dockerfile — build image chạy dịch vụ
- Makefile — lệnh phục vụ phát triển (venv, style, lint)
- example/
  - get_vsr_data_for_training.py — ví dụ thu thập dữ liệu từ API bên ngoài và ghi vào PostgreSQL
- adserving/
  - src/
    - api/
      - core_endpoints.py — endpoint thông tin dịch vụ, health
      - model_endpoints.py — quản lý model: danh sách production, thông tin, tham số, rollback, warm...
      - prediction_endpoint.py — endpoint dự đoán POST /predict
      - api_dependencies.py, models.py — DI và schema API cho core/model
    - audit/ — tích hợp audit (emit vào Redis, worker tiêu thụ, runner)
      - integration.py — hook phát sự kiện khi parse request / sau khi infer
      - hooks.py, redis_emitter.py, worker.py, runner.py — thành phần audit (nếu bật)
    - config/
      - config.py, config_manager.py — load/điều phối cấu hình (back-compat)
    - datahandler/
      - data_handler.py — chuẩn hóa/định dạng response, xử lý lỗi chi tiết
      - models.py — Pydantic models cho PredictionRequest, APIResponse, lỗi chi tiết
      - validators.py — validator đầu vào
    - deployment/
      - preloaded_model_server.py — Ray Serve Deployment nạp sẵn & theo dõi mô hình Production từ MLflow
      - request_processor.py — chuẩn hóa/validate payload dự đoán thành các “prediction task”
      - utils.py — hàm tiện ích build response/parse tên model/so ngưỡng
    - mlflow_handler/ — truy cập MLflow, cập nhật tham số (parameter updater)
    - utils/
      - logger.py, exception_handlers.py — logging và handler ngoại lệ
  - docs/
    - ARCHITECTURE.md — mô tả kiến trúc

---

## 2) Yêu cầu hệ thống

- Python 3.11+
- MLflow Tracking Server (ví dụ http://localhost:5000) để lấy model Production
- Tuỳ chọn: Redis và PostgreSQL nếu bật audit pipeline
- Hệ điều hành: Linux/Mac/Windows (Docker khuyến khích cho triển khai)

---

## 3) Cài đặt (môi trường phát triển)

- Tạo virtualenv và cài đặt:

```bash
python3 -m venv adserving_env
# Linux/Mac: source adserving_env/bin/activate
# Windows (PowerShell): adserving_env\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -e .[dev]
pre-commit install
```

- Hoặc cài đặt runtime cơ bản:

```bash
pip install -e .
```

Lưu ý: setup.py dùng README.md làm long_description khi publish; README này cần tồn tại (đã bổ sung ở repo).

---

## 4) Cấu hình

Mặc định đọc từ file config.yaml. Các khoá chính:
- mlflow.tracking_uri: URL MLflow Tracking Server (có thể override qua biến môi trường MLFLOW_TRACKING_URI)
- ray: số CPU/GPU, dashboard, v.v.
- http: host/port cho Ray Serve HTTP (nếu dùng trực tiếp)
- preload.max_load_concurrency: số model load song song
- watcher: khoảng thời gian kiểm tra, sanity check
- logging.level: mức log
- monitoring: bật/tắt và chu kỳ thu thập
- api: host/port FastAPI, prefix (mặc định: /api/v1), version
- audit: bật/tắt và tham số sampling
- redis: kết nối và tên streams dùng cho audit
- database: kết nối PostgreSQL

Ví dụ override bằng biến môi trường:
- MLFLOW_TRACKING_URI
- FASTAPI_HOST / FASTAPI_PORT (hoặc API_HOST / API_PORT)
- SERVE_HTTP_HOST / SERVE_HTTP_PORT
- RAY_ADDRESS, RAY_DASHBOARD_HOST, RAY_DASHBOARD_PORT

---

## 5) Chạy dịch vụ

- Local (Python):

```bash
# Đảm bảo MLflow server sẵn sàng, cập nhật config.yaml nếu cần
python app.py
# Mặc định FastAPI tại http://localhost:8000
# Tài liệu Swagger: http://localhost:8000/docs
```

- Docker:

```bash
docker build -t adserving:latest .
# Override MLFLOW_TRACKING_URI nếu cần
# Map port: 8000 (API), 8265 (Ray dashboard), 9090 (Prometheus nếu bật)
docker run --rm -p 8000:8000 -p 8265:8265 -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 adserving:latest
```

---

## 6) API — Đường dẫn cơ bản

Các router được mount dưới prefix cấu hình (mặc định /api/v1). Ví dụ dưới đây giả sử prefix=/api/v1.

- Core
  - GET /api/v1/                — Service info
  - GET /api/v1/health          — Health check (200/206/503 tuỳ trạng thái Serve và preload)

- Prediction
  - POST /api/v1/predict        — Suy luận bất thường theo payload đầu vào

- Model Management
  - GET /api/v1/models/production                      — Danh sách model đang preload (Production)
  - GET /api/v1/models/{model_name}/info               — Thông tin model
  - GET /api/v1/models/{model_name}/parameters         — Lấy tham số (nếu được hỗ trợ)
  - PUT /api/v1/models/{model_name}/threshold          — Cập nhật ngưỡng bất thường
  - PUT /api/v1/models/{model_name}/parameters         — Cập nhật tham số mô hình
  - POST /api/v1/models/{model_name}/rollback          — Rollback về version chỉ định
  - GET /api/v1/models/{model_name}/history            — Lịch sử cập nhật tham số
  - POST /api/v1/models/{model_name}/warm              — Làm nóng cache cho model (nếu hỗ trợ)

Lưu ý: Một số endpoint phụ thuộc implement cụ thể trong adserving/src/mlflow_handler/*.

---

## 7) Payload ví dụ — POST /api/v1/predict

Request (JSON):

```json
{
  "ma_don_vi": "DV001",
  "ma_bao_cao": "10628953",
  "ky_du_lieu": "2025-07-15",
  "data": [
    {
      "ma_tieu_chi": "TC_ABC",
      "FN01": 12.3,
      "FN02": 4.56
    },
    {
      "ma_tieu_chi": "TC_DEF",
      "FN01": 7.89,
      "FN03": 0.12
    }
  ]
}
```

Response (rút gọn):

```json
{
  "request_id": "...",
  "timestamp": "...",
  "total_time": 0.123,
  "status": "success | partial_success | error",
  "metadata": { ... },
  "request_info": { ... },
  "details": {
    "anomalies": [ { "model_name": "...", "is_anomaly": true, "anomaly_score": 1.23, ... } ],
    "failed": [ { "element_index": 1, "error": "..." } ]
  },
  "validation_errors": [ ... ]
}
```

Ràng buộc đầu vào được validate bởi datahandler/models.py (PredictionRequest) và deployment/request_processor.py. Lỗi sẽ trả về chi tiết trường vi phạm (mã lỗi, field_path, mô tả,...).

Tên model nội bộ dùng format: `<ma_don_vi>_<ma_bao_cao>_<ma_tieu_chi>_<FNxx>`.

---

## 8) MLflow & Ray Serve

- Hệ thống sẽ nạp tất cả model ở stage Production từ MLflow (preloaded_model_server.py)
- Theo dõi thay đổi Production định kỳ (watcher.interval_seconds) và làm sanity check (tuỳ chọn)
- Quyết định bất thường dựa theo ngưỡng (threshold) và điểm (score) infer; so sánh ngưỡng dương/âm được chuẩn hoá trong deployment/utils.py

---

## 9) Audit pipeline (tuỳ chọn)

- Bật/tắt trong config.yaml (audit.enabled)
- on_request_parsed() phát sự kiện training; on_inference_done() phát sự kiện kết quả suy luận (adserving/src/audit/integration.py)
- Redis Streams làm hàng đợi: streams.training_data, streams.inference_results, dlq
- Các Ray workers (adserving/src/audit/worker.py) đọc từ Redis và ghi vào PostgreSQL theo batch
- Cấu hình kết nối Redis/PostgreSQL trong config.yaml (redis.*, database.*)

---

## 10) Ví dụ thu thập dữ liệu

Thư mục example/ có script get_vsr_data_for_training.py để gọi một API bên ngoài và ghi vào DB (PostgreSQL). Cần thiết lập kết nối DB trước khi chạy.

---

## 11) Phát triển

- Kiểm tra style/lint/type:

```bash
make style
make test  # flake8 + mypy + pylint
```

- Virtualenv nhanh:

```bash
make venv
```

---

## 12) Triển khai

- Dùng Dockerfile trong repo để build image
- Sử dụng biến môi trường để override MLFLOW_TRACKING_URI, FASTAPI_HOST/PORT, vv
- Expose các port: 8000 (API), 8265 (Ray dashboard), 9090 (Prometheus nếu bật)

---

## 13) Giấy phép

Phần mềm phát hành theo giấy phép MIT (khai báo trong setup.py — License :: OSI Approved :: MIT License).
