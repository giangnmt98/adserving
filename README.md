# Anomaly Detection Serve

A comprehensive anomaly detection system using Ray Serve and MLflow to deploy and serve multiple machine learning models with ultra-scale deployment capabilities, zero-downtime deployment, auto-scaling, and comprehensive monitoring.

## Key Features

### Ultra-Scale Model Management
- **2000+ Model Capacity**: Deploy and serve thousands of models simultaneously
- **MLflow Model Registry Integration**: Automatic management of "Production" status models
- **Zero-downtime Deployment**: Blue-green deployment strategy for seamless updates
- **Intelligent Tier Management**: Hot/Warm/Cold model storage tiers for optimal performance
- **Automatic Model Updates**: Detect and load new models automatically

### Advanced Deployment System
- **Blue-Green Orchestration**: Zero-downtime environment switching
- **Ultra-Scale Deployment Manager**: Handles 2000+ models with parallel processing
- **Service Discovery**: Intelligent load balancing and service registration
- **Health Monitoring**: Comprehensive system and model health checks
- **Graceful Shutdown**: Proper resource cleanup and request draining

### Ray Serve Integration
- **Parallel Model Serving**: Deploy multiple models in parallel
- **Auto-scaling**: Automatic replica adjustment based on load
- **Resource Management**: Efficient CPU/GPU resource utilization
- **Request Routing**: Intelligent request routing to optimal model instances

### Monitoring & Observability
- **Real-time Monitoring**: Resource and performance monitoring
- **Metrics Collection**: System, deployment, and model metrics
- **Alerting System**: Configurable alerts with multiple severity levels
- **Performance Analytics**: Deployment success rates and timing analysis
- **Prometheus Integration**: Export metrics in Prometheus format

## 📋 Yêu cầu hệ thống

- Python 3.8+
- MLflow Server (cho Model Registry)
- Ray Cluster (tùy chọn, có thể chạy local)
- 4GB+ RAM (khuyến nghị 8GB+)
- GPU (tùy chọn, cho các mô hình yêu cầu GPU)

## 🛠️ Cài đặt

### 1. Clone repository và cài đặt dependencies

```bash
git clone <repository-url>
cd mlops-serve
pip install -r requirements.txt
```

### 2. Cấu hình MLflow Server

```bash
# Khởi động MLflow server
mlflow server --host 0.0.0.0 --port 5000 --default-artifact-root ./mlruns
```

### 3. Cấu hình hệ thống

Sao chép và chỉnh sửa file cấu hình:

```bash
cp config.yaml my_config.yaml
# Chỉnh sửa my_config.yaml theo nhu cầu
```

## Usage

### Starting the Server

```bash
# Start with default configuration
python app.py

# Start in production mode
python app.py --production

# Start with custom host and port
MLOPS_HOST=0.0.0.0 MLOPS_PORT=8080 python app.py

# Start with debug logging
python app.py --debug
```

### Using the Service Module

```python
from adserving.src.service import AnomalyDetectionServe

# Create service instance
service = AnomalyDetectionServe()

# Run the service (will use config.yaml)
service.run()
```

### Configuration

The system uses `config.yaml` for configuration. Key settings include:

```yaml
# API Configuration
api_host: "0.0.0.0"
api_port: 8000

# Ray Cluster Configuration
ray:
  num_cpus: 20
  object_store_memory: 8000
  
# MLflow Integration
mlflow:
  tracking_uri: "http://localhost:5000"

# Ultra-scale deployment settings
max_workers: 10
enable_auto_deployment: true
```

## 📚 API Documentation

Server cung cấp REST API đầy đủ với Swagger UI tại `http://localhost:8000/docs`

### Endpoints chính

#### Health Check
```http
GET /health
```

#### Prediction
```http
POST /api/v1/predict
Content-Type: application/json

{
    "model_name": "my_model",
    "data": {"feature1": 1.0, "feature2": 2.0},
    "request_id": "optional_id"
}
```

#### Model Management
```http
# Liệt kê tất cả mô hình
GET /api/v1/models

# Trạng thái mô hình cụ thể
GET /api/v1/models/{model_name}/status

# Deploy mô hình
POST /api/v1/models/{model_name}/deploy

# Undeploy mô hình
DELETE /api/v1/models/{model_name}/deploy
```

#### System Monitoring
```http
# Trạng thái hệ thống
GET /api/v1/system/status

# Metrics chi tiết
GET /api/v1/system/metrics

# Cấu hình hệ thống
GET /api/v1/system/config
```

## 🔧 Cấu hình

### Cấu hình cơ bản

```yaml
# MLflow
mlflow:
  tracking_uri: "http://localhost:5000"

# API
api_host: "0.0.0.0"
api_port: 8000

# Auto-deployment
enable_auto_deployment: true
```

### Cấu hình Resource Management

```yaml
deployment:
  resource_config:
    num_cpus: 2.0
    num_gpus: 1.0
    memory: 2048  # MB
  
  autoscaling:
    min_replicas: 1
    max_replicas: 5
    target_num_ongoing_requests_per_replica: 2
```

### Cấu hình Monitoring

```yaml
monitoring:
  collection_interval: 10
  alert_thresholds:
    cpu_percent: 80.0
    memory_percent: 85.0
    error_rate: 0.01
```

### Cấu hình Security

```yaml
security:
  enable_auth: true
  api_key: "your-secret-key"
  allowed_origins: ["https://yourdomain.com"]
```

## 📊 Monitoring Dashboard

Hệ thống cung cấp monitoring dashboard qua Ray Dashboard:
- Truy cập: `http://localhost:8265`
- Xem resource usage, model performance, scaling metrics
- Real-time monitoring và alerting

## 🔍 Logging

### Log Files
- `logs/mlops_serve.log`: Main application logs
- `logs/errors.log`: Error logs only
- `logs/performance.log`: Performance metrics

### Log Format
```json
{
    "timestamp": 1640995200.0,
    "level": "INFO",
    "logger": "mlops_serve",
    "message": "Model deployed successfully",
    "extra": {
        "model_name": "my_model",
        "deployment_time": 2.5
    }
}
```

## 🧪 Testing

### Unit Tests
```bash
pytest tests/
```

### API Testing
```bash
# Test prediction endpoint
curl -X POST "http://localhost:8000/api/v1/predict" \
     -H "Content-Type: application/json" \
     -d '{"model_name": "test_model", "data": {"x": 1.0}}'
```

### Load Testing
```python
import asyncio
import aiohttp

async def load_test():
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(100):
            task = session.post(
                "http://localhost:8000/api/v1/predict",
                json={"model_name": "test_model", "data": {"x": i}}
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        print(f"Completed {len(responses)} requests")

asyncio.run(load_test())
```

## 🚨 Troubleshooting

### Common Issues

#### 1. MLflow Connection Error
```
Error: Cannot connect to MLflow server
```
**Solution**: Kiểm tra MLflow server đang chạy và cấu hình `tracking_uri` đúng.

#### 2. Ray Initialization Error
```
Error: Ray cluster not available
```
**Solution**: Đảm bảo Ray cluster đang chạy hoặc sử dụng local mode.

#### 3. Model Not Found
```
Error: Model not found in registry
```
**Solution**: Kiểm tra mô hình đã được register với stage "Production" trong MLflow.

#### 4. Memory Issues
```
Error: Out of memory
```
**Solution**: Giảm `model_cache_size` hoặc tăng memory cho system.

### Debug Mode

```bash
# Chạy với debug logging
python -m mlops_serve.mlops_server --log-level DEBUG
```

### Health Checks

```bash
# Kiểm tra health của hệ thống
curl http://localhost:8000/health

# Kiểm tra metrics
curl http://localhost:8000/api/v1/system/metrics
```

## 🔄 Zero-Downtime Deployment

Hệ thống hỗ trợ zero-downtime deployment:

1. **Model Update Detection**: Tự động phát hiện mô hình mới trong MLflow
2. **Graceful Replacement**: Load mô hình mới song song với mô hình cũ
3. **Traffic Switching**: Chuyển traffic sang mô hình mới khi sẵn sàng
4. **Rollback Support**: Có thể rollback nếu mô hình mới có vấn đề

## 📈 Performance Optimization

### Auto-scaling
- Tự động scale up khi tải cao
- Scale down khi tải thấp
- Configurable thresholds và metrics

### Batching
- Automatic request batching
- Optimal batch size detection
- Reduced inference overhead

### Caching
- Model caching với LRU eviction
- Prediction result caching (tùy chọn)
- Efficient memory management

## 🔐 Security

### Authentication
- API key authentication
- JWT token support (tùy chọn)
- Role-based access control

### Network Security
- CORS configuration
- Rate limiting
- HTTPS support

### Data Security
- Input validation
- Secure logging (no sensitive data)
- Audit trails

## 📝 Best Practices

### Model Management
1. Sử dụng semantic versioning cho models
2. Test models thoroughly trước khi promote lên Production
3. Monitor model performance sau deployment
4. Implement model rollback strategy

### Resource Management
1. Set appropriate resource limits
2. Monitor resource usage
3. Use GPU efficiently
4. Implement proper cleanup

### Monitoring
1. Set up alerting cho critical metrics
2. Regular health checks
3. Log aggregation và analysis
4. Performance baseline tracking

## 🤝 Contributing

1. Fork repository
2. Tạo feature branch
3. Implement changes với tests
4. Submit pull request

## 📄 License

MIT License - xem file LICENSE để biết chi tiết.

## 🆘 Support

- GitHub Issues: Báo cáo bugs và feature requests
- Documentation: Xem docs/ folder
- Examples: Xem examples/ folder

---

**MLOps Serve** - Production-ready ML model serving với Ray Serve và MLflow 🚀