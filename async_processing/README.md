# Async Processing System for Adserving

Hệ thống xử lý bất đồng bộ dữ liệu cho adserving package, sử dụng Redis Streams, Ray Tasks và PostgreSQL để xử lý dữ liệu API một cách hiệu quả và có thể mở rộng.

## Tổng quan

Hệ thống này cung cấp:

- **Queue**: Redis Streams để lưu tạm thời thông tin request từ API
- **Async Worker**: Ray Tasks để triển khai worker bất đồng bộ
- **Database**: PostgreSQL (với tích hợp TimescaleDB tùy chọn) để lưu trữ dữ liệu training và kết quả inference

## Kiến trúc

```
API Request → Redis Streams → Ray Workers → PostgreSQL
     ↓              ↓             ↓           ↓
Training Data   Message Queue  Processing   Data Storage
Inference Results              Distributed   Performance
                              Computing     Analytics
```

## Cài đặt

### 1. Cài đặt Dependencies

```bash
pip install -r requirements.txt
```

### 2. Cài đặt Redis

```bash
# Ubuntu/Debian
sudo apt-get install redis-server

# macOS
brew install redis

# Docker
docker run -d -p 6379:6379 redis:latest
```

### 3. Cài đặt PostgreSQL

```bash
# Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib

# macOS
brew install postgresql

# Docker
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=password postgres:latest
```

### 4. Cài đặt TimescaleDB (Tùy chọn)

```bash
# Ubuntu/Debian
sudo apt-get install timescaledb-postgresql

# Docker
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=password timescale/timescaledb:latest-pg15
```

## Cấu hình

### 1. Environment Variables

Sao chép file template và cấu hình:

```bash
cp .env.template .env
```

Chỉnh sửa file `.env`:

```bash
# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# Database Configuration
DATABASE_HOST=localhost
DATABASE_PORT=5432
DATABASE_NAME=adserving
DATABASE_USER=postgres
DATABASE_PASSWORD=password

# Async Processing
ASYNC_PROCESSING_ENABLED=true
ASYNC_PROCESSING_NUM_WORKERS=4
```

### 2. Database Setup

Chạy script SQL để tạo schema:

```bash
psql -h localhost -U postgres -d adserving -f models.sql
```

## Sử dụng

### 1. Khởi tạo hệ thống

```python
from adserving.async_processing import AsyncProcessingSystem

# Khởi tạo hệ thống
system = AsyncProcessingSystem()
await system.initialize()

# Setup database schema
await system.setup_database()
```

### 2. Gửi dữ liệu training

```python
# Gửi dữ liệu training
message_id = await system.send_training_data(
    model_name="fraud_detection",
    request_id="req_123",
    input_data={
        "user_id": 1234,
        "transaction_amount": 100.50,
        "features": [0.1, 0.2, 0.3]
    },
    metadata={"api_version": "v1"},
    source_ip="192.168.1.100"
)
```

### 3. Gửi kết quả inference

```python
# Gửi kết quả inference
message_id = await system.send_inference_result(
    model_name="fraud_detection",
    request_id="req_123",
    prediction={"prediction": 0, "probability": 0.85},
    confidence=0.85,
    anomaly_score=0.15,
    is_anomaly=False,
    processing_time_ms=120
)
```

### 4. Gửi dữ liệu API hoàn chỉnh

```python
# Gửi cả training data và inference result
message_ids = await system.send_api_request_data(
    model_name="fraud_detection",
    request_id="req_123",
    input_data={"user_id": 1234, "amount": 100.50},
    prediction={"prediction": 0, "probability": 0.85},
    confidence=0.85,
    anomaly_score=0.15,
    processing_time_ms=120
)
```

### 5. Khởi động workers

```python
# Khởi động workers để xử lý messages
await system.start_workers()

# Hoặc chỉ định streams cụ thể
await system.start_workers([
    "training_data_stream",
    "inference_results_stream"
])
```

### 6. Monitoring và thống kê

```python
# Kiểm tra trạng thái hệ thống
status = await system.get_system_status()
print(f"System health: {status}")

# Lấy thống kê performance
stats = await system.get_model_performance_stats("fraud_detection", hours=24)
print(f"Model stats: {stats}")
```

## Cấu trúc Files

```
async_processing/
├── __init__.py              # Main system integration
├── config.py               # Configuration management
├── db.py                   # Database connection and repositories
├── producer.py             # Redis producer for sending data
├── consumer.py             # Ray workers for processing
├── models.sql              # Database schema
├── .env.template           # Environment configuration template
├── requirements.txt        # Python dependencies
├── example_usage.py        # Usage examples
└── README.md              # This documentation
```

## API Reference

### AsyncProcessingSystem

Lớp chính để quản lý toàn bộ hệ thống async processing.

#### Methods

- `initialize()`: Khởi tạo tất cả components
- `close()`: Đóng tất cả connections
- `setup_database(schema_path)`: Setup database schema
- `send_training_data(**kwargs)`: Gửi training data
- `send_inference_result(**kwargs)`: Gửi inference result
- `send_api_request_data(**kwargs)`: Gửi complete API data
- `start_workers(streams)`: Khởi động workers
- `stop_workers()`: Dừng workers
- `get_system_status()`: Lấy trạng thái hệ thống
- `get_model_performance_stats(model_name, hours)`: Lấy thống kê model

### Configuration Classes

- `AsyncProcessingConfig`: Main configuration class
- `RedisConfig`: Redis connection configuration
- `DatabaseConfig`: PostgreSQL connection configuration
- `ConsumerConfig`: Worker configuration

## Performance Tuning

### Redis Optimization

```bash
# Redis configuration in redis.conf
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

### PostgreSQL Optimization

```sql
-- PostgreSQL configuration
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
maintenance_work_mem = 64MB
```

### Ray Configuration

```python
# Ray initialization
ray.init(
    num_cpus=8,
    num_gpus=0,
    memory=8000000000,  # 8GB
    object_store_memory=2000000000  # 2GB
)
```

## Monitoring

### Health Checks

```python
# Kiểm tra health của từng component
status = await system.get_system_status()

# Redis health
redis_health = status['producer']['health']

# Database health
db_health = status['database']['health']

# Workers health
workers_health = status['workers']
```

### Metrics

Hệ thống cung cấp các metrics sau:

- **Throughput**: Messages processed per second
- **Latency**: Processing time per message
- **Error Rate**: Failed messages percentage
- **Queue Length**: Number of pending messages
- **Worker Health**: Status of each worker

### Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Enable debug logging for specific components
logging.getLogger('async_processing.producer').setLevel(logging.DEBUG)
logging.getLogger('async_processing.consumer').setLevel(logging.DEBUG)
```

## Troubleshooting

### Common Issues

1. **Redis Connection Failed**
   ```bash
   # Check Redis status
   redis-cli ping

   # Check Redis logs
   tail -f /var/log/redis/redis-server.log
   ```

2. **Database Connection Failed**
   ```bash
   # Check PostgreSQL status
   pg_isready -h localhost -p 5432

   # Check database exists
   psql -h localhost -U postgres -l
   ```

3. **Ray Workers Not Starting**
   ```bash
   # Check Ray status
   ray status

   # Check Ray logs
   ray logs
   ```

4. **High Memory Usage**
   ```python
   # Monitor memory usage
   import psutil
   print(f"Memory usage: {psutil.virtual_memory().percent}%")

   # Tune batch sizes
   CONSUMER_BATCH_SIZE=5  # Reduce batch size
   ```

### Debug Mode

```python
# Enable debug mode
import os
os.environ['DEBUG_ENABLED'] = 'true'
os.environ['ASYNC_PROCESSING_LOG_LEVEL'] = 'DEBUG'

# Initialize system with debug config
system = AsyncProcessingSystem()
```

## Testing

### Unit Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=async_processing tests/

# Run specific test
pytest tests/test_producer.py::test_send_training_data
```

### Integration Tests

```bash
# Run integration tests (requires Redis and PostgreSQL)
pytest tests/integration/

# Run with test database
TEST_DATABASE_NAME=adserving_test pytest tests/integration/
```

### Load Testing

```python
# Example load test
import asyncio
import time

async def load_test():
    system = AsyncProcessingSystem()
    await system.initialize()

    start_time = time.time()
    tasks = []

    for i in range(1000):
        task = system.send_training_data(
            model_name="test_model",
            request_id=f"req_{i}",
            input_data={"test": i}
        )
        tasks.append(task)

    await asyncio.gather(*tasks)

    end_time = time.time()
    print(f"Sent 1000 messages in {end_time - start_time:.2f} seconds")

asyncio.run(load_test())
```

## Production Deployment

### Docker Compose

```yaml
version: '3.8'
services:
  redis:
    image: redis:latest
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  postgres:
    image: timescale/timescaledb:latest-pg15
    ports:
      - "5432:5432"
    environment:
      POSTGRES_PASSWORD: password
      POSTGRES_DB: adserving
    volumes:
      - postgres_data:/var/lib/postgresql/data

  async_workers:
    build: .
    depends_on:
      - redis
      - postgres
    environment:
      - REDIS_HOST=redis
      - DATABASE_HOST=postgres
    command: python -m async_processing.consumer

volumes:
  redis_data:
  postgres_data:
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: async-workers
spec:
  replicas: 4
  selector:
    matchLabels:
      app: async-workers
  template:
    metadata:
      labels:
        app: async-workers
    spec:
      containers:
      - name: worker
        image: adserving/async-processing:latest
        env:
        - name: REDIS_HOST
          value: "redis-service"
        - name: DATABASE_HOST
          value: "postgres-service"
        - name: ASYNC_PROCESSING_NUM_WORKERS
          value: "2"
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes following PEP8 guidelines
4. Add tests for new functionality
5. Run tests and ensure they pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:

- Create an issue in the repository
- Contact the development team
- Check the troubleshooting section above

## Changelog

### v1.0.0
- Initial release
- Redis Streams integration
- Ray-based distributed processing
- PostgreSQL with TimescaleDB support
- Comprehensive configuration system
- Health monitoring and metrics
- Complete documentation and examples
