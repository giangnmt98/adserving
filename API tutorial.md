## **🚀 Hướng dẫn sử dụng API Endpoints**
### **1. Main Prediction Endpoint**
#### **POST `/api/v1/predict`**
**Mục đích**: Thực hiện dự đoán với model
**Input format**:
``` json
{
    "ma_don_vi": "UBND.0019",
    "ma_bao_cao": "10628953_CT", 
    "ky_du_lieu": "2024-01-01",
    "data": [
        {
            "ma_tieu_chi": "TONGCONG",
            "FN01": 1000000
        }
    ]
}
```
**Giải thích các trường**:
- `ma_don_vi`: Mã đơn vị (string) - VD: "UBND.0019"
- `ma_bao_cao`: Mã báo cáo (string) - VD: "10628953_CT"
- `ky_du_lieu`: Kỳ dữ liệu (YYYY-MM-DD format) - VD: "2024-01-01"
- `data`: Array chứa dữ liệu dự đoán
    - `ma_tieu_chi`: Mã tiêu chí (string) - VD: "TONGCONG"
    - `[field_code]`: Giá trị số (number) - VD: "FN01": 1000000

**Response**:
``` json
{
    "status": "success",
    "prediction": 0.95,
    "model_used": "UBND.0019_10628953_CT_TONGCONG_FN01",
    "confidence": 0.87,
    "processing_time": 0.123
}
```
### **2. Tier Management Endpoints**
#### **POST `/tier-management/manual-assignment`**
**Mục đích**: Thay đổi tier của model thủ công
**Input format**:
``` json
{
    "model_name": "UBND.0019_10628953_CT_TONGCONG_FN01",
    "new_tier": "HOT"
}
```
**Các tier hợp lệ**:
- `"HOT"`: Model được load sẵn, response nhanh nhất
- `"WARM"`: Model được load nhưng chưa deploy
- `"COLD"`: Model load khi có request

**Response**:
``` json
{
    "status": "success",
    "message": "Model tier changed successfully",
    "old_tier": "WARM",
    "new_tier": "HOT"
}
```
#### **GET `/tier-management/statistics`**
**Mục đích**: Xem thống kê các tier
**Response**:
``` json
{
    "tier_management": {
        "HOT": {
            "model_count": 15,
            "avg_response_time": 0.045,
            "request_count": 1250
        },
        "WARM": {
            "model_count": 30,
            "avg_response_time": 0.125,
            "request_count": 800
        },
        "COLD": {
            "model_count": 100,
            "avg_response_time": 0.850,
            "request_count": 200
        }
    }
}
```
### **3. Health Check Endpoints**
#### **GET `/health`**
**Response**:
``` json
{
    "status": "healthy",
    "deployments": {
        "model_pool_0": "healthy",
        "model_pool_1": "healthy"
    },
    "total_models": 145
}
```
#### **GET `/deployment-stats`**
**Response**:
``` json
{
    "model_pool_0": {
        "status": "healthy",
        "loaded_models": 25,
        "active_requests": 8,
        "cpu_usage": 0.65,
        "memory_usage": 0.78
    }
}
```
### **4. Model Management Endpoints**
#### **POST `/models/preload`**
**Mục đích**: Pre-load model vào deployment pool
**Input**:
``` json
{
    "model_name": "UBND.0019_10628953_CT_TONGCONG_FN01",
    "deployment_name": "model_pool_0"
}
```
#### **GET `/models/{model_name}/status`**
**Response**:
``` json
{
    "model_name": "UBND.0019_10628953_CT_TONGCONG_FN01",
    "tier": "HOT",
    "is_loaded": true,
    "is_deployed": true,
    "deployment": "model_pool_0",
    "last_used": "2024-01-15T10:30:00Z"
}
```
## **📝 Cách tạo Input phù hợp**
### **Cách xây dựng model_name**:
``` 
Format: {ma_don_vi}_{ma_bao_cao}_{ma_tieu_chi}_{field_code}
Ví dụ: "UBND.0019_10628953_CT_TONGCONG_FN01"
```
### **Các scenarios test phổ biến**:
#### **1. Normal Load Test**:
``` json
{
    "ma_don_vi": "UBND.0019",
    "ma_bao_cao": "10628953_CT",
    "ky_du_lieu": "2024-01-15",
    "data": [
        {
            "ma_tieu_chi": "TONGCONG",
            "FN01": 1500000
        }
    ]
}
```
#### **2. High Load Test**:
``` json
{
    "ma_don_vi": "UBND.0020",
    "ma_bao_cao": "10628954_CT", 
    "ky_du_lieu": "2024-01-15",
    "data": [
        {
            "ma_tieu_chi": "CHITIET",
            "FN02": 2500000
        }
    ]
}
```
#### **3. Burst Test**:
``` json
{
    "ma_don_vi": "UBND.0021",
    "ma_bao_cao": "10628955_CT",
    "ky_du_lieu": "2024-01-15", 
    "data": [
        {
            "ma_tieu_chi": "PHANLOAI",
            "FN03": 950000
        }
    ]
}
```
## **🔧 Best Practices**
### **1. Input Validation**:
- `ma_don_vi`: Không được rỗng, format "UBND.XXXX"
- `ky_du_lieu`: Format YYYY-MM-DD
- Giá trị số: Phải là number, không phải string

### **2. Error Handling**:
``` json
// Lỗi model không tồn tại
{
    "status": "error",
    "error_code": "MODEL_NOT_FOUND",
    "message": "Model UBND.0019_10628953_CT_TONGCONG_FN01 not found"
}

// Lỗi input không hợp lệ  
{
    "status": "error",
    "error_code": "INVALID_INPUT",
    "message": "Field FN01 must be a number"
}
```
### **3. Performance Tips**:
- Sử dụng HOT tier cho models thường xuyên dùng
- Batch multiple requests khi có thể
- Monitor response time và adjust tier accordingly
