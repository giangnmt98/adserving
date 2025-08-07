-- set schema 'public';

-- 1. Thông tin báo cáo (Report Information)
-- Bảng chứa thông tin về các loại báo cáo trong hệ thống
CREATE TABLE IF NOT EXISTS bao_cao (
   ma_bao_cao TEXT PRIMARY KEY,
   ten_bao_cao TEXT,
   mo_ta TEXT,
   created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
   updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 2. Thuộc tính theo từng báo cáo (Report Attributes)
-- Bảng chứa các thuộc tính/trường dữ liệu của từng báo cáo
CREATE TABLE IF NOT EXISTS thuoc_tinh (
                                          ma_bao_cao TEXT REFERENCES bao_cao(ma_bao_cao) ON DELETE CASCADE,
                                          fld_code TEXT,
                                          ten_thuoc_tinh TEXT,
                                          kieu_du_lieu TEXT DEFAULT 'TEXT',
                                          bat_buoc BOOLEAN DEFAULT FALSE,
                                          created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                                          updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                                          PRIMARY KEY (ma_bao_cao, fld_code)
);

-- 3. Đơn vị hành chính (Administrative Units)
-- Bảng chứa thông tin các đơn vị hành chính
CREATE TABLE IF NOT EXISTS don_vi (
                                      ma_don_vi TEXT PRIMARY KEY,
                                      ten_don_vi TEXT,
                                      cap_don_vi TEXT,
                                      ma_don_vi_cha TEXT,
                                      dia_chi TEXT,
                                      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                                      updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 4. Chỉ tiêu (Indicators)
-- Bảng chứa các chỉ tiêu của từng báo cáo
CREATE TABLE IF NOT EXISTS chi_tieu (
    ma_bao_cao TEXT REFERENCES bao_cao(ma_bao_cao) ON DELETE CASCADE,
    ma_tieu_chi TEXT,
    ten_chi_tieu TEXT,
    don_vi_tinh TEXT,
    mo_ta TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (ma_bao_cao, ma_tieu_chi)
);

-- 5. Dữ liệu chính (Main Data)
-- Bảng chứa dữ liệu thực tế của các báo cáo
CREATE TABLE IF NOT EXISTS bao_cao_dulieu (
    ma_don_vi TEXT REFERENCES don_vi(ma_don_vi),
    ma_bao_cao TEXT REFERENCES bao_cao(ma_bao_cao),
    ma_tieu_chi TEXT,
    fld_code TEXT,
    ky_du_lieu DATE,
    gia_tri NUMERIC,
    PRIMARY KEY (ma_don_vi, ma_bao_cao, ma_tieu_chi, fld_code, ky_du_lieu),
    FOREIGN KEY (ma_bao_cao, ma_tieu_chi) REFERENCES chi_tieu(ma_bao_cao, ma_tieu_chi),
    FOREIGN KEY (ma_bao_cao, fld_code) REFERENCES thuoc_tinh(ma_bao_cao, fld_code)
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (ma_don_vi, ma_bao_cao, ma_tieu_chi, fld_code, ky_du_lieu),
    FOREIGN KEY (ma_bao_cao, ma_tieu_chi) REFERENCES chi_tieu(ma_bao_cao, ma_tieu_chi) ON DELETE CASCADE,
    FOREIGN KEY (ma_bao_cao, fld_code) REFERENCES thuoc_tinh(ma_bao_cao, fld_code) ON DELETE CASCADE
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_bao_cao_ten ON bao_cao(ten_bao_cao);
CREATE INDEX IF NOT EXISTS idx_thuoc_tinh_ma_bao_cao ON thuoc_tinh(ma_bao_cao);
CREATE INDEX IF NOT EXISTS idx_thuoc_tinh_fld_code ON thuoc_tinh(fld_code);
CREATE INDEX IF NOT EXISTS idx_don_vi_ten ON don_vi(ten_don_vi);
CREATE INDEX IF NOT EXISTS idx_don_vi_cap ON don_vi(cap_don_vi);
CREATE INDEX IF NOT EXISTS idx_chi_tieu_ma_bao_cao ON chi_tieu(ma_bao_cao);
CREATE INDEX IF NOT EXISTS idx_chi_tieu_ma_tieu_chi ON chi_tieu(ma_tieu_chi);
CREATE INDEX IF NOT EXISTS idx_bao_cao_dulieu_ma_don_vi ON bao_cao_dulieu(ma_don_vi);
CREATE INDEX IF NOT EXISTS idx_bao_cao_dulieu_ma_bao_cao ON bao_cao_dulieu(ma_bao_cao);
CREATE INDEX IF NOT EXISTS idx_bao_cao_dulieu_ky_du_lieu ON bao_cao_dulieu(ky_du_lieu);
CREATE INDEX IF NOT EXISTS idx_bao_cao_dulieu_created_at ON bao_cao_dulieu(created_at);





-- Table for tracking processing queue status
CREATE TABLE IF NOT EXISTS queue_status (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    queue_name VARCHAR(255) NOT NULL,
    message_id VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL, -- pending, processing, completed, failed
    retry_count INTEGER DEFAULT 0,
    error_message TEXT,
    processing_started_at TIMESTAMPTZ,
    processing_completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Indexes for efficient querying
    INDEX idx_queue_status_timestamp (timestamp),
    INDEX idx_queue_status_queue_name (queue_name),
    INDEX idx_queue_status_message_id (message_id),
    INDEX idx_queue_status_status (status),
    INDEX idx_queue_status_created_at (created_at)
);





-- Table for storing simplified inference results
CREATE TABLE IF NOT EXISTS infer_result (
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_name VARCHAR(255) NOT NULL,
    request_id VARCHAR(255) NOT NULL,
    ma_tieu_chi TEXT,
    fld_code TEXT,
    is_anomaly BOOLEAN DEFAULT FALSE,
    anomaly_score FLOAT,
    threshold FLOAT,
    processing_time INTEGER,
    model_version VARCHAR(100),
    status VARCHAR(50) DEFAULT 'success',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Indexes for efficient querying
    INDEX idx_infer_result_timestamp (timestamp),
    INDEX idx_infer_result_model_name (model_name),
    INDEX idx_infer_result_request_id (request_id),
    INDEX idx_infer_result_status (status),
    INDEX idx_infer_result_anomaly_detected (is_anomaly),
    INDEX idx_infer_result_created_at (created_at)
);
