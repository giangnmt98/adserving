import requests
import datetime
import psycopg2
from psycopg2.extras import execute_values

# Cấu hình DB
conn = psycopg2.connect(
    # dbname="your_database",
    user="admin",
    password="admin",
    host="localhost",
    port=5432
)
cursor = conn.cursor()

# Cấu hình API
url = "https://report.vnsr.vn/IOC_WS_1/api/baocao/canhbao"
headers = {
    "Content-Type": "application/json",
    "Cookie": (
        "SESSIONID=!ypR3ECELFO3obkdzM3kBzYjKt4COQ0OBJF1UPrauwxuJDVX7TRQ35UMEB0krZCHbyNaF7V5V9AlUbQ==; "
        "TS01df1866=012b75ff9cdc5daf4b9545b0700bc7e9c1958151998f2f68f601153d341901294af650330d4aab64217a99a76f2594a1fa2446ec495c3032a85fc819032ceca1fbbb9bce09"
    )
}
base_payload = {
    "tokenAI": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjMiLCJuYW1lIjoiYnBjLnN5bmMiLCJpYXQiOjE1MTYyMzkwMjUsInRlbmFudF9pZCI6MTB9.OxuhRXzSYnFxYFjez4LRGWWNhfUoJYv0szJkhrPp6CM",
    "orgId": "23297903",
    "objId": "10628953"
}

start_date = datetime.date(2025, 5, 1)
end_date = datetime.date(2025, 7, 31)
delta = datetime.timedelta(days=1)

days_with_data = []

while start_date <= end_date:
    time_id = start_date.strftime("%Y%m%d")
    payload = {**base_payload, "timeId": time_id}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()

        if not data:
            start_date += delta
            continue

        ma_bao_cao = str(base_payload["objId"])
        ten_bao_cao = data[0].get("tenBaoCao", "")
        mo_ta = data[0].get("moTa", "")

        # Insert vào bảng bao_cao
        cursor.execute("""
            INSERT INTO bao_cao (ma_bao_cao, ten_bao_cao, mo_ta)
            VALUES (%s, %s, %s)
            ON CONFLICT (ma_bao_cao) DO NOTHING
        """, (ma_bao_cao, ten_bao_cao, mo_ta))

        dulieu_rows = []
        donvi_set = set()
        chitieu_set = set()
        thuoctinh_set = set()

        for report in data:
            ma_don_vi = report["maDonVi"]
            ten_don_vi = report["tenDonVi"]
            ky_du_lieu = datetime.datetime.strptime(report["kyDuLieu"], "%Y%m%d").date()
            donvi_set.add((ma_don_vi, ten_don_vi))

            for chi_tieu in report["duLieuBaoCao"]:
                ma_tieu_chi = chi_tieu["maChiTieu"]
                ten_chi_tieu = chi_tieu["tenChiTieu"]
                chitieu_set.add((ma_bao_cao, ma_tieu_chi, ten_chi_tieu))

                for item in chi_tieu["dulieu"]:
                    fld_code = item["fldCode"]
                    ten_thuoc_tinh = item["tenThuocTinh"]
                    gia_tri = float(item["giaTri"]) if item["giaTri"] else None

                    thuoctinh_set.add((ma_bao_cao, fld_code, ten_thuoc_tinh))

                    dulieu_rows.append((
                        ma_don_vi,
                        ma_bao_cao,
                        ma_tieu_chi,
                        fld_code,
                        ky_du_lieu,
                        gia_tri
                    ))

        if dulieu_rows:
            execute_values(cursor, """
                INSERT INTO don_vi (ma_don_vi, ten_don_vi)
                VALUES %s
                ON CONFLICT (ma_don_vi) DO NOTHING
            """, list(donvi_set))

            execute_values(cursor, """
                INSERT INTO chi_tieu (ma_bao_cao, ma_tieu_chi, ten_chi_tieu)
                VALUES %s
                ON CONFLICT (ma_bao_cao, ma_tieu_chi) DO NOTHING
            """, list(chitieu_set))

            execute_values(cursor, """
                INSERT INTO thuoc_tinh (ma_bao_cao, fld_code, ten_thuoc_tinh)
                VALUES %s
                ON CONFLICT (ma_bao_cao, fld_code) DO NOTHING
            """, list(thuoctinh_set))

            execute_values(cursor, """
                INSERT INTO bao_cao_dulieu (
                    ma_don_vi, ma_bao_cao, ma_tieu_chi, fld_code, ky_du_lieu, gia_tri
                )
                VALUES %s
                ON CONFLICT DO NOTHING
            """, dulieu_rows)

            conn.commit()
            days_with_data.append(time_id)

    except Exception as e:
        pass  # Lỗi ngày nào thì skip, không in

    start_date += delta

cursor.close()
conn.close()

# In tổng kết
print("Các ngày có dữ liệu đã ghi vào DB:")
for d in days_with_data:
    print(d)