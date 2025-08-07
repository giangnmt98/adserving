
# benchmark_models.py (Synchronized + CCU-aware)
import asyncio
import aiohttp
import json
import time
import random
from datetime import datetime
from statistics import mean, median, stdev

API_URL = "http://localhost:8000/api/v1/predict"
JSON_PATH = "generated_model_names.json"

# Benchmark config
CONCURRENT_USERS = 10
BENCHMARK_DURATION_SECONDS = 20
SCENARIOS = {
    # Tải đồng đều: Phân bố ngẫu nhiên từ 10-90% để mô phỏng tải bình thường trong ngày
    "uniform": lambda: round(random.uniform(10, 90), 2),

    # Tải tăng đột biến: Phân bố tam giác tập trung về 90% để mô phỏng cao điểm
    "spike": lambda: round(random.triangular(0, 100, 90), 2),

    # Tải thấp: Phân bố ngẫu nhiên từ 0-20% để mô phỏng thời điểm ít người dùng
    "low_load": lambda: round(random.uniform(0, 20), 2),

    # Tải cao: Phân bố ngẫu nhiên từ 80-100% để mô phỏng thời điểm cao điểm
    "high_load": lambda: round(random.uniform(80, 100), 2),

    # Tải đột biến: Chuyển đổi ngẫu nhiên giữa 5% và 95% để mô phỏng tải thay đổi đột ngột 
    "burst": lambda: round(random.choice([5.0, 95.0]), 2)
}

with open(JSON_PATH, "r") as f:
    model_configs = json.load(f)

def generate_payload(cfg, scenario_func):
    model_name = cfg["model_name"]
    parts = model_name.rsplit("_", 1)
    fld_code = parts[-1]
    full_parts = model_name.split("_")
    ma_don_vi = full_parts[0]
    ma_bao_cao = full_parts[1]
    ma_tieu_chi = "_".join(full_parts[2:-1])
    value = scenario_func()

    return {
        "ma_don_vi": ma_don_vi,
        "ma_bao_cao": ma_bao_cao,
        "ky_du_lieu": datetime.utcnow().strftime("%Y-%m-%d"),
        "data": [
            {
                "ma_tieu_chi": ma_tieu_chi,
                fld_code: value
            }
        ]
    }

async def send_request(session, semaphore, payload):
    async with semaphore:
        start = time.time()
        try:
            async with session.post(API_URL, json=payload, timeout=30) as resp:
                json_resp = await resp.json()
                return {
                    "status": resp.status,
                    "latency": round(time.time() - start, 3),
                    "error": None if resp.status == 200 else json_resp
                }
        except Exception as e:
            return {"status": 500, "latency": round(time.time() - start, 3), "error": str(e)}

async def run_combined_benchmark(scenario_name, scenario_func):
    semaphore = asyncio.Semaphore(CONCURRENT_USERS)
    results = []
    stop_time = time.time() + BENCHMARK_DURATION_SECONDS

    async with aiohttp.ClientSession() as session:
        print(f"Running scenario: {scenario_name}")
        print(f"Benchmark duration: {BENCHMARK_DURATION_SECONDS}s")
        print(f"Max concurrent users (CCU): {CONCURRENT_USERS}")

        t0 = time.time()
        round_count = 0
        while time.time() < stop_time:
            tasks = []
            for cfg in model_configs:
                payload = generate_payload(cfg, scenario_func)
                tasks.append(send_request(session, semaphore, payload))
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            round_count += 1

        t1 = time.time()

        latencies = [r["latency"] for r in results if r["status"] == 200]
        errors = [r for r in results if r["status"] != 200]

        print("\nBenchmark Results")
        print("=" * 60)
        print(f"Scenario: {scenario_name}")
        print(f"Rounds completed: {round_count}")
        print(f"Total requests: {len(results)}")
        print(f"Successful: {len(latencies)}")
        print(f"Failed: {len(errors)}")
        print(f"Success rate: {len(latencies)/len(results)*100:.2f}%")

        if latencies:
            print("\nLatency Stats (sec)")
            print(f"Min: {min(latencies)}")
            print(f"Max: {max(latencies)}")
            print(f"Avg: {mean(latencies):.3f}")
            print(f"Median: {median(latencies):.3f}")
            if len(latencies) > 1:
                print(f"Std Dev: {stdev(latencies):.3f}")
                print(f"P90: {round(sorted(latencies)[int(0.9 * len(latencies)) - 1], 3)}")
                print(f"P95: {round(sorted(latencies)[int(0.95 * len(latencies)) - 1], 3)}")

        throughput = len(results) / (t1 - t0)
        print("\nThroughput")
        print(f"Total time: {round(t1 - t0, 2)}s")
        print(f"Requests per second (RPS): {throughput:.2f}")
        print(f"Concurrent Users Observed: {CONCURRENT_USERS}")

        if errors:
            print("\nSample Errors:")
            for e in errors[:5]:
                print(f"Status: {e['status']}, Error: {str(e['error'])[:100]}")

        print("=" * 60)

if __name__ == "__main__":
    for name, func in SCENARIOS.items():
        asyncio.run(run_combined_benchmark(name, func))