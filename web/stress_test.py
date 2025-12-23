#!/usr/bin/env python3
import requests
import time
import os
import argparse
import random
import concurrent.futures
from pathlib import Path
from statistics import mean, median

def send_single_request(img_path, url, fov, block_filter):
    """發送單一請求並記錄時間"""
    start_time = time.perf_counter()
    try:
        with open(img_path, 'rb') as f:
            payload = {'fov': fov}
            if block_filter:
                payload['block_filter'] = block_filter
            
            resp = requests.post(url, files={'file': f}, data=payload, timeout=60)
            
        latency = (time.perf_counter() - start_time) * 1000 # 轉為 ms
        
        if resp.status_code == 200:
            return {"success": True, "latency": latency, "server_latency": resp.json().get("latency_ms", 0)}
        else:
            return {"success": False, "latency": latency, "error": f"HTTP {resp.status_code}"}
    except Exception as e:
        latency = (time.perf_counter() - start_time) * 1000
        return {"success": False, "latency": latency, "error": str(e)}

def run_benchmark(image_folder, url, fov, block_filter, num_requests, concurrency):
    """執行特定並行數的測試"""
    image_paths = list(Path(image_folder).glob("*.jpg")) + list(Path(image_folder).glob("*.png"))
    if not image_paths:
        print(f"錯誤: 在 {image_folder} 找不到圖片")
        return

    print(f"\n>>> 開始測試: 並行數={concurrency}, 總請求數={num_requests}")
    
    test_tasks = [random.choice(image_paths) for _ in range(num_requests)]
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_req = {executor.submit(send_single_request, img, url, fov, block_filter): img for img in test_tasks}
        for future in concurrent.futures.as_completed(future_to_req):
            results.append(future.result())

    # 統計數據
    successes = [r for r in results if r["success"]]
    latencies = [r["latency"] for r in results]
    
    success_rate = (len(successes) / num_requests) * 100
    avg_latency = mean(latencies) if latencies else 0
    med_latency = median(latencies) if latencies else 0

    print(f"--- 測試結果 (並行度: {concurrency}) ---")
    print(f"成功率: {success_rate:.2f}% ({len(successes)}/{num_requests})")
    print(f"平均等待時間 (Round-trip): {avg_latency:.2f} ms")
    print(f"中位數等待時間: {med_latency:.2f} ms")
    if successes:
        avg_server_internal = mean([r["server_latency"] for r in successes])
        print(f"伺服器內部平均處理時間: {avg_server_internal:.2f} ms")

def main():
    parser = argparse.ArgumentParser(description="Localization Server Stress Test Tool")
    parser.add_argument("image_folder", type=str, help="存放測試圖片的資料夾")
    parser.add_argument("--url", type=str, default="http://localhost:8000/localize", help="Server API URL")
    parser.add_argument("--fov", type=float, default=70.0)
    parser.add_argument("--requests", type=int, default=20, help="每個並行層級總共發送的請求數")
    parser.add_argument("--concurrency_list", type=str, default="1,2,4,8", help="測試的並行數清單，用逗號分隔")
    parser.add_argument("--filter", type=str, default=None, help="Block filter")

    args = parser.parse_args()
    concurrencies = [int(c) for c in args.concurrency_list.split(",")]

    print(f"=== 壓力測試啟動 ===")
    print(f"目標 URL: {args.url}")
    print(f"測試圖片目錄: {args.image_folder}")
    print(f"預計測試並行層級: {concurrencies}")
    print("====================")

    for c in concurrencies:
        run_benchmark(args.image_folder, args.url, args.fov, args.filter, args.requests, c)

if __name__ == "__main__":
    main()