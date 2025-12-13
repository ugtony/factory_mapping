#!/usr/bin/env python3
import requests
import json
import os
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Localization Server Test Client")
    
    # 必要參數: 圖片路徑
    parser.add_argument("image_path", type=str, help="Path to the query image")
    
    # 可選參數
    parser.add_argument("--fov", type=float, default=70.0, help="Camera FOV (default: 70)")
    parser.add_argument("--block-filter", type=str, default=None, help="Block filter (e.g. 'brazil360,miami360')")
    parser.add_argument("--url", type=str, default="http://localhost:8000/localize", help="Server API URL")

    args = parser.parse_args()

    # 參數設定
    url = args.url
    img_path = args.image_path
    fov = args.fov
    block_filter_str = args.block_filter

    print(f"=== Client Config ===")
    print(f"Target URL:  {url}")
    print(f"Image:       {img_path}")
    print(f"FOV:         {fov}")
    print(f"Filter:      {block_filter_str}")
    print("=====================")

    # 檢查檔案是否存在 (包含相對路徑修正)
    if not os.path.exists(img_path):
        # 嘗試在前面加 "../" (針對在 web/ 目錄下執行的情況)
        alt_path = os.path.join("..", img_path)
        if os.path.exists(alt_path):
            print(f"[Info] Image not found at '{img_path}', but found at '{alt_path}'. Using that.")
            img_path = alt_path
        else:
            print(f"[Error] Image file not found: {img_path}")
            sys.exit(1)

    try:
        with open(img_path, 'rb') as f:
            # 準備 Form Data
            payload = {'fov': fov}
            if block_filter_str:
                payload['block_filter'] = block_filter_str

            # 發送 Request
            print("Sending request...")
            resp = requests.post(
                url, 
                files={'file': f}, 
                data=payload
            )
        
        # 解析結果
        if resp.status_code == 200:
            res = resp.json()
            print("\n=== Server Response ===")
            print(json.dumps(res, indent=2))
            
            # 驗證重點摘要
            print("\n=== Summary ===")
            status = res.get('status')
            block = res.get('block', 'N/A')
            inliers = res.get('inliers', 0)
            
            print(f"Status:  {status}")
            print(f"Block:   {block}")
            print(f"Inliers: {inliers}")
            
            diag = res.get('diagnosis', {})
            if diag:
                print(f"Details: Top1={diag.get('retrieval_top1')} (Score={diag.get('retrieval_score1')})")
                
        else:
            print(f"[Error] Server returned {resp.status_code}")
            print(resp.text)

    except requests.exceptions.ConnectionError:
        print(f"[Error] Could not connect to server at {url}. Is it running?")
    except Exception as e:
        print(f"[Error] Unexpected error: {e}")

if __name__ == "__main__":
    main()