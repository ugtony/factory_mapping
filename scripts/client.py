import requests
import json

# 設定測試參數 (對齊 offline 的設定)
url = "http://localhost:8000/localize"
img_path = "data/brazil360/query/frames-000021_F.jpg"
fov = 100

print(f"Testing image: {img_path} (FOV={fov})")

try:
    with open(img_path, 'rb') as f:
        # 發送 Request
        resp = requests.post(
            url, 
            files={'file': f}, 
            data={'fov': fov}
        )
    
    # 解析結果
    if resp.status_code == 200:
        res = resp.json()
        print("\n=== Server Response ===")
        print(json.dumps(res, indent=2))
        
        # 驗證重點
        print("\n=== Verification ===")
        print(f"Block:   {res.get('block')} (Expected: brazil360)")
        print(f"Inliers: {res.get('inliers')} (Expected: ~694)")
    else:
        print(f"Error: {resp.status_code} - {resp.text}")

except FileNotFoundError:
    print(f"Error: File not found: {img_path}")
except Exception as e:
    print(f"Connection Failed: {e}")