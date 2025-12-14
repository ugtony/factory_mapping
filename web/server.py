#!/usr/bin/env python3
"""
server.py
用途：Visual Localization Service (定位後端 API)
功能：
  - 接收影像與參數 (FOV, Block Filter)
  - 執行定位 (Global Retrieval + Local Matching + PnP)
  - 回傳定位結果與詳細診斷資訊 (格式對齊 diagnosis_report.csv)
"""

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any
from pydantic import BaseModel
import numpy as np
import cv2
import time
import sys
from pathlib import Path
from scipy.spatial.transform import Rotation

# [設定路徑]
# 取得目前檔案所在的目錄 (例如 web/ 或 scripts/)
current_dir = Path(__file__).resolve().parent
# 取得專案根目錄 (假設在上一層)
project_root = current_dir.parent
# 將專案根目錄加入 Python path 以便匯入 lib
sys.path.append(str(project_root))

try:
    from lib.localization_engine import LocalizationEngine
    from lib.map_utils import colmap_to_scipy_quat
except ImportError as e:
    print(f"[Error] Import failed: {e}")
    print("Please make sure you are running this script correctly (e.g., from project root).")
    sys.exit(1)

# ==================== FastAPI 設定 ====================
app = FastAPI(title="Visual Localization Service")

# 允許跨域請求 (方便前端開發)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全域 Engine 變數
engine: LocalizationEngine = None

@app.on_event("startup")
def startup_event():
    global engine
    print("=== Starting Localization Engine ===")
    # 初始化引擎，載入模型與錨點設定
    engine = LocalizationEngine(
        project_root=project_root, 
        config_path=project_root / "project_config.env",
        anchors_path=project_root / "anchors.json"
    )
    print("✅ Server Ready!")

# ==================== Response Model ====================
class LocalizeResponse(BaseModel):
    status: str
    block: str = None
    inliers: int = 0
    map_x: float = None
    map_y: float = None
    map_yaw: float = None
    latency_ms: float = 0.0
    # 診斷資訊 (格式化後的字典)
    diagnosis: Optional[Dict[str, Any]] = None

# ==================== API Endpoint ====================
@app.post("/localize", response_model=LocalizeResponse)
async def localize_endpoint(
    file: UploadFile = File(...), 
    fov: Optional[float] = Form(None),
    block_filter: Optional[str] = Form(None)
):
    """
    定位 API
    - file: 上傳的影像檔案
    - fov: (選填) 相機視角
    - block_filter: (選填) 指定搜尋區域，逗號分隔字串 (e.g., "blockA,blockB")
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready")
    
    t0 = time.time()
    
    # 1. 讀取影像
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: 
            raise ValueError("Empty image")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    # 2. 解析 Block Filter
    filter_list = None
    if block_filter:
        # 將 "brazil360, miami360" 轉為 ['brazil360', 'miami360']
        filter_list = [b.strip() for b in block_filter.split(',') if b.strip()]

    # 3. 呼叫定位引擎
    # 注意：這裡假設 localization_engine.py 已經更新支援 block_filter 參數
    result = engine.localize(
        img, 
        fov_deg=fov, 
        return_details=False,
        block_filter=filter_list
    )
    
    dt = (time.time() - t0) * 1000
    
    # 4. 格式化診斷資訊 (使用 Engine 內建方法，確保與 CSV 報表一致)
    raw_diag = result.get('diagnosis', {})
    formatted_diag = {}
    
    if hasattr(engine, 'format_diagnosis'):
        formatted_diag = engine.format_diagnosis(raw_diag)
    else:
        # Fallback: 如果 Engine 還沒更新，直接回傳原始資料以免報錯
        print("[Warn] engine.format_diagnosis not found. Please update localization_engine.py")
        formatted_diag = raw_diag

    # 5. 處理失敗情況
    if not result['success']:
        return {
            "status": "failed", 
            "latency_ms": dt,
            "diagnosis": formatted_diag
        }
    
    # 6. 處理成功情況 (座標轉換)
    q, t = result['pose']['qvec'], result['pose']['tvec']
    trans = result['transform']
    
    # SfM 座標轉 World 座標
    q_scipy = colmap_to_scipy_quat(q)
    R_w2c = Rotation.from_quat(q_scipy).as_matrix()
    R_c2w = R_w2c.T
    cam_center_sfm = -R_c2w @ t
    view_dir = R_c2w[:, 2]
    sfm_yaw = np.degrees(np.arctan2(view_dir[1], view_dir[0]))
    
    # 轉換到 Map 座標 (若有 anchors 設定)
    if trans:
        p_map = trans['s'] * (trans['R'] @ cam_center_sfm[:2]) + trans['t']
        map_yaw = sfm_yaw + np.degrees(trans['theta'])
        map_yaw = (map_yaw + 180) % 360 - 180
    else:
        p_map = cam_center_sfm[:2]
        map_yaw = sfm_yaw

    return {
        "status": "success", 
        "block": result['block'], 
        "inliers": result['inliers'],
        "map_x": float(p_map[0]), 
        "map_y": float(p_map[1]), 
        "map_yaw": float(map_yaw),
        "latency_ms": dt,
        "diagnosis": formatted_diag
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    
    print(f"Starting Localization Server on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)