# scripts/server.py (或 web/server.py)

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import Optional, Dict, Any
from pydantic import BaseModel
import numpy as np
import cv2
import time
import sys
from pathlib import Path
from scipy.spatial.transform import Rotation

# [Plan A] Setup path to find 'lib'
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from lib.localization_engine import LocalizationEngine
from lib.map_utils import colmap_to_scipy_quat

app = FastAPI(title="Visual Localization Service")
engine: LocalizationEngine = None

@app.on_event("startup")
def startup_event():
    global engine
    engine = LocalizationEngine(
        project_root=project_root, 
        config_path=project_root / "project_config.env",
        anchors_path=project_root / "anchors.json"
    )
    print("✅ Server Ready!")

class LocalizeResponse(BaseModel):
    status: str
    block: str = None
    inliers: int = 0
    map_x: float = None
    map_y: float = None
    map_yaw: float = None
    latency_ms: float = 0.0
    # [New] 新增診斷欄位 (允許任意字典內容)
    diagnosis: Optional[Dict[str, Any]] = None

# [Modified] 新增 block_filter 參數 (Form Data)
@app.post("/localize", response_model=LocalizeResponse)
async def localize_endpoint(
    file: UploadFile = File(...), 
    fov: Optional[float] = Form(None),
    block_filter: Optional[str] = Form(None)
):
    t0 = time.time()
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: raise HTTPException(status_code=400, detail="Invalid image")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # [New] 解析逗號分隔的 block_filter 字串
    filter_list = None
    if block_filter:
        filter_list = [b.strip() for b in block_filter.split(',') if b.strip()]
        # print(f"[Server] Applying block filter: {filter_list}")

    # 呼叫 Engine，傳入 block_filter
    result = engine.localize(
        img, 
        fov_deg=fov, 
        return_details=False,
        block_filter=filter_list
    )
    
    dt = (time.time() - t0) * 1000
    
    # 提取診斷資訊 (若無則為空字典)
    diag_data = result.get('diagnosis', {})

    if not result['success']:
        return {
            "status": "failed", 
            "latency_ms": dt,
            "diagnosis": diag_data
        }
    
    q, t = result['pose']['qvec'], result['pose']['tvec']
    trans = result['transform']
    
    # 座標轉換
    q_scipy = colmap_to_scipy_quat(q)
    R_w2c = Rotation.from_quat(q_scipy).as_matrix()
    R_c2w = R_w2c.T
    cam_center_sfm = -R_c2w @ t
    view_dir = R_c2w[:, 2]
    sfm_yaw = np.degrees(np.arctan2(view_dir[1], view_dir[0]))
    
    if trans:
        p_map = trans['s'] * (trans['R'] @ cam_center_sfm[:2]) + trans['t']
        map_yaw = sfm_yaw + np.degrees(trans['theta'])
        map_yaw = (map_yaw + 180) % 360 - 180
    else:
        p_map = cam_center_sfm[:2]
        map_yaw = sfm_yaw

    return {
        "status": "success", "block": result['block'], "inliers": result['inliers'],
        "map_x": float(p_map[0]), "map_y": float(p_map[1]), "map_yaw": float(map_yaw),
        "latency_ms": dt,
        "diagnosis": diag_data
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)