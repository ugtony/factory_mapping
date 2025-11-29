# scripts/server.py
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
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

@app.post("/localize", response_model=LocalizeResponse)
async def localize_endpoint(file: UploadFile = File(...), fov: float = Form(100.0)):
    t0 = time.time()
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: raise HTTPException(status_code=400, detail="Invalid image")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    result = engine.localize(img, fov_deg=fov, return_details=False)
    dt = (time.time() - t0) * 1000
    
    if not result['success']:
        return {"status": "failed", "latency_ms": dt}
    
    q, t = result['pose']['qvec'], result['pose']['tvec']
    trans = result['transform']
    
    R_w2c = Rotation.from_quat(q).as_matrix()
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
        "latency_ms": dt
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)