#!/usr/bin/env python3
"""
web_visualizer.py
Updated: 
1. Relaxed "Smooth" mode: considers distance within local radius to allow slight rotation.
2. Dual Move Logic: 'smooth' (Single Click) vs 'nearest' (Double Click).
3. Auto-detection of anchors.json path & Consistency checks.
"""
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import argparse
import sys
import json
import numpy as np
import pycolmap
from pathlib import Path
from pydantic import BaseModel

# 添加專案根目錄以匯入 lib
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

try:
    from lib.map_utils import compute_sim2_transform, colmap_to_scipy_quat
except ImportError:
    print("[Error] Cannot import lib.map_utils.")
    sys.exit(1)

# ==================== 核心邏輯 ====================

def normalize_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def angle_diff(a1, a2):
    diff = np.abs(a1 - a2)
    return np.min([diff, 2*np.pi - diff])

def get_camera_pose_raw(img):
    if hasattr(img, "cam_from_world"):
        pose = img.cam_from_world
        if callable(pose): pose = pose()
        R_w2c = pose.rotation.matrix()
        center = img.projection_center()
    else:
        q_scipy = colmap_to_scipy_quat(img.qvec)
        from scipy.spatial.transform import Rotation
        R_w2c = Rotation.from_quat(q_scipy).as_matrix()
        center = img.projection_center()
    
    R_c2w = R_w2c.T
    view_dir = R_c2w[:, 2] 
    yaw = np.arctan2(view_dir[1], view_dir[0])
    return center, yaw

def apply_sim2(pt_sfm, yaw_sfm, trans):
    s = trans['s']
    theta = trans['theta']
    t = trans['t']
    R = trans['R']
    p_sfm_2d = np.array(pt_sfm[:2])
    p_map = s * (R @ p_sfm_2d) + t
    yaw_map = normalize_angle(yaw_sfm + theta)
    return p_map, yaw_map

# ==================== 全域資料 ====================
app = FastAPI()
all_cameras = []
scene_points = []
block_labels = []
config = {
    "visual_scale": 1.0,
    "rotation_radius": 0.5,
    "map_bounds": [0, 1, 0, 1]
}

# ==================== API 模型 ====================
class MoveRequest(BaseModel):
    target_x: float
    target_y: float
    current_idx: int
    mode: str = "smooth"  # "smooth" (Single Click) or "nearest" (Double Click)

class RotateRequest(BaseModel):
    anchor_idx: int
    mouse_x: float
    mouse_y: float

# ==================== 初始化載入 ====================
def load_data(anchors_path: Path, image_root: Path):
    global all_cameras, scene_points, config, block_labels
    
    if not anchors_path.exists():
        print(f"[Error] Anchors file not found: {anchors_path}")
        return

    with open(anchors_path, 'r') as f:
        anchors_cfg = json.load(f)

    # --- Check 1: Disk -> Config Consistency ---
    if image_root.exists():
        disk_dirs = {p.name for p in image_root.iterdir() if p.is_dir() and not p.name.startswith(".")}
        config_keys = set(anchors_cfg.keys())
        unconfigured = disk_dirs - config_keys
        if unconfigured:
            print(f"\n[Warning] Found {len(unconfigured)} folders in '{image_root}' NOT in anchors.json:")
            for b in sorted(unconfigured):
                print(f"  - {b} (Unconfigured)")
            print("-" * 50)

    temp_points = []
    all_cameras = []
    all_valid_steps = []
    block_sums = {} 

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    print(f"[Init] Loading {len(anchors_cfg)} blocks from config...")

    for i, (block_name, cfg) in enumerate(anchors_cfg.items()):
        sfm_path = Path(cfg['sfm_path'])
        if not sfm_path.exists(): sfm_path = project_root / cfg['sfm_path']
        
        # --- Check 2: Config -> Disk Consistency ---
        if not (sfm_path / "images.bin").exists():
            print(f"[Warning] Block '{block_name}' in anchors but MISSING model data at: {sfm_path}")
            continue

        recon = pycolmap.Reconstruction(sfm_path)
        trans = compute_sim2_transform(recon, cfg)
        
        if trans is None: 
            print(f"[Warning] Block '{block_name}' failed to compute Sim2 transform.")
            continue
            
        hex_color = colors[i % len(colors)]
        block_sums[block_name] = {'x': 0.0, 'y': 0.0, 'count': 0}

        # Points
        p3d_values = recon.points3D.values() if hasattr(recon, "points3D") and hasattr(recon.points3D, "values") else recon.points3D
        for j, p3d in enumerate(p3d_values):
            if j % 10 != 0: continue
            p_map, _ = apply_sim2(p3d.xyz, 0, trans)
            temp_points.append({"x": float(p_map[0]), "y": float(p_map[1])})

        # Cameras
        img_values = recon.images.values() if hasattr(recon, "images") and hasattr(recon.images, "values") else recon.images
        sorted_imgs = sorted([img for img in img_values if img.has_pose], key=lambda x: x.name)
        
        real_img_path = None
        candidates = [
            image_root / block_name / "_images_stage",
            image_root / block_name / "db",
            project_root / "data" / block_name / "db"
        ]
        for p in candidates:
            if p.exists():
                real_img_path = p
                break
        
        block_cams = []
        for img in sorted_imgs:
            center_sfm, yaw_sfm = get_camera_pose_raw(img)
            center_map, yaw_map = apply_sim2(center_sfm, yaw_sfm, trans)
            
            cam_obj = {
                'id': len(all_cameras),
                'name': img.name,
                'block': block_name,
                'file_path': str(real_img_path / img.name) if real_img_path else "",
                'x': float(center_map[0]),
                'y': float(center_map[1]),
                'yaw': float(yaw_map),
                'color': hex_color
            }
            all_cameras.append(cam_obj)
            block_cams.append(np.array(center_map[:2]))
            
            block_sums[block_name]['x'] += float(center_map[0])
            block_sums[block_name]['y'] += float(center_map[1])
            block_sums[block_name]['count'] += 1

        if len(block_cams) > 1:
            names = [img.name for img in sorted_imgs]
            f_indices = [k for k, n in enumerate(names) if "_F" in n]
            calc_centers = np.array(block_cams) if len(f_indices) < 2 else np.array([block_cams[k] for k in f_indices])
            dists = np.linalg.norm(calc_centers[1:] - calc_centers[:-1], axis=1)
            all_valid_steps.extend(dists[dists > 0.05])

    block_labels = []
    for name, data in block_sums.items():
        if data['count'] > 0:
            block_labels.append({
                "name": name,
                "x": data['x'] / data['count'],
                "y": data['y'] / data['count']
            })

    if all_valid_steps:
        median = np.median(all_valid_steps)
        config["visual_scale"] = float(median)
        config["rotation_radius"] = float(median * 0.6)
        print(f"[Init] Auto-Scale: Step={median:.3f}m")
    
    xs = [c['x'] for c in all_cameras]
    ys = [c['y'] for c in all_cameras]
    if xs:
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x, span_y = max_x - min_x, max_y - min_y
        max_span = max(span_x, span_y)
        if max_span < 1.0: max_span = 10.0
        
        cx, cy = (min_x + max_x)/2, (min_y + max_y)/2
        half = (max_span * 1.1) / 2
        config["map_bounds"] = [cx - half, cx + half, cy - half, cy + half]
    
    scene_points = temp_points
    print(f"[Init] Ready. Loaded {len(all_cameras)} cameras.")

# ==================== Endpoints ====================

@app.get("/api/init_data")
async def get_init_data():
    return {
        "config": config,
        "cameras": all_cameras,
        "points_count": len(scene_points),
        "block_labels": block_labels
    }

@app.get("/api/scene_points")
async def get_scene_points():
    return scene_points

@app.get("/api/image/{cam_id}")
async def get_image(cam_id: int):
    if cam_id < 0 or cam_id >= len(all_cameras):
        return JSONResponse(status_code=404, content={"error": "Camera not found"})
    
    path_str = all_cameras[cam_id]['file_path']
    path = Path(path_str)
    
    if not path_str:
        return JSONResponse(status_code=404, content={"error": "Image path empty"})

    if path.exists():
        return FileResponse(path)
    else:
        return JSONResponse(status_code=404, content={"error": "Image file missing"})

@app.post("/api/action/move")
async def action_move(req: MoveRequest):
    if not all_cameras: return {}
    
    centers = np.array([[c['x'], c['y']] for c in all_cameras])
    target_arr = np.array([req.target_x, req.target_y])
    dists = np.linalg.norm(centers - target_arr, axis=1)

    # 模式 A: 雙擊 = 絕對最近 (Nearest, Hit-Box Logic)
    if req.mode == "nearest":
        nearest_idx = np.argmin(dists)
        return all_cameras[nearest_idx]

    # 模式 B: 單點 = 平滑導航 (Smooth with Relaxed Angle)
    current_yaw = all_cameras[req.current_idx]['yaw']
    
    K = 15
    nearest_indices = np.argsort(dists)[:K]
    best_idx = -1
    min_score = float('inf')

    for idx in nearest_indices:
        idx = int(idx)
        cam = all_cameras[idx]
        
        # --- 權重設定 ---
        # 1. 角度差異 (Base cost)
        a_dist = angle_diff(cam['yaw'], current_yaw)
        
        # 2. 距離權重 (0.2 means 1 meter distance cost ~= 11.5 degrees rotation cost)
        # 這讓近處的照片比較容易被選中，即使角度稍微不正
        dist_cost = dists[idx] * 0.2
        
        # 3. 過遠懲罰 (避免單點一下跳去地圖另一端)
        penalty = 0.0 if dists[idx] <= 6.0 else 10.0
        
        score = a_dist + dist_cost + penalty
        
        if score < min_score:
            min_score = score
            best_idx = idx
            
    return all_cameras[best_idx]

@app.post("/api/action/rotate")
async def action_rotate(req: RotateRequest):
    anchor_cam = all_cameras[req.anchor_idx]
    dx = req.mouse_x - anchor_cam['x']
    dy = req.mouse_y - anchor_cam['y']
    target_yaw = np.arctan2(dy, dx)
    
    search_radius = config["rotation_radius"]
    best_idx = req.anchor_idx
    min_angle_dist = float('inf')
    
    for i, cam in enumerate(all_cameras):
        if cam['block'] != anchor_cam['block']: continue
        dist = np.hypot(cam['x'] - anchor_cam['x'], cam['y'] - anchor_cam['y'])
        if dist <= search_radius:
            a_dist = angle_diff(cam['yaw'], target_yaw)
            if a_dist < min_angle_dist:
                min_angle_dist = a_dist
                best_idx = i
                
    return {
        "best_camera": all_cameras[best_idx],
        "target_yaw": target_yaw
    }

# ==================== Main ====================
@app.get("/")
async def get_index():
    template_path = Path("web/templates/index.html")
    if not template_path.exists():
         template_path = Path("templates/index.html")
    
    if template_path.exists():
        return HTMLResponse(template_path.read_text(encoding='utf-8'))
    return HTMLResponse("<h1>Error: Template not found</h1><p>Please create web/templates/index.html</p>")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--anchors", type=Path, default=None)
    parser.add_argument("--image_root", type=Path, default="outputs-hloc")
    args = parser.parse_args()

    if args.anchors is None:
        args.anchors = args.image_root / "anchors.json"

    if not args.anchors.exists():
        print(f"[Warning] Anchors file not found at: {args.anchors}")
        print(f"          Please ensure anchors.json exists in {args.image_root} or specify --anchors.")

    load_data(args.anchors, args.image_root)
    
    print(f"Starting Web Visualizer on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)