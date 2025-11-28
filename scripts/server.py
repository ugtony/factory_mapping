import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import numpy as np
import cv2
import torch
import h5py
import pycolmap
import json
import time
from pathlib import Path
from scipy.spatial.transform import Rotation

# HLOC Imports
from hloc import extractors, matchers, extract_features, match_features
from hloc.utils.base_model import dynamic_load

# [New] Import shared logic
# 確保 scripts/ 目錄在 PYTHONPATH 中，或者與此檔案同目錄
try:
    from map_utils import compute_sim2_transform
except ImportError:
    # Fallback for relative import if running as module
    from .map_utils import compute_sim2_transform

class LocalizationEngine:
    def __init__(self, project_root: Path, config_path: Path, anchors_path: Path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[Init] Using device: {self.device}")
        
        # 1. Load Configuration
        self.config = {}
        if config_path.exists():
            with open(config_path) as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        k, v = line.strip().split('=', 1)
                        self.config[k.strip()] = v.strip().strip('"').strip("'")
        
        self.global_conf_name = self.config.get("GLOBAL_CONF", "netvlad")
        self.default_fov = float(self.config.get("FOV", 69.4))
        
        # 2. Load Models
        print("[Init] Loading Neural Network Models...")
        local_conf = extract_features.confs['superpoint_aachen']
        ModelLocal = dynamic_load(extractors, local_conf['model']['name'])
        self.model_extract_local = ModelLocal(local_conf['model']).eval().to(self.device)
        
        global_conf = extract_features.confs[self.global_conf_name]
        ModelGlobal = dynamic_load(extractors, global_conf['model']['name'])
        self.model_extract_global = ModelGlobal(global_conf['model']).eval().to(self.device)
        
        matcher_conf = match_features.confs['superpoint+lightglue']
        ModelMatcher = dynamic_load(matchers, matcher_conf['model']['name'])
        self.model_matcher = ModelMatcher(matcher_conf['model']).eval().to(self.device)
        
        # 3. Load Database & Anchors
        self.blocks = {}
        self._load_blocks(project_root / "outputs-hloc", anchors_path)

    def _load_blocks(self, outputs_root, anchors_path):
        anchors = {}
        if anchors_path.exists():
            with open(anchors_path, 'r') as f: anchors = json.load(f)
            print(f"[Init] Loaded anchors for: {list(anchors.keys())}")

        for block_dir in outputs_root.iterdir():
            if not block_dir.is_dir(): continue
            
            # Locate SfM model
            sfm_dir = block_dir / "sfm_aligned"
            if not (sfm_dir / "images.bin").exists(): sfm_dir = block_dir / "sfm"
            
            # Locate Feature Files
            global_h5 = block_dir / f"global-{self.global_conf_name}.h5"
            local_h5_path = block_dir / "local-superpoint_aachen.h5"
            
            if not (sfm_dir/"images.bin").exists() or not global_h5.exists() or not local_h5_path.exists():
                continue

            print(f"[Init] Loading Block: {block_dir.name}")
            
            # Load Global Descriptors
            g_names = []
            g_vecs = []
            with h5py.File(global_h5, 'r') as f:
                def visit(name, obj):
                    if isinstance(obj, h5py.Group) and 'global_descriptor' in obj:
                        g_names.append(name)
                        g_vecs.append(obj['global_descriptor'].__array__())
                f.visititems(visit)
            g_vecs = torch.from_numpy(np.array(g_vecs)).to(self.device).squeeze()
            if g_vecs.ndim == 1: g_vecs = g_vecs.unsqueeze(0)

            # Load Reconstruction
            recon = pycolmap.Reconstruction(sfm_dir)
            name_to_id = {img.name: img_id for img_id, img in recon.images.items()}
            
            # Compute Transform if anchors exist
            transform = None
            if block_dir.name in anchors:
                # [Updated] Use modularized utility
                transform = compute_sim2_transform(recon, anchors[block_dir.name])
                
                if transform:
                    s = transform['s']
                    theta_deg = np.degrees(transform['theta'])
                    f_s, f_e = transform['frames']
                    print(f"    > Aligned via {f_s} -> {f_e}")
                    print(f"    > Scale={s:.4f}, Rot={theta_deg:.2f}°")
                else:
                    print(f"[Warn] Failed to align block {block_dir.name} (anchors not found?)")

            self.blocks[block_dir.name] = {
                'recon': recon,
                'name_to_id': name_to_id,
                'global_names': g_names,
                'global_vecs': g_vecs,
                'local_h5': h5py.File(local_h5_path, 'r'),
                'transform': transform
            }

    @torch.no_grad()
    def localize(self, image_arr: np.ndarray, fov_deg: float = None):
        if fov_deg is None: fov_deg = self.default_fov
        
        # --- Step 1: Preprocessing ---
        h_orig, w_orig = image_arr.shape[:2]
        image_tensor = image_arr
        scale = 1.0
        
        # Resize if too large (Standard HLOC Preprocessing)
        resize_max = 1024
        if max(h_orig, w_orig) > resize_max:
            scale = resize_max / max(h_orig, w_orig)
            new_w, new_h = int(w_orig * scale), int(h_orig * scale)
            image_tensor = cv2.resize(image_arr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Prepare Tensors
        img_t = torch.from_numpy(image_tensor.transpose(2, 0, 1)).float().div(255.).unsqueeze(0).to(self.device)
        img_g = torch.from_numpy(cv2.cvtColor(image_tensor, cv2.COLOR_RGB2GRAY)).float().div(255.).unsqueeze(0).unsqueeze(0).to(self.device)

        # --- Step 2: Feature Extraction ---
        q_global = self.model_extract_global({'image': img_t})['global_descriptor']
        q_local = self.model_extract_local({'image': img_g})
        kpts = q_local['keypoints'][0]
        desc = q_local['descriptors'][0]
        
        # Restore coordinates to original image size
        if scale != 1.0:
            kpts = (kpts + 0.5) / scale - 0.5
        
        # --- Step 3: Camera Intrinsics ---
        fov_rad = np.deg2rad(fov_deg)
        f = 0.5 * max(w_orig, h_orig) / np.tan(fov_rad / 2.0)
        camera = pycolmap.Camera(
            model='SIMPLE_PINHOLE', 
            width=w_orig, 
            height=h_orig, 
            params=np.array([f, w_orig/2.0, h_orig/2.0], dtype=np.float64)
        )

        best_result = None

        # --- Step 4: Search in Blocks ---
        for name, block in self.blocks.items():
            # Global Retrieval
            sim = torch.matmul(q_global, block['global_vecs'].t())
            scores, indices = torch.topk(sim, k=1) # Check Top-1 candidate
            
            if scores[0].item() < 0.01: continue # Skip if low similarity

            top1_name = block['global_names'][int(indices[0])]
            db_name = top1_name
            
            if db_name not in block['local_h5']: continue
            if db_name not in block['name_to_id']: continue
            
            # Get DB Image Info
            img_obj = block['recon'].images[block['name_to_id'][db_name]]
            cam_db = block['recon'].cameras[img_obj.camera_id]
            
            # Load DB Features
            grp = block['local_h5'][db_name]
            kpts_db = torch.from_numpy(grp['keypoints'].__array__()).float().to(self.device)
            desc_db = torch.from_numpy(grp['descriptors'].__array__()).float().to(self.device)
            if desc_db.shape[0] != 256 and desc_db.shape[1] == 256: desc_db = desc_db.T

            # Local Matching (LightGlue)
            data = {
                'image0': torch.empty((1,1,h_orig,w_orig), device=self.device),
                'keypoints0': kpts.unsqueeze(0), 'descriptors0': desc.unsqueeze(0),
                'image1': torch.empty((1,1,cam_db.height,cam_db.width), device=self.device),
                'keypoints1': kpts_db.unsqueeze(0), 'descriptors1': desc_db.unsqueeze(0)
            }
            matches = self.model_matcher(data)['matches0'][0]
            valid = matches > -1
            
            if valid.sum().item() < 10: continue

            # --- Step 5: PnP Solver ---
            try:
                # Get 3D Points
                p3d_ids = np.array([p.point3D_id if p.has_point3D() else -1 for p in img_obj.points2D])
                
                m_q = torch.where(valid)[0].cpu().numpy()
                m_db = matches[valid].cpu().numpy()
                
                # Filter matches that have 3D points
                valid_3d = m_db < len(p3d_ids)
                m_q = m_q[valid_3d]
                m_db = m_db[valid_3d]
                
                target_ids = p3d_ids[m_db]
                has_3d = target_ids != -1
                
                if has_3d.sum() < 6: continue

                # Prepare PnP Input
                p2d = kpts.cpu().numpy()[m_q[has_3d]].astype(np.float64)
                p2d += 0.5 # Add 0.5 pixel offset for COLMAP
                p3d = np.array([block['recon'].points3D[i].xyz for i in target_ids[has_3d]], dtype=np.float64)
                
                # 1. Estimate & Refine
                ret = pycolmap.estimate_and_refine_absolute_pose(
                    p2d, p3d, camera, 
                    estimation_options={'ransac': {'max_error': 12.0}}
                )
                
                # Helper to handle different pycolmap versions
                def parse_ret(r):
                    if r is None: return False, None, None, 0
                    s, q, t, n = False, None, None, 0
                    
                    if isinstance(r, dict): # Old version
                        s, n = r.get('success', False), r.get('num_inliers', 0)
                        if 'qvec' in r: q, t = r['qvec'], r['tvec']
                        elif 'cam_from_world' in r:
                            obj = r['cam_from_world']
                            q, t = obj.rotation.quat, obj.translation
                            
                    elif hasattr(r, 'success'): # New version
                        s, n = r.success, r.num_inliers
                        if hasattr(r, 'cam_from_world') and r.cam_from_world is not None:
                            q, t = r.cam_from_world.rotation.quat, r.cam_from_world.translation
                    return s, q, t, n

                success, qvec, tvec, num_inliers = parse_ret(ret)

                # 2. Fallback: RANSAC Only (if refinement failed)
                if not success:
                     ret = pycolmap.estimate_absolute_pose(
                         p2d, p3d, camera, 
                         estimation_options={'ransac': {'max_error': 12.0}}
                     )
                     success, qvec, tvec, num_inliers = parse_ret(ret)
                     if qvec is not None: success = True # Accept RANSAC result

                if success and qvec is not None:
                    print(f"[Result] Block: {name}, Inliers: {num_inliers}")
                    best_result = {
                        'block': name, 
                        'pose': {'qvec': qvec, 'tvec': tvec}, 
                        'transform': block['transform'], 
                        'inliers': num_inliers
                    }
                    break # Early exit if successful

            except Exception as e:
                print(f"[Error] PnP failed for {name}: {e}")
                continue

        return best_result

# ------------------------------------------------------------------------------
# FastAPI Application
# ------------------------------------------------------------------------------
app = FastAPI(title="Visual Localization Service")
engine: LocalizationEngine = None

@app.on_event("startup")
def startup_event():
    global engine
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
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
async def localize_endpoint(
    file: UploadFile = File(...), 
    fov: float = Form(100.0)
):
    t0 = time.time()
    
    # Read Image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: 
        raise HTTPException(status_code=400, detail="Invalid image")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Run Localization
    result = engine.localize(img, fov_deg=fov)
    dt = (time.time() - t0) * 1000
    
    if not result:
        return {"status": "failed", "latency_ms": dt}
    
    q, t = result['pose']['qvec'], result['pose']['tvec']
    trans = result['transform']
    
    # Compute Camera Center in SfM coordinates
    R_w2c = Rotation.from_quat(q).as_matrix()
    R_c2w = R_w2c.T
    cam_center_sfm = -R_c2w @ t
    
    # Compute Yaw
    view_dir = R_c2w[:, 2]
    sfm_yaw = np.degrees(np.arctan2(view_dir[1], view_dir[0]))
    
    # Transform to Map Coordinates (Sim2)
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
        "latency_ms": dt
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)