# server.py (V5 - Robust Pose Parsing)
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import numpy as np
import cv2
import torch
import h5py
import pycolmap
import json
import io
from pathlib import Path
from scipy.spatial.transform import Rotation

# HLOC Imports
from hloc import extractors, matchers
from hloc import extract_features, match_features
from hloc.utils.base_model import dynamic_load

# ------------------------------------------------------------------------------
# 1. 核心定位引擎
# ------------------------------------------------------------------------------
class LocalizationEngine:
    def __init__(self, project_root: Path, config_path: Path, anchors_path: Path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[Init] Using device: {self.device}")
        
        self.config = self._load_env(config_path)
        self.global_conf_name = self.config.get("GLOBAL_CONF", "netvlad")
        self.default_fov = float(self.config.get("FOV", 69.4))
        
        print("[Init] Loading Neural Network Models...")
        
        # Local Features (SuperPoint)
        local_conf = extract_features.confs['superpoint_aachen']
        ModelLocal = dynamic_load(extractors, local_conf['model']['name'])
        self.model_extract_local = ModelLocal(local_conf['model']).eval().to(self.device)
        
        # Global Features (NetVLAD / MegaLoc)
        global_conf = extract_features.confs[self.global_conf_name]
        ModelGlobal = dynamic_load(extractors, global_conf['model']['name'])
        self.model_extract_global = ModelGlobal(global_conf['model']).eval().to(self.device)
        
        # Matcher (LightGlue)
        matcher_conf = match_features.confs['superpoint+lightglue']
        ModelMatcher = dynamic_load(matchers, matcher_conf['model']['name'])
        self.model_matcher = ModelMatcher(matcher_conf['model']).eval().to(self.device)
        
        self.blocks = {} 
        self._load_blocks(project_root / "outputs-hloc", anchors_path)

    def _load_env(self, path):
        cfg = {}
        if path.exists():
            with open(path) as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        k, v = line.strip().split('=', 1)
                        cfg[k.strip()] = v.strip().strip('"').strip("'")
        return cfg

    def _compute_sim2(self, sfm_path, anchor_cfg):
        try:
            recon = pycolmap.Reconstruction(sfm_path)
            def get_center(name):
                for img in recon.images.values():
                    if img.name == name or img.name.endswith(f"/{name}"):
                        return img.projection_center()[:2]
                return None
            p_sfm_s = get_center(anchor_cfg['start_frame'])
            p_sfm_e = get_center(anchor_cfg['end_frame'])
            if p_sfm_s is None or p_sfm_e is None: return None
            p_map_s = np.array(anchor_cfg['start_map_xy'])
            p_map_e = np.array(anchor_cfg['end_map_xy'])
            vec_sfm = p_sfm_e - p_sfm_s
            vec_map = p_map_e - p_map_s
            s = np.linalg.norm(vec_map) / (np.linalg.norm(vec_sfm) + 1e-6)
            theta = np.arctan2(vec_map[1], vec_map[0]) - np.arctan2(vec_sfm[1], vec_sfm[0])
            c, si = np.cos(theta), np.sin(theta)
            R = np.array([[c, -si], [si, c]])
            t = p_map_s - s * (R @ p_sfm_s)
            return {'s': s, 'theta': theta, 't': t, 'R': R}
        except Exception: return None

    def _load_blocks(self, outputs_root, anchors_path):
        if not anchors_path.exists(): anchors = {}
        else:
            with open(anchors_path, 'r') as f: anchors = json.load(f)

        for block_dir in outputs_root.iterdir():
            if not block_dir.is_dir(): continue
            sfm_dir = block_dir / "sfm_aligned"
            if not (sfm_dir / "images.bin").exists(): sfm_dir = block_dir / "sfm"
            global_h5 = block_dir / f"global-{self.global_conf_name}.h5"
            local_h5_path = block_dir / "local-superpoint_aachen.h5"
            
            if not (sfm_dir/"images.bin").exists() or not global_h5.exists() or not local_h5_path.exists():
                continue

            print(f"[Init] Loading Block: {block_dir.name}")
            g_names, g_vecs = self._read_global_h5(global_h5)
            recon = pycolmap.Reconstruction(sfm_dir)
            name_to_id = {img.name: img_id for img_id, img in recon.images.items()}
            transform = None
            if block_dir.name in anchors: transform = self._compute_sim2(sfm_dir, anchors[block_dir.name])
            local_h5_handle = h5py.File(local_h5_path, 'r')

            self.blocks[block_dir.name] = {
                'recon': recon,
                'name_to_id': name_to_id,
                'global_names': g_names,
                'global_vecs': torch.from_numpy(g_vecs).to(self.device),
                'local_h5': local_h5_handle,
                'transform': transform
            }

    def _read_global_h5(self, path):
        names, vecs = [], []
        with h5py.File(path, 'r') as f:
            def visit(name, obj):
                if isinstance(obj, h5py.Group) and 'global_descriptor' in obj:
                    names.append(name)
                    vecs.append(obj['global_descriptor'].__array__())
            f.visititems(visit)
        vecs = np.array(vecs)
        if vecs.ndim == 3: vecs = vecs.squeeze(1)
        if vecs.ndim == 1: vecs = vecs[np.newaxis, :]
        return names, vecs

    @torch.no_grad()
    def localize(self, image_arr: np.ndarray, fov_deg: float = None):
        if fov_deg is None: fov_deg = self.default_fov
        print(f"\n[Debug] === Start Localization (FOV={fov_deg}) ===")
        
        h_orig, w_orig = image_arr.shape[:2]
        
        # 1. Resize (Scale Consistency)
        resize_max = 1024
        scale = 1.0
        image_tensor_in = image_arr
        if max(h_orig, w_orig) > resize_max:
            scale = resize_max / max(h_orig, w_orig)
            new_w = int(w_orig * scale)
            new_h = int(h_orig * scale)
            image_tensor_in = cv2.resize(image_arr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            print(f"[Debug] Resized: {new_w}x{new_h} (Scale={scale:.4f})")

        # 2. Tensors
        img_tensor = torch.from_numpy(image_tensor_in.transpose(2, 0, 1)).float().div(255.).unsqueeze(0).to(self.device)
        img_gray = cv2.cvtColor(image_tensor_in, cv2.COLOR_RGB2GRAY)
        img_gray_tensor = torch.from_numpy(img_gray).float().div(255.).unsqueeze(0).unsqueeze(0).to(self.device)

        # 3. Features
        q_global = self.model_extract_global({'image': img_tensor})['global_descriptor']
        q_local = self.model_extract_local({'image': img_gray_tensor})
        kpts_q = q_local['keypoints'][0]
        desc_q = q_local['descriptors'][0]
        
        # Restore Coordinates & Offset
        if scale != 1.0:
            kpts_q = (kpts_q + 0.5) / scale - 0.5
        
        desc_q_batch = desc_q.unsqueeze(0)
        kpts_q_batch = kpts_q.unsqueeze(0)

        # 4. Camera
        fov_rad = np.deg2rad(fov_deg)
        f = 0.5 * max(w_orig, h_orig) / np.tan(fov_rad / 2.0)
        camera = pycolmap.Camera(
            model='SIMPLE_PINHOLE', width=w_orig, height=h_orig, 
            params=np.array([f, w_orig/2.0, h_orig/2.0], dtype=np.float64)
        )

        best_result = None
        best_inliers = -1

        # Helper Function to Parse PnP Result (Robust)
        def parse_pnp_result(res):
            if res is None: return False, 0, None
            
            success = False
            inliers = 0
            cam_from_world = None
            
            # Handle different pycolmap return types (Object vs Dict)
            if hasattr(res, 'success'): # Object
                success = res.success
                inliers = res.num_inliers
                if hasattr(res, 'cam_from_world'): cam_from_world = res.cam_from_world
            elif isinstance(res, dict): # Dict
                # Dict usually implies success in estimate_absolute_pose (returns None on fail)
                success = res.get('success', True) 
                inliers = res.get('num_inliers', 0)
                cam_from_world = res.get('cam_from_world')
                
                # Fallback for older versions returning flat dict
                if 'qvec' in res and cam_from_world is None:
                    return success, inliers, res

            # Extract qvec/tvec from Rigid3d object if present
            pose_dict = None
            if cam_from_world is not None:
                try:
                    pose_dict = {
                        'qvec': cam_from_world.rotation.quat,
                        'tvec': cam_from_world.translation,
                        'num_inliers': inliers
                    }
                except Exception as e:
                    print(f"    [Error] Parsing Rigid3d failed: {e}")
            
            return success, inliers, pose_dict

        for block_name, block_data in self.blocks.items():
            db_vecs = block_data['global_vecs']
            sim = torch.matmul(q_global, db_vecs.t())
            if sim.dim() == 2: sim = sim.squeeze(0)
            
            k = min(3, len(block_data['global_names']))
            if k == 0: continue
            scores, indices = torch.topk(sim, k=k)
            top_indices = indices.cpu().numpy()
            
            pnp_inliers_count = 0
            pose_ret = None
            recon = block_data['recon']
            name_to_id = block_data['name_to_id']
            local_h5 = block_data['local_h5']

            for idx in top_indices:
                db_name = block_data['global_names'][int(idx)]
                if db_name not in local_h5 or db_name not in name_to_id: continue
                
                img_id = name_to_id[db_name]
                img_obj = recon.images[img_id]
                cam_db = recon.cameras[img_obj.camera_id]
                w_db, h_db = cam_db.width, cam_db.height

                db_grp = local_h5[db_name]
                kpts_db = torch.from_numpy(db_grp['keypoints'].__array__()).float().to(self.device)
                desc_db = torch.from_numpy(db_grp['descriptors'].__array__()).float().to(self.device)
                if desc_db.shape[0] != 256 and desc_db.shape[1] == 256: desc_db = desc_db.transpose(0, 1)
                
                matcher_input = {
                    'keypoints0': kpts_q_batch, 'descriptors0': desc_q_batch,
                    'image0': torch.empty((1, 1, h_orig, w_orig), device=self.device),
                    'keypoints1': kpts_db.unsqueeze(0), 'descriptors1': desc_db.unsqueeze(0),
                    'image1': torch.empty((1, 1, h_db, w_db), device=self.device)
                }
                matches = self.model_matcher(matcher_input)
                matches0 = matches['matches0'][0]
                valid = matches0 > -1
                if valid.sum() < 10: continue
                
                try:
                    p3d_ids = np.array([p.point3D_id if p.has_point3D() else -1 for p in img_obj.points2D])
                    valid_3d_mask = matches0[valid].cpu().numpy() < len(p3d_ids)
                    
                    matched_q_indices = torch.where(valid)[0].cpu().numpy()[valid_3d_mask]
                    matched_db_indices = matches0[valid].cpu().numpy()[valid_3d_mask]
                    
                    target_p3d_ids = p3d_ids[matched_db_indices]
                    has_3d = target_p3d_ids != -1
                    if has_3d.sum() < 6: continue
                    
                    # Data Prep
                    p2d_q = kpts_q.cpu().numpy()[matched_q_indices[has_3d]].astype(np.float64)
                    p2d_q += 0.5 
                    
                    p3d_ids_final = target_p3d_ids[has_3d]
                    p3d_coords = np.array([recon.points3D[pid].xyz for pid in p3d_ids_final], dtype=np.float64)
                    
                    # 1. Try Refinement
                    ret = pycolmap.estimate_and_refine_absolute_pose(
                        p2d_q, p3d_coords, camera, 
                        estimation_options={'ransac': {'max_error': 12.0}}
                    )
                    success, n_inliers, pose_dict = parse_pnp_result(ret)
                    
                    # 2. Fallback RANSAC
                    if not success:
                        ret_ransac = pycolmap.estimate_absolute_pose(
                            p2d_q, p3d_coords, camera, 
                            estimation_options={'ransac': {'max_error': 12.0}}
                        )
                        s2, n2, p2 = parse_pnp_result(ret_ransac)
                        if n2 > 15 and p2 is not None:
                            print(f"    [PnP] Refine failed, but RANSAC found {n2} inliers. ACCEPTING.")
                            success, n_inliers, pose_dict = True, n2, p2
                    
                    if success and n_inliers > pnp_inliers_count and pose_dict is not None:
                        pnp_inliers_count = n_inliers
                        pose_ret = pose_dict

                except Exception as e:
                    print(f"    [Error] {e}")
                    pass

            if pnp_inliers_count > best_inliers:
                best_inliers = pnp_inliers_count
                best_result = {
                    'block': block_name,
                    'inliers': pnp_inliers_count,
                    'pose': pose_ret,
                    'transform': block_data['transform']
                }

        return best_result

# ------------------------------------------------------------------------------
# 2. FastAPI App
# ------------------------------------------------------------------------------
app = FastAPI(title="Factory Localization Server")
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
    fov: float = Form(None)
):
    import time
    t0 = time.time()
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: raise HTTPException(status_code=400, detail="Invalid image")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    result = engine.localize(img, fov_deg=fov)
    dt = (time.time() - t0) * 1000
    
    if not result or result['pose'] is None:
        print(f"[API Response] Failed. Result: {result}")
        return {"status": "failed", "latency_ms": dt}
    
    pose = result['pose']
    t = result['transform']
    
    # [Fix] 確保 qvec/tvec 存在
    if 'qvec' not in pose or 'tvec' not in pose:
        print(f"[API Response] Failed. Pose format error: {pose}")
        return {"status": "failed", "latency_ms": dt}

    qvec = pose['qvec']
    tvec = pose['tvec']
    
    R_w2c = Rotation.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()
    R_c2w = R_w2c.T
    cam_center_sfm = -R_c2w @ tvec
    view_dir = R_c2w[:, 2]
    sfm_yaw = np.degrees(np.arctan2(view_dir[1], view_dir[0]))
    
    if t:
        p_map = t['s'] * (t['R'] @ cam_center_sfm[:2]) + t['t']
        map_yaw = sfm_yaw + np.degrees(t['theta'])
        map_yaw = (map_yaw + 180) % 360 - 180
    else:
        p_map = cam_center_sfm[:2]
        map_yaw = sfm_yaw

    resp = {
        "status": "success",
        "block": result['block'],
        "inliers": result['inliers'],
        "map_x": float(p_map[0]),
        "map_y": float(p_map[1]),
        "map_yaw": float(map_yaw),
        "latency_ms": dt
    }
    print(f"[API Response] Success: {resp}")
    return resp

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)