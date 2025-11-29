# scripts/localization_engine.py
import numpy as np
import cv2
import torch
import h5py
import pycolmap
import json
from pathlib import Path
from scipy.spatial.transform import Rotation

# HLOC Imports
from hloc import extractors, matchers, extract_features, match_features
from hloc.utils.base_model import dynamic_load

# 嘗試匯入 map_utils
try:
    from map_utils import compute_sim2_transform
except ImportError:
    try:
        from .map_utils import compute_sim2_transform
    except ImportError:
        pass # 可能在某些路徑下執行會失敗，視需求處理

class LocalizationEngine:
    # 1. 在參數列表中加入 outputs_dir
    def __init__(self, project_root: Path, config_path: Path, anchors_path: Path, outputs_dir: Path = None, device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Init] LocalizationEngine using device: {self.device}")
        
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
        self.project_root = project_root
        
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
        target_outputs = outputs_dir if outputs_dir else (project_root / "outputs-hloc")  # 如果有傳入 outputs_dir 就用它，否則預設使用 project_root / "outputs-hloc"
        self.blocks = {}
        # 傳入決定的路徑
        self._load_blocks(target_outputs, anchors_path)

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
            
            if not g_vecs:
                print(f"    [Warn] No global descriptors found in {block_dir.name}")
                continue

            g_vecs = torch.from_numpy(np.array(g_vecs)).to(self.device).squeeze()
            if g_vecs.ndim == 1: g_vecs = g_vecs.unsqueeze(0)

            # Load Reconstruction
            try:
                recon = pycolmap.Reconstruction(sfm_dir)
                name_to_id = {img.name: img_id for img_id, img in recon.images.items()}
            except Exception as e:
                print(f"    [Error] Failed to load SfM: {e}")
                continue
            
            # Compute Transform
            transform = None
            if block_dir.name in anchors:
                try:
                    transform = compute_sim2_transform(recon, anchors[block_dir.name])
                    if transform:
                        s = transform['s']
                        theta_deg = np.degrees(transform['theta'])
                        print(f"    > Aligned: Scale={s:.4f}, Rot={theta_deg:.2f}°")
                except Exception as e:
                    print(f"    [Warn] Anchor alignment failed: {e}")

            self.blocks[block_dir.name] = {
                'recon': recon,
                'name_to_id': name_to_id,
                'global_names': g_names,
                'global_vecs': g_vecs,
                'local_h5_path': local_h5_path, # Store path, open on demand? Currently open handle.
                'local_h5': h5py.File(local_h5_path, 'r'),
                'transform': transform,
                'block_root': block_dir
            }

    @torch.no_grad()
    def localize(self, image_arr: np.ndarray, fov_deg: float = None, return_details: bool = False, top_k_db: int = 10):
        """
        執行 Multi-View Localization。
        top_k_db: 每個 Block 內要拿幾張最像的圖來做 PnP (HLOC 預設為 10-20，Server 追求速度可設為 3-5)
        """
        if fov_deg is None: fov_deg = self.default_fov
        
        # --- Step 1: Preprocessing ---
        h_orig, w_orig = image_arr.shape[:2]
        
        resize_max = 1024
        scale = 1.0
        new_w, new_h = w_orig, h_orig
        
        if max(h_orig, w_orig) >= resize_max:
            scale = resize_max / max(h_orig, w_orig)
            new_w, new_h = int(w_orig * scale), int(h_orig * scale)
            image_tensor = cv2.resize(image_arr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            # print(f"    [Debug] Resized {w_orig}x{h_orig} -> {new_w}x{new_h}")
        else:
            image_tensor = image_arr
            # print(f"    [Debug] No resize needed ({w_orig}x{h_orig})")

        scale_x = w_orig / new_w
        scale_y = h_orig / new_h

        img_t = torch.from_numpy(image_tensor.transpose(2, 0, 1)).float().div(255.).unsqueeze(0).to(self.device)
        img_g = torch.from_numpy(cv2.cvtColor(image_tensor, cv2.COLOR_RGB2GRAY)).float().div(255.).unsqueeze(0).unsqueeze(0).to(self.device)

        # --- Step 2: Feature Extraction ---
        q_global = self.model_extract_global({'image': img_t})['global_descriptor']
        q_local = self.model_extract_local({'image': img_g})
        kpts = q_local['keypoints'][0]
        desc = q_local['descriptors'][0]
        
        # 座標還原
        kpts[:, 0] = (kpts[:, 0] + 0.5) * scale_x - 0.5
        kpts[:, 1] = (kpts[:, 1] + 0.5) * scale_y - 0.5
        
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
        # 先用 Global Feature 篩選出最可能的 Blocks (Candidates)
        candidate_blocks = []
        for name, block in self.blocks.items():
            sim = torch.matmul(q_global, block['global_vecs'].t())
            # 這裡我們只看該 Block 內最像的那張分數，來決定要不要對這個 Block 進行細算
            score, _ = torch.max(sim, dim=1)
            score_val = score.item()
            if score_val > 0.01:
                # 儲存 sim 矩陣以便後續取 Top-K
                candidate_blocks.append((score_val, name, sim))
        
        candidate_blocks.sort(key=lambda x: x[0], reverse=True)
        
        # 逐一嘗試 Block (通常只會試第一個就成功 break)
        for _, block_name, sim_matrix in candidate_blocks:
            block = self.blocks[block_name]
            
            # [Multi-View Logic] 1. 取出該 Block 內 Top-K 張最像的圖
            k_val = min(top_k_db, sim_matrix.shape[1])
            scores, indices = torch.topk(sim_matrix, k=k_val, dim=1)
            indices = indices[0].cpu().numpy()
            
            # 準備收集所有配對到的 (2D, 3D) 點
            p2d_list = []
            p3d_list = []
            
            # 用於 Visualization 的暫存變數 (只存 Top-1 的細節)
            viz_details = {}
            
            # 2. 對這 K 張圖分別做 Matching
            for rank, db_idx in enumerate(indices):
                db_name = block['global_names'][db_idx]
                
                if db_name not in block['local_h5']: continue
                if db_name not in block['name_to_id']: continue
                
                # Load DB Features
                img_obj = block['recon'].images[block['name_to_id'][db_name]]
                cam_db = block['recon'].cameras[img_obj.camera_id]
                
                grp = block['local_h5'][db_name]
                kpts_db = torch.from_numpy(grp['keypoints'].__array__()).float().to(self.device)
                desc_db = torch.from_numpy(grp['descriptors'].__array__()).float().to(self.device)
                if desc_db.shape[0] != 256 and desc_db.shape[1] == 256: desc_db = desc_db.T

                # Match
                data = {
                    'image0': torch.empty((1,1,h_orig,w_orig), device=self.device),
                    'keypoints0': kpts.unsqueeze(0), 'descriptors0': desc.unsqueeze(0),
                    'image1': torch.empty((1,1,cam_db.height,cam_db.width), device=self.device),
                    'keypoints1': kpts_db.unsqueeze(0), 'descriptors1': desc_db.unsqueeze(0)
                }
                matches = self.model_matcher(data)['matches0'][0]
                valid = matches > -1
                
                # 如果這張圖匹配太少，就跳過
                if valid.sum().item() < 4: continue

                # 提取 2D-3D 對應
                p3d_ids = np.array([p.point3D_id if p.has_point3D() else -1 for p in img_obj.points2D])
                m_q = torch.where(valid)[0].cpu().numpy()
                m_db = matches[valid].cpu().numpy()
                
                # 確保 DB feature index 不越界
                valid_3d = m_db < len(p3d_ids)
                m_q = m_q[valid_3d]
                m_db = m_db[valid_3d]
                
                target_ids = p3d_ids[m_db]
                has_3d = target_ids != -1
                
                if has_3d.sum() < 4: continue

                # 收集資料
                p2d_local = kpts.cpu().numpy()[m_q[has_3d]].astype(np.float64)
                p2d_local += 0.5 # COLMAP offset
                
                # 從 Reconstruction 查 3D 座標
                p3d_local = []
                for pid in target_ids[has_3d]:
                    p3d_local.append(block['recon'].points3D[pid].xyz)
                p3d_local = np.array(p3d_local, dtype=np.float64)
                
                p2d_list.append(p2d_local)
                p3d_list.append(p3d_local)
                
                # 記錄 Top-1 的資訊供回傳 (Rank 0)
                if rank == 0:
                    viz_details['matched_db_name'] = db_name
                    if return_details:
                        viz_details['kpts_query'] = kpts.cpu().numpy()
                        viz_details['kpts_db'] = kpts_db.cpu().numpy()
                        viz_details['matches'] = matches.cpu().numpy()
                        
                        block_root = block['block_root']
                        stage_path = block_root / "_images_stage" / db_name
                        data_path = self.project_root / "data" / block_name / db_name
                        viz_details['db_image_path'] = stage_path if stage_path.exists() else data_path

            # 如果所有圖加起來還是沒什麼點，就換下一個 Block
            if not p2d_list: continue
            
            # 3. Aggregate Matches (大鍋炒)
            p2d_concat = np.concatenate(p2d_list, axis=0)
            p3d_concat = np.concatenate(p3d_list, axis=0)
            
            # print(f"    [Debug] Block {block_name}: Aggregated {len(p2d_concat)} points from Top-{k_val} images")

            # --- Step 5: PnP Solver (Multi-View) ---
            try:
                # 這裡的 p2d_concat 可能包含重複的 Query Keypoint (如果它同時對應到多張 DB 圖)
                # 但 pycolmap 的 RANSAC 能夠處理這種情況 (或者視為多個觀測值)
                ret = pycolmap.estimate_and_refine_absolute_pose(
                    p2d_concat, p3d_concat, camera, 
                    estimation_options={'ransac': {'max_error': 12.0}}
                )
                
                # Robust Result Parsing
                success = False
                qvec, tvec, num_inliers = None, None, 0

                if ret:
                    if isinstance(ret, dict):
                        success = ret.get('success', False)
                        num_inliers = ret.get('num_inliers', 0)
                    else:
                        success = ret.success
                        num_inliers = ret.num_inliers
                    
                    if not success:
                         ret_ransac = pycolmap.estimate_absolute_pose(
                             p2d_concat, p3d_concat, camera, 
                             estimation_options={'ransac': {'max_error': 12.0}}
                         )
                         if ret_ransac:
                            if isinstance(ret_ransac, dict):
                                if ret_ransac.get('num_inliers', 0) > 0: success = True
                            elif ret_ransac.num_inliers > 0: success = True
                            if success: ret = ret_ransac 

                    if success:
                        if isinstance(ret, dict):
                            if 'qvec' in ret: 
                                q_raw = ret['qvec']
                                tvec = ret['tvec']
                            elif 'cam_from_world' in ret:
                                q_raw = ret['cam_from_world'].rotation.quat
                                tvec = ret['cam_from_world'].translation
                        else:
                            if ret.cam_from_world:
                                q_raw = ret.cam_from_world.rotation.quat
                                tvec = ret.cam_from_world.translation

                        # Quaternion Order Check [x,y,z,w] -> [w,x,y,z]
                        if q_raw is not None:
                            qvec = np.array([q_raw[3], q_raw[0], q_raw[1], q_raw[2]])

                if success and qvec is not None:
                    # 只要 PnP 成功，我們就認定找到 Block 了
                    best_result = {
                        'success': True,
                        'block': block_name, 
                        'pose': {'qvec': qvec, 'tvec': tvec}, 
                        'transform': block['transform'], 
                        'inliers': num_inliers,
                        'matched_db_name': viz_details.get('matched_db_name', 'unknown')
                    }
                    if return_details:
                        best_result.update(viz_details)
                    
                    break # Found block, return result

            except Exception as e:
                print(f"[Error] PnP failed for {block_name}: {e}")
                continue
        
        if best_result is None:
            best_result = {'success': False, 'inliers': 0}

        return best_result