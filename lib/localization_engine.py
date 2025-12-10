# lib/localization_engine.py
import numpy as np
import cv2
import torch
import h5py
import pycolmap
import json
from pathlib import Path
from scipy.spatial.transform import Rotation
from unittest.mock import patch  # [Offline Fix] 唯一需要的額外 import

# HLOC Imports
from hloc import extractors, matchers, extract_features, match_features
from hloc.utils.base_model import dynamic_load

# [Clean Import] Relative import within the package
try:
    from .map_utils import compute_sim2_transform
except ImportError:
    # Fallback if executed as script
    from map_utils import compute_sim2_transform

class LocalizationEngine:
    def __init__(self, project_root: Path, config_path: Path, anchors_path: Path, outputs_dir: Path = None, device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Init] LocalizationEngine using device: {self.device}")
        
        self.config = {}
        if config_path.exists():
            with open(config_path) as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        k, v = line.strip().split('=', 1)
                        self.config[k.strip()] = v.strip().strip('"').strip("'")
        
        self.global_conf_name = self.config.get("GLOBAL_CONF", "netvlad")
        self.default_fov = float(self.config.get("FOV_QUERY", self.config.get("FOV", 69.4)))
        print(f"[Init] Default Query FOV: {self.default_fov}")

        self.project_root = project_root
        
        # =========================================================================
        # [Offline Fix] 設定離線模型路徑
        # =========================================================================
        # 請確認您的路徑是否正確
        CACHE_ROOT = Path("/root/.cache/torch/hub")
        SUPERPOINT_WEIGHTS = CACHE_ROOT / "checkpoints/superpoint_lightglue_v0-1_arxiv.pth"
        
        # 設定 GitHub Repo 的本地快取路徑
        MEGALOC_REPO = CACHE_ROOT / "gmberton_MegaLoc_main" 
        DINOV2_REPO = CACHE_ROOT / "facebookresearch_dinov2_main" 

        print("[Init] Loading Neural Network Models (Offline Mode)...")

        # 保存原始函數，避免無限遞迴
        _real_hub_load = torch.hub.load
        
        # 定義 Mock 函數：攔截 MegaLoc 和其依賴的 DINOv2
        def mock_hub_load(repo_or_dir, *args, **kwargs):
            # 1. 攔截 MegaLoc
            if repo_or_dir == "gmberton/MegaLoc" and MEGALOC_REPO.exists():
                print(f"  [Offline] Redirecting MegaLoc to local repo: {MEGALOC_REPO}")
                kwargs['source'] = 'local'
                return _real_hub_load(str(MEGALOC_REPO), *args, **kwargs)
            
            # 2. 攔截 DINOv2 (MegaLoc 內部會呼叫這個)
            elif repo_or_dir == "facebookresearch/dinov2":
                if DINOV2_REPO.exists():
                    print(f"  [Offline] Redirecting DINOv2 to local repo: {DINOV2_REPO}")
                    kwargs['source'] = 'local'
                    return _real_hub_load(str(DINOV2_REPO), *args, **kwargs)
                else:
                    print(f"  [Warn] DINOv2 local repo not found at {DINOV2_REPO}, it might try to connect online.")

            # 其他模型則維持原樣
            return _real_hub_load(repo_or_dir, *args, **kwargs)

        # 定義 Mock 函數：攔截 SuperPoint 的權重下載請求
        def mock_load_url(url, model_dir=None, map_location=None, progress=True, check_hash=False, file_name=None):
            print(f"  [Offline] Intercepted URL download, loading local SuperPoint: {SUPERPOINT_WEIGHTS}")
            return torch.load(SUPERPOINT_WEIGHTS, map_location=self.device)

        # 同時套用兩個 Patch
        with patch('torch.hub.load', side_effect=mock_hub_load), \
             patch('torch.hub.load_state_dict_from_url', side_effect=mock_load_url):
            
            # 1. Load Local Feature (SuperPoint)
            local_conf = extract_features.confs['superpoint_aachen']
            ModelLocal = dynamic_load(extractors, local_conf['model']['name'])
            self.model_extract_local = ModelLocal(local_conf['model']).eval().to(self.device)
            
            # 2. Load Global Feature (MegaLoc / NetVLAD)
            global_conf = extract_features.confs[self.global_conf_name]
            ModelGlobal = dynamic_load(extractors, global_conf['model']['name'])
            self.model_extract_global = ModelGlobal(global_conf['model']).eval().to(self.device)
            
            # 3. Load Matcher (LightGlue)
            matcher_conf = match_features.confs['superpoint+lightglue']
            ModelMatcher = dynamic_load(matchers, matcher_conf['model']['name'])
            self.model_matcher = ModelMatcher(matcher_conf['model']).eval().to(self.device)
        
        target_outputs = outputs_dir if outputs_dir else (project_root / "outputs-hloc")
        self.blocks = {}
        self._load_blocks(target_outputs, anchors_path)

    def _load_blocks(self, outputs_root, anchors_path):
        anchors = {}
        if anchors_path.exists():
            with open(anchors_path, 'r') as f: anchors = json.load(f)
            print(f"[Init] Loaded anchors for: {list(anchors.keys())}")

        for block_dir in outputs_root.iterdir():
            if not block_dir.is_dir(): continue
            sfm_dir = block_dir / "sfm_aligned"
            if not (sfm_dir / "images.bin").exists(): sfm_dir = block_dir / "sfm"
            global_h5 = block_dir / f"global-{self.global_conf_name}.h5"
            local_h5_path = block_dir / "local-superpoint_aachen.h5"
            
            if not (sfm_dir/"images.bin").exists() or not global_h5.exists() or not local_h5_path.exists():
                continue

            print(f"[Init] Loading Block: {block_dir.name}")
            g_names = []
            g_vecs = []
            with h5py.File(global_h5, 'r') as f:
                def visit(name, obj):
                    if isinstance(obj, h5py.Group) and 'global_descriptor' in obj:
                        g_names.append(name)
                        g_vecs.append(obj['global_descriptor'].__array__())
                f.visititems(visit)
            
            if not g_vecs: continue
            g_vecs = torch.from_numpy(np.array(g_vecs)).to(self.device).squeeze()
            if g_vecs.ndim == 1: g_vecs = g_vecs.unsqueeze(0)

            try:
                recon = pycolmap.Reconstruction(sfm_dir)
                name_to_id = {img.name: img_id for img_id, img in recon.images.items()}
            except Exception as e:
                print(f"    [Error] Failed to load SfM: {e}"); continue
            
            transform = None
            if block_dir.name in anchors:
                try:
                    transform = compute_sim2_transform(recon, anchors[block_dir.name])
                    if transform:
                        s = transform['s']
                        theta_deg = np.degrees(transform['theta'])
                        print(f"    > Aligned: Scale={s:.4f}, Rot={theta_deg:.2f}°")
                except: pass

            self.blocks[block_dir.name] = {
                'recon': recon,
                'name_to_id': name_to_id,
                'global_names': g_names,
                'global_vecs': g_vecs,
                'local_h5_path': local_h5_path,
                'local_h5': h5py.File(local_h5_path, 'r'),
                'transform': transform,
                'block_root': block_dir
            }

    @torch.no_grad()
    def localize(self, image_arr: np.ndarray, fov_deg: float = None, return_details: bool = False, top_k_db: int = 10, verbose: bool = False):
        if fov_deg is None: fov_deg = self.default_fov
        
        h_orig, w_orig = image_arr.shape[:2]
        # Resize logic...
        resize_max = 1024
        scale = 1.0
        new_w, new_h = w_orig, h_orig
        
        if max(h_orig, w_orig) >= resize_max:
            scale = resize_max / max(h_orig, w_orig)
            new_w, new_h = int(round(w_orig * scale)), int(round(h_orig * scale))
            image_tensor = cv2.resize(image_arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            image_tensor = image_arr

        scale_x = w_orig / new_w
        scale_y = h_orig / new_h
        
        img_t = torch.from_numpy(image_tensor.transpose(2, 0, 1)).float().div(255.).unsqueeze(0).to(self.device)
        img_g = torch.from_numpy(cv2.cvtColor(image_tensor, cv2.COLOR_RGB2GRAY)).float().div(255.).unsqueeze(0).unsqueeze(0).to(self.device)

        # 2. Global Feature
        q_global = self.model_extract_global({'image': img_t})['global_descriptor']
        
        # 3. Local Feature
        q_local = self.model_extract_local({'image': img_g})
        kpts = q_local['keypoints'][0]
        desc = q_local['descriptors'][0]
        
        if verbose:
            print(f"  [Log] Query Kpts: {len(kpts)} (Scale: {scale_x:.4f}, {scale_y:.4f})")
        
        # [Init Diagnosis]
        diag = {
            'num_kpts': len(kpts),
            'top1_block': 'None', 'top1_score': 0.0,
            'top2_block': 'None', 'top2_score': 0.0,
            'selected_block': 'None',
            'num_matches_2d': 0, 'num_matches_3d': 0, 'pnp_inliers': 0,
            'status': 'Fail_Unknown',
            'db_ranks': [] 
        }

        # [Fix 3] Coordinate Restoration (0.5 pixel offset for COLMAP)
        kpts[:, 0] = (kpts[:, 0] + 0.5) * scale_x
        kpts[:, 1] = (kpts[:, 1] + 0.5) * scale_y
        
        fov_rad = np.deg2rad(fov_deg)
        f = 0.5 * max(w_orig, h_orig) / np.tan(fov_rad / 2.0)
        camera = pycolmap.Camera(
            model='SIMPLE_PINHOLE', width=w_orig, height=h_orig, 
            params=np.array([f, w_orig/2.0, h_orig/2.0], dtype=np.float64)
        )

        candidate_blocks = []
        for name, block in self.blocks.items():
            sim = torch.matmul(q_global, block['global_vecs'].t())
            k_scoring = min(5, sim.shape[1])
            if k_scoring > 0:
                topk_vals, _ = torch.topk(sim, k=k_scoring, dim=1)
                score = torch.mean(topk_vals, dim=1)
            else:
                score = torch.tensor([0.0], device=self.device)
            if score.item() > 0.01: candidate_blocks.append((score.item(), name, sim))
        
        candidate_blocks.sort(key=lambda x: x[0], reverse=True)
        candidate_blocks = candidate_blocks[:3]

        if len(candidate_blocks) > 0:
            diag['top1_block'] = candidate_blocks[0][1]
            diag['top1_score'] = candidate_blocks[0][0]
        if len(candidate_blocks) > 1:
            diag['top2_block'] = candidate_blocks[1][1]
            diag['top2_score'] = candidate_blocks[1][0]

        if not candidate_blocks:
            diag['status'] = 'Fail_No_Retrieval'
            return {'success': False, 'inliers': 0, 'diagnosis': diag}

        valid_block_results = []
        best_fail_stats = diag.copy()

        for _, block_name, sim_matrix in candidate_blocks:
            if verbose:
                print(f"  [Log] Checking Block: {block_name}")

            block = self.blocks[block_name]
            k_val = min(top_k_db, sim_matrix.shape[1])
            scores, indices = torch.topk(sim_matrix, k=k_val, dim=1)
            indices = indices[0].cpu().numpy()
            
            p2d_list, p3d_list = [], []
            viz_details = {}
            current_block_stats = {'matches_2d_sum': 0, 'matches_3d_sum': 0}
            current_db_ranks = [] 
            unique_matches = set()

            for rank, db_idx in enumerate(indices):
                db_name = block['global_names'][db_idx]
                if db_name not in block['local_h5'] or db_name not in block['name_to_id']: continue
                
                img_obj = block['recon'].images[block['name_to_id'][db_name]]
                cam_db = block['recon'].cameras[img_obj.camera_id]
                
                grp = block['local_h5'][db_name]
                kpts_db = torch.from_numpy(grp['keypoints'].__array__()).float().to(self.device)
                desc_db = torch.from_numpy(grp['descriptors'].__array__()).float().to(self.device)
                if desc_db.shape[0] != 256 and desc_db.shape[1] == 256: desc_db = desc_db.T

                data = {
                    'image0': torch.empty((1,1,h_orig,w_orig), device=self.device),
                    'keypoints0': kpts.unsqueeze(0), 'descriptors0': desc.unsqueeze(0),
                    'image1': torch.empty((1,1,cam_db.height,cam_db.width), device=self.device),
                    'keypoints1': kpts_db.unsqueeze(0), 'descriptors1': desc_db.unsqueeze(0)
                }
                matches = self.model_matcher(data)['matches0'][0]
                valid = matches > -1
                n_2d = valid.sum().item()
                current_block_stats['matches_2d_sum'] += n_2d
                
                if verbose:
                    print(f"    > Rank {rank} ({db_name}): 2D Matches = {n_2d}", end="")
                
                if rank < 3:
                    current_db_ranks.append({'name': db_name, 'matches_2d': n_2d})

                if n_2d < 4: 
                    if verbose: print(" (Skipped: <4 2D)")
                    continue

                p3d_ids = np.array([p.point3D_id if p.has_point3D() else -1 for p in img_obj.points2D])
                m_q = torch.where(valid)[0].cpu().numpy()
                m_db = matches[valid].cpu().numpy()
                valid_3d = m_db < len(p3d_ids)
                m_q, m_db = m_q[valid_3d], m_db[valid_3d]
                target_ids = p3d_ids[m_db]
                has_3d = target_ids != -1
                n_3d = has_3d.sum()
                current_block_stats['matches_3d_sum'] += n_3d
                
                if n_3d < 4:
                    if verbose: print(" (Skipped: <4 3D)")
                    continue
                
                m_q_valid = m_q[has_3d]
                target_ids_valid = target_ids[has_3d]
                
                if len(m_q_valid) == 0: 
                    if verbose: print(" (No Valid Points)")
                    continue
                
                new_p2d = []
                new_p3d = []
                kpts_np = kpts.cpu().numpy()
                points3D_map = block['recon'].points3D
                
                for q_idx, tid in zip(m_q_valid, target_ids_valid):
                    q_idx = int(q_idx)
                    tid = int(tid)
                    if (q_idx, tid) not in unique_matches:
                        unique_matches.add((q_idx, tid))
                        new_p2d.append(kpts_np[q_idx])
                        new_p3d.append(points3D_map[tid].xyz)
                
                if not new_p2d:
                    if verbose: print(" (All Duplicate)")
                    continue
                
                p2d_list.append(np.array(new_p2d, dtype=np.float64))
                p3d_list.append(np.array(new_p3d, dtype=np.float64))
                
                if verbose: print(f", 3D Points = {len(new_p2d)} (Unique) -> Added")
                if rank == 0: viz_details['matched_db_name'] = db_name

            if block_name == diag['top1_block']:
                 best_fail_stats['num_matches_2d'] = current_block_stats['matches_2d_sum']
                 best_fail_stats['num_matches_3d'] = current_block_stats['matches_3d_sum']
                 best_fail_stats['db_ranks'] = current_db_ranks
                 if current_block_stats['matches_3d_sum'] == 0:
                     best_fail_stats['status'] = 'Fail_No_3D_Match'
                 else:
                     best_fail_stats['status'] = 'Fail_PnP_Error'

            if not p2d_list: continue
            p2d_concat = np.concatenate(p2d_list, axis=0)
            p3d_concat = np.concatenate(p3d_list, axis=0)
            
            if verbose:
                print(f"    [PnP Input] Total 2D-3D Correspondences: {len(p2d_concat)}")
            
            try:
                refine_opts = pycolmap.AbsolutePoseRefinementOptions()
                refine_opts.refine_focal_length = True
                refine_opts.refine_extra_params = False

                ret = pycolmap.estimate_and_refine_absolute_pose(
                    p2d_concat, p3d_concat, camera, 
                    estimation_options={'ransac': {'max_error': 12.0}},
                    refinement_options=refine_opts
                )
                success, qvec, tvec, num_inliers = False, None, None, 0
                
                if ret:
                    if isinstance(ret, dict):
                        success = ret.get('success', False)
                        num_inliers = ret.get('num_inliers', 0)
                    else:
                        success = ret.success
                        num_inliers = ret.num_inliers

                    if not success:
                         ret_ransac = pycolmap.estimate_absolute_pose(
                             p2d_concat, p3d_concat, camera, estimation_options={'ransac': {'max_error': 12.0}}
                         )
                         if ret_ransac:
                            if isinstance(ret_ransac, dict):
                                if ret_ransac.get('num_inliers', 0) > 0: success = True
                            elif ret_ransac.num_inliers > 0: success = True
                            if success: ret = ret_ransac 

                    if success and num_inliers < 15:
                        success = False
                        if verbose: print(f"    [Filter] Low inliers ({num_inliers} < 15), marking as Failed.")

                    if success:
                        if isinstance(ret, dict):
                            if 'qvec' in ret: q_raw, tvec = ret['qvec'], ret['tvec']
                            elif 'cam_from_world' in ret:
                                q_raw = ret['cam_from_world'].rotation.quat
                                tvec = ret['cam_from_world'].translation
                        else:
                            if ret.cam_from_world:
                                q_raw = ret.cam_from_world.rotation.quat
                                tvec = ret.cam_from_world.translation
                        if q_raw is not None:
                            qvec = np.array([q_raw[3], q_raw[0], q_raw[1], q_raw[2]])

                if verbose and success:
                    print(f"    [PnP Result] Success! Inliers: {num_inliers}")
                elif verbose:
                    print(f"    [PnP Result] Failed.")

                if success and qvec is not None:
                    res_diag = diag.copy()
                    res_diag.update({
                        'selected_block': block_name,
                        'num_matches_2d': current_block_stats['matches_2d_sum'],
                        'num_matches_3d': len(p2d_concat),
                        'pnp_inliers': num_inliers,
                        'status': 'Success',
                        'db_ranks': current_db_ranks 
                    })
                    
                    res = {
                        'success': True, 'block': block_name, 
                        'pose': {'qvec': qvec, 'tvec': tvec}, 
                        'transform': block['transform'], 'inliers': num_inliers,
                        'matched_db_name': viz_details.get('matched_db_name', 'unknown'),
                        'diagnosis': res_diag
                    }
                    if return_details: res.update(viz_details)
                    valid_block_results.append(res)
            except Exception as e:
                print(f"[Error] PnP failed for {block_name}: {e}"); continue
        
        if valid_block_results:
            valid_block_results.sort(key=lambda x: x['inliers'], reverse=True)
            best_result = valid_block_results[0]
            if len(valid_block_results) > 1:
                second = valid_block_results[1]
                best_result['diagnosis']['second_inliers'] = second['inliers']
                best_result['diagnosis']['second_block'] = second['block']
            else:
                best_result['diagnosis']['second_inliers'] = 0
                best_result['diagnosis']['second_block'] = "None"
        else:
            best_result = {'success': False, 'inliers': 0, 'diagnosis': best_fail_stats}
            best_result['diagnosis']['second_inliers'] = 0
            best_result['diagnosis']['second_block'] = "None"
            
        return best_result