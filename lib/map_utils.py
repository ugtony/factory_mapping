# lib/map_utils.py
import numpy as np
import pycolmap
from pathlib import Path

def colmap_to_scipy_quat(qvec_colmap):
    """
    [新增] 將 COLMAP 格式四元數 [w, x, y, z] 轉換為 Scipy 格式 [x, y, z, w]。
    統一使用此函式，避免手動交換索引時出錯。
    """
    w, x, y, z = qvec_colmap
    return np.array([x, y, z, w])

def get_sfm_center(recon: pycolmap.Reconstruction, target_name: str):
    for img in recon.images.values():
        if img.name == target_name:
            c = img.projection_center()
            return np.array([c[0], c[1]])
    candidates = []
    target_clean = Path(target_name).name
    for img in recon.images.values():
        img_clean = Path(img.name).name
        if img_clean == target_clean:
            candidates.append(img)
    if len(candidates) >= 1:
        c = candidates[0].projection_center()
        return np.array([c[0], c[1]])
    return None

def find_auto_anchors(recon: pycolmap.Reconstruction):
    all_images = sorted([img.name for img in recon.images.values()])
    if not all_images: return None, None
    f_images = [name for name in all_images if "_F." in name]
    if f_images: return f_images[0], f_images[-1]
    else: return all_images[0], all_images[-1]

def compute_sim2_transform(recon: pycolmap.Reconstruction, anchor_cfg: dict):
    start_frame = anchor_cfg.get('start_frame')
    end_frame = anchor_cfg.get('end_frame')

    if not start_frame or not end_frame:
        auto_s, auto_e = find_auto_anchors(recon)
        if not start_frame: start_frame = auto_s
        if not end_frame: end_frame = auto_e

    if not start_frame or not end_frame: return None

    p_sfm_s = get_sfm_center(recon, start_frame)
    p_sfm_e = get_sfm_center(recon, end_frame)

    if p_sfm_s is None or p_sfm_e is None: return None

    p_map_s = np.array(anchor_cfg['start_map_xy'])
    p_map_e = np.array(anchor_cfg['end_map_xy'])

    vec_sfm = p_sfm_e - p_sfm_s
    vec_map = p_map_e - p_map_s
    
    dist_sfm = np.linalg.norm(vec_sfm)
    dist_map = np.linalg.norm(vec_map)
    
    if dist_sfm < 1e-6: return None
        
    s = dist_map / dist_sfm
    theta = np.arctan2(vec_map[1], vec_map[0]) - np.arctan2(vec_sfm[1], vec_sfm[0])
    
    c, si = np.cos(theta), np.sin(theta)
    R = np.array([[c, -si], [si, c]])
    t = p_map_s - s * (R @ p_sfm_s)

    return {
        's': s, 'theta': theta, 't': t, 'R': R,
        'frames': (start_frame, end_frame)
    }