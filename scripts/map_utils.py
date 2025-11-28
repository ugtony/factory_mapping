# scripts/map_utils.py
import numpy as np
import pycolmap
from pathlib import Path

def get_sfm_center(recon: pycolmap.Reconstruction, target_name: str):
    """
    從 COLMAP Reconstruction 中讀取指定圖片的中心座標 (World Frame)。
    支援模糊比對 (e.g., "db/img.jpg" match "img.jpg")。
    """
    # 1. 精確比對
    for img in recon.images.values():
        if img.name == target_name:
            c = img.projection_center()
            return np.array([c[0], c[1]])
            
    # 2. 模糊比對
    candidates = []
    target_clean = Path(target_name).name # 只取檔名
    for img in recon.images.values():
        img_clean = Path(img.name).name
        if img_clean == target_clean:
            candidates.append(img)
            
    if len(candidates) == 1:
        c = candidates[0].projection_center()
        return np.array([c[0], c[1]])
    elif len(candidates) > 1:
        # print(f"[Warn] Multiple matches for '{target_name}'. Using first.")
        c = candidates[0].projection_center()
        return np.array([c[0], c[1]])
            
    return None

def find_auto_anchors(recon: pycolmap.Reconstruction):
    """
    自動從 SfM 模型中尋找第一張與最後一張影像。
    優先尋找 _F (Front view) 的影像。
    """
    all_images = sorted([img.name for img in recon.images.values()])
    
    if not all_images:
        return None, None

    # 優先過濾 _F
    f_images = [name for name in all_images if "_F." in name]
    
    if f_images:
        return f_images[0], f_images[-1]
    else:
        return all_images[0], all_images[-1]

def compute_sim2_transform(recon: pycolmap.Reconstruction, anchor_cfg: dict):
    """
    計算 Sim2 變換矩陣 (Scale, Rotation, Translation)。
    Input:
      recon: pycolmap Reconstruction 物件
      anchor_cfg: dict, 包含 start_map_xy, end_map_xy, 且可選 start_frame, end_frame
    Return:
      dict {'s': s, 'theta': theta, 't': t, 'R': R, 'frames': (start, end)} 或 None
    """
    # 1. 處理 Frame Name (若無則自動偵測)
    start_frame = anchor_cfg.get('start_frame')
    end_frame = anchor_cfg.get('end_frame')

    if not start_frame or not end_frame:
        # print(f"  [Auto] Detecting anchor frames...")
        auto_s, auto_e = find_auto_anchors(recon)
        if not start_frame: start_frame = auto_s
        if not end_frame: end_frame = auto_e
        
        # 回寫偵測結果到 cfg (方便 caller 知道用了哪張圖，選用)
        # anchor_cfg['auto_start_frame'] = start_frame
        # anchor_cfg['auto_end_frame'] = end_frame

    if not start_frame or not end_frame:
        return None

    # 2. 取得 SfM 座標
    p_sfm_s = get_sfm_center(recon, start_frame)
    p_sfm_e = get_sfm_center(recon, end_frame)

    if p_sfm_s is None or p_sfm_e is None:
        return None

    # 3. 取得 Map 座標
    p_map_s = np.array(anchor_cfg['start_map_xy'])
    p_map_e = np.array(anchor_cfg['end_map_xy'])

    # 4. 計算數學變換
    vec_sfm = p_sfm_e - p_sfm_s
    vec_map = p_map_e - p_map_s
    
    dist_sfm = np.linalg.norm(vec_sfm)
    dist_map = np.linalg.norm(vec_map)
    
    if dist_sfm < 1e-6:
        return None
        
    s = dist_map / dist_sfm
    theta = np.arctan2(vec_map[1], vec_map[0]) - np.arctan2(vec_sfm[1], vec_sfm[0])
    
    c, si = np.cos(theta), np.sin(theta)
    R = np.array([[c, -si], [si, c]])
    t = p_map_s - s * (R @ p_sfm_s)

    return {
        's': s, 
        'theta': theta, 
        't': t, 
        'R': R,
        'frames': (start_frame, end_frame)
    }