# lib/map_utils.py
import numpy as np
import pycolmap
from pathlib import Path
from collections import defaultdict

def colmap_to_scipy_quat(qvec_colmap):
    """
    將 COLMAP 格式四元數 [w, x, y, z] 轉換為 Scipy 格式 [x, y, z, w]。
    """
    w, x, y, z = qvec_colmap
    return np.array([x, y, z, w])

def get_sfm_center(recon: pycolmap.Reconstruction, target_name: str):
    for img in recon.images.values():
        if img.name == target_name:
            c = img.projection_center()
            return np.array([c[0], c[1]])
    
    # Fallback: 嘗試比對純檔名 (忽略路徑)
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
    """
    [Modified] 自動尋找 SfM 模型中的起始與結束 Anchor 圖片。
    策略：
    1. 解析所有有註冊(Registered)的圖片，按 frame_id 分組。
    2. 找出最早和最晚的 frame_id。
    3. 在該 frame_id 中，優先選擇 _F, 其次 _B, 最後才選任意視角。
    """
    # 1. 收集所有已註冊的圖片
    registered_images = [img.name for img in recon.images.values()]
    if not registered_images:
        return None, None

    # 2. 依照 frame_id 分組
    # 假設檔名格式為: {frame_id}_{view}.jpg (例如: frame_0001_F.jpg)
    frame_groups = defaultdict(list)
    for img_name in registered_images:
        stem = Path(img_name).stem
        parts = stem.split('_')
        
        # 防呆: 至少要有 frame_id 和 view
        if len(parts) >= 2:
            frame_id = "_".join(parts[:-1]) # 取得時間戳部分
            frame_groups[frame_id].append(img_name)
        else:
            # 若檔名格式不符，暫時用原檔名當 key (Fallback)
            frame_groups[stem].append(img_name)

    if not frame_groups:
        return None, None

    # 3. 排序找出起點與終點 Frame
    sorted_frames = sorted(frame_groups.keys())
    start_frame = sorted_frames[0]
    end_frame = sorted_frames[-1]

    # 4. 定義選擇最佳視角的函式 (Priority: F > B > Others)
    def select_best_view(img_list):
        # 優先找 F (Front)
        for img in img_list:
            if "_F." in img: return img
        # 其次找 B (Back) - 因為 B 通常也能看很遠，幾何穩健性次之
        for img in img_list:
            if "_B." in img: return img
        # 都沒有，就回傳列表中的第一張 (例如 R, L, FR...)
        return sorted(img_list)[0]

    start_anchor = select_best_view(frame_groups[start_frame])
    end_anchor = select_best_view(frame_groups[end_frame])

    print(f"[Auto Anchor] Start: {start_frame} -> Selected {start_anchor}")
    print(f"[Auto Anchor] End:   {end_frame} -> Selected {end_anchor}")

    return start_anchor, end_anchor

def compute_sim2_transform(recon: pycolmap.Reconstruction, anchor_cfg: dict):
    start_frame = anchor_cfg.get('start_frame')
    end_frame = anchor_cfg.get('end_frame')

    # 若未指定 Frame，則自動尋找
    if not start_frame or not end_frame:
        auto_s, auto_e = find_auto_anchors(recon)
        if not start_frame: start_frame = auto_s
        if not end_frame: end_frame = auto_e

    if not start_frame or not end_frame: 
        print("[Sim2] Failed to identify start/end frames.")
        return None

    p_sfm_s = get_sfm_center(recon, start_frame)
    p_sfm_e = get_sfm_center(recon, end_frame)

    if p_sfm_s is None or p_sfm_e is None: 
        print(f"[Sim2] Anchor image not found in model: {start_frame} or {end_frame}")
        return None

    p_map_s = np.array(anchor_cfg['start_map_xy'])
    p_map_e = np.array(anchor_cfg['end_map_xy'])

    vec_sfm = p_sfm_e - p_sfm_s
    vec_map = p_map_e - p_map_s
    
    dist_sfm = np.linalg.norm(vec_sfm)
    dist_map = np.linalg.norm(vec_map)
    
    if dist_sfm < 1e-6: 
        print("[Sim2] Distance between SfM anchors is too small.")
        return None
        
    s = dist_map / dist_sfm
    theta = np.arctan2(vec_map[1], vec_map[0]) - np.arctan2(vec_sfm[1], vec_sfm[0])
    
    c, si = np.cos(theta), np.sin(theta)
    R = np.array([[c, -si], [si, c]])
    t = p_map_s - s * (R @ p_sfm_s)

    return {
        's': s, 'theta': theta, 't': t, 'R': R,
        'frames': (start_frame, end_frame)
    }