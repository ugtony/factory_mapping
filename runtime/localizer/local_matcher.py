import h5py, numpy as np
from pathlib import Path
import torch

def load_local_keypoints(local_feats_path, image_key):
    """
    (此函式不變) 
    僅載入 keypoints (numpy 格式)，用於 PnP 姿態估算。
   
    """
    with h5py.File(str(local_feats_path), "r") as f:
        if image_key not in f:
            raise KeyError(f"Missing features for {image_key}")
        kpts = f[image_key]["keypoints"][()]
        return kpts

# --- [修改函式] ---
def load_local_features(local_feats_path, image_key, device):
    """
    從 H5 檔案中載入 Keypoints 和 Descriptors，
    並轉換為 LightGlue 需要的 (B, N, C) 格式。
    """
    with h5py.File(str(local_feats_path), "r") as f:
        if image_key not in f:
            raise KeyError(f"Missing features for {image_key}")
        
        grp = f[image_key]
        
        kpts = grp["keypoints"][()]
        kpts_torch = torch.from_numpy(kpts.astype(np.float32)).to(device)
        
        if "descriptors" not in grp:
             raise KeyError(f"Missing descriptors for {image_key}")
             
        desc = grp["descriptors"][()]
        desc_torch = torch.from_numpy(desc.astype(np.float32)).to(device)
        
        # [修復]
        # kpts: (M, 2) -> (1, M, 2)
        # desc: (C, M) -> (M, C) -> (1, M, C)
        return {
            "keypoints": kpts_torch[None],
            "descriptors": desc_torch.transpose(0, 1)[None] 
        }
# --- [結束修改] ---

def dummy_pairing(kpts_q, kpts_db, max_pairs=200):
    """ (此函式不變) """
    n = min(len(kpts_q), len(kpts_db), max_pairs)
    idx = np.arange(n)
    return np.stack([idx, idx], axis=1)