import h5py, numpy as np
from pathlib import Path

def load_local_keypoints(local_feats_path, image_key):
    with h5py.File(str(local_feats_path), "r") as f:
        if image_key not in f:
            raise KeyError(f"Missing features for {image_key}")
        kpts = f[image_key]["keypoints"][()]
        return kpts

def dummy_pairing(kpts_q, kpts_db, max_pairs=200):
    n = min(len(kpts_q), len(kpts_db), max_pairs)
    idx = np.arange(n)
    return np.stack([idx, idx], axis=1)
