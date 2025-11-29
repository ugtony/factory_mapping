# lib/hloc_io_utils.py
import h5py
import numpy as np
import pickle
from pathlib import Path

def load_global_descriptors_safe(h5_path):
    names = []
    vectors = []
    if not Path(h5_path).exists(): return [], np.array([])

    def visit(name, obj):
        if isinstance(obj, h5py.Group) and 'global_descriptor' in obj:
            names.append(name)
            vectors.append(obj['global_descriptor'].__array__())
            
    with h5py.File(h5_path, 'r') as f:
        f.visititems(visit)
    
    if len(vectors) == 0: return [], np.array([])
    vecs = np.array(vectors)
    if vecs.ndim == 3 and vecs.shape[1] == 1: vecs = vecs.squeeze(1)
    if vecs.ndim == 1: vecs = vecs[np.newaxis, :]
    return names, vecs

def get_matches_key(h5_file_obj, q_name, db_name):
    found_key = None
    try:
        if q_name in h5_file_obj and db_name in h5_file_obj[q_name]:
            if 'matches0' in h5_file_obj[q_name][db_name]:
                return f"{q_name}/{db_name}/matches0"
    except: pass

    def _finder(name, obj):
        nonlocal found_key
        if found_key: return
        if isinstance(obj, h5py.Dataset) and name.endswith("matches0"):
            clean_path = name.replace("/matches0", "")
            padded_path = f"/{clean_path}/"
            padded_q = f"/{q_name}/"
            padded_db = f"/{db_name}/"
            if padded_q in padded_path and padded_db in padded_path:
                found_key = name
                return
            q_enc = q_name.replace('/', '-')
            db_enc = db_name.replace('/', '-')
            if q_enc in name and db_enc in name:
                found_key = name
                return
    h5_file_obj.visititems(_finder)
    return found_key

def parse_localization_log(pkl_path):
    results = {}
    if not Path(pkl_path).exists(): return results
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        loc_data = data.get('loc', data)
        if not isinstance(loc_data, dict): return results
        for img_name, info in loc_data.items():
            if isinstance(info, dict) and 'PnP_ret' in info:
                pnp = info['PnP_ret']
                inliers = 0
                if isinstance(pnp, dict): inliers = pnp.get('num_inliers', 0)
                else: inliers = getattr(pnp, 'num_inliers', 0)
                results[img_name] = inliers
            else: results[img_name] = 0
    except Exception as e:
        print(f"[IO Error] Failed to parse log {pkl_path}: {e}")
    return results