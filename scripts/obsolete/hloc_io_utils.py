#!/usr/bin/env python3
"""
hloc_io_utils.py
封裝 HLOC 檔案讀取的 Robust 邏輯，避免結構性錯誤。
"""

import h5py
import numpy as np
import pickle
from pathlib import Path

def load_global_descriptors_safe(h5_path):
    """
    安全讀取 Global Descriptors，確保回傳 (N, D) 的 2D 陣列。
    解決 N=1 時維度被 squeeze 成 1D 的問題。
    """
    names = []
    vectors = []
    
    if not Path(h5_path).exists():
        # print(f"[IO Error] File not found: {h5_path}")
        return [], np.array([])

    def visit(name, obj):
        if isinstance(obj, h5py.Group) and 'global_descriptor' in obj:
            names.append(name)
            vectors.append(obj['global_descriptor'].__array__())
            
    with h5py.File(h5_path, 'r') as f:
        f.visititems(visit)
    
    if len(vectors) == 0:
        return [], np.array([])
    
    vecs = np.array(vectors)
    
    # 處理 (N, 1, D) -> (N, D)
    if vecs.ndim == 3 and vecs.shape[1] == 1:
        vecs = vecs.squeeze(1)
        
    # 處理 (D,) -> (1, D)  [關鍵修正]
    if vecs.ndim == 1:
        vecs = vecs[np.newaxis, :]
        
    return names, vecs

def get_matches_key(h5_file_obj, q_name, db_name):
    """
    在 matches.h5 中尋找正確的 Key。
    自動處理 HLOC 的路徑編碼問題 (例如 '/' 被換成 '-')。
    """
    found_key = None
    
    # 1. 嘗試標準層級路徑 (Standard Hierarchical)
    # HLOC 預設可能存為 q_name/db_name
    try:
        if q_name in h5_file_obj and db_name in h5_file_obj[q_name]:
            # 檢查是否有 matches0
            if 'matches0' in h5_file_obj[q_name][db_name]:
                return f"{q_name}/{db_name}/matches0"
    except: pass

    # 2. 嘗試搜尋 (針對 encoded path)
    def _finder(name, obj):
        nonlocal found_key
        if found_key: return
        
        if isinstance(obj, h5py.Dataset) and name.endswith("matches0"):
            # 移除結尾，只看路徑部分
            clean_path = name.replace("/matches0", "")
            
            # 構造「邊界保護」的搜尋字串，避免 partial match (如 2.jpg 匹配 12.jpg)
            padded_path = f"/{clean_path}/"
            padded_q = f"/{q_name}/"
            padded_db = f"/{db_name}/"
            
            # 策略 A: 完整路徑包含
            if padded_q in padded_path and padded_db in padded_path:
                found_key = name
                return

            # 策略 B: HLOC 編碼路徑匹配 ('/' -> '-')
            q_enc = q_name.replace('/', '-')
            db_enc = db_name.replace('/', '-')
            
            # 這裡無法使用 padded 檢查，因為 - 是合法字元
            # 但我們檢查是否包含 encoded string
            if q_enc in name and db_enc in name:
                found_key = name
                return

    h5_file_obj.visititems(_finder)
    return found_key

def parse_localization_log(pkl_path):
    """
    解析 _logs.pkl，自動處理 nested structure (logs['loc'][q]).
    回傳一個 dict: {image_name: num_inliers}
    """
    results = {}
    if not Path(pkl_path).exists():
        return results

    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            
        # HLOC Log 結構通常在 'loc' key 底下，如果沒有 'loc' 則假設 data 本身就是
        loc_data = data.get('loc', data)
        
        if not isinstance(loc_data, dict):
            return results

        for img_name, info in loc_data.items():
            # 確保 info 是字典且有 PnP_ret
            if isinstance(info, dict) and 'PnP_ret' in info:
                pnp = info['PnP_ret']
                # 處理 PnP_ret 可能是 object 或 dict 的情況
                inliers = 0
                if isinstance(pnp, dict):
                    inliers = pnp.get('num_inliers', 0)
                else:
                    # 嘗試讀取屬性
                    inliers = getattr(pnp, 'num_inliers', 0)
                results[img_name] = inliers
            else:
                # 雖然有紀錄但沒有 PnP 結果 (可能失敗)
                results[img_name] = 0
                
    except Exception as e:
        print(f"[IO Error] Failed to parse log {pkl_path}: {e}")
        
    return results