import h5py, numpy as np
from pathlib import Path

class NetVLADRetriever:
    # 簡單的 brute-force NetVLAD 檢索器
    def __init__(self, out_root="outputs-hloc"):
        self.out_root = Path(out_root)

    def retrieve(self, query_desc_path, candidate_blocks, top_k=10):
        # 1. 讀取 Query 特徵
        q = self._read_single_feature(query_desc_path)
        if q is None:
            raise RuntimeError(f"Could not find any dataset in query file: {query_desc_path}")
            
        qn = q / (np.linalg.norm(q) + 1e-6)
        q_shape = q.shape 
        
        # print(f"[Debug] Query descriptor shape: {q_shape}") # <-- 移除
        
        results = []
        
        for block in candidate_blocks:
            h5_path = self.out_root / block / "global-netvlad.h5"
            if not h5_path.exists():
                continue
            
            with h5py.File(h5_path, "r") as f:
                
                datasets = []
                def find_matching_datasets(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        if obj.shape == q_shape:
                            datasets.append(name)
                        # else: # <-- 移除
                            # print(f"[Debug] Skipping '{name}' (Shape {obj.shape} != {q_shape})")

                f.visititems(find_matching_datasets)
                
                if not datasets:
                    print(f"[Warn] No matching descriptors found in {h5_path}")
                    continue

                for k in datasets:
                    v = f[k][()]
                    vn = v / (np.linalg.norm(v) + 1e-6)
                    score = float(np.dot(qn, vn))
                    results.append((block, k, score)) # k 是 '.../global_descriptor'
                    
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]

    def _read_single_feature(self, path):
        """
        遞迴讀取 H5 檔案中的第一個 Dataset
        """
        path = Path(path)
        with h5py.File(path, "r") as f:
            dataset_path = None
            def find_first_dataset(name, obj):
                nonlocal dataset_path
                if isinstance(obj, h5py.Dataset):
                    dataset_path = name
                    return True 
            
            f.visititems(find_first_dataset)
            
            if dataset_path:
                # print(f"[Debug] Found query descriptor key: {dataset_path}") # <-- 移除
                return f[dataset_path][()]
                
        raise RuntimeError(f"No descriptor (dataset) found in {path}")