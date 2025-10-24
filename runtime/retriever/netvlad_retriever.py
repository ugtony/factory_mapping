import h5py, numpy as np
from pathlib import Path

class NetVLADRetriever:
    # 簡單的 brute-force NetVLAD 檢索器（預留 FAISS 介面）
    def __init__(self, out_root="outputs-hloc"):
        self.out_root = Path(out_root)

    def retrieve(self, query_desc_path, candidate_blocks, top_k=10):
        q = self._read_single_feature(query_desc_path)
        qn = q / (np.linalg.norm(q) + 1e-6)
        results = []
        for block in candidate_blocks:
            h5 = self.out_root / block / "global-netvlad.h5"
            if not h5.exists():
                continue
            with h5py.File(h5, "r") as f:
                for k in f.keys():
                    v = f[k][()]
                    vn = v / (np.linalg.norm(v) + 1e-6)
                    score = float(np.dot(qn, vn))
                    results.append((block, k, score))
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]

    def _read_single_feature(self, path):
        path = Path(path)
        with h5py.File(path, "r") as f:
            for _k in f.keys():
                return f[_k][()]
        raise RuntimeError(f"No descriptor found in {path}")
