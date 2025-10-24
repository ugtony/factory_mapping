import json
import numpy as np
from pathlib import Path

def _cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))

class WifiRouter:
    # Wi-Fi 指紋比對：輸入一次掃描結果，輸出候選 block 清單
    def __init__(self, fingerprint_path=None):
        if fingerprint_path is None:
            fingerprint_path = Path(__file__).parent / "wifi_fingerprint.json"
        self.fingerprint_path = Path(fingerprint_path)
        if not self.fingerprint_path.exists():
            raise FileNotFoundError(f"No Wi-Fi fingerprint file: {self.fingerprint_path}")
        self.db = json.loads(self.fingerprint_path.read_text())

    def predict_blocks(self, wifi_scan: dict, top_k: int = 3):
        scores = []
        for block, fp in self.db.items():
            common = set(wifi_scan.keys()) & set(fp.keys())
            if not common:
                continue
            a = [wifi_scan[k] for k in common]
            b = [fp[k] for k in common]
            sim = _cosine(a, b)
            scores.append((block, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
