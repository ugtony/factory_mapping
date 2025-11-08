#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/pairs_from_360.py
[Fix] 改為讀取 db_list 以確保路徑前綴 (e.g., db/) 與 HLOC 一致。
"""
import argparse
from pathlib import Path
from collections import defaultdict
import sys

def main():
    parser = argparse.ArgumentParser(description="Generate pairs for sliced 360 images explicitly.")
    # [修改] 改用 db_list 作為輸入
    parser.add_argument("--db_list", required=True, help="Path to HLOC db image list (e.g., outputs/block_001/db.txt)")
    parser.add_argument("--output", required=True, help="Output pairs file path.")
    parser.add_argument("--seq_window", type=int, default=5, help="Temporal window size for same-view pairs.")
    parser.add_argument("--intra_match", action="store_true", help="Enable intra-frame (cross-view) matching.")
    args = parser.parse_args()

    db_list_path = Path(args.db_list)
    if not db_list_path.exists():
        print(f"[Error] DB list not found: {db_list_path}", file=sys.stderr)
        sys.exit(1)

    # 1. 讀取影像列表
    with open(db_list_path, 'r') as f:
        images = [line.strip() for line in f if line.strip()]

    if not images:
        print(f"[Error] No images found in {db_list_path}", file=sys.stderr)
        sys.exit(1)

    # 2. 解析結構
    # 預期格式: db/{FRAME_ID}_{VIEW}.jpg
    frame_dict = defaultdict(dict)
    timestamps = []

    print(f"[Info] Scanning {len(images)} images from db_list...")
    for img_path in images:
        # 使用 Path(img_path).stem 取得不含路徑與副檔名的檔名 (e.g., "frame_001_F")
        stem = Path(img_path).stem
        parts = stem.split('_')
        if len(parts) < 2: continue
        
        view = parts[-1]
        frame_id = "_".join(parts[:-1])
        
        # 儲存完整的 img_path (包含 db/ 前綴)
        frame_dict[frame_id][view] = img_path
        if frame_id not in timestamps:
            timestamps.append(frame_id)
            
    timestamps.sort()
    print(f"[Info] Found {len(timestamps)} unique 360 frames.")

    pairs = []
    intra_count = 0
    inter_count = 0

    # 3. 生成配對
    for i, t_curr in enumerate(timestamps):
        views_curr = frame_dict[t_curr]
        view_keys = sorted(list(views_curr.keys()))
        
        # (A) Intra-frame
        if args.intra_match and len(view_keys) > 1:
            for v1_idx in range(len(view_keys)):
                for v2_idx in range(v1_idx + 1, len(view_keys)):
                    pairs.append((views_curr[view_keys[v1_idx]], views_curr[view_keys[v2_idx]]))
                    intra_count += 1

        # (B) Inter-frame
        for j in range(1, args.seq_window + 1):
            if i + j >= len(timestamps): break
            t_next = timestamps[i+j]
            views_next = frame_dict[t_next]
            for view_type in views_curr:
                if view_type in views_next:
                    pairs.append((views_curr[view_type], views_next[view_type]))
                    inter_count += 1

    print(f"[Info] Generated {len(pairs)} pairs (Intra: {intra_count}, Inter: {inter_count}).")

    # 4. 輸出
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for p1, p2 in pairs:
            f.write(f"{p1} {p2}\n")

if __name__ == "__main__":
    main()