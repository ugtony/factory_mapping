#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/pairs_from_360.py
[Modified] 支援 Backbone Strategy (脊椎策略)：
新增 --inter_frame_suffixes 參數，允許指定只有特定視角（如 'F'）能跨時間連接。
"""
import argparse
from pathlib import Path
from collections import defaultdict
import sys

def main():
    parser = argparse.ArgumentParser(description="Generate pairs for sliced 360 images explicitly.")
    # [原有參數] 保持與 build_block_model.sh 相容
    parser.add_argument("--db_list", required=True, help="Path to HLOC db image list (e.g., outputs/block_001/db.txt)")
    parser.add_argument("--output", required=True, help="Output pairs file path.")
    parser.add_argument("--seq_window", type=int, default=5, help="Temporal window size for same-view pairs.")
    parser.add_argument("--intra_match", action="store_true", help="Enable intra-frame (cross-view) matching.")
    
    # [新增參數] 用於控制脊椎策略
    parser.add_argument("--inter_frame_suffixes", type=str, default=None, 
                        help="Comma-separated list of suffixes allowed for inter-frame matching (e.g., 'F'). "
                             "If not set, all views are matched sequentially (default behavior).")
    
    args = parser.parse_args()

    db_list_path = Path(args.db_list)
    if not db_list_path.exists():
        print(f"[Error] DB list not found: {db_list_path}", file=sys.stderr)
        sys.exit(1)

    # 解析允許的 Suffixes
    allowed_inter_suffixes = None
    if args.inter_frame_suffixes:
        allowed_inter_suffixes = set(s.strip() for s in args.inter_frame_suffixes.split(',') if s.strip())
        print(f"[Info] Inter-frame matching restricted to suffixes: {allowed_inter_suffixes}")
    else:
        print(f"[Info] Inter-frame matching enabled for ALL suffixes.")

    # 1. 讀取影像列表
    with open(db_list_path, 'r') as f:
        images = [line.strip() for line in f if line.strip()]

    if not images:
        print(f"[Error] No images found in {db_list_path}", file=sys.stderr)
        sys.exit(1)

    # 2. 解析結構
    # 預期格式: db/{FRAME_ID}_{VIEW}.jpg (例如 db/frame_00123_F.jpg)
    frame_dict = defaultdict(dict)
    timestamps = []

    print(f"[Info] Scanning {len(images)} images from db_list...")
    for img_path in images:
        stem = Path(img_path).stem
        parts = stem.split('_')
        if len(parts) < 2: continue
        
        # 假設最後一個 part 是 view (F, R, B, L...)
        view = parts[-1]
        frame_id = "_".join(parts[:-1])
        
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
        
        # (A) Intra-frame (同時間點內的不同視角互連)
        # 這裡不受 Suffix 限制，保持 Rig 的剛體結構
        if args.intra_match and len(view_keys) > 1:
            for v1_idx in range(len(view_keys)):
                for v2_idx in range(v1_idx + 1, len(view_keys)):
                    pairs.append((views_curr[view_keys[v1_idx]], views_curr[view_keys[v2_idx]]))
                    intra_count += 1

        # (B) Inter-frame (跨時間點的移動路徑)
        # 這裡套用脊椎策略過濾
        for j in range(1, args.seq_window + 1):
            if i + j >= len(timestamps): break
            t_next = timestamps[i+j]
            views_next = frame_dict[t_next]
            
            for view_type in views_curr:
                # 只有當下一個時間點也有同樣的 view (例如 F 對 F) 才配對
                if view_type in views_next:
                    
                    # [過濾邏輯] 若有設定允許清單，且該 view 不在清單中，則跳過
                    if allowed_inter_suffixes is not None:
                        if view_type not in allowed_inter_suffixes:
                            continue
                            
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