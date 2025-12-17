#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/pairs_from_360.py
[Modified] 
1. 支援 Backbone Strategy (inter_frame_suffixes)
2. [New] 支援基於角度的 Intra-frame 過濾，避免無重疊的視角 (如 F-B) 互連。
"""
import argparse
from pathlib import Path
from collections import defaultdict
import sys

# 定義視角與角度的對應關係 (依照 convert360_to_pinhole.py 的定義)
VIEW_ANGLES = {
    'F': 0,   'FR': 45,  'R': 90,  'RB': 135,
    'B': 180, 'BL': -135,'L': -90, 'LF': -45
}

def get_angle_diff(suffix1, suffix2):
    """計算兩個後綴對應的角度差 (最小夾角)"""
    if suffix1 not in VIEW_ANGLES or suffix2 not in VIEW_ANGLES:
        return 999.0 # 未知視角，預設不連或全連，這裡設大一點讓 filter 決定
    
    deg1 = VIEW_ANGLES[suffix1]
    deg2 = VIEW_ANGLES[suffix2]
    
    diff = abs(deg1 - deg2)
    diff = min(diff, 360 - diff) # 處理跨越 0/360 的情況
    return diff

def main():
    parser = argparse.ArgumentParser(description="Generate pairs for sliced 360 images explicitly.")
    parser.add_argument("--db_list", required=True, help="Path to HLOC db image list")
    parser.add_argument("--output", required=True, help="Output pairs file path.")
    parser.add_argument("--seq_window", type=int, default=5, help="Temporal window size.")
    parser.add_argument("--intra_match", action="store_true", help="Enable intra-frame matching.")
    
    # [Backbone Strategy]
    parser.add_argument("--inter_frame_suffixes", type=str, default=None, 
                        help="Suffixes allowed for inter-frame matching (e.g., 'F').")
    
    # [New] Intra-frame 角度閾值
    # 預設 60 度：這會允許 45度鄰居 (F-FR) 連接，但拒絕 90度 (F-R) 和 180度 (F-B) 連接。
    # 這樣在 Dense(8view) 模式下會形成完美的環狀結構。
    parser.add_argument("--intra_max_angle", type=float, default=60.0, 
                        help="Max angular difference for intra-frame pairs. (Default: 60)")

    args = parser.parse_args()

    db_list_path = Path(args.db_list)
    if not db_list_path.exists():
        print(f"[Error] DB list not found: {db_list_path}", file=sys.stderr)
        sys.exit(1)

    # Backbone Suffix Parsing
    allowed_inter_suffixes = None
    if args.inter_frame_suffixes:
        allowed_inter_suffixes = set(s.strip() for s in args.inter_frame_suffixes.split(',') if s.strip())
        print(f"[Info] Inter-frame restricted to: {allowed_inter_suffixes}")

    # 1. Read DB List
    with open(db_list_path, 'r') as f:
        images = [line.strip() for line in f if line.strip()]

    # 2. Parse Frames
    frame_dict = defaultdict(dict)
    timestamps = []

    for img_path in images:
        stem = Path(img_path).stem
        parts = stem.split('_')
        if len(parts) < 2: continue
        
        view = parts[-1]
        frame_id = "_".join(parts[:-1])
        
        frame_dict[frame_id][view] = img_path
        if frame_id not in timestamps:
            timestamps.append(frame_id)
            
    timestamps.sort()
    print(f"[Info] Found {len(timestamps)} frames. Intra-max-angle: {args.intra_max_angle}°")

    pairs = []
    intra_count = 0
    inter_count = 0

    # 3. Generate Pairs
    for i, t_curr in enumerate(timestamps):
        views_curr = frame_dict[t_curr]
        view_keys = sorted(list(views_curr.keys()))
        
        # (A) Intra-frame: Geometric Filtering
        if args.intra_match and len(view_keys) > 1:
            for v1_idx in range(len(view_keys)):
                for v2_idx in range(v1_idx + 1, len(view_keys)):
                    v1 = view_keys[v1_idx]
                    v2 = view_keys[v2_idx]
                    
                    # [New] Check Angle Difference
                    diff = get_angle_diff(v1, v2)
                    if diff <= args.intra_max_angle:
                        pairs.append((views_curr[v1], views_curr[v2]))
                        intra_count += 1
                    # else: F and B (diff=180) skipped.

        # (B) Inter-frame: Backbone Strategy
        for j in range(1, args.seq_window + 1):
            if i + j >= len(timestamps): break
            t_next = timestamps[i+j]
            views_next = frame_dict[t_next]
            
            for view_type in views_curr:
                if view_type in views_next:
                    if allowed_inter_suffixes is not None:
                        if view_type not in allowed_inter_suffixes:
                            continue
                    pairs.append((views_curr[view_type], views_next[view_type]))
                    inter_count += 1

    # 4. Output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for p1, p2 in pairs:
            f.write(f"{p1} {p2}\n")
            
    print(f"[Success] Pairs: {len(pairs)} (Intra: {intra_count}, Inter: {inter_count})")

if __name__ == "__main__":
    main()