#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/pairs_from_360.py
[Modified V4] 
1. 參數命名：Axial/Diagonal/Lateral。
2. 預設策略調整：Lateral 預設為 1 (允許相鄰連接)，Intra 預設開啟 (Angle=90)。
3. 簡化參數：移除 --intra_match，改由 angle=0 控制關閉。
"""
import argparse
from pathlib import Path
from collections import defaultdict
import sys

# 定義視角與角度的對應關係
VIEW_ANGLES = {
    'F': 0,   'FR': 45,  'R': 90,  'RB': 135,
    'B': 180, 'BL': -135,'L': -90, 'LF': -45
}

# === [視角分組定義] ===
# Axial (軸向): 沿著移動路徑前後，視差變化最顯著，用於鎖定軌跡 (Spine)。
VIEWS_AXIAL   = {'F', 'B'} 

# Lateral (側向): 垂直於移動路徑，易受重複紋理影響。
VIEWS_LATERAL = {'L', 'R'}

# Diagonal (對角): 其餘視角 (FR, FL, RB, LB)，作為輔助約束。

def get_angle_diff(suffix1, suffix2):
    """計算兩個後綴對應的角度差 (最小夾角)"""
    if suffix1 not in VIEW_ANGLES or suffix2 not in VIEW_ANGLES:
        return 999.0 
    
    deg1 = VIEW_ANGLES[suffix1]
    deg2 = VIEW_ANGLES[suffix2]
    
    diff = abs(deg1 - deg2)
    diff = min(diff, 360 - diff)
    return diff

def main():
    parser = argparse.ArgumentParser(description="Generate pairs with Directional Window Strategy.")
    parser.add_argument("--db_list", required=True, help="Path to HLOC db image list")
    parser.add_argument("--output", required=True, help="Output pairs file path")
    
    # [Directional Window Strategy Config]
    # 1. Axial (F, B): 負責主要的縱向位移估計
    parser.add_argument("--window_axial", type=int, default=5, 
                        help="Window size for Axial views (F, B). Default: 5")
    
    # 2. Diagonal (FR, FL...): 負責輔助連接
    parser.add_argument("--window_diagonal", type=int, default=-1, 
                        help="Window size for Diagonal views. Default: -1 (Auto = axial // 2)")
    
    # 3. Lateral (L, R): 側向視角
    # [Mod V4] 預設改為 1 (允許與正隔壁幀連接，但不延伸)
    parser.add_argument("--window_lateral", type=int, default=1, 
                        help="Window size for Lateral views (L, R). Default: 1")

    # [Intra-frame Config]
    # [Mod V4] 移除 --intra_match，改由 angle 控制 (設為 0 即關閉)
    # [Mod V4] 預設改為 90 (允許 F-R 直接連接，增加剛性)
    parser.add_argument("--intra_max_angle", type=float, default=90.0, 
                        help="Max angular difference for intra-frame pairs. Set 0 to disable. (Default: 90)")

    args = parser.parse_args()

    # 參數解析與預設值邏輯
    win_axial = args.window_axial
    win_diag  = args.window_diagonal if args.window_diagonal >= 0 else (win_axial // 2)
    win_lat   = args.window_lateral
    
    # 判斷 Intra-frame 狀態
    is_intra_enabled = args.intra_max_angle > 0
    
    print("========================================")
    print(f"[Config] Directional Window Strategy:")
    print(f"  - Axial    (F, B)         : Window = {win_axial}")
    print(f"  - Diagonal (FR, FL, etc.) : Window = {win_diag}")
    print(f"  - Lateral  (L, R)         : Window = {win_lat}")
    print(f"[Config] Intra-frame Strategy:")
    print(f"  - Enabled                 : {is_intra_enabled}")
    print(f"  - Max Angle               : {args.intra_max_angle}°")
    print("========================================")

    db_list_path = Path(args.db_list)
    if not db_list_path.exists():
        print(f"[Error] DB list not found: {db_list_path}", file=sys.stderr)
        sys.exit(1)

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
    print(f"[Info] Found {len(timestamps)} frames.")

    pairs = []
    intra_count = 0
    inter_count = 0

    # 3. Generate Pairs
    for i, t_curr in enumerate(timestamps):
        views_curr = frame_dict[t_curr]
        view_keys = sorted(list(views_curr.keys()))
        
        # (A) Intra-frame: Geometric Filtering
        # 只要 angle > 0 且有多個視角，就嘗試連接
        if is_intra_enabled and len(view_keys) > 1:
            for v1_idx in range(len(view_keys)):
                for v2_idx in range(v1_idx + 1, len(view_keys)):
                    v1 = view_keys[v1_idx]
                    v2 = view_keys[v2_idx]
                    
                    diff = get_angle_diff(v1, v2)
                    if diff <= args.intra_max_angle:
                        pairs.append((views_curr[v1], views_curr[v2]))
                        intra_count += 1

        # (B) Inter-frame: Directional Window Strategy
        for view_type in views_curr:
            # 根據視角方向決定 Window Size
            current_window = 0
            
            if view_type in VIEWS_AXIAL:
                current_window = win_axial
            elif view_type in VIEWS_LATERAL:
                current_window = win_lat
            else:
                current_window = win_diag # 其餘皆為 Diagonal
            
            # 若 window <= 0，則跳過此視角的 Inter-frame
            if current_window <= 0:
                continue

            # 執行時間序列配對
            for j in range(1, current_window + 1):
                if i + j >= len(timestamps): break
                t_next = timestamps[i+j]
                views_next = frame_dict[t_next]
                
                if view_type in views_next:
                    pairs.append((views_curr[view_type], views_next[view_type]))
                    inter_count += 1

    # 4. Output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for p1, p2 in pairs:
            f.write(f"{p1} {p2}\n")
            
    print(f"[Success] Pairs: {len(pairs)} (Intra: {intra_count}, Inter: {inter_count})")
    print(f"          Output saved to {output_path}")

if __name__ == "__main__":
    main()