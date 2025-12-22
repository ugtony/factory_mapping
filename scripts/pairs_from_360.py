#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/pairs_from_360_v6.py
[Final Version]
1. 嚴格對應 User 定義的 VIEW_ANGLES (LF = -45)。
2. 實作完整的左右兩側輸送帶策略 (Cross-Over Strategy)。
3. 針對長廊場景優化：F->LF->L... 與 F->FR->R...
"""
import argparse
from pathlib import Path
from collections import defaultdict
import sys

# === [1. 嚴格定義視角] ===
# 依據您提供的轉換程式定義
VIEW_ANGLES = {
    'F': 0,   'FR': 45,  'R': 90,  'RB': 135,
    'B': 180, 'BL': -135,'L': -90, 'LF': -45
}

# === [2. 視角功能分組] ===
VIEWS_AXIAL   = {'F', 'B'} 
VIEWS_LATERAL = {'L', 'R'}

# === [3. 輸送帶流動規則 (Cross-Over Rules)] ===
# 邏輯：當相機前進 (t -> t+1) 時，物體在視野中的流動方向
# Key: 當前幀視角 (t) -> Value: 下一幀目標視角列表 (t+1)
CROSS_OVER_RULES = {
    # --- 起始點 (F) ---
    # F 同時流向左右兩邊 (分裂)
    'F':  ['FR', 'LF'],
    
    # --- 右側鏈 (Clockwise: 0 -> 45 -> 90 -> 135 -> 180) ---
    'FR': ['R'],       # 右前 -> 正右 (關鍵抗重複)
    'R':  ['RB'],      # 正右 -> 右後 (關鍵抗重複)
    'RB': ['B'],       # 右後 -> 正後
    
    # --- 左側鏈 (Counter-Clockwise: 0 -> -45 -> -90 -> -135 -> 180) ---
    'LF': ['L'],       # 左前 -> 正左 (關鍵抗重複)
    'L':  ['BL'],      # 正左 -> 左後 (關鍵抗重複)
    'BL': ['B']        # 左後 -> 正後
}

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
    parser = argparse.ArgumentParser(description="Generate pairs with Directional Window + Cross-Over Strategy.")
    parser.add_argument("--db_list", required=True, help="Path to HLOC db image list")
    parser.add_argument("--output", required=True, help="Output pairs file path")
    
    # [Directional Window Strategy]
    parser.add_argument("--window_axial", type=int, default=3, 
                        help="Window size for Axial views (F, B). Default: 3")
    
    parser.add_argument("--window_diagonal", type=int, default=1, 
                        help="Window size for Diagonal views (FR, LF, RB, BL). Default: 1")
    
    parser.add_argument("--window_lateral", type=int, default=0, 
                        help="Window size for Lateral views (L, R). Default: 0")

    # [Intra-frame Strategy]
    parser.add_argument("--intra_max_angle", type=float, default=45.0, 
                        help="Max angular difference for intra-frame pairs. Set 0 to disable.")

    # [Cross-Over Strategy]
    parser.add_argument("--enable_cross_over", action="store_true", 
                        help="Enable Cross-Over matching (e.g., LF_t -> L_t+1). Essential for corridors.")

    args = parser.parse_args()

    # 參數處理
    win_axial = args.window_axial
    win_diag  = args.window_diagonal if args.window_diagonal >= 0 else (win_axial // 2)
    win_lat   = args.window_lateral
    is_intra_enabled = args.intra_max_angle > 0
    is_crossover_enabled = args.enable_cross_over
    
    print("========================================")
    print(f"[Config] View Strategy:")
    print(f"  - Axial Window (F, B) : {win_axial}")
    print(f"  - Diag Window (LF,FR) : {win_diag}")
    print(f"  - Lat Window (L, R)   : {win_lat} (Suggest 0 for tiles)")
    print(f"  - Intra-frame Match   : {'ON' if is_intra_enabled else 'OFF'}")
    print(f"  - Cross-Over Flow     : {'ON' if is_crossover_enabled else 'OFF'} (Crucial!)")
    print("========================================")

    db_list_path = Path(args.db_list)
    if not db_list_path.exists():
        print(f"[Error] DB list not found: {db_list_path}", file=sys.stderr)
        sys.exit(1)

    # 1. 讀取與解析影像列表
    with open(db_list_path, 'r') as f:
        images = [line.strip() for line in f if line.strip()]

    frame_dict = defaultdict(dict)
    timestamps = []

    for img_path in images:
        stem = Path(img_path).stem
        parts = stem.split('_')
        if len(parts) < 2: continue
        
        # 取得最後一個 part 作為視角 (例如 LF)
        # 因為您的程式產生的一定是正確的大小寫，這裡直接比對
        view = parts[-1] 
        frame_id = "_".join(parts[:-1])
        
        if view in VIEW_ANGLES:
            frame_dict[frame_id][view] = img_path
            if frame_id not in timestamps:
                timestamps.append(frame_id)
            
    timestamps.sort()
    print(f"[Info] Found {len(timestamps)} frames.")

    pairs = []
    stats = {'intra': 0, 'inter_same': 0, 'inter_cross': 0}

    # 2. 生成配對
    for i, t_curr in enumerate(timestamps):
        views_curr = frame_dict[t_curr]
        view_keys = sorted(list(views_curr.keys()))
        
        # (A) Intra-frame: 幾何剛體連接
        if is_intra_enabled and len(view_keys) > 1:
            for v1_idx in range(len(view_keys)):
                for v2_idx in range(v1_idx + 1, len(view_keys)):
                    v1 = view_keys[v1_idx]
                    v2 = view_keys[v2_idx]
                    
                    diff = get_angle_diff(v1, v2)
                    if diff <= args.intra_max_angle:
                        pairs.append((views_curr[v1], views_curr[v2]))
                        stats['intra'] += 1

        # (B) Inter-frame: 同視角連接 (Same View)
        for view_type in views_curr:
            # 決定 Window 大小
            if view_type in VIEWS_AXIAL:
                c_win = win_axial
            elif view_type in VIEWS_LATERAL:
                c_win = win_lat
            else:
                c_win = win_diag 
            
            if c_win > 0:
                for j in range(1, c_win + 1):
                    if i + j >= len(timestamps): break
                    t_next = timestamps[i+j]
                    views_next = frame_dict[t_next]
                    
                    if view_type in views_next:
                        pairs.append((views_curr[view_type], views_next[view_type]))
                        stats['inter_same'] += 1

        # (C) Inter-frame: 輸送帶流動 (Cross-Over)
        if is_crossover_enabled:
            # 只看 t+1 (假設步距 1m, 流動最明顯發生在下一幀)
            if i + 1 < len(timestamps):
                t_next = timestamps[i+1]
                views_next = frame_dict[t_next]

                # 遍歷當前擁有的視角
                for v_curr in views_curr:
                    # 如果這個視角在規則中有定義流向 (例如 F -> [FR, LF])
                    if v_curr in CROSS_OVER_RULES:
                        target_list = CROSS_OVER_RULES[v_curr]
                        
                        for v_target in target_list:
                            # 如果下一幀有捕捉到這個目標視角
                            if v_target in views_next:
                                pairs.append((views_curr[v_curr], views_next[v_target]))
                                stats['inter_cross'] += 1

    # 3. 輸出結果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for p1, p2 in pairs:
            f.write(f"{p1} {p2}\n")
            
    print(f"[Success] Pairs Generated: {len(pairs)}")
    print(f"  - Intra-frame      : {stats['intra']}")
    print(f"  - Same-View Inter  : {stats['inter_same']}")
    print(f"  - Cross-Over Flow  : {stats['inter_cross']}")
    print(f"Output saved to {output_path}")

if __name__ == "__main__":
    main()