#!/usr/bin/env python3
import argparse
import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
import pycolmap
import matplotlib.pyplot as plt

# [New] Import shared logic
try:
    from map_utils import compute_sim2_transform, get_data_bounds  # get_data_bounds may not be in map_utils yet
except ImportError:
    from .map_utils import compute_sim2_transform

def parse_pose(qw, qx, qy, qz, tx, ty, tz):
    """解析 Pose (World-to-Camera) -> (Center, Yaw)"""
    # 1. 建立旋轉矩陣 (World to Camera)
    rot_w2c = Rotation.from_quat([qx, qy, qz, qw])
    R_w2c = rot_w2c.as_matrix()
    t_vec = np.array([tx, ty, tz])
    
    # 2. 轉為 Camera to World
    # R_c2w = R_w2c.T
    R_c2w = R_w2c.T
    center = -R_c2w @ t_vec
    
    # 3. 計算 Yaw (基於相機正前方 +Z 軸)
    # 在 COLMAP 相機座標系中，[0, 0, 1] 是正前方
    # 轉換到世界座標系： view_dir_world = R_c2w @ [0, 0, 1]^T
    # 這正好是 R_c2w 的第三個 column (index 2)
    view_dir = R_c2w[:, 2] 
    
    # 使用 arctan2(y, x) 計算平面上的方位角
    yaw = np.degrees(np.arctan2(view_dir[1], view_dir[0]))
    
    return center, yaw

def get_data_bounds(data_points, anchors_cfg):
    """[New] 計算資料邊界與跨度"""
    xs, ys = [], []
    
    for d in data_points:
        xs.append(d['x'])
        ys.append(d['y'])
        
    for cfg in anchors_cfg.values():
        xs.append(cfg['start_map_xy'][0])
        ys.append(cfg['start_map_xy'][1])
        xs.append(cfg['end_map_xy'][0])
        ys.append(cfg['end_map_xy'][1])
        
    if not xs: return (0,1,0,1), 1.0
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    span_x = max_x - min_x
    span_y = max_y - min_y
    max_span = max(span_x, span_y)
    
    return (min_x, max_x, min_y, max_y), max_span

def plot_results(output_png, data_points, anchors_cfg):
    """繪製結果圖 (包含 Poses, Anchors, Labels) - 動態畫布版"""
    
    # 1. 計算資料範圍與長寬比
    (min_x, max_x, min_y, max_y), map_span = get_data_bounds(data_points, anchors_cfg)
    
    w_range = max_x - min_x
    h_range = max_y - min_y
    
    # 加上 10% 的邊距 (Padding)
    pad_x = max(w_range * 0.1, 1.0)
    pad_y = max(h_range * 0.1, 1.0)
    
    plot_xlim = (min_x - pad_x, max_x + pad_x)
    plot_ylim = (min_y - pad_y, max_y + pad_y)
    
    final_w = plot_xlim[1] - plot_xlim[0]
    final_h = plot_ylim[1] - plot_ylim[0]
    
    # 2. 動態設定 figsize，確保畫布比例接近資料比例
    # 設定最大邊長為 14 inch
    max_fig_size = 14
    aspect = final_w / final_h
    
    if aspect > 1:
        fig_w = max_fig_size
        fig_h = max_fig_size / aspect
    else:
        fig_h = max_fig_size
        fig_w = max_fig_size * aspect
    
    # 最小保護 (避免太扁或太窄)
    fig_w = max(fig_w, 5)
    fig_h = max(fig_h, 5)

    plt.figure(figsize=(fig_w, fig_h)) 
    plt.title("Localization Results (Auto-Fit)")
    plt.xlabel("Map X")
    plt.ylabel("Map Y")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 3. 設定繪圖參數
    # 箭頭長度設為地圖最大跨度的 2%
    arrow_len = map_span * 0.02
    if arrow_len < 0.1: arrow_len = 0.5 # 最小值保護
    
    anchor_size = 150
    text_offset = arrow_len * 0.6

    unique_blocks = sorted(list(set(d['block'] for d in data_points)))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_blocks)))
    block_color_map = {b: c for b, c in zip(unique_blocks, colors)}

    # 4. 繪製相機位置
    for d in data_points:
        x, y, yaw = d['x'], d['y'], d['yaw']
        color = block_color_map.get(d['block'], 'black')
        
        plt.scatter(x, y, c=[color], s=30, label=d['block'], edgecolors='k', linewidth=0.5, alpha=0.7)
        
        dx = arrow_len * np.cos(np.deg2rad(yaw))
        dy = arrow_len * np.sin(np.deg2rad(yaw))
        
        plt.arrow(x, y, dx, dy, 
                  head_width=arrow_len*0.4, 
                  head_length=arrow_len*0.5, 
                  fc=color, ec=color, alpha=0.8)
        
        short_name = Path(d['name']).name
        plt.text(x + text_offset, y + text_offset, f"{short_name}", 
                 fontsize=6, color=color, alpha=0.8, rotation=45)

    # 5. 繪製 Anchors
    added_anchor_label = False
    for block_name, cfg in anchors_cfg.items():
        sx, sy = cfg['start_map_xy']
        ex, ey = cfg['end_map_xy']
        
        plt.scatter(sx, sy, c='red', marker='x', s=anchor_size, linewidth=2.5, 
                    label='Anchors' if not added_anchor_label else "", zorder=10)
        plt.text(sx, sy - text_offset, f" {block_name}_Start", color='red', fontsize=8, fontweight='bold', zorder=11, verticalalignment='top')
        
        plt.scatter(ex, ey, c='red', marker='x', s=anchor_size, linewidth=2.5, zorder=10)
        plt.text(ex, ey - text_offset, f" {block_name}_End", color='red', fontsize=8, fontweight='bold', zorder=11, verticalalignment='top')
        
        plt.plot([sx, ex], [sy, ey], 'r--', alpha=0.3, linewidth=1)
        added_anchor_label = True

    # 6. 設定顯示範圍與比例
    plt.xlim(plot_xlim)
    plt.ylim(plot_ylim)
    plt.axis('equal') # 重要：保持物理比例不變形

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 1))

    # 7. 儲存時裁切白邊
    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f"🖼️  Plot saved to: {output_png}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, required=True, help="hloc result file")
    parser.add_argument("--anchors", type=Path, required=True, help="anchors.json config")
    parser.add_argument("--output", type=Path, default="submission_map.txt")
    parser.add_argument("--plot", action="store_true", help="Generate visualization plot")
    args = parser.parse_args()

    with open(args.anchors, 'r') as f: anchors_cfg = json.load(f)
    transforms = {}
    print(f"[Step 1] Computing transforms...")
    
    for block_name, cfg in anchors_cfg.items():
        try:
            sfm_path = Path(cfg['sfm_path'])
            # 這裡為了計算 Transform，需短暫載入 Recon
            # 雖然有點重，但這是離線轉檔腳本，還可以接受
            recon = pycolmap.Reconstruction(sfm_path)
            
            # [Updated] Use modularized utility
            trans = compute_sim2_transform(recon, cfg)
            
            if trans:
                transforms[block_name] = trans
                s = trans['s']
                theta = trans['theta']
                print(f"  > {block_name}: Scale={s:.4f}, Rot={np.degrees(theta):.2f}°")
            else:
                print(f"  [Warn] Could not compute transform for {block_name}")
                
        except Exception as e:
            print(f"  [Error] {block_name} failed: {e}")

    if not transforms: 
        print("[Error] No valid transforms computed.")
        return

    print(f"\n[Step 2] Converting poses...")
    plot_data = []
    
    with open(args.submission, 'r') as f_in, open(args.output, 'w') as f_out:
        f_out.write("ImageName, MapX, MapY, MapYaw, BlockName\n")
        count = 0
        for line in f_in:
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split()
            if len(parts) < 8: continue
            
            try:
                name = parts[0]
                vals = list(map(float, parts[1:8]))
                if len(parts) >= 9:
                    block_name = parts[8]
                else:
                    if len(transforms) == 1:
                        block_name = list(transforms.keys())[0]
                    else:
                        continue
            except ValueError: continue
            
            if block_name not in transforms: continue
            
            sfm_center, sfm_yaw = parse_pose(*vals)
            t_data = transforms[block_name]
            s, theta, t_vec = t_data['s'], t_data['theta'], t_data['t']
            
            c, si = np.cos(theta), np.sin(theta)
            R_mat = np.array([[c, -si], [si, c]])
            p_map = s * (R_mat @ sfm_center[:2]) + t_vec
            
            map_yaw = sfm_yaw + np.degrees(theta)
            map_yaw = (map_yaw + 180) % 360 - 180
            
            f_out.write(f"{name}, {p_map[0]:.4f}, {p_map[1]:.4f}, {map_yaw:.4f}, {block_name}\n")
            
            plot_data.append({
                'name': name, 'x': p_map[0], 'y': p_map[1], 'yaw': map_yaw, 'block': block_name
            })
            count += 1

    print(f"✅ Done! Converted {count} poses to '{args.output}'")

    if args.plot and plot_data:
        png_path = args.output.with_suffix('.png')
        plot_results(png_path, plot_data, anchors_cfg)

if __name__ == "__main__":
    main()