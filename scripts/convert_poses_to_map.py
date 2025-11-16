#!/usr/bin/env python3
import argparse
import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
import pycolmap
import matplotlib.pyplot as plt

def get_sfm_center(sfm_dir, target_name):
    """從 COLMAP 模型中讀取指定圖片的中心座標 (World Frame)，支援模糊比對。"""
    sfm_path = Path(sfm_dir)
    if not (sfm_path / "images.bin").exists():
        if not (sfm_path / "images.txt").exists():
            raise FileNotFoundError(f"SfM model not found at {sfm_dir}")
            
    recon = pycolmap.Reconstruction(sfm_path)
    
    # 1. 精確比對
    for img_id, img in recon.images.items():
        if img.name == target_name:
            c = img.projection_center()
            return np.array([c[0], c[1]])
            
    # 2. 模糊比對
    candidates = []
    for img_id, img in recon.images.items():
        if img.name.endswith(f"/{target_name}") or img.name == f"db/{target_name}":
            candidates.append(img)
            
    if len(candidates) == 1:
        c = candidates[0].projection_center()
        print(f"    [Info] Fuzzy match: '{target_name}' -> '{candidates[0].name}'")
        return np.array([c[0], c[1]])
    elif len(candidates) > 1:
        print(f"    [Warn] Multiple matches for '{target_name}'. Using first.")
        c = candidates[0].projection_center()
        return np.array([c[0], c[1]])
            
    sample_names = [img.name for i, img in enumerate(recon.images.values()) if i < 5]
    raise ValueError(f"Image '{target_name}' not found in {sfm_dir}.\nSamples: {sample_names}")

def compute_sim2_transform(p_sfm_s, p_sfm_e, p_map_s, p_map_e):
    """計算 2D 相似變換 (Scale, Rotation, Translation)"""
    vec_sfm = p_sfm_e - p_sfm_s
    vec_map = p_map_e - p_map_s
    
    len_sfm = np.linalg.norm(vec_sfm)
    len_map = np.linalg.norm(vec_map)
    if len_sfm < 1e-6: raise ValueError("SfM anchors too close.")
    s = len_map / len_sfm
    
    ang_sfm = np.arctan2(vec_sfm[1], vec_sfm[0])
    ang_map = np.arctan2(vec_map[1], vec_map[0])
    theta = ang_map - ang_sfm
    
    c, si = np.cos(theta), np.sin(theta)
    R = np.array([[c, -si], [si, c]])
    t = p_map_s - s * (R @ p_sfm_s)
    
    return s, theta, t

def parse_pose(qw, qx, qy, qz, tx, ty, tz):
    """
    解析 Pose (World-to-Camera) -> (Center, Yaw)
    [Fix] 改為直接計算 +Z 軸 (Forward) 的方向，確保箭頭指向正前方。
    """
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

def plot_results(output_png, data_points, anchors_cfg):
    """繪製結果圖 (包含 Poses, Anchors, Labels)"""
    plt.figure(figsize=(14, 14)) 
    plt.title("Localization Results (Corrected Heading)")
    plt.xlabel("Map X")
    plt.ylabel("Map Y")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal')

    unique_blocks = sorted(list(set(d['block'] for d in data_points)))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_blocks)))
    block_color_map = {b: c for b, c in zip(unique_blocks, colors)}

    # 1. 繪製相機位置
    for d in data_points:
        x, y, yaw = d['x'], d['y'], d['yaw']
        color = block_color_map.get(d['block'], 'black')
        
        plt.scatter(x, y, c=[color], s=30, label=d['block'], edgecolors='k', linewidth=0.5, alpha=0.7)
        
        # 箭頭
        arrow_len = 15.0 
        dx = arrow_len * np.cos(np.deg2rad(yaw))
        dy = arrow_len * np.sin(np.deg2rad(yaw))
        plt.arrow(x, y, dx, dy, head_width=arrow_len*0.3, head_length=arrow_len*0.4, fc=color, ec=color, alpha=0.8)
        
        # 檔名標籤
        short_name = Path(d['name']).name
        plt.text(x, y, f"  {short_name}", fontsize=6, color=color, alpha=0.8, rotation=45)

    # 2. 繪製 Anchors
    added_anchor_label = False
    for block_name, cfg in anchors_cfg.items():
        sx, sy = cfg['start_map_xy']
        ex, ey = cfg['end_map_xy']
        
        plt.scatter(sx, sy, c='red', marker='x', s=150, linewidth=2.5, 
                    label='Anchors' if not added_anchor_label else "", zorder=10)
        plt.text(sx, sy, f" {block_name}_Start", color='red', fontsize=8, fontweight='bold', zorder=11, verticalalignment='bottom')
        
        plt.scatter(ex, ey, c='red', marker='x', s=150, linewidth=2.5, zorder=10)
        plt.text(ex, ey, f" {block_name}_End", color='red', fontsize=8, fontweight='bold', zorder=11, verticalalignment='bottom')
        
        plt.plot([sx, ex], [sy, ey], 'r--', alpha=0.3, linewidth=1)
        added_anchor_label = True

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
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
            p_sfm_s = get_sfm_center(cfg['sfm_path'], cfg['start_frame'])
            p_sfm_e = get_sfm_center(cfg['sfm_path'], cfg['end_frame'])
            p_map_s = np.array(cfg['start_map_xy'])
            p_map_e = np.array(cfg['end_map_xy'])
            s, theta, t = compute_sim2_transform(p_sfm_s, p_sfm_e, p_map_s, p_map_e)
            transforms[block_name] = (s, theta, t)
            print(f"  > {block_name}: Scale={s:.4f}, Rot={np.degrees(theta):.2f}°")
        except Exception as e:
            print(f"  [Error] {block_name} failed: {e}")

    if not transforms: return

    print(f"\n[Step 2] Converting poses...")
    plot_data = []
    
    with open(args.submission, 'r') as f_in, open(args.output, 'w') as f_out:
        f_out.write("ImageName, MapX, MapY, MapYaw, BlockName\n")
        count = 0
        for line in f_in:
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split()
            if len(parts) < 9: continue
            
            try:
                name = parts[0]
                vals = list(map(float, parts[1:8]))
                block_name = parts[8]
            except ValueError: continue
            
            if block_name not in transforms: continue
            
            sfm_center, sfm_yaw = parse_pose(*vals)
            s, theta, t_vec = transforms[block_name]
            
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