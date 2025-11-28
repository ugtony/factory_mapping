#!/usr/bin/env python3
import argparse
import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
import pycolmap
import matplotlib.pyplot as plt

def find_auto_anchors(sfm_dir):
    """
    [New] 自動從 SfM 模型中尋找第一張與最後一張影像。
    優先尋找 _F.jpg (360 模式的前視角)，若無則使用一般排序。
    """
    sfm_path = Path(sfm_dir)
    if not (sfm_path / "images.bin").exists() and not (sfm_path / "images.txt").exists():
        raise FileNotFoundError(f"SfM model not found at {sfm_dir}")

    recon = pycolmap.Reconstruction(sfm_path)
    
    # 取得所有影像名稱並排序
    all_images = sorted([img.name for img in recon.images.values()])
    
    if not all_images:
        raise ValueError(f"No images found in reconstruction: {sfm_dir}")

    # 1. 嘗試過濾出 _F (Front view) 的影像
    f_images = [name for name in all_images if "_F." in name]
    
    if f_images:
        # 360 模式：回傳 _F 的第一張與最後一張
        return f_images[0], f_images[-1]
    else:
        # 一般模式：直接回傳排序後的第一張與最後一張
        return all_images[0], all_images[-1]

def get_sfm_center(sfm_dir, target_name):
    """從 COLMAP 模型中讀取指定圖片的中心座標 (World Frame)，支援模糊比對。"""
    sfm_path = Path(sfm_dir)
    # 不需重複檢查路徑，pycolmap 會處理，或是由外部保證
            
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
    """解析 Pose (World-to-Camera) -> (Center, Yaw)"""
    rot_w2c = Rotation.from_quat([qx, qy, qz, qw])
    R_w2c = rot_w2c.as_matrix()
    t_vec = np.array([tx, ty, tz])
    
    R_c2w = R_w2c.T
    center = -R_c2w @ t_vec
    
    # 在 COLMAP 相機座標系中，[0, 0, 1] 是正前方
    view_dir = R_c2w[:, 2] 
    yaw = np.degrees(np.arctan2(view_dir[1], view_dir[0]))
    
    return center, yaw

def get_data_bounds(data_points, anchors_cfg):
    """計算資料邊界與跨度"""
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
    """繪製結果圖"""
    (min_x, max_x, min_y, max_y), map_span = get_data_bounds(data_points, anchors_cfg)
    
    w_range = max_x - min_x
    h_range = max_y - min_y
    pad_x = max(w_range * 0.1, 1.0)
    pad_y = max(h_range * 0.1, 1.0)
    
    plot_xlim = (min_x - pad_x, max_x + pad_x)
    plot_ylim = (min_y - pad_y, max_y + pad_y)
    
    final_w = plot_xlim[1] - plot_xlim[0]
    final_h = plot_ylim[1] - plot_ylim[0]
    
    max_fig_size = 14
    aspect = final_w / final_h
    
    if aspect > 1:
        fig_w = max_fig_size
        fig_h = max_fig_size / aspect
    else:
        fig_h = max_fig_size
        fig_w = max_fig_size * aspect
    
    fig_w = max(fig_w, 5)
    fig_h = max(fig_h, 5)

    plt.figure(figsize=(fig_w, fig_h)) 
    plt.title("Localization Results (Auto-Fit)")
    plt.xlabel("Map X")
    plt.ylabel("Map Y")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    arrow_len = map_span * 0.02
    if arrow_len < 0.1: arrow_len = 0.5 
    
    anchor_size = 150
    text_offset = arrow_len * 0.6

    unique_blocks = sorted(list(set(d['block'] for d in data_points)))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_blocks)))
    block_color_map = {b: c for b, c in zip(unique_blocks, colors)}

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

    added_anchor_label = False
    for block_name, cfg in anchors_cfg.items():
        sx, sy = cfg['start_map_xy']
        ex, ey = cfg['end_map_xy']
        
        plt.scatter(sx, sy, c='red', marker='x', s=anchor_size, linewidth=2.5, 
                    label='Anchors' if not added_anchor_label else "", zorder=10)
        # [Fix] 這裡顯示的文字可能需要根據是否為自動抓取而調整，目前維持顯示 key
        plt.text(sx, sy - text_offset, f" {block_name}_Start", color='red', fontsize=8, fontweight='bold', zorder=11, verticalalignment='top')
        
        plt.scatter(ex, ey, c='red', marker='x', s=anchor_size, linewidth=2.5, zorder=10)
        plt.text(ex, ey - text_offset, f" {block_name}_End", color='red', fontsize=8, fontweight='bold', zorder=11, verticalalignment='top')
        
        plt.plot([sx, ex], [sy, ey], 'r--', alpha=0.3, linewidth=1)
        added_anchor_label = True

    plt.xlim(plot_xlim)
    plt.ylim(plot_ylim)
    plt.axis('equal') 

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 1))

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
            sfm_path = cfg['sfm_path']
            # [Mod] 自動偵測邏輯
            # 如果 json 中沒有設定 start_frame 或 end_frame，則自動偵測
            target_start = cfg.get('start_frame')
            target_end = cfg.get('end_frame')

            if not target_start or not target_end:
                print(f"  [Auto] Detecting anchor frames for {block_name}...")
                auto_s, auto_e = find_auto_anchors(sfm_path)
                
                if not target_start:
                    target_start = auto_s
                    print(f"    -> Auto-Start: {target_start}")
                if not target_end:
                    target_end = auto_e
                    print(f"    -> Auto-End:   {target_end}")
            
            # 使用確認後的 frame name
            p_sfm_s = get_sfm_center(sfm_path, target_start)
            p_sfm_e = get_sfm_center(sfm_path, target_end)
            
            p_map_s = np.array(cfg['start_map_xy'])
            p_map_e = np.array(cfg['end_map_xy'])
            
            s, theta, t = compute_sim2_transform(p_sfm_s, p_sfm_e, p_map_s, p_map_e)
            transforms[block_name] = (s, theta, t)
            print(f"  > {block_name}: Scale={s:.4f}, Rot={np.degrees(theta):.2f}°")
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