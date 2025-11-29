# scripts/convert_poses_to_map.py
#!/usr/bin/env python3
import argparse
import json
import numpy as np
import sys
from pathlib import Path
from scipy.spatial.transform import Rotation
import pycolmap
import matplotlib.pyplot as plt

# [Plan A] Setup path to find 'lib'
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from lib.map_utils import compute_sim2_transform

def parse_pose(qw, qx, qy, qz, tx, ty, tz):
    rot_w2c = Rotation.from_quat([qx, qy, qz, qw])
    R_w2c = rot_w2c.as_matrix()
    t_vec = np.array([tx, ty, tz])
    R_c2w = R_w2c.T
    center = -R_c2w @ t_vec
    view_dir = R_c2w[:, 2]
    yaw = np.degrees(np.arctan2(view_dir[1], view_dir[0]))
    return center, yaw

def get_data_bounds(data_points, anchors_cfg):
    xs, ys = [], []
    for d in data_points: xs.append(d['x']); ys.append(d['y'])
    for cfg in anchors_cfg.values():
        xs.append(cfg['start_map_xy'][0]); ys.append(cfg['start_map_xy'][1])
        xs.append(cfg['end_map_xy'][0]); ys.append(cfg['end_map_xy'][1])
    if not xs: return (0,1,0,1), 1.0
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    span_x, span_y = max_x - min_x, max_y - min_y
    return (min_x, max_x, min_y, max_y), max(span_x, span_y)

def plot_results(output_png, data_points, anchors_cfg):
    (min_x, max_x, min_y, max_y), map_span = get_data_bounds(data_points, anchors_cfg)
    w_range, h_range = max_x - min_x, max_y - min_y
    pad_x, pad_y = max(w_range * 0.1, 1.0), max(h_range * 0.1, 1.0)
    plot_xlim, plot_ylim = (min_x - pad_x, max_x + pad_x), (min_y - pad_y, max_y + pad_y)
    
    final_w, final_h = plot_xlim[1] - plot_xlim[0], plot_ylim[1] - plot_ylim[0]
    max_fig_size, aspect = 14, final_w / final_h
    fig_w = max_fig_size if aspect > 1 else max_fig_size * aspect
    fig_h = max_fig_size / aspect if aspect > 1 else max_fig_size
    fig_w, fig_h = max(fig_w, 5), max(fig_h, 5)

    plt.figure(figsize=(fig_w, fig_h)) 
    plt.title("Localization Results (Auto-Fit)"); plt.xlabel("Map X"); plt.ylabel("Map Y")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    arrow_len = max(map_span * 0.02, 0.5)
    text_offset = arrow_len * 0.6
    unique_blocks = sorted(list(set(d['block'] for d in data_points)))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_blocks)))
    block_color_map = {b: c for b, c in zip(unique_blocks, colors)}

    for d in data_points:
        x, y, yaw = d['x'], d['y'], d['yaw']
        color = block_color_map.get(d['block'], 'black')
        plt.scatter(x, y, c=[color], s=30, label=d['block'], edgecolors='k', linewidth=0.5, alpha=0.7)
        dx, dy = arrow_len * np.cos(np.deg2rad(yaw)), arrow_len * np.sin(np.deg2rad(yaw))
        plt.arrow(x, y, dx, dy, head_width=arrow_len*0.4, head_length=arrow_len*0.5, fc=color, ec=color, alpha=0.8)
        plt.text(x + text_offset, y + text_offset, f"{Path(d['name']).name}", fontsize=6, color=color, alpha=0.8, rotation=45)

    added_anchor_label = False
    for block_name, cfg in anchors_cfg.items():
        sx, sy = cfg['start_map_xy']; ex, ey = cfg['end_map_xy']
        plt.scatter(sx, sy, c='red', marker='x', s=150, linewidth=2.5, label='Anchors' if not added_anchor_label else "", zorder=10)
        plt.text(sx, sy - text_offset, f" {block_name}_Start", color='red', fontsize=8, fontweight='bold', zorder=11, verticalalignment='top')
        plt.scatter(ex, ey, c='red', marker='x', s=150, linewidth=2.5, zorder=10)
        plt.text(ex, ey - text_offset, f" {block_name}_End", color='red', fontsize=8, fontweight='bold', zorder=11, verticalalignment='top')
        plt.plot([sx, ex], [sy, ey], 'r--', alpha=0.3, linewidth=1)
        added_anchor_label = True

    plt.xlim(plot_xlim); plt.ylim(plot_ylim); plt.axis('equal')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout(); plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f"🖼️  Plot saved to: {output_png}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--output", type=Path, default="submission_map.txt")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    with open(args.anchors, 'r') as f: anchors_cfg = json.load(f)
    transforms = {}
    print(f"[Step 1] Computing transforms...")
    for block_name, cfg in anchors_cfg.items():
        try:
            recon = pycolmap.Reconstruction(Path(cfg['sfm_path']))
            trans = compute_sim2_transform(recon, cfg)
            if trans:
                transforms[block_name] = trans
                print(f"  > {block_name}: Scale={trans['s']:.4f}, Rot={np.degrees(trans['theta']):.2f}°")
        except Exception as e: print(f"  [Error] {block_name} failed: {e}")

    if not transforms: print("[Error] No valid transforms."); return
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
                name = parts[0]; vals = list(map(float, parts[1:8]))
                block_name = parts[8] if len(parts) >= 9 else list(transforms.keys())[0]
            except ValueError: continue
            if block_name not in transforms: continue
            
            sfm_center, sfm_yaw = parse_pose(*vals)
            t_data = transforms[block_name]
            s, theta, t_vec = t_data['s'], t_data['theta'], t_data['t']
            
            c, si = np.cos(theta), np.sin(theta)
            R_mat = np.array([[c, -si], [si, c]])
            p_map = s * (R_mat @ sfm_center[:2]) + t_vec
            map_yaw = (sfm_yaw + np.degrees(theta) + 180) % 360 - 180
            
            f_out.write(f"{name}, {p_map[0]:.4f}, {p_map[1]:.4f}, {map_yaw:.4f}, {block_name}\n")
            plot_data.append({'name': name, 'x': p_map[0], 'y': p_map[1], 'yaw': map_yaw, 'block': block_name})
            count += 1

    print(f"✅ Done! Converted {count} poses to '{args.output}'")
    if args.plot and plot_data: plot_results(args.output.with_suffix('.png'), plot_data, anchors_cfg)

if __name__ == "__main__":
    main()