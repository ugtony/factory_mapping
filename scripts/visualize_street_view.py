#!/usr/bin/env python3
"""
visualize_street_view_multi_blocks.py (V10 - Block Labels)
用途：多區域 (Multi-Block) 整合版街景瀏覽器。
修正：
  - [Feature] 在地圖上自動標示 Block 名稱 (位於該區域相機群的中心)。
  - 保留所有 V9 功能 (Auto-Scale, Adaptive Radius, Visibility Fix)。
"""

import argparse
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import pycolmap
import cv2
from pathlib import Path
from matplotlib.patches import Wedge

# 添加專案根目錄以匯入 lib
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

try:
    from lib.map_utils import compute_sim2_transform, colmap_to_scipy_quat
except ImportError:
    print("[Error] Cannot import lib.map_utils. Make sure you are in the project root.")
    sys.exit(1)

# ---------------- 工具函式 ----------------

def angle_diff(a1, a2):
    diff = np.abs(a1 - a2)
    return np.min([diff, 2*np.pi - diff])

def normalize_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def get_camera_pose_raw(img):
    if hasattr(img, "cam_from_world"):
        pose = img.cam_from_world
        if callable(pose): pose = pose()
        R_w2c = pose.rotation.matrix()
        center = img.projection_center()
    else:
        q_scipy = colmap_to_scipy_quat(img.qvec)
        from scipy.spatial.transform import Rotation
        R_w2c = Rotation.from_quat(q_scipy).as_matrix()
        center = img.projection_center()
    
    R_c2w = R_w2c.T
    view_dir = R_c2w[:, 2] 
    yaw = np.arctan2(view_dir[1], view_dir[0])
    return center, yaw

def apply_sim2(pt_sfm, yaw_sfm, trans):
    s = trans['s']
    theta = trans['theta']
    t = trans['t']
    R = trans['R']
    p_sfm_2d = np.array(pt_sfm[:2])
    p_map = s * (R @ p_sfm_2d) + t
    yaw_map = normalize_angle(yaw_sfm + theta)
    return p_map, yaw_map

# ---------------- 核心邏輯 ----------------

def find_best_camera_for_move(target_x, target_y, current_yaw, cameras):
    centers = np.array([[c['x'], c['y']] for c in cameras])
    dists = np.linalg.norm(centers - np.array([target_x, target_y]), axis=1)
    
    K = 15
    nearest_indices = np.argsort(dists)[:K]
    
    best_idx = -1
    min_angle_dist = float('inf')

    for idx in nearest_indices:
        cam = cameras[idx]
        a_dist = angle_diff(cam['yaw'], current_yaw)
        
        dist_penalty = 0.0
        if dists[idx] > 5.0: dist_penalty = 10.0
        
        score = a_dist + dist_penalty
        if score < min_angle_dist:
            min_angle_dist = score
            best_idx = idx
            
    return best_idx

def find_best_camera_for_rotate(anchor_idx, mouse_x, mouse_y, cameras, search_radius):
    anchor_cam = cameras[anchor_idx]
    dx = mouse_x - anchor_cam['x']
    dy = mouse_y - anchor_cam['y']
    target_yaw = np.arctan2(dy, dx)
    target_block = anchor_cam['block']
    
    best_idx = anchor_idx
    min_angle_dist = float('inf')
    
    for i, cam in enumerate(cameras):
        if cam['block'] != target_block: continue
        dist = np.hypot(cam['x'] - anchor_cam['x'], cam['y'] - anchor_cam['y'])
        if dist <= search_radius:
            a_dist = angle_diff(cam['yaw'], target_yaw)
            if a_dist < min_angle_dist:
                min_angle_dist = a_dist
                best_idx = i
            
    return best_idx, target_yaw

def main():
    parser = argparse.ArgumentParser(description="Multi-Block Street View Simulator (V10 Labels)")
    parser.add_argument("--anchors", type=Path, default="anchors.json", help="Path to anchors.json")
    parser.add_argument("--image_root", type=Path, default="outputs-hloc", help="Root dir to find images")
    args = parser.parse_args()

    if not args.anchors.exists():
        print(f"[Error] Anchors file not found: {args.anchors}")
        sys.exit(1)

    with open(args.anchors, 'r') as f:
        anchors_cfg = json.load(f)

    all_points_x, all_points_y, all_points_c = [], [], []
    all_cameras = []
    all_valid_steps = []

    # 用於計算 Block 中心點
    block_centroids = {} 

    cmap = plt.get_cmap("tab10")
    block_colors = {}

    print(f"=== Loading {len(anchors_cfg)} Blocks ===")

    for i, (block_name, cfg) in enumerate(anchors_cfg.items()):
        sfm_path = Path(cfg['sfm_path'])
        if not sfm_path.exists(): sfm_path = project_root / cfg['sfm_path']
        
        if not (sfm_path / "images.bin").exists():
            print(f"[Warn] Skipping {block_name}: images.bin not found")
            continue

        print(f"  > Loading Block: {block_name} ...")
        recon = pycolmap.Reconstruction(sfm_path)
        trans = compute_sim2_transform(recon, cfg)
        if trans is None:
            print(f"    [Error] Failed to compute transform for {block_name}")
            continue
            
        color = cmap(i % 10)
        block_colors[block_name] = color
        
        # 初始化該 Block 的座標列表
        block_centroids[block_name] = {'x': [], 'y': []}

        # Points
        p3d_values = recon.points3D.values() if hasattr(recon, "points3D") and hasattr(recon.points3D, "values") else recon.points3D
        for j, p3d in enumerate(p3d_values):
            if j % 5 != 0: continue
            p_map, _ = apply_sim2(p3d.xyz, 0, trans)
            all_points_x.append(p_map[0])
            all_points_y.append(p_map[1])
            all_points_c.append(color)

        # Cameras
        img_values = recon.images.values() if hasattr(recon, "images") and hasattr(recon.images, "values") else recon.images
        sorted_imgs = sorted([img for img in img_values if img.has_pose], key=lambda x: x.name)
        
        img_dir = args.image_root / block_name / "_images_stage"
        if not img_dir.exists(): img_dir = args.image_root / block_name / "db"
        if not img_dir.exists(): img_dir = project_root / "data" / block_name / "db"

        block_camera_centers = []
        block_camera_names = []

        for img in sorted_imgs:
            center_sfm, yaw_sfm = get_camera_pose_raw(img)
            center_map, yaw_map = apply_sim2(center_sfm, yaw_sfm, trans)
            
            all_cameras.append({
                'name': img.name,
                'block': block_name,
                'img_path': img_dir / img.name,
                'x': center_map[0], 
                'y': center_map[1],
                'yaw': yaw_map,
                'color': color 
            })
            block_camera_centers.append(center_map[:2])
            block_camera_names.append(img.name)
            
            # 收集座標以計算中心
            block_centroids[block_name]['x'].append(center_map[0])
            block_centroids[block_name]['y'].append(center_map[1])

        # 計算步伐
        if len(block_camera_centers) > 1:
            f_indices = [k for k, name in enumerate(block_camera_names) if "_F" in name]
            if len(f_indices) < 2:
                calc_centers = np.array(block_camera_centers)
            else:
                calc_centers = np.array([block_camera_centers[k] for k in f_indices])
            
            dists = np.linalg.norm(calc_centers[1:] - calc_centers[:-1], axis=1)
            valid = dists[dists > 0.05]
            all_valid_steps.extend(valid)

    if not all_cameras:
        print("[Error] No cameras loaded.")
        sys.exit(1)

    # --- Auto Scale ---
    visual_scale = 1.0
    rotation_search_radius = 0.5
    
    if all_valid_steps:
        median_step = np.median(all_valid_steps)
        rotation_search_radius = median_step * 0.6
        visual_scale = median_step
    else:
        print("[Warn] Using default scale.")

    # --- 介面佈局 ---
    fig, (ax_map, ax_img) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [1, 1]})
    fig.canvas.manager.set_window_title("Multi-Block Map Inspector (V10)")

    # 1. 地圖：背景與相機
    ax_map.scatter(all_points_x, all_points_y, s=1, c='gray', alpha=0.1, label='Scene Cloud')
    
    cam_x = [c['x'] for c in all_cameras]
    cam_y = [c['y'] for c in all_cameras]
    cam_u = [visual_scale * 0.8 * np.cos(c['yaw']) for c in all_cameras]
    cam_v = [visual_scale * 0.8 * np.sin(c['yaw']) for c in all_cameras]
    cam_c = [c['color'] for c in all_cameras]
    
    ax_map.scatter(cam_x, cam_y, s=10, c=cam_c, alpha=0.5)
    ax_map.quiver(cam_x, cam_y, cam_u, cam_v, color=cam_c, alpha=0.6, 
                  angles='xy', scale_units='xy', scale=1, width=0.002, headwidth=3)

    # [New] 繪製 Block 名稱標籤
    for b_name, coords in block_centroids.items():
        if coords['x']:
            cx = np.mean(coords['x'])
            cy = np.mean(coords['y'])
            # 加上白色背景框，確保在雜亂背景中可讀
            ax_map.text(cx, cy, b_name, fontsize=12, fontweight='bold', 
                        color='black', ha='center', va='center',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3.0),
                        zorder=20) # 確保在最上層

    # 2. 地圖：範圍計算 (正方形) 與 Wedge 大小
    max_map_span = 10.0
    if cam_x:
        min_x, max_x = min(cam_x), max(cam_x)
        min_y, max_y = min(cam_y), max(cam_y)
        span_x, span_y = max_x - min_x, max_y - min_y
        max_map_span = max(span_x, span_y)
        if max_map_span < 1.0: max_map_span = 10.0
        
        center_x, center_y = (min_x + max_x)/2, (min_y + max_y)/2
        half = (max_map_span * 1.1) / 2
        ax_map.set_xlim(center_x - half, center_x + half)
        ax_map.set_ylim(center_y - half, center_y + half)

    base_size_from_map = max_map_span * 0.03
    base_size_from_step = visual_scale * 3.0
    wedge_radius = max(base_size_from_map, base_size_from_step)
    
    print(f"[Info] Scale: Step={visual_scale:.2f}m, MapSpan={max_map_span:.1f}m -> Wedge Radius={wedge_radius:.2f}m")

    current_fov = 60
    wedge_cam = Wedge((0, 0), wedge_radius, 0, 0, color='blue', alpha=0.6, zorder=10, label='Current View')
    wedge_mouse = Wedge((0, 0), wedge_radius * 1.2, 0, 0, color='green', alpha=0.0, zorder=9)
    ax_map.add_patch(wedge_cam)
    ax_map.add_patch(wedge_mouse)
    ax_map.set_aspect('equal')
    ax_map.set_title(f"Map (Scale={visual_scale:.2f}m): Double Click (Move) | Drag (Rotate)")
    ax_map.legend(loc='upper right', markerscale=3)

    # 右側：影像
    ax_img.axis('off')
    ax_img.set_title("Street View")
    placeholder = np.zeros((600, 800, 3), dtype=np.uint8) + 240
    img_display = ax_img.imshow(placeholder)

    state = {'current_idx': 0, 'anchor_idx': 0, 'is_dragging': False}

    def update_map_indicator(cam_idx, target_yaw=None):
        cam = all_cameras[cam_idx]
        wedge_cam.set_center((cam['x'], cam['y']))
        deg = np.degrees(cam['yaw'])
        wedge_cam.set_theta1(deg - current_fov/2)
        wedge_cam.set_theta2(deg + current_fov/2)
        
        if target_yaw is not None and state['is_dragging']:
            anchor_cam = all_cameras[state['anchor_idx']]
            wedge_mouse.set_alpha(0.3)
            wedge_mouse.set_center((anchor_cam['x'], anchor_cam['y']))
            deg_m = np.degrees(target_yaw)
            wedge_mouse.set_theta1(deg_m - 5)
            wedge_mouse.set_theta2(deg_m + 5)
        else:
            wedge_mouse.set_alpha(0.0)

    def show_image_on_ax(cam):
        path = cam['img_path']
        if not path.exists():
            frame = np.zeros((600, 800, 3), dtype=np.uint8)
            cv2.putText(frame, "Image Missing", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, str(path.name), (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
        else:
            frame = cv2.imread(str(path))
            if frame is None:
                frame = np.zeros((600, 800, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        img_display.set_data(frame)
        ax_img.set_title(f"[{cam['block']}] {cam['name']}")

    def update_view(cam_idx):
        state['current_idx'] = cam_idx
        update_map_indicator(cam_idx)
        show_image_on_ax(all_cameras[cam_idx])
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax_map: return
        if event.dblclick and event.button == 1:
            current_cam = all_cameras[state['current_idx']]
            best_idx = find_best_camera_for_move(
                event.xdata, event.ydata, current_cam['yaw'], all_cameras
            )
            if best_idx != -1:
                print(f"[Move] -> {all_cameras[best_idx]['block']} / {all_cameras[best_idx]['name']}")
                state['anchor_idx'] = best_idx
                update_view(best_idx)

    def on_press(event):
        if event.inaxes != ax_map: return
        if event.button == 1 and not event.dblclick:
            state['is_dragging'] = True
            state['anchor_idx'] = state['current_idx']

    def on_release(event):
        if event.button == 1:
            state['is_dragging'] = False
            update_map_indicator(state['current_idx'], None)
            fig.canvas.draw_idle()

    def on_motion(event):
        if state['is_dragging'] and event.xdata is not None and event.inaxes == ax_map:
            best_idx, target_yaw = find_best_camera_for_rotate(
                state['anchor_idx'], event.xdata, event.ydata, all_cameras, rotation_search_radius
            )
            if best_idx != state['current_idx']:
                state['current_idx'] = best_idx
                show_image_on_ax(all_cameras[best_idx])
            update_map_indicator(best_idx, target_yaw)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)

    # 啟動
    update_view(0)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()