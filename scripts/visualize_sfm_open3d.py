#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_sfm_open3d.py [V11]
- [Fix] 修正 Camera 圖層的可視性邏輯：使用 "legendonly" 代替 False，確保可以在 Legend 中切換 On/Off。
- [Feature] 保留 Block Filter (Query 過濾)、統計濾波 (尺度修復)、智慧縮放 (Smart Scale)。
"""

import os
import argparse
import numpy as np
import pycolmap
import webbrowser
from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer
import plotly.graph_objects as go
import plotly.io as pio
from collections import defaultdict 

# ---------- 3D 轉換輔助 ----------
def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w),   2*(x*z + y*w)],
        [2*(x*y + z*w),   1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),   2*(y*z + x*w), 1-2*(x*x+y*y)]
    ], dtype=np.float64)

def build_camera_R_from_dir(view_dir, up_ref=np.array([0., 0., 1.])):
    z = view_dir / (np.linalg.norm(view_dir) + 1e-12)
    u = up_ref if abs(np.dot(z, up_ref)) < 0.98 else np.array([0., 1., 0.])
    x = np.cross(u, z); x /= (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x); y /= (np.linalg.norm(y) + 1e-12)
    return np.stack([x, y, z], axis=1)

def make_frustum_lines(C, R, scale=0.25):
    pts_cam = np.array([
        [0,0,0],[ 0.5, 0.3, 1.0],[-0.5, 0.3, 1.0],
        [-0.5,-0.3, 1.0],[ 0.5,-0.3, 1.0]
    ], dtype=np.float64) * scale
    pts = (R @ pts_cam.T).T + C
    lines = [(0,1),(0,2),(0,3),(0,4),(1,2),(2,3),(3,4),(4,1)]
    xs, ys, zs = [], [], []
    for a,b in lines:
        xs += [pts[a,0], pts[b,0], None]
        ys += [pts[a,1], pts[b,1], None]
        zs += [pts[a,2], pts[b,2], None]
    return xs, ys, zs

# ---------- 彩色點雲擷取 ----------
def extract_colored_points(rec: pycolmap.Reconstruction):
    coords, colors = [], []
    for p in rec.points3D.values():
        coords.append(np.asarray(p.xyz, dtype=float))
        c = np.asarray(p.rgb if hasattr(p, "rgb") else p.color, dtype=float)
        c = np.clip(c, 0, 255).astype(float)
        colors.append(f"rgb({int(c[0])},{int(c[1])},{int(c[2])})")
    if not coords:
        return None, None
    return np.vstack(coords), colors

# ---------- 360 模式輔助 ----------
SUFFIXES_360 = ('_F', '_R', '_B', '_L', '_FR', '_FL', '_RB', '_BL', '_LF')
def detect_360_mode(rec: pycolmap.Reconstruction) -> bool:
    for img in rec.images.values():
        stem = os.path.splitext(img.name)[0]
        if stem.endswith(SUFFIXES_360): return True
    return False

def get_frame_id_360(image_name: str):
    stem = os.path.splitext(image_name)[0]
    if stem.endswith(SUFFIXES_360): return stem.rsplit('_', 1)[0]
    return None 

# ---------- 解析 Query 位姿 (Block Filter) ----------
def load_query_poses(path, target_block=None):
    if not path or not os.path.exists(path): return []
    poses = []
    with open(path, "r") as f:
        for line in f:
            if line.strip().startswith("#") or not line.strip(): continue
            toks = line.strip().split()
            if len(toks) < 8: continue
            
            # [Block Filter]
            if target_block and len(toks) >= 9:
                if toks[8] != target_block: continue

            name = toks[0]
            try:
                q = np.array(list(map(float, toks[1:5])), dtype=np.float64)
                t = np.array(list(map(float, toks[5:8])), dtype=np.float64)
                R_c_w = qvec2rotmat(q); R_w_c = R_c_w.T; C = -R_c_w.T @ t
                poses.append((name, C, R_w_c))
            except ValueError: continue
    print(f"[Info] Loaded {len(poses)} query poses (Filter: {target_block}).")
    return poses

# ---------- 繪製相機圖層 ----------
def create_camera_traces(image_list, total, suffix, colorscale, prefix, visible, **kwargs):
    all_xs, all_ys, all_zs = [], [], []
    names, l_xs, l_ys, l_zs = [], [], [], []
    colors = []
    denom = float(max(1, total - 1))

    for img, idx in image_list:
        try:
            C = np.array(img.projection_center()).ravel()
            v = np.array(img.viewing_direction()).ravel()
            v /= (np.linalg.norm(v) + 1e-12)
            R = build_camera_R_from_dir(v)
            xs, ys, zs = make_frustum_lines(C, R, scale=kwargs.get("frustum_scale", 0.25))
            all_xs += xs; all_ys += ys; all_zs += zs
            cval = float(idx) / denom
            colors.extend([cval] * len(xs))
            names.append(img.name)
            l_xs.append(C[0]); l_ys.append(C[1]); l_zs.append(C[2])
        except: continue
    
    # [Fix] 使用 "legendonly" 而非 False，確保在 Legend 中可以點擊開啟
    vis_val = True if visible else "legendonly"

    line = go.Scatter3d(
        x=all_xs, y=all_ys, z=all_zs, mode="lines",
        line=dict(width=2, color=colors, colorscale=colorscale),
        marker=dict(size=0.1, color=colors, colorscale=colorscale, showscale=(suffix=="100%")),
        name=f"{prefix} Cameras ({suffix})", legendgroup=f"{prefix}_cam_{suffix}",
        showlegend=True, visible=vis_val
    )
    label = go.Scatter3d(
        x=l_xs, y=l_ys, z=l_zs, mode="text", text=names,
        textfont=dict(size=9, color="red"),
        name=f"{prefix} Labels ({suffix})", legendgroup=f"{prefix}_lbl_{suffix}",
        showlegend=True, visible="legendonly"
    )
    return line, label

# ---------- 主流程 ----------
def main(sfm_dir, output_dir, port=8080, query_poses=None, no_server=False, target_block=None):
    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, "sfm_view.html")
    print(f"[Load] SfM: {sfm_dir}")
    
    try:
        rec = pycolmap.Reconstruction(sfm_dir)
    except Exception as e:
        print(f"[Error] Failed to load SfM: {e}")
        return

    is_360 = detect_360_mode(rec)
    
    # --- 0. DB Camera Stats (基準) ---
    all_cameras_sorted = []
    try: all_cameras_sorted = sorted(rec.images.values(), key=lambda x: x.name)
    except: all_cameras_sorted = sorted(rec.images.values(), key=lambda x: x.image_id)

    all_pos = []
    for im in all_cameras_sorted:
        try: all_pos.append(im.projection_center())
        except: pass
    all_pos = np.array(all_pos)

    valid_bounds = None
    scene_scale = 0.5

    if len(all_pos) > 2:
        mean_c = np.mean(all_pos, axis=0)
        std_c = np.std(all_pos, axis=0)
        mask_c = np.all(np.abs(all_pos - mean_c) < 3 * std_c, axis=1)
        filtered_pos = all_pos[mask_c]
        
        print(f"[Info] Cameras: Total={len(all_pos)}, Valid={len(filtered_pos)}")
        if len(filtered_pos) > 0:
            min_b, max_b = np.min(filtered_pos, axis=0), np.max(filtered_pos, axis=0)
            span = np.linalg.norm(max_b - min_b)
            scene_scale = max(span * 0.02, 0.5)
            margin = max(span * 0.2, 5.0)
            valid_bounds = (min_b - margin, max_b + margin)
            print(f"       Valid Bounds: {valid_bounds}")
            print(f"       Scene Scale: {scene_scale:.2f}")

    data = []

    # --- 1. Point Cloud (Filter by Camera Bounds) ---
    pts, cols = extract_colored_points(rec)
    if pts is not None and len(pts) > 0:
        if valid_bounds is not None:
            min_b, max_b = valid_bounds
            mask_p = np.all((pts >= min_b) & (pts <= max_b), axis=1)
            pts, cols = pts[mask_p], np.array(cols)[mask_p]
            print(f"[Info] Points Filtered: {len(mask_p)} -> {len(pts)}")
        
        data.append(go.Scatter3d(
            x=pts[:,0], y=pts[:,1], z=pts[:,2], mode="markers",
            marker=dict(size=2, color=cols.tolist(), opacity=0.95),
            name="Colored PCD"
        ))

    # --- 2. DB Cameras (Smart Scale) ---
    # Smart Scale Refinement based on baseline
    if len(all_cameras_sorted) > 1:
        try:
            centers = np.array([im.projection_center() for im in all_cameras_sorted])
            diffs = np.linalg.norm(centers[1:] - centers[:-1], axis=1)
            valid_diffs = diffs[diffs > 0.05]
            if len(valid_diffs) > 0:
                smart_scale = np.median(valid_diffs) * 0.4
                if smart_scale < scene_scale:
                    print(f"       [Refine] Scale by baseline: {smart_scale:.3f}")
                    scene_scale = smart_scale
        except: pass

    imgs_idx = [(im, i) for i, im in enumerate(all_cameras_sorted)]
    if imgs_idx:
        if is_360:
            groups = defaultdict(list)
            for im, idx in imgs_idx:
                fid = get_frame_id_360(im.name) or f"misc_{im.name}"
                groups[fid].append((im, idx))
            base_list = sorted(groups.keys())
            total_units = len(base_list)
            
            rates = [("100%", 100), ("50%", 50), ("25%", 25), ("12.5%", 12.5)]
            for suff, r in rates:
                target_count = max(int(total_units * (r/100.0)), 2)
                if target_count >= total_units:
                    indices = np.arange(total_units)
                else:
                    indices = np.linspace(0, total_units-1, target_count, dtype=int)
                
                indices = sorted(list(set(indices)))
                subset = []
                for i in indices: subset.extend(groups[base_list[i]])
                
                l, t = create_camera_traces(subset, len(imgs_idx), suff, 'Viridis', 'DB', suff=="100%", frustum_scale=scene_scale)
                data.append(l); data.append(t)
            
            # Front only
            fronts = [(im, i) for im, i in imgs_idx if os.path.splitext(im.name)[0].endswith("_F")]
            if fronts:
                l, t = create_camera_traces(fronts, len(imgs_idx), "Front", 'Viridis', 'DB', False, frustum_scale=scene_scale)
                data.append(l); data.append(t)
        else:
            rates = [("100%", 100), ("50%", 50), ("25%", 25), ("12.5%", 12.5)]
            for suff, r in rates:
                step = max(int(100/r), 1)
                subset = imgs_idx[::step]
                l, t = create_camera_traces(subset, len(imgs_idx), suff, 'Viridis', 'DB', suff=="100%", frustum_scale=scene_scale)
                data.append(l); data.append(t)
    
    # --- 3. Query Cameras (Apply Block Filter) ---
    qposes = load_query_poses(query_poses, target_block)
    if qposes:
        q_x, q_y, q_z = [], [], []
        q_n, q_lx, q_ly, q_lz = [], [], [], []
        for name, C, R in qposes:
            # Query Filter: also hide queries if far outside bounds (optional safety)
            if valid_bounds is not None:
                min_b, max_b = valid_bounds
                # Allow 2x margin for queries
                if not (np.all(C >= min_b - 50) and np.all(C <= max_b + 50)): continue

            xs, ys, zs = make_frustum_lines(C, R, scale=scene_scale*1.5)
            q_x+=xs; q_y+=ys; q_z+=zs
            q_n.append(name)
            q_lx.append(C[0]); q_ly.append(C[1]); q_lz.append(C[2])
        
        data.append(go.Scatter3d(
            x=q_x, y=q_y, z=q_z, mode="lines",
            line=dict(width=3, color="red"), name="Query Cameras"
        ))
        data.append(go.Scatter3d(
            x=q_lx, y=q_ly, z=q_lz, mode="text", text=q_n,
            textfont=dict(size=10, color="red"), name="Query Labels"
        ))

    # --- 4. Export ---
    fig = go.Figure(data=data)
    fig.update_layout(scene=dict(aspectmode="data"))
    pio.write_html(fig, html_path, include_plotlyjs="cdn", full_html=True)
    print(f"✅ HTML Saved: {html_path}")

    if not no_server:
        os.chdir(output_dir)
        with TCPServer(("0.0.0.0", port), SimpleHTTPRequestHandler) as httpd:
            print(f"Serving at http://localhost:{port}/{os.path.basename(html_path)}")
            try: httpd.serve_forever()
            except: pass

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sfm_dir", required=True)
    ap.add_argument("--output_dir", default="viz")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--query_poses")
    ap.add_argument("--no_server", action="store_true")
    ap.add_argument("--block_name")
    args = ap.parse_args()
    main(args.sfm_dir, args.output_dir, args.port, args.query_poses, args.no_server, args.block_name)