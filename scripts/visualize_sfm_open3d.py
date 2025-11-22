#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_sfm_open3d.py  —  彩色點雲 + DB/Query 相機（合併 legend）
- [V3] 修正 360 模式下的採樣邏輯，改為「基於 Frame ID (時間戳)」進行採樣。
- 移除 --mode 參數，改為自動偵測 360 模式（_F, _R ... 檔名）
- 從 pycolmap Reconstruction 取彩色點雲（points3D.color / .rgb）
- DB Cameras (紅色/漸層) 與 DB Labels 為各自單一 legend
- 若偵測到 360 模式，額外增加 "Front only" (_F) 視角圖層
- Query Cameras (藍色) 與 Query Labels（藍字）為各自單一 legend（可選）
- 匯出互動式 HTML；可選擇啟動簡易 HTTP server 預覽
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
from collections import defaultdict # [NEW] Import defaultdict

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

# ---------- 360 模式輔助函式 ----------
SUFFIXES_360 = ('_F', '_R', '_B', '_L', '_FR', '_FL', '_RB', '_BL', '_LF')

def detect_360_mode(rec: pycolmap.Reconstruction) -> bool:
    for img in rec.images.values():
        stem = os.path.splitext(img.name)[0]
        if stem.endswith(SUFFIXES_360):
            return True
    return False

def get_frame_id_360(image_name: str):
    """
    Extracts base frame ID from a 360 view name.
    e.g., 'db/frames-000001_F.jpg' -> 'db/frames-000001'
    Returns None if not a 360 view.
    """
    stem = os.path.splitext(image_name)[0]
    if stem.endswith(SUFFIXES_360):
        # 從最後一個 '_' 往前切
        return stem.rsplit('_', 1)[0]
    return None # Not a 360 view

# ---------- 解析 Query 位姿 ----------
def load_query_poses(path):
    if not path or not os.path.exists(path):
        print("[Info] No query poses provided or file not found.")
        return []
    poses = []
    with open(path, "r") as f:
        for line in f:
            toks = line.strip().split()
            if len(toks) < 8: continue
            name = toks[0]
            q = np.array(list(map(float, toks[1:5])), dtype=np.float64)
            t = np.array(list(map(float, toks[5:8])), dtype=np.float64)
            R_c_w = qvec2rotmat(q); R_w_c = R_c_w.T; C = -R_c_w.T @ t
            poses.append((name, C, R_w_c))
    print(f"[Info] Loaded {len(poses)} query poses from {path}.")
    return poses

# ---------- 繪製相機圖層 ----------
def create_camera_traces(
    image_list_with_indices, total_db_count, suffix, 
    colorscale, legend_prefix, default_visible=False, **kwargs
):
    all_xs, all_ys, all_zs = [], [], []
    label_names, label_xs, label_ys, label_zs = [], [], [], []
    line_color_values = [] 
    
    norm_denominator = float(max(1, total_db_count - 1))

    for img, idx in image_list_with_indices:
        try:
            C = np.array(img.projection_center()).ravel()
            v = np.array(img.viewing_direction()).ravel()
            v /= (np.linalg.norm(v) + 1e-12)
            R = build_camera_R_from_dir(v)
            
            xs, ys, zs = make_frustum_lines(C, R, scale=kwargs.get("frustum_scale", 0.25))
            all_xs += xs; all_ys += ys; all_zs += zs
            
            color_val = float(idx) / norm_denominator
            line_color_values.extend([color_val] * len(xs))
            
            label_names.append(img.name)
            label_xs.append(C[0]); label_ys.append(C[1]); label_zs.append(C[2])
        except Exception as e:
            continue
    
    line_trace = go.Scatter3d(
        x=all_xs, y=all_ys, z=all_zs,
        mode="lines",
        line=dict(width=kwargs.get("line_width", 2), color=line_color_values, colorscale=colorscale),
        marker=dict(
            size=0.1, color=line_color_values, colorscale=colorscale,
            showscale=(suffix == "100%"), 
            colorbar=dict(title="Time", tickvals=[0.0, 1.0], ticktext=['Start', 'End'])
        ),
        name=f"{legend_prefix} Cameras ({suffix})",
        legendgroup=f"{legend_prefix}_cam_{suffix}", 
        showlegend=True,
        visible=True if default_visible else "legendonly"
    )
    
    label_trace = go.Scatter3d(
        x=label_xs, y=label_ys, z=label_zs,
        mode="text",
        text=label_names,
        textposition="top center",
        textfont=dict(size=kwargs.get("text_size", 9), color="red"),
        name=f"{legend_prefix} Labels ({suffix})",
        legendgroup=f"{legend_prefix}_label_{suffix}", 
        showlegend=True,
        visible="legendonly" 
    )
    
    return line_trace, label_trace

# ---------- 主流程 ----------
def main(sfm_dir, output_dir, port=8080, query_poses=None, no_server=False):
    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, "sfm_view.html")

    print(f"[Load] Reading reconstruction from: {sfm_dir}")
    rec = pycolmap.Reconstruction(sfm_dir)

    is_360_mode = detect_360_mode(rec)
    mode_str = "360 (auto-detected)" if is_360_mode else "std"
    print(f"[Info] Mode: {mode_str}")

    data = []

    # --- 彩色點雲 ---
    pts, cols = extract_colored_points(rec)
    if pts is not None:
        data.append(go.Scatter3d(
            x=pts[:,0], y=pts[:,1], z=pts[:,2],
            mode="markers",
            marker=dict(size=2, color=cols, opacity=0.95),
            name="Colored PCD",
            legendgroup="pcd", showlegend=True
        ))
    else:
        print("[Warn] No 3D points found.")

    # --- DB Cameras (Multi-level Sampling) ---
    print("[Info] Sorting and sampling DB cameras...")
    
    try:
        all_db_images_sorted = sorted(rec.images.values(), key=lambda img: img.name)
    except Exception as e:
        all_db_images_sorted = sorted(rec.images.values(), key=lambda img: img.image_id)
        
    db_images_with_indices = [(img, i) for i, img in enumerate(all_db_images_sorted)]
    total_db_count = len(db_images_with_indices)
    db_colorscale = 'Viridis' 
    
    if total_db_count > 0:
        
        # [MODIFIED] 根據模式決定採樣的 "單位"
        if is_360_mode:
            print("[Info] 360 Mode: Grouping cameras by frame ID...")
            frame_groups = defaultdict(list)
            for img, idx in db_images_with_indices:
                frame_id = get_frame_id_360(img.name)
                if frame_id:
                    frame_groups[frame_id].append((img, idx))
                else:
                    #  fallback (例如有些圖不符合 _F 命名)
                    frame_groups[f"non_360_{img.name}"].append((img, idx))
            
            # 採樣的基礎單位是 Frame ID
            sampling_base_list = sorted(frame_groups.keys())
            total_sample_units = len(sampling_base_list)
            print(f"    > Found {total_sample_units} unique frames (time steps).")
        else:
            # 標準模式：採樣的基礎單位是單張影像
            sampling_base_list = db_images_with_indices
            total_sample_units = len(sampling_base_list)

        # 1. 標準採樣率圖層
        sample_rates = [("100%", 100.0), ("50%", 50.0), ("25%", 25.0), ("12.5%", 12.5)]
        
        for suffix, rate in sample_rates:
            target_num = int(round(total_sample_units * (rate / 100.0)))
            target_num = max(target_num, 2) 
            
            if target_num >= total_sample_units:
                indices = np.arange(total_sample_units)
            else:
                indices = np.linspace(0, total_sample_units - 1, num=target_num, dtype=int)
            
            indices = sorted(list(set(indices)))
            
            # [MODIFIED] 根據採樣單位，建立最終的相機列表
            sampled_list = []
            if is_360_mode:
                for i in indices:
                    frame_id = sampling_base_list[i]
                    sampled_list.extend(frame_groups[frame_id])
            else:
                # 標準模式
                sampled_list = [sampling_base_list[i] for i in indices]

            print(f"    > Sampling {suffix}: {len(sampled_list)} cameras (from {len(indices)} sample units)")
            
            is_100_percent = (suffix == "100%")
            line_tr, label_tr = create_camera_traces(
                sampled_list, total_db_count, suffix, db_colorscale, "DB",
                default_visible=is_100_percent
            )
            data.append(line_tr)
            data.append(label_tr)

        # 2. [360] Front-only layer (此圖層維持原邏輯：顯示 *所有* Front 視角，不受採樣率影響)
        if is_360_mode:
            print("[Info] Generating 'Front-only' camera trace (all frames)...")
            front_only_list = []
            for img, idx in db_images_with_indices:
                if os.path.splitext(img.name)[0].endswith("_F"):
                    front_only_list.append((img, idx))
            
            if front_only_list:
                print(f"    > Found {len(front_only_list)} total front-view cameras.")
                line_tr_F, label_tr_F = create_camera_traces(
                    front_only_list, total_db_count, "Front", db_colorscale, "DB",
                    default_visible=False 
                )
                data.append(line_tr_F)
                data.append(label_tr_F)
            else:
                print("[Warn] Auto-detected 360 mode but no '_F' images found.")

    else:
        print("[Warn] No DB cameras found in reconstruction.")
        
    # --- Query Cameras（可選）---
    qposes = load_query_poses(query_poses) if query_poses else []
    if qposes:
        q_xs, q_ys, q_zs = [], [], []
        q_names, q_lx, q_ly, q_lz = [], [], [], []
        for name, C, R in qposes:
            xs, ys, zs = make_frustum_lines(C, R, scale=0.35)
            q_xs += xs; q_ys += ys; q_zs += zs
            q_names.append(name)
            q_lx.append(C[0]); q_ly.append(C[1]); q_lz.append(C[2])

        data.append(go.Scatter3d(
            x=q_xs, y=q_ys, z=q_zs,
            mode="lines",
            line=dict(width=2, color="blue"),
            name="Query Cameras",
            legendgroup="qcam", showlegend=True
        ))
        data.append(go.Scatter3d(
            x=q_lx, y=q_ly, z=q_lz,
            mode="text",
            text=q_names,
            textposition="top center",
            textfont=dict(size=10, color="blue"),
            name="Query Labels",
            legendgroup="qlabel", showlegend=True
        ))

    # --- 圖表設定與輸出 ---
    fig = go.Figure(data=data)
    fig.update_layout(
        title=f"SfM Visualization (Mode: {mode_str})",
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z (Up)',
            aspectmode="data",
            camera=dict(up=dict(x=0, y=0, z=1), eye=dict(x=-0.5, y=-2.5, z=0.5)),
            dragmode="orbit",
            xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.3)', zeroline=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.3)', zeroline=False),
            zaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.3)', zeroline=False),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(itemsizing='trace', bgcolor='rgba(255,255,255,0.7)')
    )
    pio.write_html(fig, html_path, include_plotlyjs="cdn", full_html=True)
    print(f"✅ Exported HTML: {html_path}")

    # --- 簡易 HTTP server ---
    if not no_server:
        os.chdir(output_dir)
        url = f"http://localhost:{port}/{os.path.basename(html_path)}"
        print(f"[Serve] Opening {url}")
        with TCPServer(("0.0.0.0", port), SimpleHTTPRequestHandler) as httpd:
            try:
                webbrowser.open(url)
            except Exception:
                pass
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\n[Exit] Server stopped.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sfm_dir", type=str, required=True, help="Path to COLMAP/pycolmap SfM folder")
    ap.add_argument("--output_dir", type=str, default="visualization", help="Output directory for HTML")
    ap.add_argument("--port", type=int, default=8080, help="HTTP server port")
    ap.add_argument("--query_poses", type=str, help="Optional: path to hloc poses.txt")
    ap.add_argument("--no_server", action="store_true", help="Only export HTML, do not start HTTP server")
    
    args = ap.parse_args()
    
    main(args.sfm_dir, args.output_dir, args.port, args.query_poses, args.no_server)