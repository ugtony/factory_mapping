#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_sfm_open3d.py  —  彩色點雲 + DB/Query 相機（合併 legend）
- 從 pycolmap Reconstruction 取彩色點雲（points3D.color / .rgb）
- DB Cameras (紅色) 與 DB Labels（紅字）為各自單一 legend
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


# ---------- 3D 轉換輔助 ----------
def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w),   2*(x*z + y*w)],
        [2*(x*y + z*w),   1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),   2*(y*z + x*w), 1-2*(x*x+y*y)]
    ], dtype=np.float64)

def build_camera_R_from_dir(view_dir, up_ref=np.array([0., 0., 1.])):
    """由 viewing_direction 構造 cam->world 的旋轉矩陣（列向量為世界座標下的相機軸）"""
    z = view_dir / (np.linalg.norm(view_dir) + 1e-12)
    u = up_ref if abs(np.dot(z, up_ref)) < 0.98 else np.array([0., 1., 0.])
    x = np.cross(u, z); x /= (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x); y /= (np.linalg.norm(y) + 1e-12)
    return np.stack([x, y, z], axis=1)  # columns=[x_world, y_world, z_world]

def make_frustum_lines(C, R, scale=0.25):
    """回傳一個相機金字塔的折線資料（xs,ys,zs），用 None 分段"""
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


# ---------- 彩色點雲擷取（相容 color / rgb 欄位） ----------
def extract_colored_points(rec: pycolmap.Reconstruction):
    coords, colors = [], []
    for p in rec.points3D.values():
        coords.append(np.asarray(p.xyz, dtype=float))
        if hasattr(p, "color"):
            c = np.asarray(p.color)
        elif hasattr(p, "rgb"):
            c = np.asarray(p.rgb)
        else:
            c = np.array([204,204,204], dtype=float)  # 灰
        c = np.clip(c, 0, 255).astype(float)
        colors.append(f"rgb({int(c[0])},{int(c[1])},{int(c[2])})")
    if not coords:
        return None, None
    return np.vstack(coords), colors


# ---------- 解析 hloc poses.txt（Query 位姿） ----------
def load_query_poses(path):
    """
    讀取每行：name qw qx qy qz tx ty tz
    回傳 list of (name, C, R) 其中 C = -R^T t；R = cam->world
    """
    if not path or not os.path.exists(path):
        print("[Info] No query poses provided or file not found.")
        return []
    poses = []
    with open(path, "r") as f:
        for line in f:
            toks = line.strip().split()
            if len(toks) < 8: 
                continue
            name = toks[0]
            q = np.array(list(map(float, toks[1:5])), dtype=np.float64)
            t = np.array(list(map(float, toks[5:8])), dtype=np.float64)
            R = qvec2rotmat(q)       # world_R_cam
            C = -R.T @ t             # 相機中心（世界座標）
            poses.append((name, C, R))
    print(f"[Info] Loaded {len(poses)} query poses from {path}.")
    return poses


# ---------- 主流程 ----------
def main(sfm_dir, output_dir, port=8080, query_poses=None, no_server=False):
    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, "sfm_view.html")

    print(f"[Load] Reading reconstruction from: {sfm_dir}")
    rec = pycolmap.Reconstruction(sfm_dir)

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
        print("[Warn] No 3D points found; colored point cloud skipped.")

    # --- DB Cameras（單一 trace + 單一 labels）---
    db_xs, db_ys, db_zs = [], [], []
    db_names, db_lx, db_ly, db_lz = [], [], [], []
    for img in rec.images.values():
        try:
            C = np.array(img.projection_center()).ravel()
            v = np.array(img.viewing_direction()).ravel()
            v /= (np.linalg.norm(v) + 1e-12)
            R = build_camera_R_from_dir(v)
            xs, ys, zs = make_frustum_lines(C, R, scale=0.25)
            db_xs += xs; db_ys += ys; db_zs += zs
            db_names.append(img.name)
            db_lx.append(C[0]); db_ly.append(C[1]); db_lz.append(C[2])
        except Exception as e:
            # 若個別影像缺欄位就略過
            continue

    data.append(go.Scatter3d(
        x=db_xs, y=db_ys, z=db_zs,
        mode="lines",
        line=dict(width=2, color="red"),
        name="DB Cameras",
        legendgroup="dbcam", showlegend=True
    ))
    data.append(go.Scatter3d(
        x=db_lx, y=db_ly, z=db_lz,
        mode="text",
        text=db_names,
        textposition="top center",
        textfont=dict(size=9, color="red"),
        name="DB Labels",
        legendgroup="dblabel", showlegend=True
    ))

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
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                   aspectmode="data"),
        margin=dict(l=0, r=0, t=30, b=0),
        title="SfM + Cameras (DB=Red, Query=Blue)",
        legend=dict(itemsizing='trace', bgcolor='rgba(255,255,255,0.7)')
    )
    pio.write_html(fig, html_path, include_plotlyjs="cdn", full_html=True)
    print(f"✅ Exported HTML: {html_path}")

    # --- 簡易 HTTP server（可選） ---
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
