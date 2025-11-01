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

# ---------- [修改] 繪製相機圖層的輔助函式 (支援漸層色與可見度) ----------
def create_camera_traces(
    image_list_with_indices, # List of (img, idx) tuples
    total_db_count,          # Total number of cameras for normalization
    suffix, 
    colorscale,              # Plotly colorscale (e.g., 'Viridis', 'Reds')
    legend_prefix, 
    line_width=2, 
    frustum_scale=0.25, 
    text_size=9,
    default_visible=False    # <<< 新增：控制預設可見度
):
    """
    處理一個 (pycolmap.Image, index) 物件列表，並回傳兩個 Plotly traces：
    1. 相機錐台線條 (Scatter3d) - 帶有顏色梯度
    2. 相機名稱標籤 (Scatter3d)
    """
    all_xs, all_ys, all_zs = [], [], []
    label_names, label_xs, label_ys, label_zs = [], [], [], []
    line_color_values = [] # <<< 新增：用於儲存每個點的顏色對應值
    
    # 準備正規化 (避免除以 0)
    norm_denominator = float(max(1, total_db_count - 1))

    # 遍歷傳入的 (已採樣) 列表
    for img, idx in image_list_with_indices: # <<< 解開 (影像, 索引)
        try:
            C = np.array(img.projection_center()).ravel()
            v = np.array(img.viewing_direction()).ravel()
            v /= (np.linalg.norm(v) + 1e-12)
            R = build_camera_R_from_dir(v) # 使用檔案中已有的函式
            
            # 取得相機錐台線條
            xs, ys, zs = make_frustum_lines(C, R, scale=frustum_scale) # 使用檔案中已有的函式
            all_xs += xs
            all_ys += ys
            all_zs += zs
            
            # <<< 新增：為線條上的每個點指定 0.0 ~ 1.0 的顏色值 >>>
            color_val = float(idx) / norm_denominator # 正規化索引
            line_color_values.extend([color_val] * len(xs)) # 為這個相機的所有線條點指定相同顏色值
            
            # 取得標籤
            label_names.append(img.name)
            label_xs.append(C[0])
            label_ys.append(C[1])
            label_zs.append(C[2])
        except Exception as e:
            continue # 略過有問題的影像
    
    # 1. 建立線條圖層
    line_trace = go.Scatter3d(
        x=all_xs, y=all_ys, z=all_zs,
        mode="lines",
        line=dict(
            width=line_width,
            color=line_color_values, # <<< 指定顏色陣列
            colorscale=colorscale      # <<< 指定色階
        ),
        marker=dict( # <<< 必須有 marker 才能顯示 colorscale
            size=0.1, # 設為不可見
            color=line_color_values, # 顏色陣列
            colorscale=colorscale,
            showscale=(suffix == "100%"), # <<< [Req 1] 只在 100% 圖層顯示色條
            colorbar=dict(title="Time", tickvals=[0.0, 1.0], ticktext=['Start', 'End'])
        ),
        name=f"{legend_prefix} Cameras ({suffix})", # e.g., "DB Cameras (50%)"
        legendgroup=f"{legend_prefix}_cam_{suffix}", 
        showlegend=True,
        visible=True if default_visible else "legendonly" # <<< [Req 2] 根據參數設定可見度
    )
    
    # 2. 建立標籤圖層
    label_trace = go.Scatter3d(
        x=label_xs, y=label_ys, z=label_zs,
        mode="text",
        text=label_names,
        textposition="top center",
        textfont=dict(size=text_size, color="red"), # 標籤保持紅色易讀
        name=f"{legend_prefix} Labels ({suffix})", # e.g., "DB Labels (50%)"
        legendgroup=f"{legend_prefix}_label_{suffix}", 
        showlegend=True,
        visible="legendonly" # <<< [Req 2] 標籤預設全隱藏
    )
    
    return line_trace, label_trace

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

    # --- DB Cameras (Multi-level Sampling) ---
    print("[Info] Sorting and sampling DB cameras...")
    
    # 1. 排序：先依照影像名稱排序，確保順序性
    try:
        all_db_images_sorted = sorted(rec.images.values(), key=lambda img: img.name)
    except Exception as e:
        # 備用排序（如果 .name 不可用）
        all_db_images_sorted = sorted(rec.images.values(), key=lambda img: img.image_id)
        
    # <<< 修改：建立 (影像, 索引) 的元組列表 >>>
    db_images_with_indices = [(img, i) for i, img in enumerate(all_db_images_sorted)]
    total_db_count = len(db_images_with_indices)
    
    # <<< 新增：定義要用於時間漸層的色階 >>>
    db_colorscale = 'Reds'  # 從淺紅到深紅
    db_colorscale = 'Viridis'  # 從藍綠到黃色
    
    if total_db_count > 0:
        # 2. 定義您要的採樣率 (名稱, 百分比)
        sample_rates = [
            ("100%", 100.0),
            ("50%", 50.0),
            ("25%", 25.0),
            ("12.5%", 12.5)
        ]
        
        # 3. 迭代採樣率，產生圖層
        for suffix, rate in sample_rates:
            
            # 4. 計算要取的數量 (至少 2 個，即頭尾)
            target_num = int(round(total_db_count * (rate / 100.0)))
            target_num = max(target_num, 2) 
            
            if target_num >= total_db_count: # (改為 >= 增加穩健性)
                # 如果採樣數大於總數，就全取
                indices = np.arange(total_db_count)
            else:
                # 5. 使用 np.linspace 產生等距索引，這會自動包含 0 (頭) 和 total_db_count-1 (尾)
                indices = np.linspace(0, total_db_count - 1, num=target_num, dtype=int)
            
            # 確保索引唯一
            indices = sorted(list(set(indices)))
            
            # <<< 修改：根據索引取得 (影像, 索引) 列表 >>>
            sampled_list_with_indices = [db_images_with_indices[i] for i in indices]
            
            print(f"    > Sampling {suffix}: {len(sampled_list_with_indices)} cameras")
            
            # <<< 修改：判斷是否為 100% 圖層 (用於設定預設可見度) >>>
            is_100_percent_trace = (suffix == "100%")
            
            # 6. 呼叫新的輔助函式來產生線條和標籤圖層
            line_trace, label_trace = create_camera_traces(
                image_list_with_indices=sampled_list_with_indices, # <<< 傳入 (影像, 索引) 列表
                total_db_count=total_db_count,                     # <<< 傳入總數
                suffix=suffix,
                colorscale=db_colorscale,                          # <<< 傳入色階
                legend_prefix="DB",
                default_visible=is_100_percent_trace               # <<< [Req 2] 傳入可見度
            )
            
            # 7. 將圖層加入到 'data' 列表中
            data.append(line_trace)
            data.append(label_trace)
            
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
            # (Query 預設為 True)
        ))
        data.append(go.Scatter3d(
            x=q_lx, y=q_ly, z=q_lz,
            mode="text",
            text=q_names,
            textposition="top center",
            textfont=dict(size=10, color="blue"),
            name="Query Labels",
            legendgroup="qlabel", showlegend=True
            # (Query 預設為 True)
        ))

    # --- 圖表設定與輸出 ---
    fig = go.Figure(data=data)
    fig.update_layout(
        title="SfM + Cameras (DB=Red, Query=Blue)",
        scene=dict(
            xaxis_title='X', 
            yaxis_title='Y', 
            zaxis_title='Z (Up)', # 標記 Z 軸為 "Up"
            aspectmode="data", # 保持長寬比
            
            # 1. 強制 Z 軸為「上」方向
            camera=dict(
                up=dict(x=0, y=0, z=1),
                
                # 2. 設置一個合理的初始視角
                eye=dict(x=1.5, y=1.5, z=0.5) 
            ),
            
            # 3. 更改滑鼠旋轉模式為 "orbit"
            dragmode="orbit",
            
            # 4. 顯示 XY/XZ/YZ 網格平面
            xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.3)', zeroline=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.3)', zeroline=False),
            zaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.3)', zeroline=False),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
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