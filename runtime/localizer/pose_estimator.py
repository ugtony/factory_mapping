from pathlib import Path
import numpy as np
import cv2
import pycolmap
from hloc.utils.io import read_image  # 其餘地方若要用到可保留

# -----------------------------
# Resize 與相機選擇工具函式
# -----------------------------
def _get_resized_dims(orig_w, orig_h, max_side):
    """
    依照 max_side 縮放到最長邊 = max_side，維持長寬比。
    傳回 (target_w, target_h)（皆為偶數，避免某些模型/庫要求偶數邊）。
    """
    if max_side is None or max_side <= 0:
        return orig_w, orig_h

    scale = float(max_side) / float(max(orig_w, orig_h))
    if scale >= 1.0:
        # 不放大，保持原尺寸
        target_w, target_h = int(round(orig_w)), int(round(orig_h))
    else:
        target_w = int(round(orig_w * scale))
        target_h = int(round(orig_h * scale))

    # 轉成偶數（保守作法）
    if target_w % 2 == 1:
        target_w -= 1
    if target_h % 2 == 1:
        target_h -= 1
    # 避免變成 0
    target_w = max(target_w, 2)
    target_h = max(target_h, 2)

    return target_w, target_h


def _get_best_cam_model(target_w, target_h, all_cams, verbose=False):
    """
    從 SfM 模型中找出尺寸最接近的相機（以 |Δw| + |Δh| 當作距離）。
    """
    def score(cam):
        return abs(int(cam.width) - int(target_w)) + abs(int(cam.height) - int(target_h))

    best_cam = min(all_cams, key=score)

    if verbose:
        print(f"    [Info] Query (target {target_w}x{target_h}) using closest SfM cam ({best_cam.width}x{best_cam.height}) ID: {best_cam.camera_id}")

    return best_cam


def _get_query_hw(query_img_path, max_side=None):
    """
    穩定讀取 query 影像尺寸，必要時依 max_side 計算縮放後尺寸。
    直接用 cv2 讀，避免把 PIL / tensor 當成 PIL Image 而出錯。
    回傳 (hq, wq)（注意：順序為 (H, W)）。
    """
    img = cv2.imread(str(query_img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {query_img_path}")
    h, w = img.shape[:2]  # cv2: (H, W)

    target_w, target_h = _get_resized_dims(w, h, max_side)  # 傳入 (orig_w, orig_h)
    return int(target_h), int(target_w)  # 回傳 (H, W)


# -----------------------------
# 主函式：由 DB 影像估計 Query 位姿
# -----------------------------
def estimate_pose_from_db_image(
    sfm_dir: Path,
    db_image_name: str,
    matches_q2db,   # 形狀 (N, 2)，每列為 (iq, idb)
    kpts_q,         # 形狀 (N_q, 2)
    kpts_db,        # 形狀 (N_db, 2)
    query_image_path: Path,
    max_side: int
):
    """
    以檢索到的資料庫影像 (db_image_name) 與其 SfM 3D 點，對應 Query 的 2D-3D 後做 PnP。
    回傳 dict: {"qvec": ..., "tvec": ..., "inliers": ...} 或 None。
    """
    # 載入重建
    rec = pycolmap.Reconstruction(str(sfm_dir))

    # 尋找指定的 DB 影像
    img = None
    for im in rec.images.values():
        if im.name == db_image_name:
            img = im
            break
    if img is None:
        print(f"    [Error] DB image {db_image_name} not found in SfM model: {sfm_dir}")
        return None

    # --- 1) 取得 Query 影像縮放後尺寸，並選取最接近的相機內參 ---
    try:
        hq, wq = _get_query_hw(query_image_path, max_side)  # (H, W)
    except Exception as e:
        print(f"    [Error] Failed to read query image {query_image_path}: {e}")
        return None

    all_cams = list(rec.cameras.values())
    if not all_cams:
        print(f"    [Error] No cameras found in SfM model: {sfm_dir}")
        return None

    cam = _get_best_cam_model(wq, hq, all_cams, verbose=True)

    # --- 2. Prepare 2D-3D correspondences (coordinate-nearest mapping) ---
    pts2D, pts3D = [], []
    num_matches_total = len(matches_q2db)
    valid_3d_count = 0

    # 先把 COLMAP 的 points2D 取出 (xy, point3D_id)，供近鄰查找
    # 注意：xy 是以「用來重建的影像尺寸」為基準（你 build_block_model 時的尺寸）。
    p2d_list = img.points2D  # list[pycolmap.Point2D]
    if len(p2d_list) == 0:
        print("    [Error] No points2D in COLMAP image; cannot build correspondences.")
        return None

    p2d_xy = np.asarray([np.asarray(p.xy, dtype=float) for p in p2d_list], dtype=float)  # [M,2]
    p2d_3did = [int(p.point3D_id) if p.point3D_id != -1 else -1 for p in p2d_list]      # list[int]


    # 近鄰容許閾值（像素）：對同一張圖/同縮放，1~2px 通常足夠；你也可以試 3.0
    NEAREST_TOL = 2.0

    for iq, idb in matches_q2db:
        # 索引合理性
        if iq < 0 or iq >= len(kpts_q) or idb < 0 or idb >= len(kpts_db):
            continue

        # 取 DB 特徵點座標 (x,y) —— 這個座標系必須和 COLMAP 的 points2D 一致
        xy_db = np.asarray(kpts_db[idb][:2], dtype=float)  # [2]

        # 找最近的 COLMAP Point2D
        diffs = p2d_xy - xy_db[None, :]           # [M,2]
        d2 = np.sum(diffs * diffs, axis=1)        # [M]
        j = int(np.argmin(d2))
        dist = float(np.sqrt(d2[j]))

        if dist > NEAREST_TOL:
            # 太遠，當作不同點，跳過
            # print(f"    [Debug] Nearest p2d dist={dist:.2f} > tol={NEAREST_TOL}, skip")
            continue

        p3did = int(p2d_3did[j])
        if p3did == -1 or p3did not in rec.points3D:
            # 沒綁 3D 或 3D 點不在重建中
            continue

        p3d = rec.points3D[p3did]
        pts2D.append(kpts_q[iq][:2])  # 用 Query 的像素座標
        pts3D.append(p3d.xyz)         # 對應 3D 座標
        valid_3d_count += 1

    print(f"    [Debug] Total matches: {num_matches_total}, Matches with valid 3D points: {valid_3d_count}")

    min_matches_for_pnp = 4
    if valid_3d_count < min_matches_for_pnp:
        print(f"    [Info] Not enough valid 2D-3D matches ({valid_3d_count}) for PnP (minimum {min_matches_for_pnp}). Skipping.")
        return None

    pts2D = np.asarray(pts2D, dtype=float)
    pts3D = np.asarray(pts3D, dtype=float)

    # --- 3) PnP + RANSAC ---
    ransac_options = {
        "max_error": 8.0,          # 像素 reprojection 上限
        "min_inlier_ratio": 0.01,
        "min_num_trials": 100,
        "max_num_trials": 10000,
        "confidence": 0.999,
    }
    refinement_options = {
        "refine_focal_length": False,
        "refine_principal_point": False,
        "refine_extra_params": False,
    }

    ret = pycolmap.absolute_pose_estimation(
        pts2D, pts3D, cam,
        estimation_options={"ransac": ransac_options},
        refinement_options=refinement_options,
        return_covariance=False,
    )

    if ret is None:
        print(f"    [Info] pycolmap.absolute_pose_estimation returned None.")
        return None

    num_inliers = int(ret.get("num_inliers", 0))
    qvec = ret["qvec"]
    tvec = ret["tvec"]

    min_pnp_inliers = 6
    print(f"    [Debug] PnP RANSAC finished. Inliers: {num_inliers}")
    if num_inliers < min_pnp_inliers:
        print(f"    [Info] PnP found too few inliers ({num_inliers}) (minimum {min_pnp_inliers}). Skipping.")
        return None

    return {"qvec": qvec, "tvec": tvec, "inliers": num_inliers}
