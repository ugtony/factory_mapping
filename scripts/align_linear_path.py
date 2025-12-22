#!/usr/bin/env python3
"""
align_linear_path.py (V8-RANSAC-Robust)
專為直線移動的 360 切圖相機設計的對齊腳本。

修正重點 (Statistical Robustness):
1. [Robust Plane Fitting] 從單純的 SVD (L2) 改為 RANSAC 演算法。
   - 解決 L2 對離群值 (Outliers) 過於敏感的問題。
   - 隨機採樣找出最多 Inliers 的平面，再對 Inliers 做 SVD。
2. [Flow] 保持 Best Fit Plane -> Leveling -> Heading -> Scale 的流程。
"""

import sys
import shutil
import numpy as np
import pycolmap
from pathlib import Path
from scipy.spatial.transform import Rotation

def log(msg):
    print(f"[AlignLinear] {msg}")

# ==================== API Adapter ====================
class PoseAdapter:
    def __init__(self, img_sample):
        self.mode = "unknown"
        if hasattr(img_sample, "cam_from_world"):
            val = img_sample.cam_from_world
            if hasattr(val, "rotation"): self.mode = "cam_from_world_obj"
            else: self.mode = "cam_from_world_unknown"
        elif hasattr(img_sample, "qvec"): self.mode = "qvec_tvec"
        elif hasattr(img_sample, "rotation_matrix"): self.mode = "methods"
        log(f"Detected pycolmap API mode: {self.mode}")

    def get_rotation_matrix(self, img):
        if self.mode == "cam_from_world_obj":
            rot = img.cam_from_world.rotation
            if callable(getattr(rot, "matrix", None)): return rot.matrix()
            elif hasattr(rot, "matrix"): return rot.matrix
            return Rotation.from_quat(rot.quat).as_matrix()
        elif self.mode == "qvec_tvec":
            q = img.qvec
            return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        elif self.mode == "methods":
            if callable(img.rotation_matrix): return img.rotation_matrix()
            return img.rotation_matrix
        return np.eye(3) 

    def get_center(self, img):
        if callable(getattr(img, "projection_center", None)): return img.projection_center()
        elif hasattr(img, "projection_center"): return img.projection_center
        return np.zeros(3)

    def set_pose_mirror_x(self, img, flip_mat):
        R_old = self.get_rotation_matrix(img)
        C_old = self.get_center(img)
        C_new = C_old.copy(); C_new[0] *= -1 
        R_new = R_old @ flip_mat
        t_new = -R_new @ C_new
        if self.mode == "cam_from_world_obj":
            try:
                try: rot_obj = pycolmap.Rotation3d(R_new)
                except:
                    q = Rotation.from_matrix(R_new).as_quat()
                    rot_obj = pycolmap.Rotation3d(np.array([q[3], q[0], q[1], q[2]]))
                img.cam_from_world = pycolmap.Rigid3d(rot_obj, t_new)
            except: pass
        elif self.mode == "qvec_tvec":
            q = Rotation.from_matrix(R_new).as_quat()
            img.qvec = np.array([q[3], q[0], q[1], q[2]])
            img.tvec = t_new

# ==================== Robust Math Helpers ====================

def fit_plane_ransac(points, n_iter=500, threshold=0.05):
    """
    使用 RANSAC 找出最佳擬合平面
    points: (N, 3)
    Returns: normal (3,), inlier_mask (N,)
    """
    N = len(points)
    if N < 3:
        return np.array([0, 0, 1]), np.ones(N, dtype=bool)

    best_inliers = []
    best_normal = np.array([0, 0, 1])
    
    # 預先中心化可以增加數值穩定性
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid

    for _ in range(n_iter):
        # 1. Random sample 3 points
        idxs = np.random.choice(N, 3, replace=False)
        p3 = centered_points[idxs]
        
        # 2. Compute Normal (cross product)
        v1 = p3[1] - p3[0]
        v2 = p3[2] - p3[0]
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-6: continue # Collinear
        normal /= norm
        
        # 3. Count inliers (Distance to plane passing through origin)
        # dist = |dot(p, normal)|
        dists = np.abs(np.dot(centered_points, normal))
        inliers = np.where(dists < threshold)[0]
        
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_normal = normal

    # Refine: Use SVD on ONLY the inliers
    if len(best_inliers) > 2:
        final_points = centered_points[best_inliers]
        u, s, vh = np.linalg.svd(final_points)
        refined_normal = vh[-1, :]
        return refined_normal, best_inliers
    else:
        return best_normal, best_inliers

def get_rotation_between_vectors(u, v):
    u = u / (np.linalg.norm(u) + 1e-9)
    v = v / (np.linalg.norm(v) + 1e-9)
    axis = np.cross(u, v)
    sin_a = np.linalg.norm(axis)
    cos_a = np.dot(u, v)
    if sin_a < 1e-6:
        # 平行或反向
        if cos_a > 0: return np.eye(3)
        else: return -np.eye(3) # 這裡簡單翻轉，實際上應該繞任意軸轉180
    axis = axis / sin_a
    angle = np.arctan2(sin_a, cos_a)
    return Rotation.from_rotvec(axis * angle).as_matrix()

# ==================== Main Logic ====================

def rotmat2qvec_colmap(R):
    q = Rotation.from_matrix(R).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])

def create_sim3d(scale, R, t):
    try: return pycolmap.Sim3d(scale, pycolmap.Rotation3d(R), t)
    except: return pycolmap.Sim3d(scale, rotmat2qvec_colmap(R), t)

def apply_reflection_x(recon, adapter):
    log("Applying X-axis reflection (Mirror Flip)...")
    for p3d in recon.points3D.values(): p3d.xyz[0] *= -1
    flip_mat = np.diag([-1, 1, 1])
    for img in recon.images.values(): adapter.set_pose_mirror_x(img, flip_mat)
    return recon

def align_process(sfm_dir, out_dir):
    sfm_path = Path(sfm_dir)
    out_path = Path(out_dir)
    if out_path.exists(): shutil.rmtree(out_path)
    out_path.mkdir(parents=True)
    
    log(f"Loading model from {sfm_path}")
    recon = pycolmap.Reconstruction(sfm_path)
    sample_img = next(iter(recon.images.values()))
    adapter = PoseAdapter(sample_img)
    
    # Anchor Frames
    imgs_F = [img for img in recon.images.values() if "_F." in img.name]
    if len(imgs_F) < 2:
        log("Error: Not enough _F images."); return
    imgs_F.sort(key=lambda x: x.name)
    img_start, img_end = imgs_F[0], imgs_F[-1]

    # --- Step 1: Translate Start to Origin ---
    c_start = adapter.get_center(img_start)
    recon.transform(create_sim3d(1.0, np.eye(3), -c_start))

    # --- Step 2: Robust Plane Fitting (RANSAC) ---
    log("Computing Robust Plane (RANSAC) for _F cameras...")
    points_F = []
    cam_ups = [] 
    for img in recon.images.values():
        if "_F." in img.name:
            points_F.append(adapter.get_center(img))
            R = adapter.get_rotation_matrix(img)
            cam_ups.append(R.T @ np.array([0, -1, 0]))

    points_F = np.array(points_F)
    
    # 計算點雲的平均間距，用來設定 RANSAC threshold
    # 假設相機間距約 1m，若某個點偏離平面超過 0.5 * avg_dist 就算 outlier
    if len(points_F) > 1:
        dists = np.linalg.norm(points_F[1:] - points_F[:-1], axis=1)
        avg_step = np.median(dists)
        threshold = avg_step * 0.5 # 寬鬆度，可調整
    else:
        threshold = 0.1

    normal, inliers_idx = fit_plane_ransac(points_F, n_iter=1000, threshold=threshold)
    
    # 統計被剔除的點
    n_outliers = len(points_F) - len(inliers_idx)
    log(f"RANSAC Result: {len(inliers_idx)} inliers, {n_outliers} outliers removed.")
    
    # Ensure normal points Up
    avg_cam_up = np.mean(cam_ups, axis=0)
    if np.dot(normal, avg_cam_up) < 0:
        normal = -normal
        
    log(f"Robust Normal: {normal}")

    # Leveling
    target_z = np.array([0, 0, 1])
    R_level = get_rotation_between_vectors(normal, target_z)
    
    # Center points (using INLIER mean for stability)
    mean_F = np.mean(points_F[inliers_idx], axis=0) if len(inliers_idx) > 0 else np.mean(points_F, axis=0)
    
    recon.transform(create_sim3d(1.0, np.eye(3), -mean_F))
    recon.transform(create_sim3d(1.0, R_level, np.zeros(3)))
    
    # --- Step 3: Align Heading (Start->End) ---
    c_s = adapter.get_center(next(img for img in recon.images.values() if img.name == img_start.name))
    recon.transform(create_sim3d(1.0, np.eye(3), -c_s)) # Start at origin again
    
    c_e = adapter.get_center(next(img for img in recon.images.values() if img.name == img_end.name))
    vec_xy = np.array([c_e[0], c_e[1], 0]) 
    vec_xy = vec_xy / (np.linalg.norm(vec_xy) + 1e-9)
    target_y = np.array([0, 1, 0])
    
    cross_val = vec_xy[0]*target_y[1] - vec_xy[1]*target_y[0]
    dot_val = np.dot(vec_xy[:2], target_y[:2])
    yaw = np.arctan2(cross_val, dot_val)
    
    R_heading = Rotation.from_euler('z', yaw).as_matrix()
    recon.transform(create_sim3d(1.0, R_heading, np.zeros(3)))

    # --- Step 4: Mirror Check ---
    cnt_pos = 0; cnt_neg = 0
    for img in recon.images.values():
        if "_R." in img.name:
            c = adapter.get_center(img)
            if c[0] > 0.05: cnt_pos += 1
            elif c[0] < -0.05: cnt_neg += 1
    
    log(f"Mirror Check: Right={cnt_pos}, Left={cnt_neg}")
    if cnt_neg > cnt_pos:
        apply_reflection_x(recon, adapter)

    # --- Step 5: Scale ---
    c_s = adapter.get_center(next(img for img in recon.images.values() if img.name == img_start.name))
    c_e = adapter.get_center(next(img for img in recon.images.values() if img.name == img_end.name))
    dist = np.linalg.norm(c_e - c_s)
    scale = 100.0 / (dist + 1e-9)
    log(f"Scaling dist {dist:.4f} -> 100.0")
    
    recon.transform(create_sim3d(scale, np.eye(3), np.zeros(3)))
    recon.write(out_path)
    log(f"Done. Saved to {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 align_linear_path.py <in> <out>")
        sys.exit(1)
    align_process(sys.argv[1], sys.argv[2])