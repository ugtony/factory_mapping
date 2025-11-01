#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
align_sfm_model_z_up.py (V25-calib-clean-loglite)
- Y = PC1 固定（相機位移主方向）
- 僅繞 Y 調整，令平均 image-up（世界座標 -Y_cam）朝 +Z
- Kabsch #1：C0 @ R ≈ C_target（row-vector/右乘模型）
- 兩種 Sim3d 寫法（R 與 R^T）各寫一次到乾淨目錄，讀回比對 Var，擇優
- Calibration：讀回擇優結果的 centers，對 C_target 再跑一次 Kabsch 求 ΔR，套用後再置中，最終輸出
- 嚴格只用 .bin，清掉 .txt，避免混讀；相機與點雲始終同一 Reconstruction 上 transform()

日誌等級：
  --quiet=1      只輸出最後摘要（最少）
  (預設)         精簡關鍵步驟（COMPACT）
  --verbose=1    額外輸出細節（最多）
"""

import sys, shutil, tempfile
from pathlib import Path
import numpy as np
import pycolmap

try:
    from sklearn.decomposition import PCA
except ImportError:
    print("[Error] scikit-learn not found. pip install scikit-learn", file=sys.stderr)
    sys.exit(1)

# ------------- logging -------------
QUIET = False
VERBOSE = False
def log(level: str, msg: str):
    """
    level: 'Q' (always, even quiet), 'I' (compact), 'D' (verbose)
    """
    if level == 'Q':
        print(msg); return
    if QUIET:
        return
    if level == 'I':
        print(msg); return
    if level == 'D' and VERBOSE:
        print(msg)

# ---------- utils ----------
def parse_bool(s, default=False):
    if s is None: return default
    s = str(s).strip().lower()
    return s in ("1","true","yes","y","on")

def unit(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

def angle_deg(a, b):
    a = unit(a); b = unit(b)
    c = np.clip(np.dot(a, b), -1.0, 1.0)
    return float(np.degrees(np.arccos(float(c))))

def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] - R[2, 1]) / S
        z = 0.25 * S
    q = np.array([w, x, y, z], dtype=np.float64)
    if q[0] < 0: q *= -1
    return q / (np.linalg.norm(q) + 1e-12)

def build_camera_R_from_dir(view_dir, up_ref=np.array([0., 0., 1.])):
    z = unit(view_dir)
    u = up_ref if abs(np.dot(z, up_ref)) < 0.98 else np.array([0., 1., 0.])
    x = unit(np.cross(u, z))
    y = unit(np.cross(z, x))
    return np.stack([x, y, z], axis=1)

def export_ply_points(path: Path, pts: np.ndarray, rgb=(255,255,255)):
    pts = np.asarray(pts, dtype=np.float64)
    R,G,B = rgb
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p in pts:
            f.write(f"{p[0]} {p[1]} {p[2]} {R} {G} {B}\n")

# ---------- axes: Y locked, rotate around Y ----------
def build_axes_y_locked(pc1, pc2, pc3, up_vecs):
    Y = unit(pc1)
    Z0 = pc3 - np.dot(pc3, Y) * Y
    if np.linalg.norm(Z0) < 1e-10:
        Z0 = pc2 - np.dot(pc2, Y) * Y
    if np.linalg.norm(Z0) < 1e-10:
        tmp = np.array([0.,0.,1.]) if abs(np.dot(Y,[0,0,1])) < 0.9 else np.array([1.,0.,0.])
        Z0 = tmp - np.dot(tmp, Y) * Y
    Z0 = unit(Z0)
    X0 = unit(np.cross(Y, Z0))
    Z0 = unit(np.cross(X0, Y))

    if len(up_vecs) == 0:
        return X0, Y, Z0

    U = np.stack(up_vecs, axis=0)
    U_perp = U - (U @ Y[:,None]).reshape(-1,1) * Y[None,:]
    ubar = U_perp.mean(axis=0)
    if np.linalg.norm(ubar) < 1e-10:
        return X0, Y, Z0
    ubar = unit(ubar)

    a = float(np.dot(ubar, X0))
    b = float(np.dot(ubar, Z0))
    theta = np.arctan2(a, b)

    c, s = np.cos(theta), np.sin(theta)
    X = unit(c*X0 - s*Z0)
    Z = unit(s*X0 + c*Z0)
    X = unit(np.cross(Y, Z))
    Z = unit(np.cross(X, Y))
    return X, Y, Z

# ---------- Kabsch ----------
def kabsch_solve(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    H = A.T @ B
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R

# ---------- file helpers ----------
def clean_model_dir(dirpath: Path):
    dirpath.mkdir(parents=True, exist_ok=True)
    patterns = ["cameras.bin","images.bin","points3D.bin",
                "cameras.txt","images.txt","points3D.txt"]
    for p in patterns:
        f = dirpath / p
        if f.exists():
            f.unlink()

def copy_model_bin(src_dir: Path, dst_dir: Path):
    clean_model_dir(dst_dir)
    for name in ("cameras.bin","images.bin","points3D.bin"):
        src = src_dir / name
        if not src.exists():
            log('Q', f"[Error] Missing {name} in {src_dir}")
            sys.exit(2)
        shutil.copy2(src, dst_dir / name)

def apply_and_measure(out_dir: Path, R_apply: np.ndarray):
    rec_out = pycolmap.Reconstruction(out_dir)
    rec_out.transform(pycolmap.Sim3d(1.0, rotmat2qvec(R_apply), np.zeros(3)))
    centers_rot = np.stack([np.asarray(img.projection_center(), dtype=np.float64)
                            for img in rec_out.images.values() if img.has_pose], axis=0)
    t_center = -centers_rot.mean(axis=0)
    rec_out.transform(pycolmap.Sim3d(1.0, rotmat2qvec(np.eye(3)), t_center))
    rec_out.write(out_dir)
    rec_chk = pycolmap.Reconstruction(out_dir)
    centers_after = np.stack([np.asarray(img.projection_center(), dtype=np.float64)
                              for img in rec_chk.images.values() if img.has_pose], axis=0)
    var_after = centers_after.var(axis=0)
    log('D', f"[Sanity] read-back: #cams={len(rec_chk.images)}, #points3D={len(rec_chk.points3D)}")
    return var_after, centers_after

# ---------- main ----------
def align_model_single_pass(
    sfm_dir: str,
    out_dir: str | None = None,
    dump: bool = True,
    export_ply: bool = False,
) -> int:
    sfm_path = Path(sfm_dir)
    if not (sfm_path/"images.bin").exists():
        log('Q', f"[Error] Not a valid SfM directory: {sfm_path}")
        return 1

    final_out = Path(out_dir) if out_dir else Path(str(sfm_path) + "_aligned")
    clean_model_dir(final_out)

    log('I', f"[Load] {sfm_path}")
    rec = pycolmap.Reconstruction(sfm_path)

    centers, up_vecs = [], []
    for img in rec.images.values():
        if not img.has_pose: continue
        centers.append(np.asarray(img.projection_center(), dtype=np.float64))
        try:
            v = np.asarray(img.viewing_direction(), dtype=np.float64).ravel()
            Rwc = build_camera_R_from_dir(v)
            up_vecs.append(-Rwc[:,1])
        except Exception:
            pass
    if len(centers) < 3:
        log('Q', "[Warn] Not enough registered images (<3). Abort.")
        return 0

    centers = np.stack(centers, axis=0)
    C0 = centers - centers.mean(axis=0)

    pca = PCA(n_components=3).fit(centers)
    vals = pca.explained_variance_
    PC1, PC2, PC3 = pca.components_
    log('I', f"[PCA] Var(PC1,PC2,PC3): {vals}")

    Xf, Yf, Zf = build_axes_y_locked(PC1, PC2, PC3, up_vecs)
    log('D', f"[Angles] Y-PC1={angle_deg(Yf,PC1):.4f}°, X-PC2={angle_deg(Xf,PC2):.4f}°, Z-PC3={angle_deg(Zf,PC3):.4f}°")

    S_cols = np.stack([Xf, Yf, Zf], axis=1)
    C_target = C0 @ S_cols
    theory_var = C_target.var(axis=0)
    log('I', f"[Target] Var(X,Y,Z): {theory_var} (Y max, Z min)")

    R_kabsch = kabsch_solve(C0, C_target)
    if VERBOSE:
        ortho_err = np.linalg.norm(R_kabsch @ R_kabsch.T - np.eye(3))
        log('D', f"[Kabsch] |R R^T - I|_F={ortho_err:.3e}, det={np.linalg.det(R_kabsch):.6f}")
        log('D', f"[Check] Var(C0@R)={(C0@R_kabsch).var(axis=0)}")

    with tempfile.TemporaryDirectory(prefix="sfm_align_") as tmpdir:
        tmpdir = Path(tmpdir)
        candA = tmpdir / "cand_Rt"
        candB = tmpdir / "cand_R"
        copy_model_bin(sfm_path, candA)
        copy_model_bin(sfm_path, candB)

        varA, centersA = apply_and_measure(candA, R_kabsch.T)
        varB, centersB = apply_and_measure(candB, R_kabsch)

        dA = float(np.abs(varA - theory_var).sum())
        dB = float(np.abs(varB - theory_var).sum())
        log('I', f"[Select] Rt L1={dA:.4g}, R L1={dB:.4g}")

        if dA <= dB:
            chosen_dir, chosen_var, chosen_centers, tag = candA, varA, centersA, "R^T"
        else:
            chosen_dir, chosen_var, chosen_centers, tag = candB, varB, centersB, "R"

        # Calibration ΔR
        rec_chosen = pycolmap.Reconstruction(chosen_dir)
        centers_chosen = np.stack([
            np.asarray(img.projection_center(), dtype=np.float64)
            for img in rec_chosen.images.values() if img.has_pose
        ], axis=0)
        A = centers_chosen - centers_chosen.mean(axis=0, keepdims=True)
        B = C_target       - C_target.mean(axis=0, keepdims=True)
        R_fix = kabsch_solve(A, B)

        with tempfile.TemporaryDirectory(prefix="sfm_align_fix_") as fixdir:
            fixdir = Path(fixdir)
            copy_model_bin(chosen_dir, fixdir)

            rec_fix = pycolmap.Reconstruction(fixdir)
            rec_fix.transform(pycolmap.Sim3d(1.0, rotmat2qvec(R_fix.T), np.zeros(3)))
            centers_rot2 = np.stack([
                np.asarray(img.projection_center(), dtype=np.float64)
                for img in rec_fix.images.values() if img.has_pose
            ], axis=0)
            t_center2 = -centers_rot2.mean(axis=0)
            rec_fix.transform(pycolmap.Sim3d(1.0, rotmat2qvec(np.eye(3)), t_center2))
            rec_fix.write(fixdir)

            rec_final_chk = pycolmap.Reconstruction(fixdir)
            centers_final = np.stack([
                np.asarray(img.projection_center(), dtype=np.float64)
                for img in rec_final_chk.images.values() if img.has_pose
            ], axis=0)
            var_final = centers_final.var(axis=0)
            log('I', f"[Calib] Var(X,Y,Z): {var_final}")

            clean_model_dir(final_out)
            for name in ("cameras.bin","images.bin","points3D.bin"):
                shutil.copy2(fixdir / name, final_out / name)

        chosen_var = var_final
        chosen_centers = centers_final
        tag = f"{tag}+ΔR"

    log('I', f"[After] Var(X,Y,Z): {chosen_var}")
    log('I', f"[Write] {final_out}")

    # Summary（quiet 等級也會輸出）
    rec_chk = pycolmap.Reconstruction(final_out)
    cams, pts = len(rec_chk.images), len(rec_chk.points3D)
    log('Q', f"✅ Aligned (mode={tag})  cams={cams} points={pts}  out={final_out}")
    if dump:
        dbg = final_out / "debug"; dbg.mkdir(exist_ok=True)
        np.save(dbg/"basis_xyz_rows_new.npy", np.stack([Xf, Yf, Zf], axis=0))
        np.save(dbg/"R_kabsch.npy", R_kabsch)
        np.save(dbg/"centers_before.npy", centers)
        np.save(dbg/"centers_after.npy", chosen_centers)
        np.save(dbg/"theory_var.npy", theory_var)
        log('I', f"[Dump] {dbg}")
    if export_ply:
        dbg = final_out / "debug"
        export_ply_points(dbg/"centers_before.ply", centers, (200,200,255))
        export_ply_points(dbg/"centers_after.ply", chosen_centers, (255,200,200))
        if pts > 0:
            pts_after = np.stack([np.asarray(p.xyz, dtype=np.float64)
                                  for p in rec_chk.points3D.values()], axis=0)
            export_ply_points(dbg/"points3D_after.ply", pts_after, (200,255,200))
        log('I', f"[PLY] {dbg}")
    return 0

# ---------- CLI ----------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <sfm_dir> [--out_dir=<dir>] [--dump=1] [--export-ply=0] [--quiet=0] [--verbose=0]")
        sys.exit(1)

    sfm_dir = sys.argv[1]
    out_dir = None
    dump = True
    export_ply = False
    for a in sys.argv[2:]:
        if a.startswith("--out_dir="):
            out_dir = a.split("=",1)[1]
        elif a.startswith("--dump="):
            dump = parse_bool(a.split("=",1)[1], True)
        elif a.startswith("--export-ply="):
            export_ply = parse_bool(a.split("=",1)[1], False)
        elif a.startswith("--quiet="):
            QUIET = parse_bool(a.split("=",1)[1], False)
        elif a.startswith("--verbose="):
            VERBOSE = parse_bool(a.split("=",1)[1], False)

    # quiet 與 verbose 同時設時，以 quiet 為主
    if QUIET: VERBOSE = False

    sys.exit(align_model_single_pass(
        sfm_dir=sfm_dir,
        out_dir=out_dir,
        dump=dump,
        export_ply=export_ply,
    ))
