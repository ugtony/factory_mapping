from pathlib import Path
import numpy as np
import pycolmap

def estimate_pose_from_db_image(sfm_dir, db_image_name, matches_q2db, kpts_q, kpts_db):
    rec = pycolmap.Reconstruction(str(sfm_dir))
    img = None
    for im in rec.images.values():
        if im.name == db_image_name:
            img = im
            break
    if img is None:
        raise KeyError(f"DB image {db_image_name} not found in SfM model.")

    pts2D, pts3D = [], []
    for iq, idb in matches_q2db:
        if idb < 0 or idb >= len(kpts_db):
            continue
        p2d = img.points2D[idb]
        if p2d.point3D_id == -1:
            continue
        p3d = rec.points3D[p2d.point3D_id]
        pts2D.append(kpts_q[iq][:2])
        pts3D.append(p3d.xyz)
    if len(pts3D) < 6:
        return None

    pts2D = np.asarray(pts2D, dtype=float)
    pts3D = np.asarray(pts3D, dtype=float)
    cam = rec.cameras[img.camera_id]
    ok, qvec, tvec, inliers = pycolmap.absolute_pose_estimation(
        pts2D, pts3D, cam, estimation="ransac", refinement_options={"refine_focal_length": False}
    )
    if not ok:
        return None
    return {"qvec": qvec, "tvec": tvec, "inliers": int(inliers.sum()) if hasattr(inliers, "sum") else int(inliers)}
