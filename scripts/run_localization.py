#!/usr/bin/env python3
"""
run_localization.py [V9 - Log Fix]
統一的室內定位腳本。

修正歷程：
1. [Fix] 修正 Log 讀取邏輯：支援 HLOC 的 nested log 結構 (logs['loc'][q_name])，解決 Inliers 讀不到導致為 0 的問題。
2. [Fix] 針對 matches.h5 路徑編碼問題增加相容性檢查。
3. [Fix] 視覺化 Key 搜尋邏輯優化。
4. [Fix] 增加舊資料清理機制。
"""

import argparse
import sys
import os
import pickle
import numpy as np
import h5py
import shutil
from pathlib import Path
from collections import defaultdict
import cv2

# 嘗試載入同目錄下的 visualize_sfm_open3d
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.append(str(SCRIPT_DIR))

try:
    import visualize_sfm_open3d
    HAS_VIZ_3D = True
except ImportError:
    HAS_VIZ_3D = False

# ==========================================
# 1. HLOC Imports
# ==========================================
try:
    from hloc.localize_sfm import main as localize_sfm
    from hloc.extract_features import main as extract_features, confs as extract_confs
    from hloc.pairs_from_retrieval import main as pairs_from_retrieval
    from hloc.match_features import main as match_features, confs as match_confs
except ImportError:
    print("[Error] HLOC not found or import failed.")
    sys.exit(1)

# ==========================================
# 2. 工具函式
# ==========================================
def load_shell_config(config_path):
    cfg = {}
    if not config_path.exists(): return cfg
    print(f"[Init] Loading config from {config_path}")
    try:
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): continue
                if '=' in line:
                    k, v = line.split('=', 1)
                    cfg[k.strip()] = v.strip().strip('"').strip("'")
    except Exception as e:
        print(f"[Warn] Config load failed: {e}")
    return cfg

def generate_intrinsics(query_list, image_dir, output_path, fov_deg):
    fov_rad = np.deg2rad(fov_deg)
    lines = []
    print(f"[Auto] Generating intrinsics for {len(query_list)} images (FOV={fov_deg}°)...")
    for name in query_list:
        path = image_dir / name
        if not path.exists(): continue
        img = cv2.imread(str(path))
        if img is None: continue
        h, w = img.shape[:2]
        max_side = max(w, h)
        f = 0.5 * max_side / np.tan(0.5 * fov_rad)
        cx, cy = w / 2.0, h / 2.0
        lines.append(f"{name} SIMPLE_PINHOLE {w} {h} {f:.4f} {cx} {cy}")
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

def load_global_descriptors(h5_path):
    names, vectors = [], []
    def visit(name, obj):
        if isinstance(obj, h5py.Group) and 'global_descriptor' in obj:
            names.append(name)
            vectors.append(obj['global_descriptor'].__array__())
    with h5py.File(h5_path, 'r') as f:
        f.visititems(visit)
    
    if len(vectors) == 0: return [], np.array([])
    
    vecs = np.array(vectors)
    if vecs.ndim == 3 and vecs.shape[1] == 1: vecs = vecs.squeeze(1)
    if vecs.ndim == 1: vecs = vecs[np.newaxis, :] 
    
    return names, vecs

# ==========================================
# 3. 視覺化函式
# ==========================================
def run_viz_retrieval(query_list, pairs_path, query_root, db_root, out_dir, max_figs=10):
    try: from PIL import Image
    except ImportError: return
    
    if out_dir.exists(): shutil.rmtree(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    
    pairs = defaultdict(list)
    with open(pairs_path, 'r') as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 2: pairs[p[0]].append(p[1])
    count = 0
    for q_name in query_list:
        if q_name not in pairs: continue
        dbs = pairs[q_name][:5]
        try: q_img = Image.open(query_root / q_name).convert("RGB")
        except: continue
        W = 800
        def resize_w(im): 
            w, h = im.size
            return im.resize((W, int(h * (W/w))), Image.BILINEAR)
        q_viz = resize_w(q_img)
        db_imgs = []
        for db_name in dbs:
            try: db_imgs.append(resize_w(Image.open(db_root / db_name).convert("RGB")))
            except: pass
        if not db_imgs: continue
        total_h = q_viz.size[1] + sum(im.size[1] for im in db_imgs)
        canvas = Image.new("RGB", (W, total_h), (255,255,255))
        y = 0
        canvas.paste(q_viz, (0, y)); y += q_viz.size[1]
        for im in db_imgs: canvas.paste(im, (0, y)); y += im.size[1]
        out_name = Path(q_name).stem.replace('/', '_') + "_retrieval.jpg"
        canvas.save(out_dir / out_name, quality=80)
        count += 1
        if count >= max_figs: break
    print(f"[Viz] Generated {count} retrieval visualizations.")

def run_viz_matches(query_list, pairs_path, local_feats, matches_h5, query_root, db_root, out_dir, max_figs=5):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError: return
    
    if out_dir.exists(): shutil.rmtree(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    
    top1_pairs = {}
    with open(pairs_path, 'r') as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 2 and p[0] not in top1_pairs: top1_pairs[p[0]] = p[1]
    count = 0
    with h5py.File(local_feats, 'r') as ffeat, h5py.File(matches_h5, 'r') as fmat:
        for q in query_list:
            if q not in top1_pairs: continue
            db = top1_pairs[q]
            pair_key = None
            
            def find_key_robust(name, obj):
                nonlocal pair_key
                if pair_key: return
                if isinstance(obj, h5py.Dataset) and name.endswith("matches0"):
                    clean_path = name.replace("/matches0", "")
                    padded_name = f"/{clean_path}/"
                    padded_q = f"/{q}/"
                    padded_db = f"/{db}/"
                    if (padded_q in padded_name and padded_db in padded_name) or \
                       (q.replace('/','-') in name and db.replace('/','-') in name):
                        pair_key = name
                        
            fmat.visititems(find_key_robust)
            
            if not pair_key: continue
            matches = fmat[pair_key].__array__()
            valid = matches > -1
            if valid.sum() < 10: continue
            kpts_q = ffeat[q]['keypoints'].__array__()
            kpts_db = ffeat[db]['keypoints'].__array__()
            pts_q = kpts_q[np.where(valid)[0]][:, :2]
            pts_db = kpts_db[matches[valid]][:, :2]
            try:
                im_q = np.array(Image.open(query_root / q).convert("RGB"))
                im_db = np.array(Image.open(db_root / db).convert("RGB"))
            except: continue
            H = max(im_q.shape[0], im_db.shape[0])
            W1, W2 = im_q.shape[1], im_db.shape[1]
            canvas = np.zeros((H, W1 + W2, 3), dtype=np.uint8)
            canvas[:im_q.shape[0], :W1] = im_q
            canvas[:im_db.shape[0], W1:] = im_db
            fig = plt.figure(figsize=(12, 6)); ax = fig.add_subplot(111)
            ax.imshow(canvas); ax.axis('off')
            indices = np.arange(len(pts_q))
            if len(indices) > 100: np.random.shuffle(indices); indices = indices[:100]
            for idx in indices:
                ax.plot([pts_q[idx, 0], pts_db[idx, 0] + W1], [pts_q[idx, 1], pts_db[idx, 1]], color="lime", alpha=0.5, linewidth=0.5)
            plt.savefig(out_dir / (Path(q).stem.replace('/', '_') + "_matches.jpg"), bbox_inches='tight', dpi=150)
            plt.close(fig); count += 1
            if count >= max_figs: break
    print(f"[Viz] Generated {count} match visualizations.")

# ==========================================
# 4. 主程式
# ==========================================
def main():
    project_root = SCRIPT_DIR.parent
    config = load_shell_config(project_root / "project_config.env")
    
    default_fov = float(config.get("FOV", 69.4))
    default_global = config.get("GLOBAL_CONF", "netvlad")

    parser = argparse.ArgumentParser(description="Unified HLOC Localization Pipeline (Robust Top-K)")
    parser.add_argument("--query_dir", type=Path, required=True, help="Directory containing query images")
    parser.add_argument("--reference", "--ref", dest="reference", type=Path, required=True,
                        help="Path to a SINGLE block OR a ROOT directory (Multi Block)")
    parser.add_argument("--global-conf", type=str, default=default_global, help=f"Global model (default: {default_global})")
    parser.add_argument("--fov", type=float, default=default_fov, help=f"Query camera FOV (default: {default_fov})")
    parser.add_argument("--num_retrieval", type=int, default=10, help="Number of pairs per block")
    parser.add_argument("--top_k", type=int, default=3, help="Number of candidate blocks to verify per query")
    
    parser.add_argument("--viz_retrieval", action="store_true")
    parser.add_argument("--viz_matches", action="store_true")
    parser.add_argument("--viz_3d", action="store_true")
    args = parser.parse_args()

    ref_path = args.reference
    is_single_block = (ref_path / "sfm").exists() or (ref_path / "sfm_aligned").exists()
    mode = "SINGLE" if is_single_block else "MULTI"
    print(f"=== Localization Pipeline: {mode} Mode ===")
    print(f"Reference: {ref_path} | Global: {args.global_conf} | Top-K: {args.top_k}")

    work_dir = args.query_dir.parent / f"query_processed_{args.global_conf}"
    work_dir.mkdir(exist_ok=True, parents=True)
    
    query_images = sorted([p.relative_to(args.query_dir).as_posix() 
                           for p in args.query_dir.glob("**/*") 
                           if p.suffix.lower() in {'.jpg','.jpeg','.png'}])
    if not query_images: print("[Error] No query images."); return

    query_list_path = work_dir / "queries_with_intrinsics.txt"
    generate_intrinsics(query_images, args.query_dir, query_list_path, args.fov)
    
    print("[Step 1] Extracting features for queries...")
    global_conf = extract_confs[args.global_conf]
    local_conf = extract_confs['superpoint_aachen']
    
    feats_global_q = extract_features(global_conf, args.query_dir, export_dir=work_dir, image_list=query_list_path)
    feats_local_q = extract_features(local_conf, args.query_dir, export_dir=work_dir, image_list=query_list_path)

    block_tasks = defaultdict(list)
    
    if mode == "SINGLE":
        for q in query_images: block_tasks[ref_path].append(q)
    else:
        print("[Step 2] Scoring blocks (Global Retrieval)...")
        q_names, q_vecs = load_global_descriptors(feats_global_q)
        
        candidate_blocks = []
        for d in ref_path.iterdir():
            if d.is_dir() and (d / f"global-{args.global_conf}.h5").exists():
                candidate_blocks.append(d)
        
        print(f"    > Comparing {len(query_images)} queries against {len(candidate_blocks)} blocks...")
        
        for block_dir in candidate_blocks:
            db_g = block_dir / f"global-{args.global_conf}.h5"
            try:
                db_names, db_vecs = load_global_descriptors(db_g)
                if len(db_vecs) == 0: continue
                sim = np.dot(q_vecs, db_vecs.T)
                topk_db = min(5, sim.shape[1])
                if topk_db > 0:
                    block_scores = np.sort(sim, axis=1)[:, -topk_db:].mean(axis=1)
                else:
                    block_scores = np.zeros(len(q_names))
                
                for i, score in enumerate(block_scores):
                    if not hasattr(main, "query_block_scores"): main.query_block_scores = defaultdict(list)
                    main.query_block_scores[q_names[i]].append((score, block_dir))
                    
            except Exception as e:
                print(f"[Warn] Failed to score {block_dir.name}: {e}")

        assigned_count = 0
        if hasattr(main, "query_block_scores"):
            for q_name, scores in main.query_block_scores.items():
                scores.sort(key=lambda x: x[0], reverse=True)
                top_k_blocks = scores[:args.top_k]
                for score, block_dir in top_k_blocks:
                    if score > 0:
                        block_tasks[block_dir].append(q_name)
                        assigned_count += 1
        print(f"    > Assigned {assigned_count} localization tasks (Top-{args.top_k} strategy).")

    results_pool = defaultdict(list)

    for block_dir, q_list in block_tasks.items():
        if not q_list: continue
        block_name = block_dir.name
        print(f"\n--- Processing Block: {block_name} ({len(q_list)} queries) ---")
        
        block_q_list_path = work_dir / f"queries_{block_name}.txt"
        with open(block_q_list_path, 'w') as f:
            with open(query_list_path, 'r') as f_all:
                for line in f_all:
                    if line.split()[0] in q_list:
                        f.write(line)

        sfm_dir = block_dir / "sfm_aligned"
        if not (sfm_dir / "images.bin").exists(): sfm_dir = block_dir / "sfm"
        db_global = block_dir / f"global-{args.global_conf}.h5"
        
        pairs_path = work_dir / f"pairs_{block_name}.txt"
        matches_path = work_dir / f"matches_{block_name}.h5"
        results_path = work_dir / f"results_{block_name}.txt"
        
        local_candidates = list(block_dir.glob("local-*.h5"))
        if not local_candidates: continue
        db_local = local_candidates[0]
        
        pairs_from_retrieval(
            feats_global_q, pairs_path, num_matched=args.num_retrieval,
            query_list=block_q_list_path, db_list=None, db_descriptors=db_global
        )
        
        merged_feats = work_dir / f"feats_merged_{block_name}.h5"
        if merged_feats.exists(): merged_feats.unlink()
        
        if matches_path.exists(): matches_path.unlink()
        
        with h5py.File(merged_feats, 'w') as f_out:
            with h5py.File(feats_local_q, 'r') as f_q:
                for k in f_q: 
                    if k not in f_out: f_out[k] = h5py.ExternalLink(str(feats_local_q), k)
            with h5py.File(db_local, 'r') as f_db:
                def visit_link(name, obj):
                    if isinstance(obj, h5py.Group) and 'keypoints' in obj:
                        if name not in f_out: f_out[name] = h5py.ExternalLink(str(db_local), name)
                f_db.visititems(visit_link)

        match_features(
            match_confs['superpoint+lightglue'], pairs_path, 
            features=merged_feats, matches=matches_path
        )
        
        try:
            localize_sfm(
                sfm_dir, block_q_list_path, pairs_path, 
                merged_feats, matches_path, results_path,
                covisibility_clustering=False
            )
        except Exception as e:
            print(f"[Error] Localization failed: {e}"); continue

        log_path = Path(str(results_path) + "_logs.pkl")
        logs = {}
        if log_path.exists():
            with open(log_path, 'rb') as f: logs = pickle.load(f)
        
        # [Fix V9] Handle nested log structure: logs['loc'][q_name]
        # This fixes the "0 inliers" bug because we were looking in the wrong place
        loc_logs = logs.get('loc', logs)

        if results_path.exists():
            with open(results_path, 'r') as f:
                for line in f:
                    if line.startswith('#'): continue
                    p = line.strip().split()
                    q_name = p[0]
                    n_inliers = 0
                    
                    # Use loc_logs to access the correct dictionary
                    if q_name in loc_logs:
                        entry = loc_logs[q_name]
                        if 'PnP_ret' in entry:
                            n_inliers = entry['PnP_ret'].get('num_inliers', 0)
                    
                    results_pool[q_name].append({
                        'block': block_name,
                        'inliers': n_inliers,
                        'pose_str': line.strip()
                    })

        db_root = project_root / "data" / block_name
        if not db_root.exists(): db_root = args.query_dir.parent 
        viz_root = work_dir / "viz" / block_name
        if args.viz_retrieval: run_viz_retrieval(q_list, pairs_path, args.query_dir, db_root, viz_root/"retrieval")
        if args.viz_matches: run_viz_matches(q_list, pairs_path, merged_feats, matches_path, args.query_dir, db_root, viz_root/"matches")
        if args.viz_3d and HAS_VIZ_3D:
            viz_out = viz_root / "3d"
            visualize_sfm_open3d.main(str(sfm_dir), str(viz_out), query_poses=str(results_path), no_server=True)

    final_results_file = work_dir / "final_poses.txt"
    print("\n[Step 4] Merging results based on Geometric Verification (Inliers)...")
    
    final_poses = []
    localized_count = 0
    
    for q in query_images:
        candidates = results_pool.get(q, [])
        if not candidates:
            print(f"  [Warn] {q}: No successful localization candidates.")
            continue
        
        candidates.sort(key=lambda x: x['inliers'], reverse=True)
        best = candidates[0]
        
        debug_info = " vs ".join([f"{c['block']}={c['inliers']}" for c in candidates])
        print(f"  > {q}: Winner={best['block']} ({best['inliers']} inliers) | Candidates: [{debug_info}]")
        
        final_poses.append(f"{best['pose_str']} {best['block']}")
        localized_count += 1

    if final_poses:
        print(f"✅ Successfully localized {localized_count}/{len(query_images)} images.")
        with open(final_results_file, 'w') as f:
            f.write("# ImageName, Qw, Qx, Qy, Qz, Tx, Ty, Tz, BlockName\n")
            for line in final_poses:
                f.write(line + "\n")
        print(f"Final Poses: {final_results_file}")
    else:
        print("\n[Warn] No images successfully localized.")

if __name__ == "__main__":
    main()