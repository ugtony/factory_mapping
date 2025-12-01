#!/usr/bin/env python3
"""
run_localization_original.py [DEBUG Version]
Modified to print intermediate values for comparison with localization_engine.py
"""

import argparse
import sys
import os
import numpy as np
import h5py
import shutil
from pathlib import Path
from collections import defaultdict
import cv2

# 設定 path 以便 import 同目錄下的模組
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.append(str(SCRIPT_DIR))

# 引入我們剛建立的工具模組
try:
    from hloc_io_utils import load_global_descriptors_safe, get_matches_key, parse_localization_log
except ImportError:
    print("[Error] Could not import hloc_io_utils.")
    sys.exit(1)

try:
    from hloc.localize_sfm import main as localize_sfm
    from hloc.extract_features import main as extract_features, confs as extract_confs
    from hloc.pairs_from_retrieval import main as pairs_from_retrieval
    from hloc.match_features import main as match_features, confs as match_confs
except ImportError:
    print("[Error] HLOC not found or import failed.")
    sys.exit(1)

def load_shell_config(config_path):
    cfg = {}
    if not config_path.exists(): return cfg
    try:
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): continue
                if '=' in line:
                    k, v = line.split('=', 1)
                    cfg[k.strip()] = v.strip().strip('"').strip("'")
    except Exception: pass
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

# --- DEBUG FUNCTIONS ---
def inspect_global_h5(h5_path, query_name):
    if not Path(h5_path).exists(): return
    with h5py.File(h5_path, 'r') as f:
        if query_name in f:
            vec = f[query_name]['global_descriptor'].__array__()
            print(f"[DEBUG] Global Desc Sum: {np.sum(vec):.6f}")
        else:
            print(f"[DEBUG] Global H5 missing query: {query_name}")

def inspect_local_h5(h5_path, query_name):
    if not Path(h5_path).exists(): return
    with h5py.File(h5_path, 'r') as f:
        if query_name in f:
            kpts = f[query_name]['keypoints'].__array__()
            print(f"[DEBUG] Raw Keypoints Count: {len(kpts)}")
            if len(kpts) > 0:
                print(f"[DEBUG] Stored Sample Kpt[0]: {kpts[0]}")
        else:
            print(f"[DEBUG] Local H5 missing query: {query_name}")

def inspect_pairs(pairs_path, query_name):
    if not Path(pairs_path).exists(): return
    top_db = None
    with open(pairs_path, 'r') as f:
        for line in f:
            q, db = line.strip().split()
            if q == query_name:
                print(f"[DEBUG] Rank 0 DB Image: {db}")
                top_db = db
                break
    return top_db

def inspect_matches(matches_h5, query_name, db_name):
    if not Path(matches_h5).exists() or not db_name: return
    with h5py.File(matches_h5, 'r') as f:
        key = get_matches_key(f, query_name, db_name)
        if key:
            matches = f[key].__array__()
            valid = matches > -1
            print(f"[DEBUG] Rank 0 Matches Found: {np.sum(valid)}")
        else:
            print(f"[DEBUG] Matches missing for pair: {query_name} - {db_name}")

# --- Main ---
def main():
    project_root = SCRIPT_DIR.parent
    config = load_shell_config(project_root / "project_config.env")
    
    default_fov = float(config.get("FOV", 69.4))
    default_global = config.get("GLOBAL_CONF", "netvlad")

    parser = argparse.ArgumentParser(description="Unified HLOC Localization Pipeline (DEBUG)")
    parser.add_argument("--query_dir", type=Path, required=True, help="Directory containing query images")
    parser.add_argument("--reference", "--ref", dest="reference", type=Path, required=True)
    parser.add_argument("--global-conf", type=str, default=default_global)
    parser.add_argument("--fov", type=float, default=default_fov)
    parser.add_argument("--num_retrieval", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=3)
    
    # Dummy args to match interface
    parser.add_argument("--viz_retrieval", action="store_true")
    parser.add_argument("--viz_matches", action="store_true")
    parser.add_argument("--viz_3d", action="store_true")
    args = parser.parse_args()

    ref_path = args.reference
    is_single_block = (ref_path / "sfm").exists() or (ref_path / "sfm_aligned").exists()
    mode = "SINGLE" if is_single_block else "MULTI"

    work_dir = args.query_dir.parent / f"query_processed_{args.global_conf}"
    work_dir.mkdir(exist_ok=True, parents=True)
    
    query_images = sorted([p.relative_to(args.query_dir).as_posix() 
                           for p in args.query_dir.glob("**/*") 
                           if p.suffix.lower() in {'.jpg','.jpeg','.png'}])
    if not query_images: print("[Error] No query images."); return
    
    # 選擇第一張圖進行 DEBUG
    target_query = query_images[0]
    print(f"\n[DEBUG] Target Query Image: {target_query}")
    
    # Load Image Size for Debug
    img_path = args.query_dir / target_query
    im_arr = cv2.imread(str(img_path))
    if im_arr is not None:
         print(f"[DEBUG] Input Image Size: {im_arr.shape[1]}x{im_arr.shape[0]}")

    query_list_path = work_dir / "queries_with_intrinsics.txt"
    generate_intrinsics(query_images, args.query_dir, query_list_path, args.fov)
    
    print("[Step 1] Extracting features for queries...")
    global_conf = extract_confs[args.global_conf]
    local_conf = extract_confs['superpoint_aachen']
    
    feats_global_q = extract_features(global_conf, args.query_dir, export_dir=work_dir, image_list=query_list_path)
    # [DEBUG]
    inspect_global_h5(feats_global_q, target_query)

    feats_local_q = extract_features(local_conf, args.query_dir, export_dir=work_dir, image_list=query_list_path)
    # [DEBUG]
    inspect_local_h5(feats_local_q, target_query)

    block_tasks = defaultdict(list)
    
    if mode == "SINGLE":
        for q in query_images: block_tasks[ref_path].append(q)
    else:
        print("[Step 2] Scoring blocks (Global Retrieval)...")
        q_names, q_vecs = load_global_descriptors_safe(feats_global_q)
        candidate_blocks = [d for d in ref_path.iterdir() if d.is_dir() and (d / f"global-{args.global_conf}.h5").exists()]
        
        for block_dir in candidate_blocks:
            db_g = block_dir / f"global-{args.global_conf}.h5"
            try:
                db_names, db_vecs = load_global_descriptors_safe(db_g)
                if len(db_vecs) == 0: continue
                sim = np.dot(q_vecs, db_vecs.T)
                topk_db = min(5, sim.shape[1])
                block_scores = np.sort(sim, axis=1)[:, -topk_db:].mean(axis=1) if topk_db > 0 else np.zeros(len(q_names))
                
                for i, score in enumerate(block_scores):
                    if not hasattr(main, "query_block_scores"): main.query_block_scores = defaultdict(list)
                    main.query_block_scores[q_names[i]].append((score, block_dir))
            except Exception: pass

        if hasattr(main, "query_block_scores"):
            for q_name, scores in main.query_block_scores.items():
                scores.sort(key=lambda x: x[0], reverse=True)
                # [DEBUG]
                if q_name == target_query:
                    print(f"[DEBUG] Top Block: {scores[0][1].name} (Score: {scores[0][0]:.4f})")
                
                top_k_blocks = scores[:args.top_k]
                for score, block_dir in top_k_blocks:
                    if score > 0: block_tasks[block_dir].append(q_name)

    results_pool = defaultdict(list)

    for block_dir, q_list in block_tasks.items():
        if not q_list: continue
        block_name = block_dir.name
        
        # 只對包含 target_query 的 block 顯示詳細 debug
        is_target_block = (target_query in q_list)
        if is_target_block:
             print(f"[DEBUG] --- Checking Block: {block_name} ---")

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
        
        # [DEBUG]
        if is_target_block:
             top_db_name = inspect_pairs(pairs_path, target_query)
        
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
        
        # [DEBUG]
        if is_target_block and top_db_name:
             inspect_matches(matches_path, target_query, top_db_name)
        
        try:
            localize_sfm(
                sfm_dir, block_q_list_path, pairs_path, 
                merged_feats, matches_path, results_path,
                covisibility_clustering=False
            )
        except Exception as e:
            print(f"[Error] Localization failed: {e}"); continue

        log_path = Path(str(results_path) + "_logs.pkl")
        inliers_map = parse_localization_log(log_path)
        
        if is_target_block:
             n_in = inliers_map.get(target_query, 0)
             print(f"[DEBUG] PnP Success! Inliers: {n_in}")

        # ... (後續合併邏輯省略，不影響 Debug) ...

if __name__ == "__main__":
    main()