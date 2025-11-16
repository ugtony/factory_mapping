#!/usr/bin/env python3
import argparse
import numpy as np
import h5py
from pathlib import Path
from collections import defaultdict
import logging
import sys

# ... (Import 部分保持不變) ...
try:
    from hloc.pipelines.Cambridge.utils import (
        get_best_covisibility_clustering, get_best_hierarchical_clustering, 
        evaluate_Aachen, get_covisibility_clusters, get_hierarchical_clusters
    )
    from hloc.localize_sfm import main as localize_sfm
    from hloc.match_features import main as match_features, confs as match_confs
    from hloc.pairs_from_retrieval import main as pairs_from_retrieval
    from hloc.extract_features import main as extract_features, confs as extract_confs
except ImportError:
    print("[Warn] Standard HLOC imports failed. Trying fallback...")
    try:
        from hloc.extract_features import main as extract_features, confs as extract_confs
        from hloc.pairs_from_retrieval import main as pairs_from_retrieval
        from hloc.match_features import main as match_features, confs as match_confs
        from hloc.localize_sfm import main as localize_sfm
    except ImportError as e:
        print(f"[Error] Critical HLOC import failed: {e}")
        sys.exit(1)

# ... (load_global_descriptors 函式保持不變) ...
def load_global_descriptors(h5_path):
    names = []
    vectors = []
    target_keys = ['global_descriptor', 'descriptors', 'global_descriptors']
    def visit_fn(name, obj):
        if isinstance(obj, h5py.Group):
            for k in target_keys:
                if k in obj:
                    vec = obj[k].__array__()
                    names.append(name)
                    vectors.append(vec)
                    return
    with h5py.File(h5_path, 'r') as f:
        f.visititems(visit_fn)
    if not names: return [], np.array([])
    vectors = np.array(vectors).squeeze()
    if vectors.ndim == 1: vectors = vectors[np.newaxis, :]
    return names, vectors

# ==========================================
# [New] 產生帶有內參的 Query List
# ==========================================
def generate_query_list_with_intrinsics(output_path, image_names, width, height, fov_deg):
    """
    產生符合 COLMAP 格式的 query list，包含 SIMPLE_PINHOLE 內參。
    格式: name SIMPLE_PINHOLE W H f cx cy
    """
    fov_rad = np.deg2rad(fov_deg)
    f = 0.5 * width / np.tan(0.5 * fov_rad)
    cx = width / 2.0
    cy = height / 2.0
    
    model = "SIMPLE_PINHOLE"
    
    with open(output_path, 'w') as f_out:
        for name in image_names:
            # 寫入格式: Name Model Width Height Params(f, cx, cy)
            line = f"{name} {model} {width} {height} {f:.4f} {cx} {cy}\n"
            f_out.write(line)

# ==========================================
# 主程式
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Multi-block localization pipeline")
    parser.add_argument("--query_dir", type=Path, required=True, help="Folder containing query images")
    parser.add_argument("--outputs_root", type=Path, required=True, help="Root of hloc outputs")
    parser.add_argument("--results_file", type=Path, default="submission.txt", help="Final poses output file")
    parser.add_argument("--num_retrieval", type=int, default=10)
    
    # [New Args] 相機參數 (必須與您轉檔時的設定一致)
    parser.add_argument("--width", type=int, default=1024, help="Query image width")
    parser.add_argument("--height", type=int, default=768, help="Query image height")
    parser.add_argument("--fov", type=float, default=100.0, help="Query image FOV (degrees)")
    
    args = parser.parse_args()

    # 1. 掃描 Blocks
    block_dirs = [d for d in args.outputs_root.iterdir() if d.is_dir() and (d / "global-netvlad.h5").exists()]
    print(f"[Info] Found {len(block_dirs)} blocks: {[b.name for b in block_dirs]}")

    if not block_dirs:
        print("[Error] No valid blocks found.")
        return

    # 2. 處理 Query 特徵
    work_dir = args.query_dir.parent / "query_processed"
    work_dir.mkdir(exist_ok=True, parents=True)
    
    query_list_path = work_dir / "queries_with_intrinsics.txt"
    query_images = sorted([p.relative_to(args.query_dir).as_posix() for p in args.query_dir.glob("**/*") if p.suffix.lower() in {'.jpg','.png','.jpeg'}])
    
    # [FIX] 產生帶有內參的列表 (localize_sfm 需要，extract_features 也相容)
    print(f"[Info] Generating query list with intrinsics (W={args.width}, H={args.height}, FOV={args.fov})...")
    generate_query_list_with_intrinsics(
        query_list_path, query_images, args.width, args.height, args.fov
    )
    
    print(f"[Info] Extracting Query features for {len(query_images)} images...")
    
    # Extract Global (NetVLAD)
    global_feats_q = extract_features(
        extract_confs['netvlad'], args.query_dir, export_dir=work_dir, image_list=query_list_path
    )
    
    # Extract Local (SuperPoint)
    local_feats_q = extract_features(
        extract_confs['superpoint_aachen'], args.query_dir, export_dir=work_dir, image_list=query_list_path
    )

    # 3. Block Selection
    query_best_block = {} 
    query_scores = defaultdict(float)

    print("[Info] Selecting best blocks for queries...")
    
    q_names, q_vectors = load_global_descriptors(global_feats_q)
    
    for block in block_dirs:
        db_global_path = block / "global-netvlad.h5"
        db_names, db_vectors = load_global_descriptors(db_global_path)
        
        if len(db_names) == 0:
            print(f"[Warn] Block {block.name} is empty or invalid format.")
            continue
            
        print(f"  > Checking {block.name} ({len(db_names)} images)...")

        # 計算相似度
        sim = np.dot(q_vectors, db_vectors.T)
        topk = min(5, len(db_names))
        if topk > 0:
            block_scores = np.sort(sim, axis=1)[:, -topk:].mean(axis=1)
        else:
            block_scores = np.zeros(len(q_names))

        for i, score in enumerate(block_scores):
            q_name = q_names[i]
            if score > query_scores[q_name]:
                query_scores[q_name] = score
                query_best_block[q_name] = block

    # 4. 分組定位
    queries_by_block = defaultdict(list)
    for q, block in query_best_block.items():
        queries_by_block[block].append(q)

    final_poses = {}

    for block, q_list in queries_by_block.items():
        print(f"--- Processing {len(q_list)} queries in {block.name} ---")
        
        # [FIX] 針對每個 Block 產生對應的 Query List (同樣需要內參)
        block_q_list_path = work_dir / f"queries_{block.name}.txt"
        generate_query_list_with_intrinsics(
            block_q_list_path, q_list, args.width, args.height, args.fov
        )
            
        sfm_dir = block / "sfm_aligned"
        if not (sfm_dir / "images.bin").exists(): sfm_dir = block / "sfm"
        
        db_local_candidates = list(block.glob("local-*.h5"))
        if not db_local_candidates:
            print(f"[Warn] No local features found in {block.name}, skipping.")
            continue
        db_local = db_local_candidates[0]
        db_global = block / "global-netvlad.h5"
        
        pairs_path = work_dir / f"pairs_{block.name}.txt"
        
        # (A) Retrieval
        pairs_from_retrieval(
            global_feats_q, pairs_path, num_matched=args.num_retrieval,
            query_list=block_q_list_path, db_list=None, db_descriptors=db_global
        )
        
        # (B) Matching
        matches_path = work_dir / f"matches_{block.name}.h5"
        merged_features_path = work_dir / f"features_{block.name}.h5"
        
        if merged_features_path.exists(): merged_features_path.unlink()
        
        with h5py.File(merged_features_path, 'w') as f_out:
            with h5py.File(local_feats_q, 'r') as f_q:
                for k in f_q:
                    if k not in f_out:
                        f_out[k] = h5py.ExternalLink(str(local_feats_q.absolute()), k)
            with h5py.File(db_local, 'r') as f_db:
                 def link_visitor(name, obj):
                     if isinstance(obj, h5py.Group) and 'keypoints' in obj:
                         if name not in f_out:
                             f_out[name] = h5py.ExternalLink(str(db_local.absolute()), name)
                 f_db.visititems(link_visitor)

        match_features(
            match_confs['superpoint+lightglue'], pairs_path, 
            features=merged_features_path, matches=matches_path
        )
        
        # (C) Localization
        block_results_path = work_dir / f"results_{block.name}.txt"
        try:
            # 這裡傳入的 block_q_list_path 現在包含內參了，所以不會報錯
            localize_sfm(
                sfm_dir, block_q_list_path, pairs_path,
                merged_features_path, matches_path, block_results_path,
                covisibility_clustering=False
            )
        except Exception as e:
            print(f"[Error] Localization failed: {e}")
            # Print full stacktrace for easier debugging if it fails again
            import traceback
            traceback.print_exc()
            continue

        if block_results_path.exists():
            with open(block_results_path, 'r') as f:
                for line in f:
                    if line.startswith('#'): continue
                    parts = line.strip().split()
                    if len(parts) >= 8:
                        final_poses[parts[0]] = line.strip()

    # ==========================================
    # 5. 輸出結果 (修改版：加入 Block 名稱欄位)
    # ==========================================
    print(f"[Info] Writing results to {args.results_file}...")
    
    with open(args.results_file, 'w') as f:
        # 修改 Header，增加 BlockName
        f.write("# ImageName, Qw, Qx, Qy, Qz, Tx, Ty, Tz, BlockName\n")
        
        success_count = 0
        for q in query_images:
            if q in final_poses:
                # 取得該 Query 被分配到的 Block 名稱
                # query_best_block[q] 儲存的是 Path 物件，我們取 .name
                block_path = query_best_block.get(q)
                block_name = block_path.name if block_path else "Unknown"
                
                # 寫入原本的 Pose 字串，並在後面加上 Block 名稱
                f.write(f"{final_poses[q]} {block_name}\n")
                success_count += 1
            else:
                # 可選：失敗的也可以寫入，標記為 Failed (方便除錯)
                # f.write(f"{q} # Failed\n") 
                print(f"[Warn] Failed to localize: {q}")

    print(f"✅ All done. Results: {args.results_file} ({success_count}/{len(query_images)} localized)")

if __name__ == "__main__":
    main()