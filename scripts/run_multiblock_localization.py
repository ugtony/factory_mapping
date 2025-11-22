#!/usr/bin/env python3
import argparse
import numpy as np
import h5py
from pathlib import Path
from collections import defaultdict
import logging
import sys
from PIL import Image  # [NEW] 需要用來讀取圖片尺寸

# ==========================================
# 1. HLOC Import (相容性修復版)
# ==========================================
try:
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

# ==========================================
# 2. 工具函式：強健的 H5 讀取器
# ==========================================
def load_global_descriptors(h5_path):
    """
    遞迴掃描 H5 檔案，找出所有包含 global descriptors 的圖片。
    """
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
        
    if not names:
        return [], np.array([])
    
    vectors = np.array(vectors).squeeze()
    if vectors.ndim == 1: vectors = vectors[np.newaxis, :]
    return names, vectors

# ==========================================
# 3. 產生帶有內參的 Query List (自動判斷版)
# ==========================================
def generate_query_list_with_intrinsics(output_path, image_names, query_root_dir, fov_deg):
    """
    [修改版] 自動讀取影像尺寸，並依據長邊 FOV 計算焦距。
    解決直拍/橫拍混用問題，無需手動輸入 width/height。
    """
    fov_rad = np.deg2rad(fov_deg)
    model = "SIMPLE_PINHOLE"
    
    print(f"[Auto] Generating intrinsics list based on Long-Side FOV = {fov_deg}°...")

    with open(output_path, 'w') as f_out:
        for name in image_names:
            img_path = query_root_dir / name
            if not img_path.exists():
                print(f"[Warn] Image not found: {img_path}, skipping.")
                continue

            # 1. 實際讀取圖片取得尺寸
            try:
                with Image.open(img_path) as img:
                    w_real, h_real = img.size
            except Exception as e:
                print(f"[Error] Cannot read image size: {name}, skipping. ({e})")
                continue

            # 2. 判斷長邊，計算焦距
            # 邏輯：Focal Length = (Long_Side / 2) / tan(FOV / 2)
            # 這樣無論直拍橫拍，焦距都會是正確的
            max_side = max(w_real, h_real)
            f = 0.5 * max_side / np.tan(0.5 * fov_rad)
            
            # 3. 主點設為中心
            cx = w_real / 2.0
            cy = h_real / 2.0
            
            # 寫入 COLMAP 格式
            line = f"{name} {model} {w_real} {h_real} {f:.4f} {cx} {cy}\n"
            f_out.write(line)

# ==========================================
# 4. 主程式
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Multi-block localization pipeline (Auto-Intrinsics)")
    parser.add_argument("--query_dir", type=Path, required=True, help="Folder containing query images")
    parser.add_argument("--outputs_root", type=Path, required=True, help="Root of hloc outputs")
    parser.add_argument("--results_file", type=Path, default="submission.txt", help="Final poses output file")
    
    parser.add_argument(
        "--global_model", type=str, default="netvlad", 
        choices=list(extract_confs.keys()), 
        help="Global feature extractor model name."
    )
    
    parser.add_argument("--num_retrieval", type=int, default=10)
    # [修改] 移除 width/height，只保留 FOV
    parser.add_argument("--fov", type=float, default=69.4, help="Query camera Max-Side FOV (e.g. iPhone 1x = 69.4)")
    
    args = parser.parse_args()

    # 檢查 Image 模組
    try:
        from PIL import Image
    except ImportError:
        print("[Error] PIL (Pillow) is required. Please install it: pip install Pillow")
        sys.exit(1)

    try:
        global_conf = extract_confs[args.global_model]
    except KeyError:
        print(f"[Error] Model '{args.global_model}' not found in hloc extract_confs.")
        sys.exit(1)
        
    global_feature_filename_DB = f"global-{args.global_model}.h5" 
    
    print(f"[Info] Using global model: {args.global_model}")
    print(f"[Info] DB Blocks must use filename: {global_feature_filename_DB}")

    # 1. 掃描 Blocks
    block_dirs = [d for d in args.outputs_root.iterdir() if d.is_dir() and (d / global_feature_filename_DB).exists()]
    print(f"[Info] Found {len(block_dirs)} blocks: {[b.name for b in block_dirs]}")

    if not block_dirs:
        print("[Error] No valid blocks found.")
        return

    # 2. 處理 Query 特徵
    work_dir = args.query_dir.parent / "query_processed"
    work_dir.mkdir(exist_ok=True, parents=True)
    
    query_list_path = work_dir / "queries_with_intrinsics.txt"
    query_images = sorted([p.relative_to(args.query_dir).as_posix() for p in args.query_dir.glob("**/*") if p.suffix.lower() in {'.jpg','.png','.jpeg'}])
    
    print(f"[Info] Found {len(query_images)} query images.")
    print(f"[Info] Generating intrinsics (Auto-detect size, FOV={args.fov})...")
    
    # [修改] 呼叫自動內參生成函式
    generate_query_list_with_intrinsics(
        query_list_path, query_images, args.query_dir, args.fov
    )
    
    print(f"[Info] Extracting Query features...")
    
    # Extract Global
    global_feats_q = extract_features(
        global_conf, args.query_dir, export_dir=work_dir, image_list=query_list_path
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
    
    if len(q_names) == 0:
         print("[Error] Failed to extract global descriptors for queries.")
         return

    for block in block_dirs:
        db_global_path = block / global_feature_filename_DB
        db_names, db_vectors = load_global_descriptors(db_global_path)
        
        if len(db_names) == 0:
            print(f"[Warn] Block {block.name} is empty or invalid format.")
            continue
            
        # print(f"  > Checking {block.name} ({len(db_names)} images)...")

        sim = np.dot(q_vectors, db_vectors.T)
        topk = min(5, len(db_names))
        block_scores = np.sort(sim, axis=1)[:, -topk:].mean(axis=1) if topk > 0 else np.zeros(len(q_names))

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
        
        block_q_list_path = work_dir / f"queries_{block.name}.txt"
        
        # [修改] 針對此 Block 的影像再次生成內參列表 (因為 localize_sfm 需要獨立的 query list)
        generate_query_list_with_intrinsics(
            block_q_list_path, q_list, args.query_dir, args.fov
        )
            
        sfm_dir = block / "sfm_aligned"
        if not (sfm_dir / "images.bin").exists(): sfm_dir = block / "sfm"
        
        db_local_candidates = list(block.glob("local-*.h5"))
        if not db_local_candidates:
            print(f"[Warn] No local features found in {block.name}, skipping.")
            continue
        db_local = db_local_candidates[0]
        db_global = block / global_feature_filename_DB
        
        pairs_path = work_dir / f"pairs_{block.name}.txt"
        
        # (A) Retrieval
        pairs_from_retrieval(
            global_feats_q, pairs_path, num_matched=args.num_retrieval,
            query_list=block_q_list_path, db_list=None, db_descriptors=db_global
        )
        
        # (B) Matching & Merge
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
            localize_sfm(
                sfm_dir, block_q_list_path, pairs_path,
                merged_features_path, matches_path, block_results_path,
                covisibility_clustering=False
            )
        except Exception as e:
            print(f"[Error] Localization failed: {e}")
            continue

        if block_results_path.exists():
            with open(block_results_path, 'r') as f:
                for line in f:
                    if line.startswith('#'): continue
                    parts = line.strip().split()
                    if len(parts) >= 8:
                        final_poses[parts[0]] = line.strip()

    # 5. 輸出
    with open(args.results_file, 'w') as f:
        f.write("# ImageName, Qw, Qx, Qy, Qz, Tx, Ty, Tz, BlockName\n")
        count = 0
        for q in query_images:
            if q in final_poses:
                block_path = query_best_block.get(q)
                block_name = block_path.name if block_path else "Unknown"
                f.write(f"{final_poses[q]} {block_name}\n")
                count += 1
            else:
                # print(f"[Warn] Failed to localize: {q}")
                pass

    print(f"✅ All done. Results: {args.results_file} ({count}/{len(query_images)} localized)")

if __name__ == "__main__":
    main()