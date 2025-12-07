#!/usr/bin/env python3
"""
run_localization_original.py [Advanced Diagnosis Version]
用途：執行標準 HLOC 定位流程，並產生詳細的診斷報告與 Log 分析。
特點：
1. 產生 diagnosis_report_original.csv，格式與新版對齊。
2. 執行後會自動分析中間檔 (matches.h5, _logs.pkl)，印出 2D/3D 匹配數統計。
"""

import argparse
import sys
import os
import numpy as np
import h5py
import shutil
import csv
import pickle
from pathlib import Path
from collections import defaultdict
import cv2

# 設定 path 以便 import 同目錄下的模組
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.append(str(SCRIPT_DIR))

# 引入工具模組
try:
    from hloc_io_utils import load_global_descriptors_safe, get_matches_key, parse_localization_log
except ImportError:
    print("[Error] Could not import hloc_io_utils. Please create scripts/hloc_io_utils.py first.")
    sys.exit(1)

try:
    import visualize_sfm_open3d
    HAS_VIZ_3D = True
except ImportError:
    HAS_VIZ_3D = False

# HLOC Imports
try:
    from hloc.localize_sfm import main as localize_sfm
    from hloc.extract_features import main as extract_features, confs as extract_confs
    from hloc.pairs_from_retrieval import main as pairs_from_retrieval
    from hloc.match_features import main as match_features, confs as match_confs
except ImportError:
    print("[Error] HLOC not found or import failed.")
    sys.exit(1)

# --- Config Loading ---
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

# --- Main ---
def main():
    project_root = SCRIPT_DIR.parent
    config = load_shell_config(project_root / "project_config.env")
    
    default_fov = float(config.get("FOV", 69.4))
    default_global = config.get("GLOBAL_CONF", "netvlad")

    parser = argparse.ArgumentParser(description="Unified HLOC Localization Pipeline (Diagnosis Version)")
    parser.add_argument("--query_dir", type=Path, required=True, help="Directory containing query images")
    parser.add_argument("--reference", "--ref", dest="reference", type=Path, required=True)
    parser.add_argument("--global-conf", type=str, default=default_global)
    parser.add_argument("--fov", type=float, default=default_fov)
    parser.add_argument("--num_retrieval", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=3)
    
    # [New] 報告輸出路徑
    parser.add_argument("--report", type=Path, default="diagnosis_report_original.csv", help="Output diagnosis CSV report")
    
    parser.add_argument("--viz_3d", action="store_true")
    args = parser.parse_args()

    ref_path = args.reference
    is_single_block = (ref_path / "sfm").exists() or (ref_path / "sfm_aligned").exists()
    mode = "SINGLE" if is_single_block else "MULTI"
    
    print(f"=== Localization Pipeline: {mode} Mode ===")
    
    # 初始化診斷資料結構
    diagnosis_data = defaultdict(lambda: {
        'top1_block': 'None', 'top1_score': 0.0,
        'top2_block': 'None', 'top2_score': 0.0,
        'block_inliers': defaultdict(int)
    })

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
    
    # --- Scoring / Retrieval Phase ---
    if mode == "SINGLE":
        for q in query_images: 
            block_tasks[ref_path].append(q)
            diagnosis_data[q]['top1_block'] = ref_path.name
            diagnosis_data[q]['top1_score'] = 1.0
    else:
        print("[Step 2] Scoring blocks (Global Retrieval)...")
        q_names, q_vecs = load_global_descriptors_safe(feats_global_q)
        
        candidate_blocks = [d for d in ref_path.iterdir() if d.is_dir() and (d / f"global-{args.global_conf}.h5").exists()]
        
        # 暫存分數
        if not hasattr(main, "query_block_scores"): main.query_block_scores = defaultdict(list)

        for block_dir in candidate_blocks:
            db_g = block_dir / f"global-{args.global_conf}.h5"
            try:
                db_names, db_vecs = load_global_descriptors_safe(db_g)
                if len(db_vecs) == 0: continue
                sim = np.dot(q_vecs, db_vecs.T)
                topk_db = min(5, sim.shape[1])
                if topk_db > 0:
                    block_scores = np.sort(sim, axis=1)[:, -topk_db:].mean(axis=1)
                else:
                    block_scores = np.zeros(len(q_names))
                
                for i, score in enumerate(block_scores):
                    main.query_block_scores[q_names[i]].append((score, block_dir))
                    
            except Exception as e:
                print(f"[Warn] Failed to score {block_dir.name}: {e}")

        # Assign Tasks & Fill Diagnosis Data
        for q_name, scores in main.query_block_scores.items():
            scores.sort(key=lambda x: x[0], reverse=True)
            
            if len(scores) > 0:
                diagnosis_data[q_name]['top1_block'] = scores[0][1].name
                diagnosis_data[q_name]['top1_score'] = float(scores[0][0])
            if len(scores) > 1:
                diagnosis_data[q_name]['top2_block'] = scores[1][1].name
                diagnosis_data[q_name]['top2_score'] = float(scores[1][0])

            top_k_blocks = scores[:args.top_k]
            for score, block_dir in top_k_blocks:
                if score > 0:
                    block_tasks[block_dir].append(q_name)

    results_pool = defaultdict(list)

    # --- Localization Phase ---
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
        
        # 1. Retrieval
        pairs_from_retrieval(
            feats_global_q, pairs_path, num_matched=args.num_retrieval,
            query_list=block_q_list_path, db_list=None, db_descriptors=db_global
        )
        
        # 2. Merge Features
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

        # 3. Match
        match_features(
            match_confs['superpoint+lightglue'], pairs_path, 
            features=merged_feats, matches=matches_path
        )
        
        # 4. Localize (PnP)
        try:
            localize_sfm(
                sfm_dir, block_q_list_path, pairs_path, 
                merged_feats, matches_path, results_path,
                covisibility_clustering=False
            )
        except Exception as e:
            print(f"[Error] Localization failed: {e}"); continue

        # ==========================================
        # [New] 詳細 Log 分析區塊 (Post-Mortem Analysis)
        # ==========================================
        print(f"\n[Detailed Log Analysis for Block: {block_name}]")
        
        # 1. 讀取 _logs.pkl 取得 PnP 統計
        log_path = Path(str(results_path) + "_logs.pkl")
        loc_logs = {}
        if log_path.exists():
            with open(log_path, 'rb') as f:
                raw_log = pickle.load(f)
                loc_logs = raw_log.get('loc', raw_log)
        
        # 2. 讀取 matches.h5 與 pairs.txt
        matches_h5 = h5py.File(matches_path, 'r')
        pairs_map = defaultdict(list)
        if pairs_path.exists():
            with open(pairs_path, 'r') as f:
                for line in f:
                    p = line.strip().split()
                    if len(p) >= 2: pairs_map[p[0]].append(p[1])

        # 3. 針對每個 Query 印出詳細資訊
        for q_name in q_list:
            # (A) Retrieval Info
            retrieved_dbs = pairs_map.get(q_name, [])
            
            # (B) 2D Matches Info
            matches_info = []
            for db_name in retrieved_dbs[:3]: # 只看前 3 名
                pair_key = get_matches_key(matches_h5, q_name, db_name)
                n_matches = 0
                if pair_key:
                    # [Fix] pair_key 已經是 dataset 路徑 (例如 q/db/matches0)，不用再加 ['matches0']
                    m = matches_h5[pair_key].__array__()
                    n_matches = (m > -1).sum()
                matches_info.append(f"{db_name}({n_matches})")
            
            # (C) PnP Stats
            pnp_input = 0
            inliers = 0
            if q_name in loc_logs:
                log_info = loc_logs[q_name]
                pnp_input = log_info.get('num_matches', 0) # HLOC log 中 num_matches 指的是 2D-3D 對應數
                pnp_ret = log_info.get('PnP_ret', {})
                if isinstance(pnp_ret, dict):
                    inliers = pnp_ret.get('num_inliers', 0)
                else:
                    inliers = getattr(pnp_ret, 'num_inliers', 0)
            
            # 簡潔輸出
            print(f"  Q: {q_name} | PnP Input: {pnp_input} -> Inliers: {inliers}")
            print(f"     Top-3 Matches: {', '.join(matches_info)}")

            # 存入診斷資料
            diagnosis_data[q_name]['block_inliers'][block_name] = inliers

        matches_h5.close()
        print("==========================================\n")

        # 讀取 Pose 結果供合併用
        if results_path.exists():
            with open(results_path, 'r') as f:
                for line in f:
                    if line.startswith('#'): continue
                    p = line.strip().split()
                    q_name = p[0]
                    n_inliers = diagnosis_data[q_name]['block_inliers'][block_name]
                    results_pool[q_name].append({
                        'block': block_name,
                        'inliers': n_inliers,
                        'pose_str': line.strip()
                    })

    # --- Result Merging & Report Generation ---
    final_poses = []
    localized_count = 0
    
    print(f"\n[Info] Generating diagnosis report: {args.report}")
    
    with open(args.report, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_header = [
            "ImageName", "Status", 
            "Selected_Block", "PnP_Inliers",
            "Top1_Block", "Top1_Score", 
            "Top2_Block", "Top2_Score"
        ]
        csv_writer.writerow(csv_header)
        
        for q in query_images:
            candidates = results_pool.get(q, [])
            diag = diagnosis_data[q]
            
            if candidates:
                candidates.sort(key=lambda x: x['inliers'], reverse=True)
                best = candidates[0]
                selected_block = best['block']
                inliers = best['inliers']
                status = "Success" if inliers > 10 else "Failed"
                
                final_poses.append(f"{best['pose_str']} {selected_block}")
                localized_count += 1
            else:
                selected_block = "None"
                inliers = 0
                status = "Failed"
                if diag['block_inliers']:
                    best_fail_block = max(diag['block_inliers'], key=diag['block_inliers'].get)
                    selected_block = best_fail_block + "(Fail)"

            row = [
                q, status, 
                selected_block, inliers,
                diag['top1_block'], f"{diag['top1_score']:.4f}",
                diag['top2_block'], f"{diag['top2_score']:.4f}"
            ]
            csv_writer.writerow(row)

    # 輸出最終 Pose
    final_results_file = work_dir / "final_poses.txt"
    if final_poses:
        print(f"✅ Successfully localized {localized_count}/{len(query_images)} images.")
        with open(final_results_file, 'w') as f:
            f.write("# ImageName, Qw, Qx, Qy, Qz, Tx, Ty, Tz, BlockName\n")
            for line in final_poses:
                f.write(line + "\n")
        print(f"Final Poses: {final_results_file}")
    else:
        print("\n[Warn] No images successfully localized.")

    if args.viz_3d and HAS_VIZ_3D and final_poses:
         print("[Viz] Generating 3D visualization...")
         target_block = final_poses[0].split()[-1] 
         sfm_dir = args.reference / target_block / "sfm_aligned"
         if not (sfm_dir/"images.bin").exists(): sfm_dir = args.reference / target_block / "sfm"
         viz_out = work_dir / "viz_3d" / target_block
         visualize_sfm_open3d.main(str(sfm_dir), str(viz_out), query_poses=str(final_results_file), no_server=True)

if __name__ == "__main__":
    main()