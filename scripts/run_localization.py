#!/usr/bin/env python3
import argparse
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv
from pathlib import Path
from collections import Counter

# [Plan A] Setup path to find 'lib'
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from lib.localization_engine import LocalizationEngine
# [New] Import 3D visualization tool
try:
    from scripts import visualize_sfm_open3d
except ImportError:
    import visualize_sfm_open3d

def draw_matches(query_img, db_img_path, kpts_q, kpts_db, matches, out_path):
    try:
        if not Path(db_img_path).exists():
            print(f"  [Viz] DB image not found: {db_img_path}"); return
        im_q = query_img
        im_db = cv2.imread(str(db_img_path))
        im_db = cv2.cvtColor(im_db, cv2.COLOR_BGR2RGB)
        valid = matches > -1
        mkpts_q = kpts_q[valid]
        mkpts_db = kpts_db[matches[valid]]
        
        H = max(im_q.shape[0], im_db.shape[0])
        W1, W2 = im_q.shape[1], im_db.shape[1]
        canvas = np.zeros((H, W1 + W2, 3), dtype=np.uint8)
        canvas[:im_q.shape[0], :W1] = im_q
        canvas[:im_db.shape[0], W1:] = im_db
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(canvas); ax.axis('off')
        indices = np.arange(len(mkpts_q))
        if len(indices) > 100: np.random.shuffle(indices); indices = indices[:100]
        for i in indices:
            pt1 = mkpts_q[i]
            pt2 = mkpts_db[i]
            ax.plot([pt1[0], pt2[0] + W1], [pt1[1], pt2[1]], color="lime", linewidth=0.5, alpha=0.7)
        plt.tight_layout(); plt.savefig(out_path, dpi=100); plt.close(fig)
    except Exception as e: print(f"  [Viz Error] {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Path to hloc outputs root")
    parser.add_argument("--fov", type=float, default=None)
    parser.add_argument("--output", type=Path, default="offline_results.txt")
    parser.add_argument("--report", type=Path, default="diagnosis_report.csv", help="Output diagnosis CSV report")
    parser.add_argument("--viz", action="store_true", help="Visualize both 2D matches and 3D point cloud")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed logging from localization engine")
    args = parser.parse_args()

    engine = LocalizationEngine(
        project_root=project_root,
        config_path=project_root / "project_config.env",
        anchors_path=project_root / "anchors.json",
        outputs_dir=args.reference
    )
    
    fov = args.fov if args.fov else engine.default_fov
    print(f"=== Starting Offline Localization (FOV={fov}) ===")
    query_files = sorted([p for p in args.query_dir.glob("*") if p.is_file() and p.suffix.lower() in {'.jpg','.png','.jpeg'}])
    
    # Setup Viz Directories
    viz_dir_2d = args.query_dir.parent / "viz_offline"
    viz_dir_3d = args.query_dir.parent / "viz_3d"
    if args.viz: 
        viz_dir_2d.mkdir(exist_ok=True, parents=True)

    results_lines = []
    success_blocks = [] # Track which blocks were matched
    success_count = 0
    
    # Initialize CSV Report
    print(f"[Info] Diagnosis report will be saved to: {args.report}")
    csv_file = open(args.report, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_header = [
        "ImageName", "Status", 
        "Selected_Block", "PnP_Inliers",
        "Second_Block", "Second_Inliers",  # <--- [新增] 這兩欄
        "Top1_Block", "Top1_Score", 
        "Top2_Block", "Top2_Score",
        "R1_Name", "R1_Match",
        "R2_Name", "R2_Match",
        "R3_Name", "R3_Match",
        "Num_Keypoints", "Num_Matches_2D", "Num_Matches_3D"
    ]
    csv_writer.writerow(csv_header)

    for q_path in query_files:
        print(f"Processing {q_path.name}...", end=" ", flush=True)
        # 若 verbose 開啟，換行以免 Log 擠在一起
        if args.verbose: print("") 

        img = cv2.imread(str(q_path))
        if img is None: print("[Error] Read failed"); continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        ret = engine.localize(img, fov_deg=fov, return_details=args.viz, verbose=args.verbose)
        
        # Extract Diagnosis Data
        diag = ret.get('diagnosis', {})
        ranks = diag.get('db_ranks', [])
        while len(ranks) < 3: ranks.append({'name': 'None', 'matches_2d': 0})
        
        # Determine the owner block for these ranks
        rank_owner = diag.get('selected_block')
        if rank_owner == 'None' or not rank_owner:
            rank_owner = diag.get('top1_block', 'None')
            
        def fmt_rank_name(name):
            if name == 'None': return 'None'
            return f"{rank_owner}/{name}"

        # Write to CSV
        row = [
            q_path.name,
            diag.get('status', 'Unknown'),
            diag.get('selected_block', 'None'),
            diag.get('pnp_inliers', 0),
            diag.get('second_block', 'None'),
            diag.get('second_inliers', 0),
            diag.get('top1_block', 'None'),
            f"{diag.get('top1_score', 0.0):.4f}",
            diag.get('top2_block', 'None'),
            f"{diag.get('top2_score', 0.0):.4f}",
            
            fmt_rank_name(ranks[0]['name']), ranks[0]['matches_2d'],
            fmt_rank_name(ranks[1]['name']), ranks[1]['matches_2d'],
            fmt_rank_name(ranks[2]['name']), ranks[2]['matches_2d'],
            
            diag.get('num_kpts', 0),
            diag.get('num_matches_2d', 0),
            diag.get('num_matches_3d', 0)
        ]
        csv_writer.writerow(row)
        csv_file.flush()

        if ret['success']:
            if args.verbose:
                print(f"  ✅ [Result] Block: {ret['block']} ({ret['inliers']} inliers)")
            else:
                print(f"✅ Block: {ret['block']} ({ret['inliers']} inliers)")
            
            success_count += 1
            success_blocks.append(ret['block'])
            
            qvec, tvec = ret['pose']['qvec'], ret['pose']['tvec']
            line = f"{q_path.name} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} {tvec[0]} {tvec[1]} {tvec[2]} {ret['block']}"
            results_lines.append(line)
            
            if args.viz and 'matches' in ret:
                draw_matches(img, ret['db_image_path'], ret['kpts_query'], ret['kpts_db'], ret['matches'], viz_dir_2d / f"{q_path.stem}_matches.jpg")
        else:
            status = diag.get('status', 'Failed')
            top1 = diag.get('top1_block', 'None')
            score = diag.get('top1_score', 0.0)
            
            r1_name = fmt_rank_name(ranks[0]['name'])
            r1_m = ranks[0]['matches_2d']
            print(f"❌ {status} (Top1: {top1}, Score: {score:.2f}) -> BestDB: {r1_name} ({r1_m} matches)")

    csv_file.close()

    with open(args.output, 'w') as f:
        f.write("# ImageName Qw Qx Qy Qz Tx Ty Tz BlockName\n")
        for line in results_lines: f.write(line + "\n")
    print(f"\nSummary: {success_count}/{len(query_files)} localized.")
    print(f"Results saved to: {args.output}")
    print(f"Full diagnosis report: {args.report}")

    if args.viz and success_count > 0:
        print("\n=== Generating 3D Visualizations ===")
        unique_blocks = set(success_blocks)
        
        for block_name in unique_blocks:
            print(f"[Viz] Processing Block: {block_name}...")
            
            ref_root = args.reference if args.reference else (project_root / "outputs-hloc")
            sfm_path = ref_root / block_name / "sfm_aligned"
            if not (sfm_path / "images.bin").exists():
                 sfm_path = ref_root / block_name / "sfm"
            
            block_viz_dir = viz_dir_3d / block_name
            block_viz_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                visualize_sfm_open3d.main(
                    sfm_dir=str(sfm_path),
                    output_dir=str(block_viz_dir),
                    query_poses=str(args.output),
                    no_server=True,
                    target_block=block_name
                )
            except Exception as e:
                print(f"[Viz] Failed for {block_name}: {e}")

if __name__ == "__main__":
    main()