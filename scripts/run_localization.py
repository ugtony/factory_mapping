# scripts/run_localization.py
#!/usr/bin/env python3
import argparse
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# [Plan A] Setup path to find 'lib'
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from lib.localization_engine import LocalizationEngine

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
    parser.add_argument("--viz", action="store_true")
    args = parser.parse_args()

    engine = LocalizationEngine(
        project_root=project_root,
        config_path=project_root / "project_config.env",
        anchors_path=project_root / "anchors.json",
        outputs_dir=args.reference
    )
    
    fov = args.fov if args.fov else engine.default_fov
    print(f"=== Starting Offline Localization (FOV={fov}) ===")
    query_files = sorted([p for p in args.query_dir.glob("**/*") if p.suffix.lower() in {'.jpg','.png','.jpeg'}])
    viz_dir = args.query_dir.parent / "viz_offline"
    if args.viz: viz_dir.mkdir(exist_ok=True, parents=True)

    results_lines = []
    success_count = 0
    for q_path in query_files:
        print(f"Processing {q_path.name}...", end=" ", flush=True)
        img = cv2.imread(str(q_path))
        if img is None: print("[Error] Read failed"); continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        ret = engine.localize(img, fov_deg=fov, return_details=args.viz)
        
        if ret['success']:
            print(f"✅ Block: {ret['block']} ({ret['inliers']} inliers)")
            success_count += 1
            qvec, tvec = ret['pose']['qvec'], ret['pose']['tvec']
            line = f"{q_path.name} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} {tvec[0]} {tvec[1]} {tvec[2]} {ret['block']}"
            results_lines.append(line)
            if args.viz and 'matches' in ret:
                draw_matches(img, ret['db_image_path'], ret['kpts_query'], ret['kpts_db'], ret['matches'], viz_dir / f"{q_path.stem}_matches.jpg")
        else: print("❌ Failed")

    with open(args.output, 'w') as f:
        f.write("# ImageName Qw Qx Qy Qz Tx Ty Tz BlockName\n")
        for line in results_lines: f.write(line + "\n")
    print(f"\nSummary: {success_count}/{len(query_files)} localized.")

if __name__ == "__main__":
    main()