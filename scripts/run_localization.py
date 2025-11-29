# scripts/run_localization.py
#!/usr/bin/env python3
import argparse
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Import core engine
try:
    from localization_engine import LocalizationEngine
except ImportError:
    sys.path.append(str(Path(__file__).parent))
    from localization_engine import LocalizationEngine

def draw_matches(query_img, db_img_path, kpts_q, kpts_db, matches, out_path):
    """
    Visualization helper using matplotlib.
    """
    try:
        if not Path(db_img_path).exists():
            print(f"  [Viz] DB image not found: {db_img_path}")
            return
            
        im_q = query_img
        im_db = cv2.imread(str(db_img_path))
        im_db = cv2.cvtColor(im_db, cv2.COLOR_BGR2RGB)
        
        # Filter valid matches
        valid = matches > -1
        mkpts_q = kpts_q[valid]
        mkpts_db = kpts_db[matches[valid]]
        
        # Canvas
        H = max(im_q.shape[0], im_db.shape[0])
        W1, W2 = im_q.shape[1], im_db.shape[1]
        canvas = np.zeros((H, W1 + W2, 3), dtype=np.uint8)
        canvas[:im_q.shape[0], :W1] = im_q
        canvas[:im_db.shape[0], W1:] = im_db
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(canvas)
        ax.axis('off')
        
        # Draw lines (sample if too many)
        indices = np.arange(len(mkpts_q))
        if len(indices) > 100:
            np.random.shuffle(indices)
            indices = indices[:100]
            
        for i in indices:
            pt1 = mkpts_q[i]
            pt2 = mkpts_db[i]
            ax.plot([pt1[0], pt2[0] + W1], [pt1[1], pt2[1]], color="lime", linewidth=0.5, alpha=0.7)
            
        plt.tight_layout()
        plt.savefig(out_path, dpi=100)
        plt.close(fig)
    except Exception as e:
        print(f"  [Viz Error] {e}")

def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    parser = argparse.ArgumentParser(description="Offline Localization Tester (Using Shared Engine)")
    parser.add_argument("--query_dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Path to hloc outputs root (containing block folders)")
    parser.add_argument("--fov", type=float, default=None, help="Override FOV in config")
    parser.add_argument("--output", type=Path, default="offline_results.txt")
    parser.add_argument("--viz", action="store_true", help="Generate match visualizations")
    args = parser.parse_args()

    # 1. Init Engine
    engine = LocalizationEngine(
        project_root=project_root,
        config_path=project_root / "project_config.env",
        anchors_path=project_root / "anchors.json",
        outputs_dir=args.reference
    )
    
    fov = args.fov if args.fov else engine.default_fov
    print(f"=== Starting Offline Localization (FOV={fov}) ===")
    
    # 2. Scan Queries
    query_files = sorted([p for p in args.query_dir.glob("**/*") if p.suffix.lower() in {'.jpg','.png','.jpeg'}])
    print(f"Found {len(query_files)} query images.")
    
    viz_dir = args.query_dir.parent / "viz_offline"
    if args.viz: 
        viz_dir.mkdir(exist_ok=True, parents=True)
        print(f"Visualization will be saved to: {viz_dir}")

    results_lines = []
    
    # 3. Loop & Test
    success_count = 0
    for q_path in query_files:
        print(f"Processing {q_path.name}...", end=" ", flush=True)
        
        img = cv2.imread(str(q_path))
        if img is None:
            print("[Error] Read failed")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Call Engine (Same logic as Server)
        ret = engine.localize(img, fov_deg=fov, return_details=args.viz)
        
        if ret['success']:
            print(f"✅ Block: {ret['block']} ({ret['inliers']} inliers)")
            success_count += 1
            
            # Format Output line (compatible with HLOC format)
            # ImageName Qw Qx Qy Qz Tx Ty Tz Block
            qvec = ret['pose']['qvec']
            tvec = ret['pose']['tvec']
            line = f"{q_path.name} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} {tvec[0]} {tvec[1]} {tvec[2]} {ret['block']}"
            results_lines.append(line)
            
            # Visualization
            if args.viz and 'matches' in ret:
                out_name = viz_dir / f"{q_path.stem}_matches.jpg"
                draw_matches(
                    img, ret['db_image_path'], 
                    ret['kpts_query'], ret['kpts_db'], 
                    ret['matches'], out_name
                )
        else:
            print("❌ Failed")

    # 4. Save Results
    with open(args.output, 'w') as f:
        f.write("# ImageName Qw Qx Qy Qz Tx Ty Tz BlockName\n")
        for line in results_lines:
            f.write(line + "\n")
            
    print(f"\nSummary: {success_count}/{len(query_files)} localized.")
    print(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main()