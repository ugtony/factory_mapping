#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/analyze_reconstruction_quality.py [V5.1-Fix-ImportError]

[V5.1 修正]
修正當沒有安裝 tabulate 時，會在 Exception Handler 中發生 UnboundLocalError 的問題。
將 headers 定義移至 try 區塊外。
"""

import argparse
import numpy as np
import pycolmap
from pathlib import Path
from collections import defaultdict
import re

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze SfM quality with view retention metrics.")
    parser.add_argument("--outputs", type=Path, default=Path("outputs-hloc"), help="Root directory of HLOC outputs.")
    parser.add_argument("--block", type=str, default=None, help="Specific block name to analyze.")
    parser.add_argument("--scalar_intra", type=float, default=0.5, help="Scalar for Intra-Shift threshold. Default: 0.5")
    parser.add_argument("--scalar_step", type=float, default=1.0, help="Scalar for Step-Dev threshold. Default: 1.0")
    return parser.parse_args()

def get_frame_info(filename):
    stem = Path(filename).stem
    try:
        fid, suffix = stem.rsplit('_', 1)
    except ValueError:
        return stem, 0, "Unknown"

    nums = re.findall(r'\d+', fid)
    if nums:
        ts = int(nums[-1])
    else:
        ts = 0
        
    return fid, ts, suffix

def analyze_model(model_path, args):
    if not (model_path / "images.bin").exists():
        return None

    try:
        recon = pycolmap.Reconstruction(model_path)
    except Exception as e:
        print(f"[Error] Failed to load {model_path}: {e}")
        return None

    if len(recon.images) == 0:
        return {'status': 'Empty', 'n_images': 0}

    # 1. Group images & Count Views
    frames = defaultdict(list)
    frame_ts = {} 
    view_counts = defaultdict(int)

    for img_id, img in recon.images.items():
        fname = img.name
        fid, ts, suffix = get_frame_info(fname)
        
        center = img.projection_center()
        frames[fid].append(center)
        frame_ts[fid] = ts
        view_counts[suffix] += 1

    sorted_fids = sorted(frames.keys(), key=lambda k: (frame_ts[k], k))
    n_frames = len(sorted_fids)
    
    if n_frames < 2:
        return {'status': 'TooFewFrames', 'n_frames': n_frames}

    # 2. Metric: View Retention
    view_stats = {}
    for v, count in view_counts.items():
        rate = (count / n_frames) * 100
        view_stats[v] = rate

    if view_stats:
        min_view = min(view_stats, key=view_stats.get)
        min_view_rate = view_stats[min_view]
    else:
        min_view, min_view_rate = "None", 0.0

    # 3. Metric: Intra-frame Shift
    intra_diffs = []
    frame_centers = {} 

    for fid in sorted_fids:
        centers = np.array(frames[fid])
        centroid = np.mean(centers, axis=0)
        frame_centers[fid] = centroid
        
        if len(centers) > 1:
            dists = np.linalg.norm(centers - centroid, axis=1)
            intra_diffs.extend(dists)

    avg_intra_shift = np.mean(intra_diffs) if intra_diffs else 0.0
    
    # 4. Metric: Inter-frame Steps
    steps = []
    path_points = []
    for fid in sorted_fids:
        path_points.append(frame_centers[fid])
        
    path_points = np.array(path_points)
    for i in range(len(path_points) - 1):
        dist = np.linalg.norm(path_points[i+1] - path_points[i])
        steps.append(dist)
    
    steps = np.array(steps)
    if len(steps) == 0: return {'status': 'NoSteps', 'n_frames': n_frames}

    step_median = np.median(steps)
    step_std = np.std(steps)
    baseline_step = max(step_median, 0.01)

    # 5. Thresholds & Grading
    thresh_intra = baseline_step * args.scalar_intra
    thresh_step_dev = baseline_step * args.scalar_step
    
    quality = "Excellent"
    reasons = []

    if step_std > thresh_step_dev * 2:
        quality = "Bad"
        reasons.append(f"Jump Motion")
    elif step_std > thresh_step_dev:
        if quality != "Bad": quality = "Warning"
        reasons.append(f"Irregular Pace")

    if avg_intra_shift > thresh_intra * 2:
        if quality != "Bad": quality = "Bad"
        reasons.append(f"Rig Broken")
    elif avg_intra_shift > thresh_intra:
        if quality == "Excellent": quality = "Warning"
        reasons.append(f"Loose Rig")

    if min_view_rate < 50.0:
        if quality != "Bad": quality = "Bad"
        reasons.append(f"Missing Views ({min_view}:{min_view_rate:.0f}%)")
    elif min_view_rate < 80.0:
        if quality == "Excellent": quality = "Warning"
        reasons.append(f"Low Retention ({min_view}:{min_view_rate:.0f}%)")

    if not reasons: reasons.append("Stable")

    return {
        'n_frames': n_frames,
        'median_step': step_median,
        'intra_val': avg_intra_shift,
        'intra_thresh': thresh_intra,
        'step_std': step_std,
        'step_thresh': thresh_step_dev,
        'view_stats': view_stats,
        'min_view': min_view,
        'min_view_rate': min_view_rate,
        'quality': quality,
        'reason': ", ".join(reasons)
    }

def main():
    args = parse_args()
    results = []
    details = {} 
    
    search_dir = args.outputs
    if args.block:
        block_dirs = [search_dir / args.block]
    else:
        block_dirs = sorted([d for d in search_dir.iterdir() if d.is_dir()])

    print(f"Scanning {len(block_dirs)} blocks...")

    for block_dir in block_dirs:
        model_path = block_dir / "sfm_aligned"
        used_aligned = True
        if not (model_path / "images.bin").exists():
            model_path = block_dir / "sfm"
            used_aligned = False
        
        if not (model_path / "images.bin").exists():
            continue

        res = analyze_model(model_path, args)
        if res and 'quality' in res:
            name = block_dir.name + ("*" if not used_aligned else "")
            min_ret_str = f"{res['min_view_rate']:.0f}% ({res['min_view']})"
            
            row = [
                name,
                f"{res['n_frames']}",
                f"{res['median_step']:.2f}",
                f"{res['intra_val']:.3f}/{res['intra_thresh']:.2f}",
                f"{res['step_std']:.2f}/{res['step_thresh']:.2f}",
                min_ret_str,
                res['quality'],
                res['reason']
            ]
            results.append(row)
            details[name] = res['view_stats']

    if not results:
        print("No valid models found.")
        return

    # [Fix] 將 headers 定義移出 try 區塊
    headers = ["Block", "Frms", "Step", "Intra(Act/Lim)", "StepDev(Act/Lim)", "MinRet", "Quality", "Reason"]

    try:
        from tabulate import tabulate
        print(tabulate(results, headers=headers, tablefmt="simple_grid"))
    except ImportError:
        # Fallback formatting
        fmt = "{:<15} | {:<5} | {:<7} | {:<16} | {:<18} | {:<12} | {:<9} | {}"
        print(fmt.format(*headers))
        print("-" * 120)
        for r in results:
            print(fmt.format(*r))

    print("\n[Detailed View Retention]")
    for name, stats in details.items():
        order = ['F', 'FR', 'R', 'RB', 'B', 'BL', 'L', 'LF']
        sorted_keys = sorted(stats.keys(), key=lambda x: order.index(x) if x in order else 99)
        stat_str = ", ".join([f"{k}:{stats[k]:.0f}%" for k in sorted_keys])
        print(f"  - {name:<15}: {stat_str}")

if __name__ == "__main__":
    main()