#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_two_frames.py
用途：指定兩張影像 (img1, img2) 與對應的 HLOC 特徵檔，
      使用 LightGlue 進行即時匹配並繪製結果，用於除錯「為什麼這兩張圖沒接上？」

使用範例：
  python scripts/visualize_two_frames.py \
    --data_root data/school \
    --local_feats outputs-hloc/school/local-superpoint_aachen.h5 \
    --img1 db/frames-000011.jpg \
    --img2 db/frames-000006.jpg
"""

import argparse
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import sys

# 嘗試匯入 LightGlue
try:
    from lightglue import LightGlue
    from lightglue.utils import rbd
except ImportError:
    print("[Error] 請先安裝 lightglue: pip install lightglue", file=sys.stderr)
    sys.exit(1)

def load_feature_from_h5(h5_path, img_key, device):
    """從 H5 載入特定影像的 SuperPoint 特徵，並轉為 LightGlue 格式"""
    with h5py.File(h5_path, 'r') as f:
        if img_key not in f:
            raise KeyError(f"在 {h5_path} 中找不到 {img_key} 的特徵")
        
        grp = f[img_key]
        kpts = torch.from_numpy(grp['keypoints'][()]).float()
        desc = torch.from_numpy(grp['descriptors'][()]).float()
        
        # HLOC SuperPoint 儲存格式通常為:
        # keypoints: (N, 2)
        # descriptors: (D, N)  (D=256)
        
        # LightGlue 預期輸入:
        # keypoints: (1, N, 2)
        # descriptors: (1, N, D)
        
        if desc.shape[0] == 256 and desc.shape[1] != 256:
             desc = desc.transpose(0, 1)
             
        return {
            'keypoints': kpts.unsqueeze(0).to(device),
            'descriptors': desc.unsqueeze(0).to(device),
        }

def draw_matches(img1_path, img2_path, kpts1, kpts2, matches0, output_path):
    """繪製兩張圖的匹配連線"""
    # 1. 讀取影像
    try:
        im1 = np.array(Image.open(img1_path).convert("RGB"))
        im2 = np.array(Image.open(img2_path).convert("RGB"))
    except Exception as e:
        print(f"[Error] 無法讀取影像: {e}", file=sys.stderr)
        return

    # 2. 建立並排畫布
    H1, W1, C = im1.shape
    H2, W2, C = im2.shape
    H, W = max(H1, H2), W1 + W2
    canvas = np.zeros((H, W, C), dtype=np.uint8)
    canvas[:H1, :W1, :] = im1
    canvas[:H2, W1:W1+W2, :] = im2
    
    # 3. 繪圖
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(canvas)
    ax.axis('off')
    
    # 4. 提取有效匹配點
    # matches0 是 (N1,) 的陣列，數值為 -1 (無匹配) 或對應的 img2 索引
    valid = matches0 > -1
    mkpts1 = kpts1[valid]
    mkpts2 = kpts2[matches0[valid]]
    
    num_matches = len(mkpts1)
    print(f"[Info] 找到 {num_matches} 組匹配")

    # 5. 繪製連線與點 (隨機抽樣以免太密集)
    MAX_DRAW = 300
    if num_matches > MAX_DRAW:
        indices = np.random.choice(num_matches, MAX_DRAW, replace=False)
        mkpts1 = mkpts1[indices]
        mkpts2 = mkpts2[indices]
    
    # 繪製連線
    for (x1, y1), (x2, y2) in zip(mkpts1, mkpts2):
        # 顏色隨機，增加辨識度
        color = np.random.rand(3,)
        ax.plot([x1, x2 + W1], [y1, y2], '-', color=color, linewidth=0.8, alpha=0.7)
        ax.scatter([x1, x2 + W1], [y1, y2], s=5, color=color, zorder=2)

    ax.set_title(f"Matches: {num_matches} (Shown: {min(num_matches, MAX_DRAW)}) \n L: {Path(img1_path).name} | R: {Path(img2_path).name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Output] 視覺化結果已儲存至: {output_path}")

def main():
    example_text = """範例指令:
  python scripts/visualize_two_frames.py \\
    --data_root data/school \\
    --local_feats outputs-hloc/school/local-superpoint_aachen.h5 \\
    --img1 db/frames-000011.jpg \\
    --img2 db/frames-000006.jpg"""

    parser = argparse.ArgumentParser(
        description="視覺化任意兩張 DB 影像的 LightGlue 匹配效果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=example_text
    )
    parser.add_argument("--data_root", required=True, help="資料根目錄 (e.g., data/block_001)")
    parser.add_argument("--local_feats", required=True, help="Local features H5 路徑")
    parser.add_argument("--img1", required=True, help="影像 1 的相對路徑 (e.g., db/frame_000100.jpg)")
    parser.add_argument("--img2", required=True, help="影像 2 的相對路徑 (e.g., db/frame_000105.jpg)")
    parser.add_argument("--output", default="matches_debug.jpg", help="輸出圖片路徑")
    args = parser.parse_args()

    root = Path(args.data_root)
    p1 = root / args.img1
    p2 = root / args.img2
    
    if not p1.exists():
        print(f"[Error] 找不到影像 1: {p1}", file=sys.stderr)
        sys.exit(1)
    if not p2.exists():
        print(f"[Error] 找不到影像 2: {p2}", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] 使用裝置: {device}")

    # 1. 載入 LightGlue 模型
    print("[Info] 載入 LightGlue (SuperPoint) 模型...")
    matcher = LightGlue(features='superpoint').eval().to(device)

    # 2. 從 H5 載入特徵 (確保與建圖時一致)
    print(f"[Info] 從 {args.local_feats} 載入特徵...")
    try:
        feats1 = load_feature_from_h5(args.local_feats, args.img1, device)
        feats2 = load_feature_from_h5(args.local_feats, args.img2, device)
    except KeyError as e:
        print(f"[Error] {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[Error] 讀取 H5 特徵失敗: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. 執行匹配
    print(f"[Info] 正在匹配 {args.img1} <--> {args.img2} ...")
    with torch.no_grad():
        matches = matcher({'image0': feats1, 'image1': feats2})
        # [修正] 正確使用 rbd 處理整個字典，再取出 matches0
        matches_rbd = rbd(matches)
        matches0 = matches_rbd['matches0']

    # 4. 繪圖
    kpts1 = feats1['keypoints'].squeeze(0).cpu().numpy()
    kpts2 = feats2['keypoints'].squeeze(0).cpu().numpy()
    matches0_np = matches0.cpu().numpy()
    
    draw_matches(p1, p2, kpts1, kpts2, matches0_np, args.output)

if __name__ == "__main__":
    main()