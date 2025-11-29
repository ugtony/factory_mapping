#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
resize_images.py

遞迴地掃描輸入資料夾，將所有影像 (.jpg, .png) 縮放到指定的
最長邊 (max_side)，並保持長寬比。

這有助於在 HLOC pipeline 開始前，將 query 影像的尺寸
與 db 影像的尺寸標準化，避免 PnP 姿態估計時發生尺寸不匹配。

用法:
  python scripts/resize_images.py <input_dir> <output_dir> [--max_side 1024]

範例:
  # 將 data/street/query/ 中的所有影像縮放到 1024，儲存到 data/street/query_1024/
  python scripts/resize_images.py data/street/query data/street/query_1024 --max_side 1024
"""

import argparse
from pathlib import Path
from PIL import Image
import os
import sys

# -----------------------------
# 縮放工具函式
# (複製自 runtime/localizer/pose_estimator.py 以保持一致性)
# -----------------------------
def _get_resized_dims(orig_w, orig_h, max_side):
    """
    依照 max_side 縮放到最長邊 = max_side，維持長寬比。
    傳回 (target_w, target_h)（皆為偶數）。
    """
    if max_side is None or max_side <= 0:
        return orig_w, orig_h

    scale = float(max_side) / float(max(orig_w, orig_h))
    if scale >= 1.0:
        # 不放大，保持原尺寸
        target_w, target_h = int(round(orig_w)), int(round(orig_h))
    else:
        target_w = int(round(orig_w * scale))
        target_h = int(round(orig_h * scale))

    # 轉成偶數（保守作法）
    if target_w % 2 == 1:
        target_w -= 1
    if target_h % 2 == 1:
        target_h -= 1
    # 避免變成 0
    target_w = max(target_w, 2)
    target_h = max(target_h, 2)

    return target_w, target_h

# -----------------------------
# 主處理函式
# -----------------------------
def process_images(input_dir, output_dir, max_side, quality=92):
    """
    掃描資料夾並轉換影像。
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.is_dir():
        print(f"[Error] 輸入路徑不是一個有效的資料夾: {input_dir}")
        return

    # 1. 找到所有影像檔案
    extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    files_to_process = []
    for ext in extensions:
        files_to_process.extend(input_dir.rglob(f"*{ext}"))

    if not files_to_process:
        print(f"[Info] 在 {input_dir} 中找不到任何影像檔案。")
        return

    print(f"[Info] 找到 {len(files_to_process)} 張影像。開始處理 (max_side={max_side})...")

    processed_count = 0
    skipped_count = 0
    failed_count = 0

    # 2. 迭代並轉換
    for input_path in files_to_process:
        try:
            # 2a. 計算輸出路徑 (保持子資料夾結構)
            # e.g., input_dir/sub/img.jpg -> output_dir/sub/img.jpg
            relative_path = input_path.relative_to(input_dir)
            output_path = output_dir / relative_path

            # 2b. 建立輸出資料夾
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 2c. 如果輸出檔案已存在，則跳過
            if output_path.exists():
                skipped_count += 1
                continue

            # 2d. 讀取影像
            with Image.open(input_path) as img:
                orig_w, orig_h = img.size

                # 2e. 計算目標尺寸
                target_w, target_h = _get_resized_dims(orig_w, orig_h, max_side)

                # 2f. 執行縮放
                if (target_w, target_h) == (orig_w, orig_h):
                    # 不需要縮放，直接儲存 (或複製)
                    img_to_save = img
                    print(f"  > 複製: {relative_path} (已是 {orig_w}x{orig_h})")
                else:
                    # 使用 LANCZOS 進行高品質縮放
                    img_to_save = img.resize((target_w, target_h), Image.LANCZOS)
                    print(f"  > 縮放: {relative_path} ({orig_w}x{orig_h} -> {target_w}x{target_h})")

                # 2g. 處理透明度 (儲存 JPG 時)
                if output_path.suffix.lower() in ['.jpg', '.jpeg']:
                    if img_to_save.mode in ("RGBA", "P"):
                        img_to_save = img_to_save.convert("RGB")
                    img_to_save.save(output_path, quality=quality, optimize=True)
                else:
                    # 儲存 PNG (或其他格式)
                    img_to_save.save(output_path)

                processed_count += 1

        except Exception as e:
            print(f"  [FAIL] 處理 {input_path.name} 失敗: {e}")
            failed_count += 1

    print("\n--- 縮放總結 ---")
    print(f"成功處理/縮放: {processed_count}")
    print(f"已跳過 (檔案已存在): {skipped_count}")
    print(f"失敗: {failed_count}")


def main():
    parser = argparse.ArgumentParser(
        description="遞迴地縮放資料夾中的影像，保持最長邊不大於 max_side。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "input_dir", 
        type=str, 
        help="包含原始影像的輸入資料夾。"
    )
    
    parser.add_argument(
        "output_dir", 
        type=str,
        help="儲存縮放後影像的輸出資料夾。"
    )
    
    parser.add_argument(
        "--max_side", 
        type=int, 
        default=1024,
        help="影像最長邊的最大像素值 (預設: 1024)。"
    )
    
    parser.add_argument(
        "-q", "--quality", 
        type=int, 
        default=92,
        help="JPEG 儲存品質 (1-95)。 (預設: 92)"
    )
    
    args = parser.parse_args()
    
    process_images(args.input_dir, args.output_dir, args.max_side, args.quality)

if __name__ == "__main__":
    main()