
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_heic.py

遞迴地掃描指定資料夾，並將所有 .HEIC / .HEIF 檔案轉換為 .JPG 檔案。
這有助於在 HLOC pipeline 開始前，將影像格式標準化。

用法:
  python scripts/convert_heic.py <target_directory> [--quality 90] [--delete]

依賴:
  pip install pillow pillow-heif
"""

import argparse
from pathlib import Path
from PIL import Image

try:
    # 導入 pillow_heif 會自動向 PIL 註冊 HEIC/HEIF 讀取器
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    print("[Error] 'pillow-heif' library not found.")
    print("Please install it first: pip install pillow-heif")
    exit(1)


def convert_heic_to_jpg(target_dir, quality=90, delete_original=False):
    """
    掃描資料夾並轉換影像。
    """
    target_dir = Path(target_dir)
    if not target_dir.is_dir():
        print(f"[Error] Path is not a valid directory: {target_dir}")
        return

    print(f"[Info] Scanning {target_dir} recursively for .HEIC/.HEIF files...")

    # 1. 找到所有 HEIC/HEIF 檔案
    # (參考 HEIC 副檔名)
    extensions = ['.HEIC', '.heic', '.HEIF', '.heif'] 
    files_to_convert = []
    for ext in extensions:
        # 使用 rglob 進行遞迴掃描
        files_to_convert.extend(target_dir.rglob(f"*{ext}"))

    if not files_to_convert:
        print("[Info] No HEIC/HEIF files found.")
        return

    print(f"[Info] Found {len(files_to_convert)} files to process.")
    
    converted_count = 0
    skipped_count = 0
    failed_count = 0

    # 2. 迭代並轉換
    for heic_path in files_to_convert:
        # 將 .HEIC 替換為 .jpg
        jpg_path = heic_path.with_suffix('.jpg')

        # 2a. 如果 JPG 已存在，則跳過
        if jpg_path.exists():
            print(f"  > Skipping: {jpg_path.name} already exists.")
            skipped_count += 1
            continue

        try:
            # 2b. 讀取 HEIC
            print(f"  > Converting: {heic_path.relative_to(target_dir)}")
            image = Image.open(heic_path)

            # 2c. 移除 Alpha 通道（JPG 不支援）
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")
            
            # 2d. 儲存為 JPG
            image.save(jpg_path, quality=quality)
            converted_count += 1

            # 2e. (可選) 刪除原始檔案
            if delete_original:
                heic_path.unlink()
                print(f"    > Deleted original: {heic_path.name}")

        except Exception as e:
            print(f"  [FAIL] Failed to convert {heic_path.name}: {e}")
            failed_count += 1
    
    print("\n--- Conversion Summary ---")
    print(f"Successfully converted: {converted_count}")
    print(f"Skipped (JPG exists): {skipped_count}")
    print(f"Failed: {failed_count}")


def main():
    parser = argparse.ArgumentParser(description="Convert HEIC/HEIF files to JPG recursively.")
    
    parser.add_argument(
        "directory", 
        type=str, 
        help="The target directory to scan (e.g., 'data/street/query')."
    )
    
    parser.add_argument(
        "-q", "--quality", 
        type=int, 
        default=90,
        help="JPEG quality (1-100). Default: 90"
    )
    
    parser.add_argument(
        "--delete", 
        action="store_true",
        help="Delete the original HEIC/HEIF file after successful conversion."
    )
    
    args = parser.parse_args()
    
    # 確保 pillow-heif 已安裝
    try:
        import pillow_heif
    except ImportError:
        print("[Error] 'pillow-heif' is required but not installed.")
        print("Please run: pip install pillow-heif")
        return

    convert_heic_to_jpg(args.directory, args.quality, args.delete)

if __name__ == "__main__":
    main()