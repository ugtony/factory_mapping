#!/usr/bin/env bash
set -euo pipefail

# batch_build_blocks.sh
# 用途：掃描母資料夾，對所有包含 raw/ 影片的子區塊執行建模。

if [ $# -lt 1 ]; then
  echo "Usage: $0 <PARENT_DATA_DIR> [args for build_block_model...]"
  echo "Example: $0 data --fps=2 --mode=360"
  exit 1
fi

PARENT_DIR=$(realpath "$1")
shift

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_SCRIPT="${SCRIPT_DIR}/build_block_model.sh"

if [ ! -f "${BUILD_SCRIPT}" ]; then
    echo "[Error] Cannot find build_block_model.sh at ${BUILD_SCRIPT}"
    exit 1
fi

if [ ! -d "${PARENT_DIR}" ]; then
    echo "[Error] Parent directory not found: ${PARENT_DIR}"
    exit 1
fi

echo "================================================================"
echo "[Batch] Scanning parent directory: ${PARENT_DIR}"
echo "================================================================"

count=0
success_count=0
fail_count=0

# [Update] 定義支援的副檔名列表
VIDEO_EXTS=(-iname "*.mp4" -o -iname "*.mov" -o -iname "*.m4v" -o -iname "*.avi" -o -iname "*.mkv" -o -iname "*.flv" -o -iname "*.wmv" -o -iname "*.insv" -o -iname "*.360" -o -iname "*.mts" -o -iname "*.m2ts" -o -iname "*.webm" -o -iname "*.ts")

for block_dir in "${PARENT_DIR}"/*; do
    if [ ! -d "${block_dir}" ]; then continue; fi
    
    block_name=$(basename "${block_dir}")
    raw_dir="${block_dir}/raw"
    
    has_video=""
    if [ -d "${raw_dir}" ]; then
        # [Update] 使用擴充後的列表進行偵測
        has_video=$(find "${raw_dir}" -maxdepth 1 -type f \( "${VIDEO_EXTS[@]}" \) -print -quit)
    fi
    
    if [ -n "$has_video" ]; then
        echo ""
        echo "----------------------------------------------------------------"
        echo "[Batch] Found raw videos in block: ${block_name}"
        echo "        > Target: ${block_dir}"
        echo "----------------------------------------------------------------"
        
        if bash "${BUILD_SCRIPT}" "${block_dir}" "$@"; then
            ((success_count++))
        else
            echo "[Batch] [Error] Build failed for ${block_name}"
            ((fail_count++))
        fi
        ((count++))
    fi
done

echo ""
echo "================================================================"
echo "[Batch] Summary"
echo "Total Blocks Found: ${count}"
echo "Success: ${success_count}"
echo "Failed:  ${fail_count}"
echo "================================================================"