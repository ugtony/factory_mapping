#!/usr/bin/env bash
set -euo pipefail

# batch_build_blocks.sh
# [Modified] 用途：掃描母資料夾，對包含有效 db_360/ (有圖片) 或 raw/ (有影片) 的子區塊執行建模。
# 新增功能：支援 --prefix=STR 參數，僅處理名稱符合特定前綴的區塊。

if [ $# -lt 1 ]; then
  echo "Usage: $0 <PARENT_DATA_DIR> [options] [args for build_block_model...]"
  echo "Example: $0 data --prefix=AreaA --fps=2"
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

# -------------------------------------------------------------
# 參數解析：分離出 --prefix，其餘保留給 build_block_model.sh
# -------------------------------------------------------------
TARGET_PREFIX=""
PASSTHROUGH_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix=*)
      TARGET_PREFIX="${1#*=}"
      shift
      ;;
    *)
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
  esac
done

echo "================================================================"
echo "[Batch] Scanning parent directory: ${PARENT_DIR}"
if [ -n "${TARGET_PREFIX}" ]; then
    echo "[Batch] Filter Prefix: '${TARGET_PREFIX}'"
fi
echo "================================================================"

count=0
success_count=0
fail_count=0

# 定義支援的副檔名列表
# 1. 影片格式 (用於 raw/)
VIDEO_EXTS=(-iname "*.mp4" -o -iname "*.mov" -o -iname "*.m4v" -o -iname "*.avi" -o -iname "*.mkv" -o -iname "*.flv" -o -iname "*.wmv" -o -iname "*.insv" -o -iname "*.360" -o -iname "*.mts" -o -iname "*.m2ts" -o -iname "*.webm" -o -iname "*.ts")
# 2. 圖片格式 (用於 db_360/)
IMAGE_EXTS=(-iname "*.jpg" -o -iname "*.png" -o -iname "*.jpeg")

for block_dir in "${PARENT_DIR}"/*; do
    if [ ! -d "${block_dir}" ]; then continue; fi
    
    block_name=$(basename "${block_dir}")

    # -------------------------------------------------------------
    # [Filter] 檢查 Prefix (若有指定)
    # -------------------------------------------------------------
    if [ -n "${TARGET_PREFIX}" ]; then
        if [[ "${block_name}" != "${TARGET_PREFIX}"* ]]; then
            continue
        fi
    fi

    raw_dir="${block_dir}/raw"
    db360_dir="${block_dir}/db_360"
    
    should_run=false
    found_reason=""

    # -------------------------------------------------------------
    # 檢查條件：db_360/ 內有圖片 或 raw/ 內有影片
    # -------------------------------------------------------------
    
    # 條件 1: 檢查 db_360 是否存在且含有圖片
    has_db_images=""
    if [ -d "${db360_dir}" ]; then
        has_db_images=$(find "${db360_dir}" -maxdepth 1 -type f \( "${IMAGE_EXTS[@]}" \) -print -quit)
    fi

    if [ -n "$has_db_images" ]; then
        should_run=true
        found_reason="Found images in db_360/"
    
    # 條件 2: 如果沒有有效的 db_360，檢查是否有 raw 且裡面有影片
    elif [ -d "${raw_dir}" ]; then
        has_video=$(find "${raw_dir}" -maxdepth 1 -type f \( "${VIDEO_EXTS[@]}" \) -print -quit)
        if [ -n "$has_video" ]; then
            should_run=true
            found_reason="Found videos in raw/"
        fi
    fi

    # -------------------------------------------------------------
    # 執行區塊
    # -------------------------------------------------------------
    if [ "$should_run" = true ]; then
        echo ""
        echo "----------------------------------------------------------------"
        echo "[Batch] Processing Block: ${block_name}"
        echo "        > Reason: ${found_reason}"
        echo "        > Path:   ${block_dir}"
        echo "----------------------------------------------------------------"
        
        # 使用 PASSTHROUGH_ARGS 傳遞剩餘參數
        if bash "${BUILD_SCRIPT}" "${block_dir}" "${PASSTHROUGH_ARGS[@]}"; then
            echo "[Batch] ✅ Success: ${block_name}"
            ((success_count+=1))
        else
            echo "[Batch] ❌ Failed: ${block_name}"
            ((fail_count+=1))
        fi
        ((count+=1))
    fi
done

echo ""
echo "================================================================"
echo "[Batch] Summary"
echo "Total Blocks Processed: ${count}"
echo "Success: ${success_count}"
echo "Failed:  ${fail_count}"
echo "================================================================"