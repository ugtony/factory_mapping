#!/usr/bin/env bash
# scripts/run_dense_mvs.sh
#
# 用途：接續 build_block_model.sh 的輸出，執行 COLMAP MVS 以產生稠密點雲 (fused.ply)。
# 注意：此過程非常耗時且需要大量 GPU VRAM 與硬碟空間。
#
# 用法：bash scripts/run_dense_mvs.sh <BLOCK_DATA_DIR> [--max_size=2000]
# 範例：bash scripts/run_dense_mvs.sh data/block_001

set -euo pipefail

# -------- 0. 參數解析 --------
if [ $# -lt 1 ]; then
  echo "Usage: $0 <BLOCK_DATA_DIR> [--max_size=N]"
  exit 1
fi

DATA_DIR="$(realpath "$1")"
shift

MAX_IMAGE_SIZE=2000 # 預設將影像縮小至最長邊 2000px 以節省顯卡記憶體

while [ $# -gt 0 ]; do
  case "$1" in
    --max_size=*) MAX_IMAGE_SIZE="${1#*=}" ;;
    *) echo "[Warn] Unknown argument: $1";;
  esac
  shift
done

# -------- 1. 路徑設定 (對齊 build_block_model.sh 的結構) --------
BLOCK_NAME="$(basename "${DATA_DIR}")"
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "${SCRIPT_DIR}/..")"
OUT_ROOT="${PROJECT_ROOT}/outputs-hloc"
OUT_DIR="${OUT_ROOT}/${BLOCK_NAME}"

# 優先使用對齊過的模型 (若有)，否則使用原始 SfM
if [ -d "${OUT_DIR}/sfm_aligned" ] && [ -f "${OUT_DIR}/sfm_aligned/images.bin" ]; then
    SFM_INPUT="${OUT_DIR}/sfm_aligned"
    echo "[Info] Found aligned SfM model, using it for MVS."
elif [ -d "${OUT_DIR}/sfm" ] && [ -f "${OUT_DIR}/sfm/images.bin" ]; then
    SFM_INPUT="${OUT_DIR}/sfm"
    echo "[Info] Using raw SfM model for MVS."
else
    echo "[Error] No valid SfM model found in ${OUT_DIR}/sfm or ${OUT_DIR}/sfm_aligned"
    echo "Please run build_block_model.sh first."
    exit 1
fi

DENSE_DIR="${OUT_DIR}/dense"
mkdir -p "${DENSE_DIR}"

echo "========================================"
echo "[Info] Block: ${BLOCK_NAME}"
echo "[Info] Input SfM: ${SFM_INPUT}"
echo "[Info] Output Dense: ${DENSE_DIR}"
echo "[Info] Max Image Size: ${MAX_IMAGE_SIZE}"
echo "========================================"

# -------- 2. 執行 COLMAP MVS Pipeline --------

# [Step 1] Image Undistortion
# 將影像去畸變，準備給 MVS 使用。
echo "[1/3] Running Image Undistortion..."
colmap image_undistorter \
    --image_path "${DATA_DIR}" \
    --input_path "${SFM_INPUT}" \
    --output_path "${DENSE_DIR}" \
    --output_type COLMAP \
    --max_image_size "${MAX_IMAGE_SIZE}"

# [Step 2] Patch Match Stereo
# 計算深度圖與法向量圖 (最耗時步驟)。
# 如果您的 GPU 很強，可以試著增加 --PatchMatchStereo.window_radius
# 若是 360 影像轉換而來，建議開啟幾何一致性檢查 (geom_consistency true)
echo "[2/3] Running Patch Match Stereo (this may take a while)..."
colmap patch_match_stereo \
    --workspace_path "${DENSE_DIR}" \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true

# [Step 3] Stereo Fusion
# 將深度圖融合為單一点雲檔案。
echo "[3/3] Running Stereo Fusion..."
colmap stereo_fusion \
    --workspace_path "${DENSE_DIR}" \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path "${DENSE_DIR}/fused.ply"

echo "========================================"
echo "✅ Dense reconstruction complete!"
echo "Result saved to: ${DENSE_DIR}/fused.ply"
echo "You can view it using MeshLab or CloudCompare."