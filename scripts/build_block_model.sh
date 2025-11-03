#!/usr/bin/env bash
# build_block_model.sh
# 針對單一 block 區域執行離線建模（HLOC pipeline）。
#
# - 自動檢查 local H5 特徵的完整性，若不符則重建。
# - 自動潔淨化 pairs list，避免 H5 中不存在的影像鍵名。
# - [STEP 9] 內建 HLOC SfM 模型選擇 bug 的修復程序（選最大模型）。
# - [STEP 10] 對齊 SfM 至 Z-Up（採用 align_sfm_model_z_up.py），並輸出視覺化。
# - 支援 staging（鏡射）或直接讀取影像。
# - 可選啟動內建 HTTP 伺服器預覽 HTML。

set -euo pipefail
set -x # 開啟 debug 輸出，若嫌吵可註解此行

if [ $# -lt 1 ]; then
  echo "Usage: $0 <BLOCK_DATA_DIR>"
  echo "Example: $0 data/block_001"
  exit 1
fi

# -------- 1. 路徑與環境設定 --------
DATA_DIR="$(realpath "$1")"
BLOCK_NAME="$(basename "${DATA_DIR}")"

# 以腳本所在目錄作為專案根目錄
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "${SCRIPT_DIR}/..")"
OUT_ROOT="${PROJECT_ROOT}/outputs-hloc"
OUT_DIR="${OUT_ROOT}/${BLOCK_NAME}"
LOG_DIR="${OUT_DIR}/logs"
VIZ_DIR="${OUT_DIR}/visualization"
DBG_DIR="${OUT_DIR}/debug"

mkdir -p "${OUT_DIR}" "${LOG_DIR}" "${VIZ_DIR}" "${DBG_DIR}"

# Python
if [ -x "/opt/conda/bin/python" ]; then
  PY="/opt/conda/bin/python"
else
  PY="${PY:-python3}"
fi
echo "[Info] Using Python executable at: $PY"
echo "[Info] Processing Block: ${BLOCK_NAME}"
echo "[Info] Output directory: ${OUT_DIR}"

# -------- 2. 組態設定 --------
GLOBAL_CONF="netvlad"
LOCAL_CONF="superpoint_aachen"
MATCHER_CONF="superpoint+lightglue"

USE_STAGE="${USE_STAGE:-1}"         # 1=使用 staging；0=直接用 ${DATA_DIR}/db
REBUILD_SFM="${REBUILD_SFM:-0}"     # 1=每次重建 SfM 目錄；0=沿用
ALIGN_SFM="${ALIGN_SFM:-1}"         # 1=執行 Z-Up 對齊；0=跳過
NUM_RETRIEVAL="${NUM_RETRIEVAL:-0}" # 檢索配對數
SEQ_WINDOW="${SEQ_WINDOW:-5}"       # 時序配對窗口

# NUM_RETRIEVAL 和 SEQ_WINDOW 不能同時為 0
if [ "${NUM_RETRIEVAL}" -le 0 ] && [ "${SEQ_WINDOW}" -le 0 ]; then
  echo "[Error] NUM_RETRIEVAL and SEQ_WINDOW cannot both be 0."
  echo "        At least one must be > 0 for pair generation."
  exit 1
fi

# -------- 3. 核心檔案路徑 --------
DB_LIST="${OUT_DIR}/db.txt"
LOCAL_FEATS="${OUT_DIR}/local-${LOCAL_CONF}.h5"
GLOBAL_FEATS="${OUT_DIR}/global-${GLOBAL_CONF}.h5"
PAIRS_DB="${OUT_DIR}/pairs-db-retrieval_and_seq.txt"
PAIRS_DB_CLEAN="${OUT_DIR}/_pairs-db-retrieval.clean.txt"
DB_MATCHES="${OUT_DIR}/db-matches-${MATCHER_CONF}.h5"
SFM_DIR="${OUT_DIR}/sfm"
SFM_MODELS_DIR="${SFM_DIR}/models"
STAGE="${OUT_DIR}/_images_stage"
# 對齊後輸出目錄（以專案根與 OUT_DIR 為基準）
SFM_ALIGNED="${OUT_DIR}/sfm_aligned"

ALIGN_SCRIPT="${PROJECT_ROOT}/scripts/align_sfm_model_z_up.py"
VIZ_SCRIPT="${PROJECT_ROOT}/scripts/visualize_sfm_open3d.py"

# -------- 4. HLOC Pipeline (共 8 步) --------

echo "[1/8] Generating DB image list (db.txt)..."
(cd "${DATA_DIR}" && find db -maxdepth 3 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | sort) > "${DB_LIST}"
if [ ! -s "${DB_LIST}"; then
  echo "[Error] No images found in ${DATA_DIR}/db. Aborting."
  exit 1
fi
echo "    > Found $(wc -l < "${DB_LIST}") DB images."

echo "[2/8] Checking integrity of local features H5 (${LOCAL_FEATS})..."
${PY} - <<PY || { echo "[Check] H5 is stale/corrupted. Deleting."; rm -f "${LOCAL_FEATS}"; }
import h5py, sys
from pathlib import Path
db_paths=[l.strip() for l in Path("${DB_LIST}").read_text().splitlines() if l.strip()]
ok=True
try:
    with h5py.File("${LOCAL_FEATS}","r") as f:
        for p in db_paths:
            if p not in f or "keypoints" not in f[p]:
                print(f"[Check] Missing key '{p}'.", file=sys.stderr); ok=False; break
except Exception as e:
    print(f"[Check] Cannot open H5: {e}", file=sys.stderr); ok=False
sys.exit(0 if ok else 1)
PY

echo "[3/8] Extracting LOCAL features (${LOCAL_CONF})..."
${PY} -m hloc.extract_features --conf "${LOCAL_CONF}" \
  --image_dir "${DATA_DIR}" --image_list "${DB_LIST}" \
  --export_dir "${OUT_DIR}" --feature_path "${LOCAL_FEATS}"

echo "[4/8] Extracting GLOBAL features (${GLOBAL_CONF})..."
${PY} -m hloc.extract_features --conf "${GLOBAL_CONF}" \
  --image_dir "${DATA_DIR}" --image_list "${DB_LIST}" \
  --export_dir "${OUT_DIR}" --feature_path "${GLOBAL_FEATS}"

echo "[5/8] Building DB pairs (retrieval + sequential)..."
if [ -f "${PAIRS_DB}" ]; then
  echo "    > Exists: ${PAIRS_DB}. Skipping pair generation."
else
  ${PY} "${PROJECT_ROOT}/scripts/pairs_from_retrieval_and_sequential.py" \
    --db_list "${DB_LIST}" \
    --global_feats "${GLOBAL_FEATS}" \
    --num_retrieval "${NUM_RETRIEVAL}" \
    --seq_window "${SEQ_WINDOW}" \
    --output "${PAIRS_DB}"
fi

echo "[6/8] Cleaning pairs list (ensure all keys exist in H5)..."
${PY} - <<PY
from pathlib import Path
import h5py, sys
pairs_in = Path("${PAIRS_DB}"); pairs_out = Path("${PAIRS_DB_CLEAN}")
db_list = set([l.strip() for l in Path("${DB_LIST}").read_text().splitlines() if l.strip()])
try:
    with h5py.File("${LOCAL_FEATS}","r") as f, open(pairs_in,"r") as fi, open(pairs_out,"w") as fo:
        keep=0; drop=0
        for line in fi:
            s=line.strip().split()
            if len(s)<2: continue
            a,b = s[0], s[1]
            ok = (a in db_list) and (b in db_list)
            ok = ok and (a in f) and (b in f) and ("keypoints" in f[a]) and ("keypoints" in f[b])
            if ok:
                fo.write(line); keep+=1
            else:
                drop+=1
        print(f"[Check] Pairs cleaned: {keep} kept, {drop} dropped. -> {pairs_out}")
except Exception as e:
    print(f"[Error] Clean pairs failed: {e}", file=sys.stderr); sys.exit(1)
PY

PAIRS_USE="${PAIRS_DB_CLEAN}"

echo "[7/8] Matching DB pairs (${MATCHER_CONF})..."
${PY} - <<PY
from pathlib import Path
from hloc import match_features
match_features.main(
    conf=match_features.confs["${MATCHER_CONF}"],
    pairs=Path("${PAIRS_USE}"),
    features=Path("${LOCAL_FEATS}"),
    matches=Path("${DB_MATCHES}")
)
PY

echo "[8/8] Preparing images for SfM (Staging/Direct)..."
set +x
if [ "${USE_STAGE}" = "1" ]; then
  echo "    > Using staging at: ${STAGE}"
  rm -rf "${STAGE}"; mkdir -p "${STAGE}"
  while IFS= read -r rel; do
    src="${DATA_DIR}/${rel}"
    dst="${STAGE}/${rel}"
    mkdir -p "$(dirname "${dst}")"
    ln -sf "${src}" "${dst}"
  done < "${DB_LIST}"
  IMG_DIR="${STAGE}"
else
  IMG_DIR="${DATA_DIR}"
  echo "    > Using direct path: ${IMG_DIR}"
fi
set -x

if [ "${REBUILD_SFM}" = "1" ]; then
  echo "    > REBUILD_SFM=1. Cleaning previous SfM directory."
  rm -rf "${SFM_DIR}"
fi

if [ -f "${SFM_DIR}/images.bin" ]; then
  echo "    > SfM exists at ${SFM_DIR}. Skipping reconstruction."
else
  echo "[Run] SfM reconstruction (COLMAP via HLOC)..."
  ${PY} -m hloc.reconstruction \
    --image_dir "${IMG_DIR}" \
    --pairs "${PAIRS_USE}" \
    --features "${LOCAL_FEATS}" \
    --matches "${DB_MATCHES}" \
    --sfm_dir "${SFM_DIR}" | tee "${LOG_DIR}/reconstruction.log"
  echo "[Done] SfM model: ${SFM_DIR}"
fi

# -------- STEP 9) HLOC Model Selection BUGFIX（選最大模型） --------
echo "[9] Verifying HLOC selected the largest SfM model..."
NEED_SWAP=false
BEST_MODEL_PATH=""
MAX_IMG_COUNT=0
ROOT_IMG_COUNT=0

if [ -f "${SFM_DIR}/images.bin" ]; then
  ROOT_IMG_COUNT=$(${PY} -c "import pycolmap, sys; rec=pycolmap.Reconstruction(sys.argv[1]); print(len(rec.images))" "${SFM_DIR}")
else
  echo "[Fix]   > Warning: No images.bin in ${SFM_DIR}. Will search in models/."
  MAX_IMG_COUNT=1
fi

if [ -d "${SFM_MODELS_DIR}" ]; then
  for MODEL_DIR in "${SFM_MODELS_DIR}"/*; do
    if [ -d "${MODEL_DIR}" ] && [ -f "${MODEL_DIR}/images.bin" ]; then
      CURRENT_IMG_COUNT=$(${PY} -c "import pycolmap, sys; rec=pycolmap.Reconstruction(sys.argv[1]); print(len(rec.images))" "${MODEL_DIR}")
      if [ "${CURRENT_IMG_COUNT}" -gt "${MAX_IMG_COUNT}" ]; then
        MAX_IMG_COUNT="${CURRENT_IMG_COUNT}"
        BEST_MODEL_PATH="${MODEL_DIR}"
      fi
    fi
  done
fi

if [ "${ROOT_IMG_COUNT}" -lt "${MAX_IMG_COUNT}" ] && [ -n "${BEST_MODEL_PATH}" ]; then
  echo "[Fix]   > Root (${ROOT_IMG_COUNT}) < Best (${MAX_IMG_COUNT}). Swapping..."
  NEED_SWAP=true
else
  echo "[Fix]   > Selection OK (Root: ${ROOT_IMG_COUNT})."
fi

if [ "${NEED_SWAP}" = true ]; then
  TMP_SWAP_DIR="${SFM_DIR}/_tmp_swap_$(date +%s)"
  mkdir -p "${TMP_SWAP_DIR}"
  find "${SFM_DIR}" -maxdepth 1 -type f -name "*.bin" -exec mv {} "${TMP_SWAP_DIR}/" \;
  find "${BEST_MODEL_PATH}" -maxdepth 1 -type f -name "*.bin" -exec mv {} "${SFM_DIR}/" \;
  find "${TMP_SWAP_DIR}" -maxdepth 1 -type f -name "*.bin" -exec mv {} "${BEST_MODEL_PATH}/" \;
  rm -rf "${TMP_SWAP_DIR}"
  echo "[Fix]   > Swap complete. ${SFM_DIR} now holds the largest model (${MAX_IMG_COUNT} images)."
fi

# -------- STEP 10) Align to Z-Up + Visualization --------
# 對齊：輸出到 ${SFM_ALIGNED}；視覺化預設使用對齊後模型（若跳過對齊則回退到原 sfm）
echo "[10] Align to Z-Up & Visualize"
USE_VIZ_DIR="${VIZ_DIR}"
VIZ_INPUT_DIR="${SFM_DIR}"  # 預設先指向原始；若對齊成功則改用對齊後

if [ "${ALIGN_SFM}" = "1" ] && [ -f "${ALIGN_SCRIPT}" ] && [ -f "${SFM_DIR}/images.bin" ]; then
  echo "    > Aligning SfM to Z-Up with: ${ALIGN_SCRIPT}"
  "${PY}" "${ALIGN_SCRIPT}" "${SFM_DIR}" "--out_dir=${SFM_ALIGNED}" "--dump=0" "--export-ply=0"
  if [ -f "${SFM_ALIGNED}/images.bin" ]; then
    echo "    > Alignment OK. Using aligned model for visualization: ${SFM_ALIGNED}"
    VIZ_INPUT_DIR="${SFM_ALIGNED}"
  else
    echo "    > Alignment failed or missing output. Fallback to original: ${SFM_DIR}"
  fi
else
  if [ "${ALIGN_SFM}" != "1" ]; then
    echo "    > ALIGN_SFM=0, skip alignment. Visualize original SfM."
  else
    echo "    > Alignment script or model missing. Visualize original SfM."
  fi
fi

if [ -f "${VIZ_SCRIPT}" ]; then
  echo "    > Exporting interactive HTML visualization..."
  VIZ_ARGS=( --sfm_dir "${VIZ_INPUT_DIR}" --output_dir "${USE_VIZ_DIR}" )
  if [ -f "${OUT_DIR}/poses.txt" ]; then
    echo "      - Found poses.txt, include query cameras."
    VIZ_ARGS+=( --query_poses "${OUT_DIR}/poses.txt" )
  fi
  if [ "${START_SERVER:-0}" = "1" ]; then
    VIZ_ARGS+=( --port "${PORT:-8080}" )
    echo "      - START_SERVER=1 (port ${PORT:-8080})"
  else
    VIZ_ARGS+=( --no_server )
  fi
  "${PY}" "${VIZ_SCRIPT}" "${VIZ_ARGS[@]}"
  echo "    > HTML: ${USE_VIZ_DIR}/sfm_view.html"
else
  echo "    > Skip visualization: ${VIZ_SCRIPT} not found"
fi

echo "✅ All steps completed for Block: ${BLOCK_NAME}"
