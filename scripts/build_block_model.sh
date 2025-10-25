#!/usr/bin/env bash
# build_block_model.sh
# 針對單一 block 區域執行離線建模（HLOC pipeline）。
#
# - 自動檢查 local H5 特徵的完整性，若不符則重建。
# - 自動潔淨化 pairs list，避免 H5 中不存在的影像鍵名。
# - [HOTFIX] 內建 HLOC SfM 模型選擇 bug 的修復程序 (9.1)。
# - 支援 staging（鏡射）或直接讀取影像。
# - 支援重建後自動視覺化 (10)。

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

# 優先使用 /opt/conda/bin/python（若存在），確保和 HLOC Docker 環境一致
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
REBUILD_SFM="${REBUILD_SFM:-0}"     # 1=每次重建 SfM 目錄；0=沿用（警告：可能不一致）

# -------- 3. 核心檔案路徑 --------
DB_LIST="${OUT_DIR}/db.txt"
LOCAL_FEATS="${OUT_DIR}/local-${LOCAL_CONF}.h5"
GLOBAL_FEATS="${OUT_DIR}/global-${GLOBAL_CONF}.h5"
PAIRS_DB="${OUT_DIR}/pairs-db-retrieval_and_seq.txt" # 原始 pair 檔
PAIRS_DB_CLEAN="${OUT_DIR}/_pairs-db-retrieval.clean.txt" # 潔淨化後 pair 檔
DB_MATCHES="${OUT_DIR}/db-matches-${MATCHER_CONF}.h5"
SFM_DIR="${OUT_DIR}/sfm"
STAGE="${OUT_DIR}/_images_stage"   # Staging 鏡射資料夾

# -------- 4. HLOC Pipeline (共 9 步) --------

echo "[1/9] Generating DB image list (db.txt)..."
(cd "${DATA_DIR}" && find db -maxdepth 3 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | sort) > "${DB_LIST}"
if [ ! -s "${DB_LIST}" ]; then
  echo "[Error] No images found in ${DATA_DIR}/db. Aborting."
  exit 1
fi
echo "    > Found $(wc -l < "${DB_LIST}") DB images."

echo "[2/9] Checking integrity of local features H5 (${LOCAL_FEATS})..."
# 檢查 H5 是否包含 db.txt 中的所有 keypoints。若 H5 檔案過舊或不完整，則刪除它。
${PY} - <<PY || { echo "[Check] H5 file is stale or corrupted. Deleting."; rm -f "${LOCAL_FEATS}"; }
import h5py, sys
from pathlib import Path
db_paths=[l.strip() for l in Path("${DB_LIST}").read_text().splitlines() if l.strip()]
ok=True
try:
    with h5py.File("${LOCAL_FEATS}","r") as f:
        for p in db_paths:
            if p not in f or "keypoints" not in f[p]:
                print(f"[Check] Stale H5: Missing key '{p}'.", file=sys.stderr); ok=False; break
except Exception as e:
    print(f"[Check] Cannot open H5 file: {e}", file=sys.stderr); ok=False
sys.exit(0 if ok else 1)
PY

echo "[3/9] Extracting LOCAL features (${LOCAL_CONF})..."
${PY} -m hloc.extract_features --conf "${LOCAL_CONF}" \
  --image_dir "${DATA_DIR}" --image_list "${DB_LIST}" \
  --export_dir "${OUT_DIR}" --feature_path "${LOCAL_FEATS}"

echo "[4/9] Extracting GLOBAL features (${GLOBAL_CONF})..."
${PY} -m hloc.extract_features --conf "${GLOBAL_CONF}" \
  --image_dir "${DATA_DIR}" --image_list "${DB_LIST}" \
  --export_dir "${OUT_DIR}" --feature_path "${GLOBAL_FEATS}"

echo "[5/9] Building DB pairs (retrieval + sequential)..."
if [ -f "${PAIRS_DB}" ]; then
  echo "    > File exists: ${PAIRS_DB}. Skipping pair generation."
else
  # 使用您專案中的 pairs_from_retrieval_and_sequential.py
  #
  ${PY} "${PROJECT_ROOT}/scripts/pairs_from_retrieval_and_sequential.py" \
    --db_list "${DB_LIST}" \
    --global_feats "${GLOBAL_FEATS}" \
    --num_retrieval 5 \
    --seq_window 5 \
    --output "${PAIRS_DB}"
fi

echo "[6/9] Cleaning pairs list (ensuring all images exist in H5)..."
# 潔淨化 pairs：僅保留存在於 H5 且有 keypoints 的影像
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
            # 檢查 (a,b) 是否都在 db_list 且 都在 H5 中
            ok = (a in db_list) and (b in db_list)
            ok = ok and (a in f) and (b in f) and ("keypoints" in f[a]) and ("keypoints" in f[b])
            if ok:
                fo.write(line); keep+=1
            else:
                drop+=1
        print(f"[Check] Pairs cleaned: {keep} kept, {drop} dropped. Output: {pairs_out}")
except Exception as e:
    print(f"[Error] Failed to clean pairs: {e}", file=sys.stderr)
    sys.exit(1)
PY

# 使用潔淨版 pairs 進行後續匹配與重建
PAIRS_USE="${PAIRS_DB_CLEAN}"

echo "[7/9] Matching DB pairs (${MATCHER_CONF})..."
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

echo "[8/9] Preparing image directory for SfM (Staging)..."
set +x
if [ "${USE_STAGE}" = "1" ]; then
  echo "    > Using staging (symlink mirror) at: ${STAGE}"
  rm -rf "${STAGE}"; mkdir -p "${STAGE}"
  while IFS= read -r rel; do
    src="${DATA_DIR}/${rel}"
    dst="${STAGE}/${rel}"
    mkdir -p "$(dirname "${dst}")"
    ln -sf "${src}" "${dst}"
  done < "${DB_LIST}"
  IMG_DIR="${STAGE}"
else
  IMG_DIR="${DATA_DIR}" # 注意：hloc 會自動找 db/ 子目錄
  echo "    > Using direct path: ${IMG_DIR}"
fi
set -x

# 視需要重建 SfM 目錄（避免殘留 DB 混入上一輪影像）
if [ "${REBUILD_SFM}" = "1" ]; then
  echo "    > REBUILD_SFM=1. Cleaning previous SfM directory."
  rm -rf "${SFM_DIR}"
fi

if [ -f "${SFM_DIR}/images.bin" ]; then
  echo "    > File exists: ${SFM_DIR}/. Skipping SfM."
else
  echo "[9/9] Running SfM (COLMAP reconstruction)..."
  ${PY} -m hloc.reconstruction \
    --image_dir "${IMG_DIR}" \
    --pairs "${PAIRS_USE}" \
    --features "${LOCAL_FEATS}" \
    --matches "${DB_MATCHES}" \
    --sfm_dir "${SFM_DIR}" | tee "${LOG_DIR}/reconstruction.log"

  echo "[Done] SfM reconstruction finished."
  echo "    > Model output at: ${SFM_DIR}"
fi

# -------- 9.1) [BUGFIX] HLOC Model Selection Hotfix --------
# HLOC 的 reconstruction.py (as of writing) 可能會產生多個模型片段
# (sfm/models/0, sfm/models/1, ...)，但有時會錯誤地將「最小」
#（而非最大）的模型 .bin 檔案移動到 sfm/ 根目錄。
#
# 此區塊會：
# 1. 檢查 HLOC 放在 sfm/ 根目錄的模型影像數 (ROOT_IMG_COUNT)
# 2. 遍歷所有 sfm/models/* 子模型，找出影像數最多的 (MAX_IMG_COUNT)
# 3. 如果 ROOT_IMG_COUNT < MAX_IMG_COUNT，則將兩者交換。
echo "[Fix] Verifying HLOC selected the largest SfM model..."

NEED_SWAP=false
BEST_MODEL_PATH=""
MAX_IMG_COUNT=0
ROOT_IMG_COUNT=0

# 1. 取得 HLOC 已選擇的模型（sfm/ 根目錄）的影像數
if [ -f "${SFM_DIR}/images.bin" ]; then
  ROOT_IMG_COUNT=$(${PY} -c "import pycolmap, sys; \
    rec = pycolmap.Reconstruction(sys.argv[1]); \
    print(len(rec.images))" "${SFM_DIR}")
else
  echo "[Fix]   > Warning: No images.bin found in ${SFM_DIR}. Skipping check."
  # 觸發下一步驟，嘗試從 models/ 尋找
  MAX_IMG_COUNT=1 # 設為非 0，以便觸發下面的比較
fi

# 2. 迭代所有 sfm/models/* 子目錄，找出影像數最多的模型
for MODEL_DIR in "${SFM_DIR}/models"/*; do
  if [ -d "${MODEL_DIR}" ] && [ -f "${MODEL_DIR}/images.bin" ]; then
    CURRENT_IMG_COUNT=$(${PY} -c "import pycolmap, sys; \
      rec = pycolmap.Reconstruction(sys.argv[1]); \
      print(len(rec.images))" "${MODEL_DIR}")
    
    if [ "${CURRENT_IMG_COUNT}" -gt "${MAX_IMG_COUNT}" ]; then
      MAX_IMG_COUNT="${CURRENT_IMG_COUNT}"
      BEST_MODEL_PATH="${MODEL_DIR}"
    fi
  fi
done

# 3. 比較：如果 HLOC 選的 (ROOT) 小於我們找到的 (MAX)，就觸發交換
if [ "${ROOT_IMG_COUNT}" -lt "${MAX_IMG_COUNT}" ]; then
  echo "[Fix]   > HLOC-selected model (Root: ${ROOT_IMG_COUNT} images) is NOT the largest."
  echo "[Fix]   > Found better model at ${BEST_MODEL_PATH} (${MAX_IMG_COUNT} images)."
  echo "[Fix]   > Swapping models..."
  NEED_SWAP=true
else
  echo "[Fix]   > HLOC selection OK (Root: ${ROOT_IMG_COUNT} images). No swap needed."
fi

if [ "${NEED_SWAP}" = true ]; then
  TMP_SWAP_DIR="${SFM_DIR}/_tmp_swap_$(date +%s)"
  mkdir -p "${TMP_SWAP_DIR}"
  
  # 1. 把 sfm/ (root) 的錯誤 .bin 檔案移到 tmp
  # (只移動 .bin，保留 .db 和 .log)
  find "${SFM_DIR}" -maxdepth 1 -type f -name "*.bin" -exec mv {} "${TMP_SWAP_DIR}/" \;

  # 2. 把 BEST_MODEL_PATH (正確模型) 的 .bin 檔案移到 root
  find "${BEST_MODEL_PATH}" -maxdepth 1 -type f -name "*.bin" -exec mv {} "${SFM_DIR}/" \;

  # 3. 把 tmp (錯誤模型) 移回 BEST_MODEL_PATH (保持 models/ 結構完整)
  find "${TMP_SWAP_DIR}" -maxdepth 1 -type f -name "*.bin" -exec mv {} "${BEST_MODEL_PATH}/" \;
  rm -rf "${TMP_SWAP_DIR}"

  echo "[Fix]   > Swap complete. ${SFM_DIR} now holds the largest model (${MAX_IMG_COUNT} images)."
fi
# -------- [BUGFIX] End --------


# -------- 10. 視覺化（可選）--------
VIZ_SCRIPT="${PROJECT_ROOT}/scripts/visualize_sfm_open3d.py"
if [ -f "${VIZ_SCRIPT}" ]; then
  echo "[Viz] Exporting interactive HTML visualization..."
  
  # 基本參數
  VIZ_ARGS=( --sfm_dir "${SFM_DIR}" --output_dir "${VIZ_DIR}" )

  # 若該 block 已有 poses.txt，則一起畫出 Query 相機
  if [ -f "${OUT_DIR}/poses.txt" ]; then
    echo "    > Found poses.txt, adding query cameras to visualization."
    VIZ_ARGS+=( --query_poses "${OUT_DIR}/poses.txt" )
  fi
  
  # 可用環境變數啟動內建 HTTP 伺服器（預設關閉）
  # 例： START_SERVER=1 PORT=18080 bash scripts/build_block_model.sh data/xxx
  if [ "${START_SERVER:-0}" = "1" ]; then
    VIZ_ARGS+=( --port "${PORT:-8080}" )
    echo "    > START_SERVER=1. Starting HTTP server on port ${PORT:-8080}..."
  else
    VIZ_ARGS+=( --no_server )
  fi

  # 執行視覺化
  "${PY}" "${VIZ_SCRIPT}" "${VIZ_ARGS[@]}"
  echo "[Viz] HTML exported to: ${VIZ_DIR}/sfm_view.html"
else
  echo "[Viz] Skip visualization: ${VIZ_SCRIPT} not found"
fi

echo "✅ All steps completed for Block: ${BLOCK_NAME}"