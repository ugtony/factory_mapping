#!/usr/bin/env bash
# build_block_model.sh
# 針對單一 block 區域執行離線建模（HLOC pipeline）。
#
# [V4 最終修正]
# - Fix FOV check: 使用 awk -v 確保數值比較正確 (解決 100.0 沒觸發的問題)。
# - Fix Matcher call: 確保使用 Python API 傳遞 Path 物件 (解決 ValueError)。

set -euo pipefail

# -------- 0. 參數解析 --------
if [ $# -lt 1 ]; then
  echo "Usage: $0 <BLOCK_DATA_DIR> [--mode=std|360] [--dense] [--fov=100.0] [...]"
  exit 1
fi

DATA_DIR="$(realpath "$1")"
shift

# 預設參數
CAM_MODE="std"
DENSE_360=0
FOV_360=100.0

while [ $# -gt 0 ]; do
  case "$1" in
    --mode=*) CAM_MODE="${1#*=}" ;;
    --dense)  DENSE_360=1 ;;
    --fov=*)  FOV_360="${1#*=}" ;;
    *) ;;
  esac
  shift
done

# -------- 1. 路徑與環境設定 --------
BLOCK_NAME="$(basename "${DATA_DIR}")"
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "${SCRIPT_DIR}/..")"
OUT_ROOT="${PROJECT_ROOT}/outputs-hloc"
OUT_DIR="${OUT_ROOT}/${BLOCK_NAME}"
LOG_DIR="${OUT_DIR}/logs"
VIZ_DIR="${OUT_DIR}/visualization"
DBG_DIR="${OUT_DIR}/debug"

mkdir -p "${OUT_DIR}" "${LOG_DIR}" "${VIZ_DIR}" "${DBG_DIR}"

if [ -x "/opt/conda/bin/python" ]; then PY="/opt/conda/bin/python"; else PY="${PY:-python3}"; fi

echo "========================================"
echo "[Info] Block: ${BLOCK_NAME} | Mode: ${CAM_MODE^^}"
echo "[Info] Output: ${OUT_DIR}"
echo "========================================"

# -------- 2. 組態設定 --------
GLOBAL_CONF="netvlad"
LOCAL_CONF="superpoint_aachen"
MATCHER_CONF="superpoint+lightglue"

USE_STAGE="${USE_STAGE:-1}"
REBUILD_SFM="${REBUILD_SFM:-0}"
ALIGN_SFM="${ALIGN_SFM:-1}"
NUM_RETRIEVAL="${NUM_RETRIEVAL:-10}"
SEQ_WINDOW="${SEQ_WINDOW:-5}"

# -------- 3. 核心檔案路徑 --------
DB_LIST="${OUT_DIR}/db.txt"
LOCAL_FEATS="${OUT_DIR}/local-${LOCAL_CONF}.h5"
GLOBAL_FEATS="${OUT_DIR}/global-${GLOBAL_CONF}.h5"
PAIRS_DB="${OUT_DIR}/pairs-raw.txt"
PAIRS_DB_CLEAN="${OUT_DIR}/pairs-clean.txt"
DB_MATCHES="${OUT_DIR}/db-matches-${MATCHER_CONF}.h5"
SFM_DIR="${OUT_DIR}/sfm"
SFM_MODELS_DIR="${SFM_DIR}/models"
STAGE="${OUT_DIR}/_images_stage"
SFM_ALIGNED="${OUT_DIR}/sfm_aligned"

ALIGN_SCRIPT="${PROJECT_ROOT}/scripts/align_sfm_model_z_up.py"
VIZ_SCRIPT="${PROJECT_ROOT}/scripts/visualize_sfm_open3d.py"
CONVERT_360_SCRIPT="${PROJECT_ROOT}/scripts/convert360_to_pinhole.py"
PAIRS_360_SCRIPT="${PROJECT_ROOT}/scripts/pairs_from_360.py"
PAIRS_STD_SCRIPT="${PROJECT_ROOT}/scripts/pairs_from_retrieval_and_sequential.py"

# -------- [Step 0] 360 前處理 --------
if [ "${CAM_MODE}" = "360" ]; then
  echo "[0] (360 Mode) Converting Equirectangular to Pinhole..."
  SRC_360="${DATA_DIR}/db_360"
  DST_DB="${DATA_DIR}/db"
  if [ ! -d "${SRC_360}" ]; then echo "[Error] Missing ${SRC_360}"; exit 1; fi
  
  CONVERT_ARGS=( "--input_dir" "${SRC_360}" "--output_dir" "${DST_DB}" "--fov" "${FOV_360}" )
  if [ "${DENSE_360}" = "1" ]; then CONVERT_ARGS+=( "--dense" ); fi
  "${PY}" "${CONVERT_360_SCRIPT}" "${CONVERT_ARGS[@]}"
fi

# -------- HLOC Pipeline --------
echo "[1] Generating DB image list (db.txt)..."
if [ ! -d "${DATA_DIR}/db" ]; then echo "[Error] ${DATA_DIR}/db not found."; exit 1; fi
(cd "${DATA_DIR}" && find db -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.png' \) | sort) > "${DB_LIST}"
if [ ! -s "${DB_LIST}" ]; then echo "[Error] No images in db/."; exit 1; fi
echo "    > Found $(wc -l < "${DB_LIST}") DB images."

echo "[2] Checking integrity of local features H5..."
${PY} - <<PY || { echo "[Check] H5 stale/corrupted. Deleting."; rm -f "${LOCAL_FEATS}"; }
import h5py, sys; from pathlib import Path
db_paths=[l.strip() for l in Path("${DB_LIST}").read_text().splitlines() if l.strip()]
ok=True
try:
    with h5py.File("${LOCAL_FEATS}","r") as f:
        for p in db_paths:
            if p not in f or "keypoints" not in f[p]:
                print(f"[Check] Missing key '{p}'", file=sys.stderr); ok=False; break
except Exception: ok=False
sys.exit(0 if ok else 1)
PY

echo "[3] Extracting LOCAL features (${LOCAL_CONF})..."
${PY} -m hloc.extract_features --conf "${LOCAL_CONF}" \
  --image_dir "${DATA_DIR}" --image_list "${DB_LIST}" \
  --export_dir "${OUT_DIR}" --feature_path "${LOCAL_FEATS}"

echo "[4] Extracting GLOBAL features (${GLOBAL_CONF})..."
${PY} -m hloc.extract_features --conf "${GLOBAL_CONF}" \
  --image_dir "${DATA_DIR}" --image_list "${DB_LIST}" \
  --export_dir "${OUT_DIR}" --feature_path "${GLOBAL_FEATS}"

echo "[5] Building DB pairs..."
if [ "${CAM_MODE}" = "360" ]; then
  echo "    > [360 Mode] Using explicit geometric pairing..."
  PAIRS_ARGS=( "--db_list" "${DB_LIST}" "--output" "${PAIRS_DB}" "--seq_window" "${SEQ_WINDOW}" )
  
  # [Fix V4] 使用 awk -v 確保數值比較正確
  IS_FOV_GT_90=$(awk -v f="${FOV_360}" 'BEGIN {print (f > 90 ? 1 : 0)}')
  
  if [ "${IS_FOV_GT_90}" -eq 1 ]; then
      echo "      - FOV > 90 (${FOV_360}), enabling intra-frame matching."
      PAIRS_ARGS+=( "--intra_match" )
  fi
  "${PY}" "${PAIRS_360_SCRIPT}" "${PAIRS_ARGS[@]}"
else
  echo "    > [Std Mode] Using retrieval + sequential pairing..."
  "${PY}" "${PAIRS_STD_SCRIPT}" \
    --db_list "${DB_LIST}" --global_feats "${GLOBAL_FEATS}" \
    --num_retrieval "${NUM_RETRIEVAL}" --seq_window "${SEQ_WINDOW}" \
    --output "${PAIRS_DB}"
fi

echo "[6] Cleaning pairs list..."
${PY} - <<PY
from pathlib import Path; import h5py, sys
pairs_in = Path("${PAIRS_DB}"); pairs_out = Path("${PAIRS_DB_CLEAN}")
db_list = set([l.strip() for l in Path("${DB_LIST}").read_text().splitlines() if l.strip()])
try:
    with h5py.File("${LOCAL_FEATS}","r") as f, open(pairs_in,"r") as fi, open(pairs_out,"w") as fo:
        keep=0; drop=0
        for line in fi:
            s=line.strip().split()
            if len(s)<2: continue
            a,b = s[0], s[1]
            if (a in db_list) and (b in db_list) and (a in f) and (b in f):
                fo.write(line); keep+=1
            else: drop+=1
        print(f"    > Pairs cleaned: {keep} kept, {drop} dropped.")
except Exception as e: print(f"[Error] Clean failed: {e}", file=sys.stderr); sys.exit(1)
PY
PAIRS_USE="${PAIRS_DB_CLEAN}"

echo "[7] Matching DB pairs (${MATCHER_CONF})..."
# [Fix V4] 強制使用 Python API 呼叫，避免 CLI 路徑解析錯誤
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

echo "[8] Running SfM reconstruction..."
if [ "${REBUILD_SFM}" = "1" ]; then rm -rf "${SFM_DIR}"; fi
if [ -f "${SFM_DIR}/images.bin" ]; then
  echo "    > SfM exists. Skipping."
else
  if [ "${USE_STAGE}" = "1" ]; then
      rm -rf "${STAGE}"; mkdir -p "${STAGE}/db"
      ln -sf "$(realpath "${DATA_DIR}/db")"/* "${STAGE}/db/"
      IMG_DIR="${STAGE}"
  else
      IMG_DIR="${DATA_DIR}"
  fi
  ${PY} -m hloc.reconstruction \
    --image_dir "${IMG_DIR}" --pairs "${PAIRS_USE}" \
    --features "${LOCAL_FEATS}" --matches "${DB_MATCHES}" \
    --sfm_dir "${SFM_DIR}" | tee "${LOG_DIR}/reconstruction.log"
fi

echo "[9] Verifying SfM model..."
ROOT_IMG_COUNT=0
if [ -f "${SFM_DIR}/images.bin" ]; then
    ROOT_IMG_COUNT=$(${PY} -c "import pycolmap, sys; print(len(pycolmap.Reconstruction(sys.argv[1]).images))" "${SFM_DIR}" 2>/dev/null || echo 0)
fi
echo "    > Root model has ${ROOT_IMG_COUNT} images."

echo "[10] Align to Z-Up & Visualize"
if [ "${ALIGN_SFM}" = "1" ] && [ "${ROOT_IMG_COUNT}" -gt 2 ]; then
  "${PY}" "${ALIGN_SCRIPT}" "${SFM_DIR}" "--out_dir=${SFM_ALIGNED}" > "${LOG_DIR}/align.log" 2>&1
  FINAL_SFM="${SFM_ALIGNED}"
else
  FINAL_SFM="${SFM_DIR}"
fi
if [ -f "${VIZ_SCRIPT}" ] && [ -d "${FINAL_SFM}" ]; then
  "${PY}" "${VIZ_SCRIPT}" --sfm_dir "${FINAL_SFM}" --output_dir "${VIZ_DIR}" --no_server >/dev/null
fi

echo "✅ Completed: ${BLOCK_NAME} (Mode: ${CAM_MODE})"