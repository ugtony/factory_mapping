#!/usr/bin/env bash
# build_block_model.sh [V12-Configurable-SeqWindow]
# - Update: 支援更多影片格式 (mp4, mov, mkv, insv, etc.)
# - Update: 自動偵測 raw/ 目錄並呼叫 extract_frames.sh 進行抽幀。
# - Update: 使用 FOV_MODEL 作為建模視角
# - New: 若 db 或 db_360 已有圖片，則自動跳過抽幀與 360 轉換步驟。
# - New: 支援從 project_config.env 或 CLI (--seq-window) 設定 SEQ_WINDOW。
# - Debug: set -x 開啟詳細指令輸出

set -euox pipefail

# -------- 0. 參數解析與設定檔讀取 --------
if [ $# -lt 1 ]; then
  echo "Usage: $0 <BLOCK_DATA_DIR> [--mode=std|360] [--dense] [--fov=FLOAT] [--seq-window=INT] [--global-conf=STR] [--fps=FLOAT]"
  exit 1
fi

DATA_DIR="$(realpath "$1")"
shift

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "${SCRIPT_DIR}/..")"
CONFIG_FILE="${PROJECT_ROOT}/project_config.env"

# 1. 硬編碼預設值
DEFAULT_MODE="std"
DEFAULT_DENSE=0
DEFAULT_FOV="AUTO"
DEFAULT_GLOBAL="netvlad"
DEFAULT_FPS=2
DEFAULT_SEQ_WINDOW=3
DEFAULT_INTER_FRAME_SUFFIXES=""

# 2. 嘗試載入設定檔
if [ -f "${CONFIG_FILE}" ]; then
  echo "[Init] Loading config from ${CONFIG_FILE}..."
  source "${CONFIG_FILE}"
fi

# 3. 套用設定檔值 (改用 FOV_MODEL)
CAM_MODE="${MODE:-$DEFAULT_MODE}"
if [[ "${DENSE:-$DEFAULT_DENSE}" =~ ^(1|true|True)$ ]]; then DENSE_360=1; else DENSE_360=0; fi

# 優先使用 FOV_MODEL，若無則為 AUTO
FOV_MODEL_VAL="${FOV_MODEL:-$DEFAULT_FOV}"

GLOBAL_CONF="${GLOBAL_CONF:-$DEFAULT_GLOBAL}"
EXTRACT_FPS="${FPS:-$DEFAULT_FPS}"
SEQ_WINDOW_VAL="${SEQ_WINDOW:-$DEFAULT_SEQ_WINDOW}"
INTER_FRAME_SUFFIXES="${INTER_FRAME_SUFFIXES:-$DEFAULT_INTER_FRAME_SUFFIXES}"

# 4. 解析 CLI 參數
while [ $# -gt 0 ]; do
  case "$1" in
    --mode=*) CAM_MODE="${1#*=}" ;;
    --dense)  DENSE_360=1 ;;
    --fov=*)  FOV_MODEL_VAL="${1#*=}" ;; # 允許 CLI 覆蓋 FOV
    --seq-window=*) SEQ_WINDOW_VAL="${1#*=}" ;; # 允許 CLI 覆蓋 SEQ_WINDOW
    --fps=*)  EXTRACT_FPS="${1#*=}" ;;
    --inter-frame-suffixes=*) INTER_FRAME_SUFFIXES="${1#*=}" ;;
    --global-conf=*|--global_conf=*|--global_model=*) GLOBAL_CONF="${1#*=}" ;;
    --global-conf|--global_conf|--global_model) GLOBAL_CONF="$2"; shift ;;
    *) ;;
  esac
  shift
done

# 智慧 FOV 預設值邏輯
if [ "${CAM_MODE}" = "360" ] && [ "${FOV_MODEL_VAL}" = "AUTO" ]; then
  if [ "${DENSE_360}" = "1" ]; then FOV_MODEL_VAL=100.0; else FOV_MODEL_VAL=120.0; fi
fi

# -------- 1. 路徑與環境設定 --------
BLOCK_NAME="$(basename "${DATA_DIR}")"
OUT_ROOT="${PROJECT_ROOT}/outputs-hloc"
OUT_DIR="${OUT_ROOT}/${BLOCK_NAME}"
LOG_DIR="${OUT_DIR}/logs"
VIZ_DIR="${OUT_DIR}/visualization"
DBG_DIR="${OUT_DIR}/debug"

mkdir -p "${OUT_DIR}" "${LOG_DIR}" "${VIZ_DIR}" "${DBG_DIR}"

if [ -x "/opt/conda/bin/python" ]; then PY="/opt/conda/bin/python"; else PY="${PY:-python3}"; fi

echo "========================================"
echo "[Info] Block: ${BLOCK_NAME}"
echo "[Info] Mode: ${CAM_MODE^^}"
echo "[Info] Global Model: ${GLOBAL_CONF}"
echo "[Info] Seq Window: ${SEQ_WINDOW_VAL}"
if [ "${CAM_MODE}" = "360" ]; then
  [ "${DENSE_360}" = "1" ] && V_TYPE="Dense(8)" || V_TYPE="Sparse(4)"
  echo "[Info] 360 Settings: ${V_TYPE}, Modeling FOV=${FOV_MODEL_VAL}"
fi
echo "[Info] Output: ${OUT_DIR}"
echo "========================================"

# -------- 2. 組態設定 --------
LOCAL_CONF="superpoint_aachen"
MATCHER_CONF="superpoint+lightglue"
REBUILD_SFM="${REBUILD_SFM:-0}"
ALIGN_SFM="${ALIGN_SFM:-1}"
NUM_RETRIEVAL="${NUM_RETRIEVAL:-10}"
SEQ_WINDOW="${SEQ_WINDOW_VAL}"

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
EXTRACT_SCRIPT="${PROJECT_ROOT}/scripts/extract_frames.sh"

# -------- [Step -1] 自動抽幀 (若有 raw/) --------
RAW_DIR="${DATA_DIR}/raw"
VIDEO_EXTS=(-iname "*.mp4" -o -iname "*.mov" -o -iname "*.m4v" -o -iname "*.avi" -o -iname "*.mkv" -o -iname "*.flv" -o -iname "*.wmv" -o -iname "*.insv" -o -iname "*.360" -o -iname "*.mts" -o -iname "*.m2ts" -o -iname "*.webm" -o -iname "*.ts")

if [ -d "${RAW_DIR}" ]; then
  HAS_VIDEO=$(find "${RAW_DIR}" -maxdepth 1 -type f \( "${VIDEO_EXTS[@]}" \) -print -quit)
  
  if [ -n "$HAS_VIDEO" ]; then
      if [ "${CAM_MODE}" = "360" ]; then
          TARGET_EXT_DIR="${DATA_DIR}/db_360"
      else
          TARGET_EXT_DIR="${DATA_DIR}/db"
      fi

      # [New] 檢查目標資料夾是否已有圖片，若有則跳過
      HAS_IMAGES=""
      if [ -d "${TARGET_EXT_DIR}" ]; then
          HAS_IMAGES=$(find "${TARGET_EXT_DIR}" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.png" \) -print -quit)
      fi

      if [ -n "$HAS_IMAGES" ]; then
          echo "[-1] Found existing images in ${TARGET_EXT_DIR}. Skipping extraction."
      else
          echo "[-1] Found video files in raw/. Auto-extracting frames (FPS=${EXTRACT_FPS})..."
          bash "${EXTRACT_SCRIPT}" "${RAW_DIR}" "${TARGET_EXT_DIR}" \
              --fps "${EXTRACT_FPS}" \
              --prefix "frames" \
              --ext "jpg"
          echo "     > Extraction done. Output: ${TARGET_EXT_DIR}"
      fi
  else
      echo "[-1] raw/ directory exists but no supported video files found. Skipping extraction."
  fi
fi

# -------- [Step 0] 360 前處理 --------
if [ "${CAM_MODE}" = "360" ]; then
  SRC_360="${DATA_DIR}/db_360"
  DST_DB="${DATA_DIR}/db"

  # [New] 檢查目標資料夾是否已有圖片，若有則跳過
  HAS_DB_IMAGES=""
  if [ -d "${DST_DB}" ]; then
      HAS_DB_IMAGES=$(find "${DST_DB}" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.png" \) -print -quit)
  fi

  if [ -n "$HAS_DB_IMAGES" ]; then
      echo "[0] (360 Mode) Target DB ${DST_DB} is not empty. Skipping Pinhole conversion."
  else
      echo "[0] (360 Mode) Converting Equirectangular to Pinhole..."
      if [ ! -d "${SRC_360}" ]; then echo "[Error] Missing ${SRC_360}"; exit 1; fi
      
      CONVERT_ARGS=( "--input_dir" "${SRC_360}" "--output_dir" "${DST_DB}" "--fov" "${FOV_MODEL_VAL}" )
      if [ "${DENSE_360}" = "1" ]; then CONVERT_ARGS+=( "--dense" ); fi
      "${PY}" "${CONVERT_360_SCRIPT}" "${CONVERT_ARGS[@]}"
  fi
fi

# -------- HLOC Pipeline --------
echo "[1] Generating DB image list (db.txt)..."
if [ ! -d "${DATA_DIR}/db" ]; then echo "[Error] ${DATA_DIR}/db not found."; exit 1; fi

(cd "${DATA_DIR}" && find db -maxdepth 3 -type f \( -iname '*.jpg' -o -iname '*.png' \) | sort) > "${DB_LIST}"

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
                ok=False; break
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

  # 若有設定 INTER_FRAME_SUFFIXES，則加入參數
  if [ -n "${INTER_FRAME_SUFFIXES}" ]; then
      echo "      - Backbone Strategy: Inter-frame restricted to '${INTER_FRAME_SUFFIXES}'"
      PAIRS_ARGS+=( "--inter_frame_suffixes" "${INTER_FRAME_SUFFIXES}" )
  fi

  IS_FOV_GT_90=$(awk -v f="${FOV_MODEL_VAL}" 'BEGIN {print (f > 90 ? 1 : 0)}')
  if [ "${IS_FOV_GT_90}" -eq 1 ]; then
      echo "      - FOV > 90 (${FOV_MODEL_VAL}), enabling intra-frame matching."
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
  echo "    > [Fix] Staging images from db.txt to clean directory..."
  rm -rf "${STAGE}"
  mkdir -p "${STAGE}"
  
  count=0
  while read -r rel_path; do
      [ -z "$rel_path" ] && continue
      
      SRC_FILE="${DATA_DIR}/${rel_path}"
      DST_FILE="${STAGE}/${rel_path}"
      DST_DIR="$(dirname "${DST_FILE}")"
      
      if [ -f "${SRC_FILE}" ]; then
          mkdir -p "${DST_DIR}"
          ln "${SRC_FILE}" "${DST_FILE}" 2>/dev/null || cp "${SRC_FILE}" "${DST_FILE}"
          count=$((count+1))
      fi
  done < "${DB_LIST}"
  
  echo "      - Staged ${count} images to ${STAGE}"
  IMG_DIR="${STAGE}"

  # 執行 hloc 重建
  ${PY} -m hloc.reconstruction \
    --image_dir "${IMG_DIR}" --pairs "${PAIRS_USE}" \
    --features "${LOCAL_FEATS}" --matches "${DB_MATCHES}" \
    --sfm_dir "${SFM_DIR}" | tee "${LOG_DIR}/reconstruction.log"
fi

echo "[9] Verifying SfM model..."
NEED_SWAP=false; BEST_MODEL_PATH=""; MAX_IMG_COUNT=0; ROOT_IMG_COUNT=0
if [ -f "${SFM_DIR}/images.bin" ]; then
    ROOT_IMG_COUNT=$(${PY} -c "import pycolmap, sys; print(len(pycolmap.Reconstruction(sys.argv[1]).images))" "${SFM_DIR}" 2>/dev/null || echo 0)
else
    echo "    [Warn] No images.bin in root. Searching models/..."
fi
if [ -d "${SFM_MODELS_DIR}" ]; then
  for MODEL_DIR in "${SFM_MODELS_DIR}"/*; do
    if [ -d "${MODEL_DIR}" ] && [ -f "${MODEL_DIR}/images.bin" ]; then
      CNT=$(${PY} -c "import pycolmap, sys; print(len(pycolmap.Reconstruction(sys.argv[1]).images))" "${MODEL_DIR}" 2>/dev/null || echo 0)
      if [ "${CNT}" -gt "${MAX_IMG_COUNT}" ]; then MAX_IMG_COUNT="${CNT}"; BEST_MODEL_PATH="${MODEL_DIR}"; fi
    fi
  done
fi
if [ "${ROOT_IMG_COUNT}" -lt "${MAX_IMG_COUNT}" ] && [ -n "${BEST_MODEL_PATH}" ]; then
  echo "    [Fix] Swapping root (${ROOT_IMG_COUNT}) with best (${MAX_IMG_COUNT}) from ${BEST_MODEL_PATH##*/}..."
  TMP="${SFM_DIR}/_swap"; mkdir -p "${TMP}"
  find "${SFM_DIR}" -maxdepth 1 -type f -name "*.bin" -exec mv {} "${TMP}/" \;
  find "${BEST_MODEL_PATH}" -maxdepth 1 -type f -name "*.bin" -exec mv {} "${SFM_DIR}/" \;
  find "${TMP}" -maxdepth 1 -type f -name "*.bin" -exec mv {} "${BEST_MODEL_PATH}/" \;
  rm -rf "${TMP}"; ROOT_IMG_COUNT=${MAX_IMG_COUNT}
else
  echo "    > Selection OK (Root: ${ROOT_IMG_COUNT}, Max: ${MAX_IMG_COUNT})."
fi

echo "[10] Align to Z-Up & Visualize"
if [ "${ALIGN_SFM}" = "1" ] && [ "${ROOT_IMG_COUNT}" -gt 2 ]; then
  "${PY}" "${ALIGN_SCRIPT}" "${SFM_DIR}" "--out_dir=${SFM_ALIGNED}" > "${LOG_DIR}/align.log" 2>&1
  FINAL_SFM="${SFM_ALIGNED}"
else
  FINAL_SFM="${SFM_DIR}"
fi
if [ -f "${VIZ_SCRIPT}" ] && [ -d "${FINAL_SFM}" ]; then
  "${PY}" "${VIZ_SCRIPT}" \
    --sfm_dir "${FINAL_SFM}" \
    --output_dir "${VIZ_DIR}" \
    --no_server >/dev/null
fi

echo "✅ Completed: ${BLOCK_NAME} (Mode: ${CAM_MODE})"