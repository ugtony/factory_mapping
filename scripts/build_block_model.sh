#!/usr/bin/env bash
# build_block_model.sh
# 單一 block 的離線建模（HLOC pipeline），避免把 query 圖混入 SfM。
# - 預設啟用 staging（只把 db.txt 指定的影像鏡射到 _images_stage 再跑 SfM）
# - 自動檢查/修復 local H5 完整性；pairs 潔淨化避免不在 H5 的影像
# - 優先使用 /opt/conda/bin/python（若存在），確保和 HLOC 安裝環境一致

set -euo pipefail
set -x

if [ $# -lt 1 ]; then
  echo "Usage: $0 <BLOCK_DATA_DIR>"
  exit 1
fi

# -------- 路徑與環境 --------
DATA_DIR="$(realpath "$1")"
BLOCK_NAME="$(basename "${DATA_DIR}")"

# 以腳本所在目錄作為專案根（穩定）
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "${SCRIPT_DIR}/..")"
OUT_ROOT="${PROJECT_ROOT}/outputs-hloc"
OUT_DIR="${OUT_ROOT}/${BLOCK_NAME}"
LOG_DIR="${OUT_DIR}/logs"
VIZ_DIR="${OUT_DIR}/visualization"
DBG_DIR="${OUT_DIR}/debug"

mkdir -p "${OUT_DIR}" "${LOG_DIR}" "${VIZ_DIR}" "${DBG_DIR}"

# Python 路徑（優先用 conda 的 python）
if [ -x "/opt/conda/bin/python" ]; then
  PY="/opt/conda/bin/python"
else
  PY="${PY:-python3}"
fi

# -------- 組態 --------
GLOBAL_CONF="netvlad"
LOCAL_CONF="superpoint_aachen"
MATCHER_CONF="superpoint+lightglue"

NUM_DB_PAIRS="${NUM_DB_PAIRS:-10}"    # 每張 DB 找 K 張
USE_STAGE="${USE_STAGE:-1}"           # 1=使用 staging；0=直接用 ${DATA_DIR}/db
REBUILD_SFM="${REBUILD_SFM:-1}"       # 1=每次重建 SfM 目錄；0=沿用（僅限一致情況）

# 輸出檔
DB_LIST="${OUT_DIR}/db.txt"
LOCAL_FEATS="${OUT_DIR}/local-${LOCAL_CONF}.h5"
GLOBAL_FEATS="${OUT_DIR}/global-${GLOBAL_CONF}.h5"
PAIRS_DB="${OUT_DIR}/pairs-db-retrieval.txt"
PAIRS_DB_CLEAN="${OUT_DIR}/_pairs-db-retrieval.clean.txt"
DB_MATCHES="${OUT_DIR}/db-matches-${MATCHER_CONF}.h5"
SFM_DIR="${OUT_DIR}/sfm"
STAGE="${OUT_DIR}/_images_stage"   # staging 鏡射資料夾

# -------- 1) 產生 db.txt（只收集 db 子資料夾） --------
(cd "${DATA_DIR}" && find db -maxdepth 3 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | sort) > "${DB_LIST}"

# -------- 2) 保證 local 特徵 H5 完整（不完整就刪掉重抽）--------
# 為何：避免舊 H5 缺少新增影像的 keypoints，導致 reconstruction KeyError
${PY} - <<PY || rm -f "${LOCAL_FEATS}"
import h5py, sys
from pathlib import Path
db_paths=[l.strip() for l in Path("${DB_LIST}").read_text().splitlines() if l.strip()]
ok=True
try:
    with h5py.File("${LOCAL_FEATS}","r") as f:
        for p in db_paths:
            try:
                _=f[p]["keypoints"]
            except Exception:
                print("[MISS local]", p); ok=False; break
except Exception as e:
    print("[ERR open local feats]:", e); ok=False
sys.exit(0 if ok else 1)
PY

echo "[1/5] Extracting LOCAL features…"
${PY} -m hloc.extract_features --conf "${LOCAL_CONF}" \
  --image_dir "${DATA_DIR}" --image_list "${DB_LIST}" \
  --export_dir "${OUT_DIR}" --feature_path "${LOCAL_FEATS}"

# -------- 3) 確保 GLOBAL 特徵存在（可被跳過視版本實作）--------
echo "[2/5] Extracting GLOBAL features…"
${PY} -m hloc.extract_features --conf "${GLOBAL_CONF}" \
  --image_dir "${DATA_DIR}" --image_list "${DB_LIST}" \
  --export_dir "${OUT_DIR}" --feature_path "${GLOBAL_FEATS}"

# -------- 4) 產 pairs（DB 自檢索）--------
echo "[3/5] Building retrieval pairs…"
python3 scripts/pairs_from_retrieval_and_sequential.py \
  --db_list "${DB_LIST}" \
  --global_feats "${GLOBAL_FEATS}" \
  --num_retrieval 5 \
  --seq_window 5 \
  --output "${PAIRS_DB}"

# -------- 4.1) 潔淨化 pairs：僅保留存在於 H5 且有 keypoints 的影像 --------
# 避免 pairs 包含不在 LOCAL_FEATS 的鍵名（或誤混入 query/）
${PY} - <<PY
from pathlib import Path
import h5py
pairs_in = Path("${PAIRS_DB}"); pairs_out = Path("${PAIRS_DB_CLEAN}")
db_list = set([l.strip() for l in Path("${DB_LIST}").read_text().splitlines() if l.strip()])
with h5py.File("${LOCAL_FEATS}","r") as f, open(pairs_in,"r") as fi, open(pairs_out,"w") as fo:
    keep=0; drop=0
    for line in fi:
        s=line.strip().split()
        if len(s)<2: 
            continue
        a,b = s[0], s[1]
        ok = (a in db_list) and (b in db_list)
        ok = ok and (a in f) and (b in f) and ("keypoints" in f[a]) and ("keypoints" in f[b])
        if ok:
            fo.write(line); keep+=1
        else:
            drop+=1
    print(f"[pairs clean] keep={keep} drop={drop} -> {pairs_out}")
PY

# 用潔淨版 pairs
PAIRS_USE="${PAIRS_DB_CLEAN}"

# -------- 5) DB 內匹配（LightGlue）--------
echo "[4/5] Matching DB pairs (${MATCHER_CONF})…"
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

# -------- 6) 準備影像來源（staging 或直接 db/）--------
if [ "${USE_STAGE}" = "1" ]; then
  echo "[Prep] Create staging mirror at: ${STAGE}"
  rm -rf "${STAGE}"; mkdir -p "${STAGE}"
  while IFS= read -r rel; do
    src="${DATA_DIR}/${rel}"
    dst="${STAGE}/${rel}"
    mkdir -p "$(dirname "${dst}")"
    ln -sf "${src}" "${dst}"
  done < "${DB_LIST}"
  IMG_DIR="${STAGE}"
else
  IMG_DIR="${DATA_DIR}/db"
fi

# 視需要重建 SfM 目錄（避免殘留 DB 混入上一輪影像）
if [ "${REBUILD_SFM}" = "1" ]; then
  rm -rf "${SFM_DIR}"
fi

# -------- 7) SfM 重建 --------
echo "[5/5] Running SfM…"
${PY} -m hloc.reconstruction \
  --image_dir "${IMG_DIR}" \
  --pairs "${PAIRS_USE}" \
  --features "${LOCAL_FEATS}" \
  --matches "${DB_MATCHES}" \
  --sfm_dir "${SFM_DIR}" | tee "${LOG_DIR}/reconstruction.log"

echo "✅ Done: ${BLOCK_NAME}"
echo "SfM model stored at: ${SFM_DIR}"

# -------- 8) 視覺化：匯出互動式 HTML（不開 server）--------
VIZ_SCRIPT="${PROJECT_ROOT}/scripts/visualize_sfm_open3d.py"
if [ -f "${VIZ_SCRIPT}" ]; then
  echo "[Viz] Export interactive HTML..."
  VIZ_ARGS=( --sfm_dir "${SFM_DIR}" --output_dir "${VIZ_DIR}" --no_server )

  # 若該 block 已有 poses.txt，則一起畫出 Query 相機
  if [ -f "${OUT_DIR}/poses.txt" ]; then
    VIZ_ARGS+=( --query_poses "${OUT_DIR}/poses.txt" )
  fi

  # 可用環境變數啟動內建 HTTP 伺服器（預設關閉）
  # 例： START_SERVER=1 PORT=18080 bash scripts/build_block_model.sh data/xxx
  if [ "${START_SERVER:-0}" = "1" ]; then
    VIZ_ARGS=( --sfm_dir "${SFM_DIR}" --output_dir "${VIZ_DIR}" --port "${PORT:-8080}" )
    # 如有 poses 一樣附上
    if [ -f "${OUT_DIR}/poses.txt" ]; then
      VIZ_ARGS+=( --query_poses "${OUT_DIR}/poses.txt" )
    fi
  fi

  "${PY}" "${VIZ_SCRIPT}" "${VIZ_ARGS[@]}"
  echo "[Viz] HTML exported to: ${VIZ_DIR}/sfm_view.html"
else
  echo "[Viz] Skip: ${VIZ_SCRIPT} not found"
fi
