#!/bin/bash
set -euo pipefail
set -x
if [ $# -lt 1 ]; then
  echo "Usage: $0 <BLOCK_DATA_DIR>"
  exit 1
fi
DATA_DIR=$(realpath "$1")
BLOCK_NAME=$(basename "${DATA_DIR}")
OUT_DIR="$(dirname "${DATA_DIR}")/../../outputs-hloc/${BLOCK_NAME}"
OUT_DIR=$(realpath "${OUT_DIR}")
mkdir -p "${OUT_DIR}" "${OUT_DIR}/logs" "${OUT_DIR}/visualization" "${OUT_DIR}/debug"
GLOBAL_CONF="netvlad"
LOCAL_CONF="superpoint_aachen"
MATCHER_CONF="superpoint+lightglue"
NUM_DB_PAIRS=10
DB_LIST="${OUT_DIR}/db.txt"
LOCAL_FEATS="${OUT_DIR}/local-${LOCAL_CONF}.h5"
GLOBAL_FEATS="${OUT_DIR}/global-${GLOBAL_CONF}.h5"
DB_MATCHES="${OUT_DIR}/db-matches-${MATCHER_CONF}.h5"
PAIRS_DB="${OUT_DIR}/pairs-db-retrieval.txt"
SFM_DIR="${OUT_DIR}/sfm"
(cd "${DATA_DIR}" && find db -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \)) | sort > "${DB_LIST}"
python3 -m hloc.extract_features --conf ${LOCAL_CONF} --image_dir "${DATA_DIR}" --image_list "${DB_LIST}" --export_dir "${OUT_DIR}" --feature_path "${LOCAL_FEATS}"
python3 -m hloc.extract_features --conf ${GLOBAL_CONF} --image_dir "${DATA_DIR}" --image_list "${DB_LIST}" --export_dir "${OUT_DIR}" --feature_path "${GLOBAL_FEATS}"
python3 -m hloc.pairs_from_retrieval --query_list "${DB_LIST}" --db_list "${DB_LIST}" --descriptors "${GLOBAL_FEATS}" --db_descriptors "${GLOBAL_FEATS}" --num_matched ${NUM_DB_PAIRS} --output "${PAIRS_DB}"
python3 - <<'PY'
from pathlib import Path
from hloc import match_features
match_features.main(
    conf=match_features.confs["superpoint+lightglue"],
    pairs=Path("${PAIRS_DB}"),
    features=Path("${LOCAL_FEATS}"),
    matches=Path("${DB_MATCHES}")
)
PY
python3 -m hloc.reconstruction --image_dir "${DATA_DIR}" --pairs "${PAIRS_DB}" --features "${LOCAL_FEATS}" --matches "${DB_MATCHES}" --sfm_dir "${SFM_DIR}" | tee "${OUT_DIR}/logs/reconstruction.log"
echo "Done: ${BLOCK_NAME}"
