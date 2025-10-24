#!/bin/bash
set -euo pipefail
set -x
ROOT_DIR="$(dirname "$(realpath "$0")")/.."
DATA_ROOT="${ROOT_DIR}/data"
OUT_ROOT="${ROOT_DIR}/outputs-hloc"
BUILD_SCRIPT="${ROOT_DIR}/scripts/build_block_model.sh"
MISSING=0
OK=0
for BLOCK_PATH in "${DATA_ROOT}"/block_*; do
  [ -d "${BLOCK_PATH}" ] || continue
  BLOCK_NAME=$(basename "${BLOCK_PATH}")
  OUT_DIR="${OUT_ROOT}/${BLOCK_NAME}"
  SFM_DIR="${OUT_DIR}/sfm"
  LOCAL_FEATS="${OUT_DIR}/local-superpoint_aachen.h5"
  GLOBAL_FEATS="${OUT_DIR}/global-netvlad.h5"
  MATCHES="${OUT_DIR}/db-matches-superpoint+lightglue.h5"
  NEED=false
  [ ! -d "${SFM_DIR}" ] || [ ! -f "${SFM_DIR}/points3D.bin" ] && NEED=true
  [ ! -f "${LOCAL_FEATS}" ] && NEED=true
  [ ! -f "${GLOBAL_FEATS}" ] && NEED=true
  [ ! -f "${MATCHES}" ] && NEED=true
  if [ "${NEED}" = true ]; then
    bash "${BUILD_SCRIPT}" "${BLOCK_PATH}" | tee "${OUT_DIR}/logs/rebuild.log"
    ((MISSING++))
  else
    ((OK++))
  fi
done
echo "Summary: OK=${OK} Rebuilt=${MISSING}"
