#!/usr/bin/env bash
set -euo pipefail

# extract_frames.sh [Multi-Format Support]
# 用途：抽幀並直接輸出到目標目錄（不建立子資料夾），檔名包含來源影片名稱以避免衝突。

if [ $# -lt 2 ]; then
  echo "Usage: $0 <input_video_or_dir> <output_dir> [--fps 2] [--max-side 1024] [--ext jpg] [--jpeg-q 3] [--prefix frames]"
  exit 1
fi

INPUT_PATH=$(realpath "$1")
OUT_DIR=$(realpath "$2")
shift 2

FPS=2
MAX_SIDE=1024
EXT="jpg"
JPEQ_Q=3
PREFIX="frames"

# 解析可選參數
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fps)       FPS="$2"; shift 2;;
    --max-side)  MAX_SIDE="$2"; shift 2;;
    --ext)       EXT="$2"; shift 2;;
    --jpeg-q)    JPEQ_Q="$2"; shift 2;;
    --prefix)    PREFIX="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

mkdir -p "$OUT_DIR"

build_scale_filter() {
  local maxside="$1"
  if [[ "$maxside" -le 0 ]]; then
    echo "fps=${FPS},format=yuv420p,setsar=1"
  else
    echo "fps=${FPS},scale='if(gt(iw,ih),${maxside},-2)':'if(gt(iw,ih),-2,${maxside})':flags=lanczos,format=yuv420p,setsar=1"
  fi
}

SCALE_FILTER=$(build_scale_filter "$MAX_SIDE")

MANIFEST="${OUT_DIR}/frames_manifest.tsv"
if [ ! -f "$MANIFEST" ]; then
    echo -e "src_video\tframe_file\ttimestamp_sec" > "$MANIFEST"
fi

process_one_video() {
  local vpath="$1"
  local vbase
  vbase=$(basename "$vpath")
  local stem="${vbase%.*}"
  
  local out_sub="${OUT_DIR}"
  echo "[Info] Extracting from: $vbase -> $out_sub (Unique Pattern: ${PREFIX}_${stem}_...)"

  local out_pattern="${out_sub}/${PREFIX}_${stem}_%06d.${EXT}"

  if [[ "$EXT" == "jpg" || "$EXT" == "jpeg" ]]; then
    ffmpeg -hide_banner -loglevel warning -y \
      -i "$vpath" \
      -vf "${SCALE_FILTER}" \
      -q:v "${JPEQ_Q}" \
      -vsync vfr \
      "$out_pattern"
  else
    ffmpeg -hide_banner -loglevel warning -y \
      -i "$vpath" \
      -vf "${SCALE_FILTER}" \
      -vsync vfr \
      "$out_pattern"
  fi

  local idx=1
  shopt -s nullglob
  for f in "${out_sub}/${PREFIX}_${stem}_"*".${EXT}"; do
    ts=$(awk -v i="$idx" -v fps="$FPS" 'BEGIN { printf "%.3f", (i-1)/fps }')
    echo -e "${vbase}\t$(realpath --relative-to="${OUT_DIR}" "$f")\t${ts}" >> "$MANIFEST"
    idx=$((idx+1))
  done
  shopt -u nullglob
}

# [Update] 支援更多格式的列表
VIDEO_EXTS=(-iname "*.mp4" -o -iname "*.mov" -o -iname "*.m4v" -o -iname "*.avi" -o -iname "*.mkv" -o -iname "*.flv" -o -iname "*.wmv" -o -iname "*.insv" -o -iname "*.360" -o -iname "*.mts" -o -iname "*.m2ts" -o -iname "*.webm" -o -iname "*.ts" -o -iname "*.heic" -o -iname "*.heif")

if [[ -f "$INPUT_PATH" ]]; then
  process_one_video "$INPUT_PATH"
elif [[ -d "$INPUT_PATH" ]]; then
  # 使用 find 搭配陣列展開
  mapfile -t VIDEOS < <(find "$INPUT_PATH" -type f \( "${VIDEO_EXTS[@]}" \) | sort)
  if [[ ${#VIDEOS[@]} -eq 0 ]]; then
    echo "[Warn] No video files found under: $INPUT_PATH"
    exit 0
  fi
  for v in "${VIDEOS[@]}"; do
    process_one_video "$v"
  done
else
  echo "[Error] Input path not found: $INPUT_PATH"
  exit 1
fi

echo "[Done] Frames extracted to: ${OUT_DIR}"