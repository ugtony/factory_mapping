#!/usr/bin/env bash
set -euo pipefail

# 抽幀 + 縮圖（限制最長邊），支援單一影片或資料夾
# 需求：ffmpeg, ffprobe
#
# 用法：
#   scripts/extract_frames.sh <input_video_or_dir> <output_dir> \
#       [--fps 2] [--max-side 1024] [--ext jpg] [--jpeg-q 3] [--prefix db]
#
# 參數說明：
#   --fps <float>        每秒抽幀數 (default: 2)
#   --max-side <int>     縮圖最長邊像素，維持等比例；若設 0 則不縮 (default: 1024)
#   --ext <jpg|png>      幀輸出副檔名 (default: jpg)
#   --jpeg-q <int>       JPEG 品質（ffmpeg -q:v，2~5 建議，2 最好但檔案較大）(default: 3)
#   --prefix <name>      檔名前綴（資料夾名 / 影片名的前綴）(default: frames)
#
# 範例：
#   scripts/extract_frames.sh data/videos ./data/frames --fps 2 --max-side 1024 --prefix db
#   scripts/extract_frames.sh data/clip.mov ./data/frames --fps 1 --max-side 960 --ext jpg

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

# 生成 scale 濾鏡：限制最長邊，保持等比例，並確保偶數邊（-2）
# 如果 MAX_SIDE=0 就不縮放
build_scale_filter() {
  local maxside="$1"
  if [[ "$maxside" -le 0 ]]; then
    echo "fps=${FPS},format=yuv420p,setsar=1"
  else
    # 如果寬>高，寬縮到 max，高自適應；否則高縮到 max，寬自適應（-2 讓維持偶數）
    echo "fps=${FPS},scale='if(gt(iw,ih),${maxside},-2)':'if(gt(iw,ih),-2,${maxside})':flags=lanczos,format=yuv420p,setsar=1"
  fi
}

SCALE_FILTER=$(build_scale_filter "$MAX_SIDE")

# 建立清單
MANIFEST="${OUT_DIR}/frames_manifest.tsv"
echo -e "src_video\tframe_file\ttimestamp_sec" > "$MANIFEST"

process_one_video() {
  local vpath="$1"
  local vbase
  vbase=$(basename "$vpath")
  local stem="${vbase%.*}"
  local out_sub="${OUT_DIR}/${PREFIX}_${stem}"
  mkdir -p "$out_sub"

  echo "[Info] Extracting from: $vbase -> $out_sub (fps=${FPS}, max-side=${MAX_SIDE}, ext=${EXT})"

  # 先抽幀並縮圖
  # -vsync vfr：可搭配 fps 濾鏡更穩定
  # 注意：ffmpeg 新版預設會自動套用旋轉 metadata（autorotate），這裡不額外處理。
  if [[ "$EXT" == "jpg" || "$EXT" == "jpeg" ]]; then
    ffmpeg -hide_banner -loglevel warning -y \
      -i "$vpath" \
      -vf "${SCALE_FILTER}" \
      -q:v "${JPEQ_Q}" \
      -vsync vfr \
      "${out_sub}/${PREFIX}-%06d.${EXT}"
  else
    # PNG 沒有 -q:v 的概念
    ffmpeg -hide_banner -loglevel warning -y \
      -i "$vpath" \
      -vf "${SCALE_FILTER}" \
      -vsync vfr \
      "${out_sub}/${PREFIX}-%06d.${EXT}"
  fi

  # 以 ffprobe 取每個輸出幀的時間戳（近似），寫入 manifest
  # 注意：純輸出圖片不帶 EXIF/時間，這裡以「第 n 張 / FPS」估算秒數；若要嚴格對時，要改用 select + showinfo 再 parse。
  local idx=1
  shopt -s nullglob
  for f in "${out_sub}/${PREFIX}-"*".${EXT}"; do
    # 以 idx/FPS 粗估時間
    # 使用 awk 做浮點運算，避免 bash 浮點問題
    ts=$(awk -v i="$idx" -v fps="$FPS" 'BEGIN { printf "%.3f", (i-1)/fps }')
    echo -e "${vbase}\t$(realpath --relative-to="${OUT_DIR}" "$f")\t${ts}" >> "$MANIFEST"
    idx=$((idx+1))
  done
  shopt -u nullglob
}

# 列舉輸入
if [[ -f "$INPUT_PATH" ]]; then
  process_one_video "$INPUT_PATH"
elif [[ -d "$INPUT_PATH" ]]; then
  # 常見影片副檔名
  mapfile -t VIDEOS < <(find "$INPUT_PATH" -type f \( -iname "*.mp4" -o -iname "*.mov" -o -iname "*.m4v" -o -iname "*.avi" -o -iname "*.mkv" -o -iname "*.heic" -o -iname "*.heif" \) | sort)
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

echo "[Done] Frames at: ${OUT_DIR}"
echo "[Done] Manifest:  ${MANIFEST}"
