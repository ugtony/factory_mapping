#!/bin/bash
# run_query_pipeline.sh
#
# 針對一個「已經用 build_block_model.sh 建模完成」的 block，
# 執行一組新的 query 影像的定位。
#
# [更新功能]
# 1. 支援 --fov 與 --global-conf 參數。
# 2. 自動根據 FOV 計算內參 (解決手機直/橫拍問題)。
# 3. 保留完整的視覺化輸出 (Retrieval + Matches)。
#
set -euo pipefail

# --- 0. 參數解析 ---
POSITIONAL_ARGS=()
FOV_DEG="69.4"          # 預設 FOV (iPhone 1x)
GLOBAL_CONF="netvlad"   # 預設 Global Model

while [[ $# -gt 0 ]]; do
  case $1 in
    --fov=*)
      FOV_DEG="${1#*=}"
      shift
      ;;
    --fov)
      FOV_DEG="$2"
      shift 2
      ;;
    --global-conf=*|--global_conf=*)
      GLOBAL_CONF="${1#*=}"
      shift
      ;;
    --global-conf|--global_conf)
      GLOBAL_CONF="$2"
      shift 2
      ;;
    -*|--*)
      echo "未知參數: $1"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

# 恢復位置參數
set -- "${POSITIONAL_ARGS[@]}"

if [ $# -lt 2 ]; then
  echo "用法: $0 <BLOCK_DATA_DIR> <BLOCK_OUTPUT_DIR> [--fov=FLOAT] [--global-conf=STR]"
  echo ""
  echo "參數說明:"
  echo "  <BLOCK_DATA_DIR>:   包含 'db/' 和 'query/' 影像的資料夾"
  echo "  <BLOCK_OUTPUT_DIR>: 包含 'sfm/' 和特徵 H5 檔的資料夾"
  echo "  --fov:              Query 相機的長邊視野角 (預設: 69.4)"
  echo "  --global-conf:      全域特徵模型名稱 (預設: netvlad)"
  echo ""
  exit 1
fi

# --- 1. 環境與路徑設定 ---
if [ -x "/opt/conda/bin/python" ]; then
  PY="/opt/conda/bin/python"
else
  PY="${PY:-python3}"
fi

BLOCK_DATA_DIR=$(realpath "$1")
BLOCK_OUT_DIR=$(realpath "$2")

# 檢查輸入資料夾結構
if [ ! -d "${BLOCK_DATA_DIR}/db" ]; then
  echo "[Error] 輸入的資料夾缺少 'db/' 子目錄: ${BLOCK_DATA_DIR}"
  exit 1
fi
if [ ! -d "${BLOCK_DATA_DIR}/query" ]; then
  echo "[Error] 輸入的資料夾缺少 'query/' 子目錄: ${BLOCK_DATA_DIR}"
  exit 1
fi

# 查詢結果將儲存在 block 輸出目錄下的一個新資料夾
QUERY_OUT_DIR="${BLOCK_OUT_DIR}/query_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${QUERY_OUT_DIR}"

echo "--- 執行 HLOC 查詢 Pipeline (Auto-Intrinsics) ---"
echo "影像根目錄 (Input): ${BLOCK_DATA_DIR}"
echo "Block 輸出根目錄 (Input): ${BLOCK_OUT_DIR}"
echo "查詢結果 (Output): ${QUERY_OUT_DIR}"
echo "-------------------------------------"
echo "設定 FOV: ${FOV_DEG}"
echo "設定 Global Model: ${GLOBAL_CONF}"
echo "-------------------------------------"

# --- 2. 組態設定 ---
LOCAL_CONF="superpoint_aachen"
MATCHER_CONF="superpoint+lightglue"
NUM_Q_PAIRS=10   # 每個 query 影像檢索 K 張最像的 DB 影像

# --- 3. 檔案路徑定義 ---
SFM_DIR_RAW="${BLOCK_OUT_DIR}/sfm"
SFM_DIR_ALIGNED="${BLOCK_OUT_DIR}/sfm_aligned"
DB_LIST="${BLOCK_OUT_DIR}/db.txt"
LOCAL_FEATS_H5="${BLOCK_OUT_DIR}/local-${LOCAL_CONF}.h5"
# 根據參數動態決定 Global 特徵檔名
GLOBAL_FEATS_DB_H5="${BLOCK_OUT_DIR}/global-${GLOBAL_CONF}.h5"

# 優先使用對齊後 SfM 模型
if [ -f "${SFM_DIR_ALIGNED}/images.bin" ]; then
  SFM_DIR="${SFM_DIR_ALIGNED}"
  echo "[Info] 使用對齊後 SfM 模型: ${SFM_DIR_ALIGNED}"
elif [ -f "${SFM_DIR_RAW}/images.bin" ]; then
  SFM_DIR="${SFM_DIR_RAW}"
  echo "[Info] 找不到 sfm_aligned，改用原始 SfM 模型: ${SFM_DIR_RAW}"
else
  echo "[Error] 找不到 SfM 模型 (images.bin)。"
  exit 1
fi

# 本次 query 產生的新檔案
Q_LIST_RAW="${QUERY_OUT_DIR}/query_raw.txt"
Q_LIST_INFERRED="${QUERY_OUT_DIR}/query_inferred_intrinsics.txt"
GLOBAL_FEATS_Q_H5="${QUERY_OUT_DIR}/q-${GLOBAL_CONF}.h5"
PAIRS_Q2DB="${QUERY_OUT_DIR}/pairs-q2db-retrieval.txt"
Q_MATCHES_H5="${QUERY_OUT_DIR}/q-matches-${MATCHER_CONF}.h5"
RESULTS_TXT="${QUERY_OUT_DIR}/poses.txt"
VIZ_DIR="${QUERY_OUT_DIR}/visualization"
mkdir -p "${VIZ_DIR}/retrieval" "${VIZ_DIR}/localization"

# --- 4. 檢查必要檔案 ---
if [ ! -f "${LOCAL_FEATS_H5}" ]; then
  echo "[Error] 缺少 Local 特徵檔案: ${LOCAL_FEATS_H5}"
  exit 1
fi
if [ ! -f "${GLOBAL_FEATS_DB_H5}" ]; then
  echo "[Error] 缺少 Global 特徵檔案: ${GLOBAL_FEATS_DB_H5}"
  echo "       (請確認 build_block_model.sh 是否使用了 --global=${GLOBAL_CONF})"
  exit 1
fi
if [ ! -f "${DB_LIST}" ]; then
   echo "[Error] 找不到 ${DB_LIST}。這個檔案應由 build_block_model.sh 產生。"
   exit 1
fi

# --- 5. 執行 Pipeline ---
echo "[1/8] 掃描 Query 影像 (from ${BLOCK_DATA_DIR}/query)..."
( cd "${BLOCK_DATA_DIR}" && find query -maxdepth 3 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | sort ) > "${Q_LIST_RAW}"
if [ ! -s "${Q_LIST_RAW}" ]; then
  echo "[Error] 在 ${BLOCK_DATA_DIR}/query 中找不到任何影像。中止。"
  exit 1
fi
echo "    > 找到 $(wc -l < "${Q_LIST_RAW}") 張 Query 影像。"

echo "[2/8] 擷取 Query 局部特徵 (${LOCAL_CONF})..."
${PY} -m hloc.extract_features --conf ${LOCAL_CONF} \
  --image_dir "${BLOCK_DATA_DIR}" \
  --image_list "${Q_LIST_RAW}" \
  --export_dir "${QUERY_OUT_DIR}" \
  --feature_path "${LOCAL_FEATS_H5}"

echo "[3/8] 擷取 Query 全域特徵 (${GLOBAL_CONF})..."
${PY} -m hloc.extract_features --conf ${GLOBAL_CONF} \
  --image_dir "${BLOCK_DATA_DIR}" \
  --image_list "${Q_LIST_RAW}" \
  --export_dir "${QUERY_OUT_DIR}" \
  --feature_path "${GLOBAL_FEATS_Q_H5}"

echo "[4/8] 檢索 Query-DB 配對 (Retrieval)..."
${PY} -m hloc.pairs_from_retrieval \
  --query_list "${Q_LIST_RAW}" \
  --db_list "${DB_LIST}" \
  --descriptors "${GLOBAL_FEATS_Q_H5}" \
  --db_descriptors "${GLOBAL_FEATS_DB_H5}" \
  --num_matched ${NUM_Q_PAIRS} \
  --output "${PAIRS_Q2DB}"

echo "[5/8] 匹配 Query-DB 特徵點 (${MATCHER_CONF})..."
${PY} - <<PY
from pathlib import Path
from hloc import match_features
match_features.main(
    conf=match_features.confs["${MATCHER_CONF}"],
    pairs=Path("${PAIRS_Q2DB}"),
    features=Path("${LOCAL_FEATS_H5}"),
    matches=Path("${Q_MATCHES_H5}")
)
PY

echo "[6/8] 自動生成 Query 內參 (FOV=${FOV_DEG})..."
export Q_LIST_RAW Q_LIST_INFERRED BLOCK_DATA_DIR FOV_DEG
${PY} - <<'PY'
import os
import cv2
import numpy as np
from pathlib import Path

Q_RAW     = Path(os.environ["Q_LIST_RAW"])
Q_TXT     = Path(os.environ["Q_LIST_INFERRED"])
DATA_ROOT = Path(os.environ["BLOCK_DATA_DIR"])
FOV       = float(os.environ["FOV_DEG"])

query_names = [l.strip() for l in Q_RAW.read_text().splitlines() if l.strip()]
print(f"處理 {len(query_names)} 張 query 影像待處理...")

output_lines = []
fov_rad = np.deg2rad(FOV)
model_name = "SIMPLE_PINHOLE"

for qname in query_names:
    qpath = DATA_ROOT / qname
    if not qpath.exists():
        print(f"[Warn] Query 影像不存在, 跳過: {qpath}")
        continue
        
    # 讀取圖片取得尺寸
    img = cv2.imread(str(qpath))
    if img is None:
        print(f"[Warn] 無法讀取 Query 影像, 跳過: {qpath}")
        continue
        
    h, w = img.shape[:2]
    
    # 根據長邊 FOV 計算焦距 (解決直拍/橫拍問題)
    max_side = max(w, h)
    f = 0.5 * max_side / np.tan(fov_rad / 2.0)
    cx = w / 2.0
    cy = h / 2.0
    
    # COLMAP 格式: NAME MODEL W H PARAMS...
    line = f"{qname} {model_name} {w} {h} {f:.4f} {cx} {cy}"
    output_lines.append(line)

Q_TXT.write_text('\n'.join(output_lines) + '\n')
print(f"✅ 已將 {len(output_lines)} 筆 query 內參寫入: {Q_TXT}")
PY

echo "[7/8] 執行定位 (localize_sfm)..."
( cd "${BLOCK_DATA_DIR}" && ${PY} -m hloc.localize_sfm \
  --reference_sfm "${SFM_DIR}" \
  --queries "${Q_LIST_INFERRED}" \
  --retrieval "${PAIRS_Q2DB}" \
  --features "${LOCAL_FEATS_H5}" \
  --matches "${Q_MATCHES_H5}" \
  --results "${RESULTS_TXT}" )

echo "[8/8] 產生視覺化報告..."

# 視覺化：檢索結果 (Retrieval) - 原始完整版
export DATA_ROOT="${BLOCK_DATA_DIR}"
export Q_LIST_RAW PAIRS_Q2DB VIZ_DIR
export MAX_VIZ_IMAGES_RETRIEVAL=10
${PY} - <<'PY'
import os, h5py, math
from pathlib import Path
from PIL import Image
import matplotlib; matplotlib.use("Agg")
DATA_ROOT = Path(os.environ["DATA_ROOT"])
Q_LIST_RAW = Path(os.environ["Q_LIST_RAW"])
PAIRS_Q2DB = Path(os.environ["PAIRS_Q2DB"])
VIZ_DIR = Path(os.environ["VIZ_DIR"]) / "retrieval"
VIZ_DIR.mkdir(parents=True, exist_ok=True)
from collections import defaultdict
queries = [l.strip() for l in Q_LIST_RAW.read_text().splitlines() if l.strip()]
pairs = []
with open(PAIRS_Q2DB, "r") as f:
    for line in f:
        p = line.strip().split()
        if len(p) >= 2: pairs.append((p[0], p[1]))
q2db = defaultdict(list)
for q, db in pairs: q2db[q].append(db)
def load_img(path): return Image.open(DATA_ROOT / path).convert("RGB")
MAXV = int(os.environ.get("MAX_VIZ_IMAGES_RETRIEVAL", "10"))
count = 0
for q in queries:
    if q not in q2db: continue
    dbs = q2db[q][:20]
    if not dbs: continue
    try: qimg = load_img(q)
    except Exception: continue
    W = 900
    def resize_w(im, W): w, h = im.size; return im.resize((W, int(h * (W / w))), Image.BILINEAR)
    qviz = resize_w(qimg, W); db_rows = []
    for db in dbs:
        try: db_rows.append(resize_w(load_img(db), W))
        except Exception: pass
    if not db_rows: continue
    total_h = qviz.size[1] + sum(im.size[1] for im in db_rows)
    canvas = Image.new("RGB", (W, total_h), (255,255,255))
    y = 0; canvas.paste(qviz, (0,y)); y += qviz.size[1]
    for im in db_rows: canvas.paste(im, (0,y)); y += im.size[1]
    out = VIZ_DIR / (Path(q).stem.replace('/', '_') + "_retrieval.jpg")
    canvas.save(out, quality=90); count += 1
    if count >= MAXV: break
print(f"[Retrieval Viz] 寫入 {count} 張圖片到 {VIZ_DIR}")
PY

# 視覺化：本地化匹配 (Localization) - 原始完整版
export LOCAL_FEATS="${LOCAL_FEATS_H5}"
export Q_MATCHES="${Q_MATCHES_H5}"
export MAX_VIZ_IMAGES_LOCALIZATION=5
${PY} - <<'PY'
import os, h5py, numpy as np, random
from pathlib import Path
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
DATA_ROOT = Path(os.environ["DATA_ROOT"])
Q_LIST_RAW = Path(os.environ["Q_LIST_RAW"])
PAIRS_Q2DB = Path(os.environ["PAIRS_Q2DB"])
LOCAL_FEATS = Path(os.environ["LOCAL_FEATS"])
Q_MATCHES = Path(os.environ["Q_MATCHES"])
VIZ_DIR = Path(os.environ["VIZ_DIR"]) / "localization"
VIZ_DIR.mkdir(parents=True, exist_ok=True)
MAXV = int(os.environ.get("MAX_VIZ_IMAGES_LOCALIZATION", "5"))
queries = [l.strip() for l in Q_LIST_RAW.read_text().splitlines() if l.strip()]
q2db_top1 = {}
with open(PAIRS_Q2DB, "r") as f:
    for line in f:
        p = line.strip().split()
        if len(p) >= 2:
            q, db = p[0], p[1]
            if q not in q2db_top1: q2db_top1[q] = db

def draw_matches(q, db, pts_q, pts_db, out_path, original_match_count):
    im_q = np.array(Image.open(DATA_ROOT / q).convert("RGB"))
    im_db = np.array(Image.open(DATA_ROOT / db).convert("RGB"))
    H = max(im_q.shape[0], im_db.shape[0])
    canvas = np.ones((H, im_q.shape[1] + im_db.shape[1], 3), dtype=np.uint8) * 255
    canvas[:im_q.shape[0], :im_q.shape[1]] = im_q
    canvas[:im_db.shape[0], im_q.shape[1]:] = im_db
    fig = plt.figure(figsize=(12,6)); ax = fig.add_subplot(1,1,1)
    ax.imshow(canvas); ax.axis('off'); shift_x = im_q.shape[1]
    
    num_matches_sampled = len(pts_q)
    
    if num_matches_sampled > 0:
        line_width = 0.5 
        circle_size = 5    
        line_alpha = 0.6
        
        N = num_matches_sampled
        colors = plt.cm.get_cmap('turbo', N)(np.linspace(0, 1, N))
        
        for i in range(N):
            color = colors[i]
            (x1, y1) = pts_q[i]
            (x2, y2) = pts_db[i]
            
            ax.plot([x1, x2 + shift_x], [y1, y2], 
                    linewidth=line_width, color=color, alpha=line_alpha)
            
            ax.scatter(x1, y1, 
                       s=circle_size, color=color, marker='o', alpha=line_alpha + 0.2)
                       
            ax.scatter(x2 + shift_x, y2, 
                       s=circle_size, color=color, marker='o', alpha=line_alpha + 0.2)

    ax.set_title(f"{q} ↔ {db} (matches={original_match_count})")
    fig.savefig(out_path, bbox_inches='tight', dpi=160); plt.close(fig)

with h5py.File(str(LOCAL_FEATS), 'r') as ffeat, h5py.File(str(Q_MATCHES), 'r') as fmat:
    matches_nodes = []
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and name.endswith("/matches0"):
            matches_nodes.append(name)
    fmat.visititems(lambda n, o: visit(n, o))
    print(f"[Info] 找到 {len(matches_nodes)} 個 'matches0' datasets")
    done = 0
    for q in queries:
        if q not in q2db_top1: continue
        db = q2db_top1[q]
        if q not in ffeat or db not in ffeat: continue
        found = None
        cand_q = [q, q.replace('/', '-'), q.replace('/', '_')]
        cand_db= [db, db.replace('/', '-'), db.replace('/', '_')]
        for node in matches_nodes:
            if any(c in node for c in cand_q) and any(c in node for c in cand_db):
                found = node; break
        if found is None:
            q_short = q.split('/')[-1]; db_short= db.split('/')[-1]
            for node in matches_nodes:
                if q_short in node and db_short in node: found = node; break
        if found is None:
            print(f"[Skip] 找不到 ({q}, {db}) 的 matches0 dataset")
            continue
        
        m0 = np.array(fmat[found])
        kpts_q = np.array(ffeat[q]['keypoints']); kpts_db= np.array(ffeat[db]['keypoints'])
        
        valid = m0 > -1
        if valid.sum() == 0: pts_q = np.empty((0,2)); pts_db = np.empty((0,2))
        else: 
            idx_q = np.where(valid)[0]
            idx_db = m0[valid]
            pts_q = kpts_q[idx_q][:, :2]
            pts_db = kpts_db[idx_db][:, :2]
        
        MAX_LINES_TO_DRAW = 200
        num_matches_original = len(pts_q)
        
        if num_matches_original > MAX_LINES_TO_DRAW:
            indices = random.sample(range(num_matches_original), MAX_LINES_TO_DRAW)
            pts_q = pts_q[indices] 
            pts_db = pts_db[indices]
            
        out = VIZ_DIR / (Path(q).stem.replace('/', '_') + "_matches.jpg")
        try: 
            draw_matches(q, db, pts_q, pts_db, out, num_matches_original)
            done += 1
        except Exception as e: 
            print(f"[Warn] 繪製 {q} 失敗: {e}")
            
        if done >= MAXV: break
print(f"[Localization Viz] 寫入 {done} 張圖片到 {VIZ_DIR}")
PY

# 視覺化：HTML 匯出
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIZ_PY_SCRIPT="${SCRIPT_DIR}/visualize_sfm_open3d.py"
if [ -f "${VIZ_PY_SCRIPT}" ]; then
  echo "匯出互動式 HTML (包含 query 相機)..."
  ${PY} "${VIZ_PY_SCRIPT}" \
    --sfm_dir "${SFM_DIR}" \
    --output_dir "${VIZ_DIR}" \
    --query_poses "${RESULTS_TXT}" \
    --no_server
else
  echo "[Warn] Noy ${VIZ_PY_SCRIPT}。跳過 HTML 視覺化。"
fi

# --- 最終總結 ---
set +x
echo ""
echo "✅ 查詢 Pipeline 執行完畢。"
echo "所有查詢結果位於: ${QUERY_OUT_DIR}"
echo ""
if [ -f "${RESULTS_TXT}" ]; then
  echo "--- 定位姿態結果 (${RESULTS_TXT}) ---"
  cat "${RESULTS_TXT}"
else
  echo "定位失敗。沒有姿態結果產生。"
fi