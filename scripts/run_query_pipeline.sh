#!/bin/bash
# run_query_pipeline.sh
#
# 針對一個「已經用 build_block_model.sh 建模完成」的 block，
# 執行一組新的 query 影像的定位。
#
# 它會重用 DB 特徵和 SfM 模型，只對 query 影像進行特徵提取、
# 匹配和 hloc.localize_sfm。
#
set -euo pipefail
set -x

if [ $# -lt 2 ]; then
  echo "用法: $0 <BLOCK_DATA_DIR> <BLOCK_OUTPUT_DIR>"
  echo ""
  echo "參數說明:"
  echo "  <BLOCK_DATA_DIR>:   包含 'db/' 和 'query/' 影像的資料夾 (例如: data/block_001)"
  echo "  <BLOCK_OUTPUT_DIR>: 包含 'sfm/' 和特徵 H5 檔的資料夾 (例如: outputs-hloc/block_001)"
  echo ""
  echo "範例: bash scripts/run_query_pipeline.sh data/block_001 outputs-hloc/block_001"
  exit 1
fi

# --- 1. 路徑設定 ---
BLOCK_DATA_DIR=$(realpath "$1") # e.g., data/block_001 (包含 db/ 和 query/)
BLOCK_OUT_DIR=$(realpath "$2")  # e.g., outputs-hloc/block_001 (包含 sfm/ 和 *.h5)

# 檢查輸入資料夾結構
if [ ! -d "${BLOCK_DATA_DIR}/db" ]; then
  echo "[Error] 輸入的資料夾缺少 'db/' 子目錄: ${BLOCK_DATA_DIR}"
  exit 1
fi
if [ ! -d "${BLOCK_DATA_DIR}/query" ]; then
  echo "[Error] 輸入的資料夾缺少 'query/' 子目錄: ${BLOCK_DATA_DIR}"
  exit 1
fi

# 查詢結果將儲存在 block 輸出目錄下的一個新資料夾，避免覆蓋
QUERY_OUT_DIR="${BLOCK_OUT_DIR}/query_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${QUERY_OUT_DIR}"

echo "--- 執行 HLOC 查詢 Pipeline ---"
echo "DB/SfM 模型 (Input):  ${BLOCK_OUT_DIR}"
echo "影像根目錄 (Input): ${BLOCK_DATA_DIR} (使用此處的 db/ 和 query/)"
echo "查詢結果 (Output): ${QUERY_OUT_DIR}"
echo "-------------------------------------"

# --- 2. 組態設定 (必須與 build_block_model.sh 一致) ---
GLOBAL_CONF="netvlad"
LOCAL_CONF="superpoint_aachen"
MATCHER_CONF="superpoint+lightglue"
NUM_Q_PAIRS=10   # 每個 query 影像要檢索 K 張最像的 DB 影像

# --- 3. 檔案路徑定義 ---

# [EXISTING] 指向 build_block_model.sh 已產生的檔案
SFM_DIR="${BLOCK_OUT_DIR}/sfm"
DB_LIST="${BLOCK_OUT_DIR}/db.txt" # build_block_model.sh 產生的 DB 清單
LOCAL_FEATS_H5="${BLOCK_OUT_DIR}/local-${LOCAL_CONF}.h5"
GLOBAL_FEATS_DB_H5="${BLOCK_OUT_DIR}/global-${GLOBAL_CONF}.h5"

# [NEW] 此腳本為 query 產生的新檔案
Q_LIST_RAW="${QUERY_OUT_DIR}/query_raw.txt"
Q_LIST_INFERRED="${QUERY_OUT_DIR}/query_inferred_intrinsics.txt"
GLOBAL_FEATS_Q_H5="${QUERY_OUT_DIR}/q-${GLOBAL_CONF}.h5"
PAIRS_Q2DB="${QUERY_OUT_DIR}/pairs-q2db-retrieval.txt"
Q_MATCHES_H5="${QUERY_OUT_DIR}/q-matches-${MATCHER_CONF}.h5"
RESULTS_TXT="${QUERY_OUT_DIR}/poses.txt"
VIZ_DIR="${QUERY_OUT_DIR}/visualization"
mkdir -p "${VIZ_DIR}/retrieval"
mkdir -p "${VIZ_DIR}/localization"

# --- 4. 檢查 DB 所需檔案是否存在 ---
if [ ! -f "${LOCAL_FEATS_H5}" ] || [ ! -f "${GLOBAL_FEATS_DB_H5}" ] || [ ! -f "${SFM_DIR}/images.bin" ]; then
  echo "[Error] 指定的 Block 輸出目錄缺少必要檔案:"
  echo "  應有 SfM model: ${SFM_DIR}/images.bin"
  echo "  應有 Local Feats: ${LOCAL_FEATS_H5}"
  echo "  應有 Global Feats: ${GLOBAL_FEATS_DB_H5}"
  echo "請先執行 'scripts/build_block_model.sh' 來產生模型。"
  exit 1
fi
if [ ! -f "${DB_LIST}" ]; then
   echo "[Error] 找不到 ${DB_LIST}。這個檔案應由 build_block_model.sh 產生。"
   exit 1
fi

# --- 5. 執行 Query Pipeline (擷取自 HLOC 完整流程) ---

echo "[1/8] 掃描 Query 影像 (from ${BLOCK_DATA_DIR}/query)..."
# HLOC 期望的路徑是相對於 --image_dir (即 BLOCK_DATA_DIR)
# 所以清單中的路徑會是 "query/IMG_001.jpg"
(cd "${BLOCK_DATA_DIR}" && find query -maxdepth 3 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | sort) > "${Q_LIST_RAW}"
if [ ! -s "${Q_LIST_RAW}" ]; then
  echo "[Error] 在 ${BLOCK_DATA_DIR}/query 中找不到任何影像。中止。"
  exit 1
fi
echo "    > 找到 $(wc -l < "${Q_LIST_RAW}") 張 Query 影像。"

echo "[2/8] 擷取 Query 的局部特徵 (${LOCAL_CONF})..."
# HLOC 會自動將 query 特徵「附加」到現有的 H5 檔案中
# 我們使用 --image_dir "${BLOCK_DATA_DIR}"，因為 Q_LIST_RAW 中的路徑是相對於它
python3 -m hloc.extract_features --conf ${LOCAL_CONF} \
  --image_dir "${BLOCK_DATA_DIR}" \
  --image_list "${Q_LIST_RAW}" \
  --export_dir "${QUERY_OUT_DIR}" \
  --feature_path "${LOCAL_FEATS_H5}"

echo "[3/8] 擷取 Query 的全域特徵 (${GLOBAL_CONF})..."
# Query 的 global features 應獨立儲存
python3 -m hloc.extract_features --conf ${GLOBAL_CONF} \
  --image_dir "${BLOCK_DATA_DIR}" \
  --image_list "${Q_LIST_RAW}" \
  --export_dir "${QUERY_OUT_DIR}" \
  --feature_path "${GLOBAL_FEATS_Q_H5}"

echo "[4/8] 檢索 Query-DB 配對 (Retrieval)..."
# 比較 Q (GLOBAL_FEATS_Q_H5) 與 DB (GLOBAL_FEATS_DB_H5)
python3 -m hloc.pairs_from_retrieval \
  --query_list "${Q_LIST_RAW}" \
  --db_list "${DB_LIST}" \
  --descriptors "${GLOBAL_FEATS_Q_H5}" \
  --db_descriptors "${GLOBAL_FEATS_DB_H5}" \
  --num_matched ${NUM_Q_PAIRS} \
  --output "${PAIRS_Q2DB}"

echo "[5/8] 匹配 Query-DB 特徵點 (${MATCHER_CONF})..."
python3 - <<PY
from pathlib import Path
from hloc import match_features
match_features.main(
    conf=match_features.confs["${MATCHER_CONF}"],
    pairs=Path("${PAIRS_Q2DB}"),
    features=Path("${LOCAL_FEATS_H5}"),
    matches=Path("${Q_MATCHES_H5}")
)
PY

echo "[6/8] 從 SfM 推斷 Query 影像的相機內參..."
# 這裡的 DATA_ROOT 必須是 BLOCK_DATA_DIR，因為 Q_LIST_RAW 的路徑是相對於它
export SFM_DIR Q_LIST_RAW Q_LIST_INFERRED BLOCK_DATA_DIR 
python3 - <<'PY'
import os
from pathlib import Path
import pycolmap, cv2
# 從環境變數讀取路徑
SFM_DIR   = Path(os.environ["SFM_DIR"])
Q_RAW     = Path(os.environ["Q_LIST_RAW"])
Q_TXT     = Path(os.environ["Q_LIST_INFERRED"])
DATA_ROOT = Path(os.environ["BLOCK_DATA_DIR"]) # 影像根目錄

rec = pycolmap.Reconstruction(SFM_DIR)
cams = list(rec.cameras.values())
if not cams:
    print(f"[Error] SfM 模型 {SFM_DIR} 中沒有找到相機。")
    exit(1)

# 幫相機尺寸找到最接近的相機模型的函數
def get_best_cam_model(qw, qh, all_cams):
    def score(cam): return abs(cam.width - qw) + abs(cam.height - qh)
    cam = min(all_cams, key=score) # 找出尺寸最接近的
    try: model_name = cam.model.name
    except Exception: from pycolmap import CameraModel; model_name = CameraModel.name(cam.model_id)
    params_str = " ".join(map(str, cam.params))
    return cam, model_name, params_str

query_names = [l.strip() for l in Q_RAW.read_text().splitlines() if l.strip()]
print(f"找到 {len(query_names)} 張 query 影像待處理...")
output_lines = []

for qname in query_names:
    qpath = DATA_ROOT / qname # 組合出完整影像路徑
    if not qpath.exists():
        print(f"[Warn] Query 影像不存在, 跳過: {qpath}")
        continue
    img = cv2.imread(str(qpath))
    if img is None:
        print(f"[Warn] 無法讀取 Query 影像, 跳過: {qpath}")
        continue
    
    qh, qw = img.shape[:2]
    # 為這張影像的尺寸找到最佳的相機內參
    cam, model_name, params_str = get_best_cam_model(qw, qh, cams)
    line = f"{qname} {model_name} {cam.width} {cam.height} {params_str}"
    output_lines.append(line)
    if (cam.width, cam.height) != (qw, qh):
        print(f"[info] Query '{qname}' ({qw}x{qh}) 使用 SfM cam ({cam.width}x{cam.height})")

Q_TXT.write_text('\n'.join(output_lines) + '\n')
print(f"✅ 已將 {len(output_lines)} 筆 query 內參寫入: {Q_TXT}")
PY

# --- 💥 步驟 7 修正 💥 ---
echo "[7/8] 執行定位 (localize_sfm)..."
# 我們必須 cd 到影像根目錄 (BLOCK_DATA_DIR)，
# 因為 hloc.localize_sfm 會根據 query list (e.g., "query/IMG_1397.jpg")
# 從「當前工作目錄」去尋找影像。
# 由於所有其他路徑都已是絕對路徑 (realpath)，所以這樣做是安全的。
(cd "${BLOCK_DATA_DIR}" && python3 -m hloc.localize_sfm \
  --reference_sfm "${SFM_DIR}" \
  --queries "${Q_LIST_INFERRED}" \
  --retrieval "${PAIRS_Q2DB}" \
  --features "${LOCAL_FEATS_H5}" \
  --matches "${Q_MATCHES_H5}" \
  --results "${RESULTS_TXT}")
# --- 💥 修正結束 💥 ---

echo "[8/8] 產生視覺化報告..."

# 視覺化：檢索結果 (Retrieval)
# DATA_ROOT 設為 BLOCK_DATA_DIR，因為 q 和 db 路徑都是相對於它
export DATA_ROOT="${BLOCK_DATA_DIR}"
export Q_LIST_RAW PAIRS_Q2DB VIZ_DIR
export MAX_VIZ_IMAGES_RETRIEVAL=10
python3 - <<'PY'
# (此 Python 腳本區塊與您範本中的 Step 11 完全相同)
import os, io, h5py, math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")  # headless
from matplotlib import pyplot as plt
DATA_ROOT = Path(os.environ["DATA_ROOT"])
Q_LIST_RAW = Path(os.environ["Q_LIST_RAW"])
PAIRS_Q2DB = Path(os.environ["PAIRS_Q2DB"])
VIZ_DIR = Path(os.environ["VIZ_DIR"]) / "retrieval"
VIZ_DIR.mkdir(parents=True, exist_ok=True)
MAXV = int(os.environ.get("MAX_VIZ_IMAGES_RETRIEVAL", "10"))
from collections import defaultdict
queries = [l.strip() for l in Q_LIST_RAW.read_text().splitlines() if l.strip()]
pairs = []
with open(PAIRS_Q2DB, "r") as f:
    for line in f:
        p = line.strip().split();
        if len(p) >= 2: pairs.append((p[0], p[1]))
q2db = defaultdict(list)
for q, db in pairs: q2db[q].append(db)
def load_img(path): return Image.open(DATA_ROOT / path).convert("RGB")
count = 0
for q in queries:
    if q not in q2db: continue
    dbs = q2db[q][:20]
    if len(dbs) == 0: continue
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

# 視覺化：本地化匹配 (Localization)
# 這裡也一樣, DATA_ROOT 是 BLOCK_DATA_DIR
export LOCAL_FEATS="${LOCAL_FEATS_H5}" # 確保使用正確的 H5
export Q_MATCHES="${Q_MATCHES_H5}"
export MAX_VIZ_IMAGES_LOCALIZATION=5
python3 - <<'PY'
# (此 Python 腳本區塊與您範本中的 Step 12 基本相同)
import os, h5py, numpy as np
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")
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
        p = line.strip().split();
        if len(p) >= 2: q, db = p[0], p[1];
        if q not in q2db_top1: q2db_top1[q] = db
def draw_matches(q, db, pts_q, pts_db, out_path):
    im_q = np.array(Image.open(DATA_ROOT / q).convert("RGB"))
    im_db = np.array(Image.open(DATA_ROOT / db).convert("RGB"))
    H = max(im_q.shape[0], im_db.shape[0])
    canvas = np.ones((H, im_q.shape[1] + im_db.shape[1], 3), dtype=np.uint8) * 255
    canvas[:im_q.shape[0], :im_q.shape[1]] = im_q
    canvas[:im_db.shape[0], im_q.shape[1]:] = im_db
    fig = plt.figure(figsize=(12,6)); ax = fig.add_subplot(1,1,1)
    ax.imshow(canvas); ax.axis('off'); shift_x = im_q.shape[1]
    for (x1,y1),(x2,y2) in zip(pts_q, pts_db):
        ax.plot([x1, x2+shift_x], [y1, y2], linewidth=0.5)
    ax.set_title(f"{q} ↔ {db} (matches={len(pts_q)})")
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
        else: idx_q = np.where(valid)[0]; idx_db = m0[valid]; pts_q = kpts_q[idx_q][:, :2]; pts_db = kpts_db[idx_db][:, :2]
        out = VIZ_DIR / (Path(q).stem.replace('/', '_') + "_matches.jpg")
        try: draw_matches(q, db, pts_q, pts_db, out); done += 1
        except Exception as e: print(f"[Warn] 繪製 {q} 失敗: {e}")
        if done >= MAXV: break
print(f"[Localization Viz] 寫入 {done} 張圖片到 {VIZ_DIR}")
PY

# 視覺化：HTML 匯出
# 假設 visualize_sfm_open3d.py 在這個腳本的同一層目錄 (scripts/)
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIZ_PY_SCRIPT="${SCRIPT_DIR}/visualize_sfm_open3d.py"

if [ -f "${VIZ_PY_SCRIPT}" ]; then
  echo "匯出互動式 HTML (包含 query 相機)..."
  python3 "${VIZ_PY_SCRIPT}" \
    --sfm_dir "${SFM_DIR}" \
    --output_dir "${VIZ_DIR}" \
    --query_poses "${RESULTS_TXT}" \
    --no_server # 只產生檔案，不啟動伺服器
else
  echo "[Warn] 找不到 ${VIZ_PY_SCRIPT}。跳過 HTML 視覺化。"
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