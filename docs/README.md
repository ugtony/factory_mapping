# 🏭 Factory Indoor Localization & Mapping System

本專案是一套針對大型室內工廠環境設計的影像式定位系統（Visual Localization）。
基於 **HLOC (Hierarchical Localization)** 框架，並針對工廠環境的特殊性（相似場景多、360 全景影像、多區域管理）進行了深度優化。

---

## 🌟 核心特色 (Key Features)

* **360° 全景支援**：內建 Equirectangular 到 Pinhole 的轉換工具，支援 4 視角 (Sparse) 或 8 視角 (Dense) 建模。
* **穩健的多區域定位 (Robust Multi-Block)**：
    * 採用 **Top-K Block Candidate** 策略，結合 **MegaLoc/NetVLAD** 全域檢索。
    * 引入 **幾何驗證 (Geometric Verification)** 機制，比較 PnP Inliers 數量來決定最佳區域，有效解決工廠內部的視覺混淆 (Visual Aliasing) 問題。
* **自動化管線**：
    * **Auto-Intrinsics**：自動根據輸入影像的 FOV 計算內參（解決手機直拍/橫拍問題）。
    * **Configurable**：透過 `.env` 檔統一管理全域參數。
* **完整的視覺化除錯**：
    * 支援 Open3D 互動式 3D 場景。
    * 自動生成 Retrieval 與 Matching 連線圖，並具備防呆機制（避免幽靈檔案）。
* **地圖座標整合**：內建工具可將 HLOC 的局部座標轉換為全廠區的統一 2D 地圖座標。

---

## 📁 建議資料夾結構

```text
/factory_mapping_project/
│
├── project_config.env          # [New] 全域設定檔 (FOV, Global Model 等)
│
├── data/                       # 原始影像資料
│   ├── block_A/
│   │   ├── db_360/             # (選用) 原始 360 全景圖
│   │   ├── db/                 # 建模用的 Pinhole 影像
│   │   └── query/              # 測試用的查詢影像
│   └── ...
│
├── outputs-hloc/               # HLOC 產出的模型與特徵檔
│   ├── block_A/
│   │   ├── sfm_aligned/        # 轉正後的 COLMAP 模型
│   │   ├── global-netvlad.h5   # 全域特徵
│   │   └── local-superpoint... # 局部特徵
│   └── ...
│
├── scripts/                    # 核心腳本
│   ├── hloc_io_utils.py        # [Core] HLOC 檔案讀取模組 (必須)
│   ├── run_localization.py     # [Core] 統一的定位入口 (支援 Single/Multi)
│   ├── build_block_model.sh    # [Core] 單一區塊建模腳本
│   ├── convert_poses_to_map.py # [Tool] 座標轉換工具
│   ├── convert360_to_pinhole.py
│   └── visualize_sfm_open3d.py
│
└── docs/
    └── anchors.json            # 座標轉換用的錨點設定
```

## ⚙️ 環境設定 (Configuration)
請在專案根目錄建立 project_config.env，所有腳本都會自動讀取此檔案：

```bash
# project_config.env
# 攝影機模式: std (一般手機) 或 360 (全景相機)
MODE="360"

# 全域特徵模型: netvlad, megaloc, dino_v2
GLOBAL_CONF="netvlad"

# Query 相機或 360 拆圖後的水平視角 (FOV)
FOV=100.0
```

## 🚀 使用流程 (Workflow)
### Step 0. 前處理 (360 轉 Pinhole)
如果您使用的是 Insta360 等全景相機，需先轉換為多視角平面圖：

```bash
# 將 db_360 內的圖片轉換到 db 資料夾 (8視角 Dense 模式)
python scripts/convert360_to_pinhole.py \
  --input_dir data/block_A/db_360 \
  --output_dir data/block_A/db \
  --dense --fov 100
```

### Step 1. 建立區域模型 (Modeling)
對單一區域進行特徵提取與 SfM 重建：

```bash
# 自動讀取 env 設定，建立 block_A 模型
bash scripts/build_block_model.sh data/block_A
```

### Step 2. 執行定位 (Localization)
這是本系統最強大的部分，支援兩種模式。

#### 模式 A：單一區塊測試 (Single Block)

```bash
python scripts/run_localization.py \
  --query_dir data/block_A/query \
  --reference outputs-hloc/block_A \
  --viz_3d --viz_matches
```

#### 模式 B：多區塊自動定位 (Multi-Block) 
系統會自動掃描 outputs-hloc 下的所有區塊，利用 Global Retrieval 找出最可能的 Top-K (預設 3) 個區域，並透過特徵匹配數量 (Inliers) 決定最終勝出者。

```bash
python scripts/run_localization.py \
  --query_dir data/unknown_queries \
  --reference outputs-hloc/ \
  --top_k 3 \
  --viz_retrieval
```

### Step 3. 座標轉換 (Map Conversion)
將 HLOC 輸出的 6DoF 姿態轉換到工廠平面圖座標 (需準備 anchors.json)：

```bash
python scripts/convert_poses_to_map.py \
  --submission data/block_A/query_processed_netvlad/final_poses.txt \
  --anchors docs/anchors.json \
  --plot
```