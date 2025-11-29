# 🏭 Factory Indoor Localization & Mapping System

本專案是一套針對大型室內工廠環境設計的影像式定位系統（Visual Localization）。
基於 **HLOC (Hierarchical Localization)** 框架，並針對工廠環境的特殊性（相似場景多、360 全景影像、多區域管理）進行了深度優化與架構重構。

---

## 🌟 核心特色 (Key Features)

* **邏輯統一 (Unified Architecture)**
    * **Offline (測試)** 與 **Online (伺服器)** 共用同一套核心引擎 `localization_engine.py`。
    * 徹底解決了「離線測試準確，上線卻定位失敗」的常見問題 (如 Resize 策略、座標還原精度、PnP 參數一致性)。
    * 確保 "所測即所得" (What you test is what you deploy)。

* **穩健的多視角定位 (Robust Multi-View PnP)**
    * 採用 **Top-K Retrieval + Match Aggregation** 策略。
    * 即使單張匹配特徵不足，也能透過聚合多張視角 (Top-K) 的幾何約束來達成高精度定位。
    * 顯著提升 Inliers 數量與抗遮擋能力。

* **360° 全景支援**
    * 內建 Equirectangular 到 Pinhole 的轉換工具。
    * 支援 Dense (8視角) 或 Sparse (4視角) 建模模式。

* **自動化與可配置**
    * **Auto-Intrinsics**：自動根據輸入影像的 FOV 計算內參（解決手機直拍/橫拍問題）。
    * **Configurable**：透過 `.env` 檔統一管理全域參數。

---

## 📁 建議資料夾結構

```plaintext
/factory_mapping_project/
│
├── project_config.env          # [Config] 全域設定檔 (FOV, Global Model 等)
│
├── data/                       # [Data] 原始影像資料
│   ├── block_A/
│   │   ├── raw/                # (選用) 原始影片檔
│   │   ├── db/                 # 建模用的 Pinhole 影像
│   │   └── query/              # 測試用的查詢影像
│   └── ...
│
├── outputs-hloc/               # [Model] HLOC 產出的模型與特徵檔
│   ├── block_A/
│   │   ├── sfm_aligned/        # 轉正後的 COLMAP 模型
│   │   ├── global-netvlad.h5   # 全域特徵
│   │   └── local-superpoint... # 局部特徵
│   └── ...
│
├── scripts/                    # [Code] 核心腳本
│   ├── localization_engine.py  # [Core] 核心定位引擎 (所有邏輯的真理來源)
│   ├── server.py               # [Online] FastAPI 伺服器 (呼叫 Engine)
│   ├── run_localization.py     # [Offline] 批次測試腳本 (呼叫 Engine)
│   ├── client.py               # [Tool] 測試 Server API 的客戶端
│   ├── build_block_model.sh    # [Build] 單一區塊建模腳本
│   ├── convert_poses_to_map.py # [Tool] 座標轉換工具
│   └── ...
│
└── docs/
    ├── HLOC_DEPLOYMENT_NOTES.md # 技術細節與除錯筆記
    └── anchors.json            # 座標轉換用的錨點設定
```

---

## ⚙️ 環境設定 (Configuration)

請在專案根目錄建立 `project_config.env`，所有腳本都會自動讀取此檔案：

# project_config.env 範例
MODE="360"           # 攝影機模式: std 或 360
DENSE=1              # 360模式: 1=8視角, 0=4視角
GLOBAL_CONF="netvlad"# 全域特徵模型
FOV=100.0            # Query 相機或 360 拆圖後的水平視角

---

## 🚀 使用流程 (Workflow)

### Step 0. 前處理 (360 轉 Pinhole)
如果您使用的是 Insta360 等全景相機，需先轉換為多視角平面圖。
(若目錄下有 `raw/` 影片檔，build_block_model.sh 亦會嘗試自動抽幀)

python scripts/convert360_to_pinhole.py \
  --input_dir data/block_A/db_360 \
  --output_dir data/block_A/db \
  --dense --fov 100

### Step 1. 建立區域模型 (Modeling)
對單一區域進行特徵提取與 SfM 重建。

# 自動讀取 env 設定，建立 block_A 模型
bash scripts/build_block_model.sh data/block_A

### Step 2. 離線測試 (Offline Testing)
使用與 Server 完全相同的邏輯進行批次測試與視覺化驗證。
這是開發階段最重要的步驟。

# 指定 Query 資料夾與模型根目錄
python scripts/run_localization.py \
  --query_dir data/block_A/query \
  --reference outputs-hloc/ \
  --viz  # (選用) 輸出匹配視覺化圖至 viz_offline/

### Step 3. 啟動服務 (Online Server)
啟動 FastAPI 服務，載入所有模型至記憶體。

python scripts/server.py
# 服務預設跑在 Port 8000

### Step 4. 呼叫服務 (Client Request)
測試 API 回傳結果。

python scripts/client.py

### Step 5. 座標轉換 (Map Conversion)
將定位輸出的 6DoF 姿態轉換到工廠平面圖座標 (需準備 anchors.json)。

python scripts/convert_poses_to_map.py \
  --submission offline_results.txt \
  --anchors docs/anchors.json \
  --plot