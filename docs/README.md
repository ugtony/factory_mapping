# 🏭 Factory Indoor Localization & Mapping Project

本專案旨在建立大型室內工廠的影像式定位（Image-based Localization）系統，  
結合 HLOC (Hierarchical Localization) 框架與 Wi-Fi 區域路由資訊，  
達成分區建模、多區定位、可擴充的高精度 6DoF 定位系統。

---

## 📁 資料夾結構概述

```
/factory_mapping_project/
│
├── data/ # 原始輸入資料（依 block 區域分）
│ ├── block_001/
│ │ ├── db/ # 用於建模的資料庫影像
│ │ ├── query/ # 試驗定位影像（非必要）
│ │ ├── metadata/ # Wi-Fi / 感測器資訊
│ │ └── notes/ # 拍攝紀錄與補拍說明
│ ├── block_002/
│ └── ...
│
├── outputs-hloc/ # 各 block 的 HLOC 重建結果
│ ├── block_001/
│ │ ├── global-netvlad.h5
│ │ ├── local-superpoint_aachen.h5
│ │ ├── sfm/ # pycolmap Reconstruction
│ │ ├── visualization/ # Open3D / HTML 視覺化
│ │ ├── logs/
│ │ └── debug/
│ └── ...
│
├── runtime/ # 線上定位（Runtime）模組
│ ├── router/ # Wi-Fi 區域預選
│ ├── retriever/ # Global feature retrieval (NetVLAD / DINOv2)
│ ├── localizer/ # Local feature matching + Pose estimation
│ └── run_localization.py # 整合入口
│
├── scripts/ # Bash 腳本：Offline 建模
│ ├── build_block_model.sh # 單一子圖自動建模（本文件附範例）
│ ├── rebuild_failed_blocks.sh
│ ├── merge_sfm_models.sh
│ └── visualize_sfm_open3d.py
│
├── checkpoints/ # 共用特徵模型（NetVLAD / SuperPoint / LightGlue）
│ ├── netvlad.pth
│ ├── superpoint_v1.pth
│ └── lightglue.pth
│
└── docs/
├── README.md
├── project_plan.md
├── block_status.xlsx
└── debug_report/

/factory_mapping_project/
│
├── data/ # 原始輸入資料（依 block 區域分）
│ ├── block_001/
│ │ ├── db/ # 用於建模的資料庫影像
│ │ ├── query/ # 試驗定位影像（非必要）
│ │ ├── metadata/ # Wi-Fi / 感測器資訊
│ │ └── notes/ # 拍攝紀錄與補拍說明
│ ├── block_002/
│ └── ...
│
├── outputs-hloc/ # 各 block 的 HLOC 重建結果
│ ├── block_001/
│ │ ├── global-netvlad.h5
│ │ ├── local-superpoint_aachen.h5
│ │ ├── sfm/ # pycolmap Reconstruction
│ │ ├── visualization/ # Open3D / HTML 視覺化
│ │ ├── logs/
│ │ └── debug/
│ └── ...
│
├── runtime/ # 線上定位（Runtime）模組
│ ├── router/ # Wi-Fi 區域預選
│ ├── retriever/ # Global feature retrieval (NetVLAD / DINOv2)
│ ├── localizer/ # Local feature matching + Pose estimation
│ └── run_localization.py # 整合入口
│
├── scripts/ # Bash 腳本：Offline 建模
│ ├── build_block_model.sh # 單一子圖自動建模（本文件附範例）
│ ├── rebuild_failed_blocks.sh
│ ├── merge_sfm_models.sh
│ └── visualize_sfm_open3d.py
│
├── checkpoints/ # 共用特徵模型（NetVLAD / SuperPoint / LightGlue）
│ ├── netvlad.pth
│ ├── superpoint_v1.pth
│ └── lightglue.pth
│
└── docs/
├── README.md
├── project_plan.md
├── block_status.xlsx
└── debug_report/
```


---

## 🧠 系統運作概念

### 1️⃣ Offline 建模階段（Bash）
- 每個 block（子圖）獨立執行 HLOC pipeline：
  - 提取 Global 與 Local 特徵
  - 建立 SfM 模型
  - 匹配 / 驗證 / 可視化
- 可單獨重建、刪除、偵錯，不互相干擾。

### 2️⃣ Runtime 定位階段（Python）
- 根據 Wi-Fi 指紋選出候選子圖（如 block_1, block_7, block_10）
- 在候選子圖中進行：
  - global retrieval（暫為 brute-force，未來可接 FAISS）
  - local matching（SuperPoint + LightGlue）
  - pose estimation（pycolmap PnP）
- 最終輸出最佳 6DoF 姿態。

---

## 🧱 優點與特性

| 特性 | 說明 |
|------|------|
| **分區建模** | 每個 block 可獨立生成 SfM 結果與索引 |
| **模組化設計** | runtime 與建模完全解耦，便於測試 |
| **漸進擴充** | 可隨時加入新區域影像，不需重建整體模型 |
| **除錯方便** | 每區有獨立 log / visualization / debug |
| **跨場景可延伸** | 支援多廠區、多樓層管理 |

---

## ⚙️ Pipeline 概覽

```bash
# Step 1. 建立單一區域模型
bash scripts/build_block_model.sh data/block_001

# Step 2. 在 runtime 中進行定位
python runtime/run_localization.py --query query.jpg --wifi wifi_scan.json
