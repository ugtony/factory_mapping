# HLOC & PyColmap 部署與除錯筆記

這份文件記錄了將 HLOC/COLMAP 定位流程從 Research Script (`run_localization.py`) 遷移至 Production Server (`server.py`) 時遇到的關鍵問題與解決方案。

## 1. 影像預處理的一致性 (Image Preprocessing)

在 Server 端接收 Client 上傳的圖片時，必須嚴格遵守 HLOC 訓練或建庫時的預處理邏輯，否則特徵提取 (Feature Extraction) 會產生偏差。

* **縮放 (Resizing)**: HLOC 的 `superpoint_aachen` 設定預設會將圖片長邊縮放至 `1024` px。Server 端必須實作相同的邏輯：
    ```python
    resize_max = 1024
    if max(h, w) > resize_max:
        scale = resize_max / max(h, w)
        # 務必使用 INTER_LINEAR，保持與 HLOC/OpenCV 預設一致
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    ```
* **座標還原 (Coordinate Restoration)**: 特徵點 (Keypoints) 是在縮放後的圖片上提取的。在傳入 PnP 求解器前，必須將其座標映射回原始圖片尺寸：
    ```python
    if scale != 1.0:
        kpts = (kpts + 0.5) / scale - 0.5
    ```
* **像素中心偏移 (Pixel Offset)**: COLMAP 系統假設像素中心為 `(0.5, 0.5)`，而某些深度學習模型輸出為 `(0, 0)`。在傳入 `pycolmap` 前，通常需要加上 `0.5` 的偏移量。

## 2. PyColmap PnP 求解器的陷阱

`pycolmap` 的 Python 綁定 (Bindings) 在不同版本間的回傳格式不同，且演算法行為受參數影響巨大。

* **回傳格式 (Return Type)**:
    * 新版 `pycolmap`: 回傳一個物件 (Object)，屬性為 `.success`, `.cam_from_world` 等。
    * 舊版 `pycolmap`: 回傳一個字典 (Dict)，鍵值為 `'success'`, `'qvec'`, `'tvec'`。
    * **解決方案**: 撰寫一個 helper function 同時檢查 `hasattr` 和 `isinstance(dict)`。
* **RANSAC vs Refinement**:
    * 有時 RANSAC 找到了大量的 Inliers (例如 >200)，但隨後的非線性優化 (Refinement) 因為幾何約束過強而失敗 (回傳 `success=False`)。
    * **策略**: 如果 `estimate_and_refine_absolute_pose` 失敗，應嘗試退回使用純 RANSAC 的 `estimate_absolute_pose`，或者檢查是否仍有 `cam_from_world` 數據殘留。
* **相機內參 (Intrinsics)**: 務必確保 Server 端初始化的 `pycolmap.Camera` 尺寸是圖片的**原始尺寸**，而非縮放後的尺寸。

## 3. 座標轉換 (Sim2 Transformation)

從 COLMAP 的 SfM 座標系轉換到工廠平面圖 (Map Coordinates) 是最容易出錯的環節。

* **四元數順序 (Quaternion Order)**:
    * **PyColmap / Eigen**: 內部儲存順序通常為 `[x, y, z, w]` (imaginary parts first)。
    * **COLMAP 檔案 / HLOC**: 儲存順序為 `[w, x, y, z]` (real part first)。
    * **Scipy (`Rotation.from_quat`)**: 嚴格要求輸入為 `[x, y, z, w]`。
    * **陷阱**: 如果直接將 COLMAP 格式的 `[w, x, y, z]` 餵給 Scipy，會導致旋轉計算錯誤，進而導致位移 (Translation) 計算完全錯誤。
    * **驗證**: 務必確認 `pycolmap` 回傳的 pose 物件中，quaternion 的屬性順序。通常 `cam_from_world.rotation.quat` 已經是 `[x, y, z, w]`，**不需要**手動調換順序。

* **錨點校正 (Anchors)**:
    * 用於計算 Sim2 變換矩陣的錨點圖片，必須在 SfM 重建中**存在**且**位置準確**。
    * 建議使用 `debug_anchors.py` 腳本，定期驗證錨點在當前模型中的 SfM 座標是否跑掉 (因為每次重新 Run SfM，座標系都會變動)。

## 4. 記憶體管理 (OOM Issues)

在有限資源的環境 (如 Docker 容器或邊緣裝置) 同時運行 Server 和 Debug Script 容易導致 OOM。

* **模型載入**: NetVLAD 和 SuperPoint 模型較大。Debug 時建議採用「用完即丟」策略：`load model -> extract -> del model -> gc.collect()`。
* **GPU 資源**: 如果 Server 佔用了 GPU VRAM，Debug Script 若未指定 `CUDA_VISIBLE_DEVICES` 或強制使用 CPU，可能會因記憶體爭奪而被 OS 砍掉 (Killed)。

## 5. 除錯工具建議

建立一個 `debug_tools/` 目錄，保留以下腳本以備不時之需：
1.  **`compare_pipeline.py`**: 同時執行 Server 邏輯與 HLOC 原生邏輯，針對同一張圖比對每個步驟的中間產出 (Mean RGB, Keypoint count, Match count, PnP inliers)。
2.  **`check_anchors.py`**: 讀取 `anchors.json` 與 `sfm_model`，直接計算並印出 Sim2 轉換矩陣，確認與 Server Log 是否一致。