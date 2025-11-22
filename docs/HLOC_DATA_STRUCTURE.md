# HLOC Data Structure Reference

此文件記錄 HLOC (Hierarchical Localization) 產生的關鍵檔案結構，用於避免常見的讀取錯誤。

## 1. 特徵檔案 (`.h5` Features)
包含 Global (NetVLAD/Megaloc) 或 Local (SuperPoint) 特徵。

### 結構特徵
* **Group/Dataset**: 通常是層級式結構，Key 為影像檔名 (相對路徑)。
* **Global Features**:
    * Key: `global_descriptor`
    * Shape: `(D,)` 或 `(1, D)`。
    * **⚠️ 常見陷阱**: 當只有 1 張 Query 時，`np.array()` 讀出來可能是 `(1, D)`，若隨意使用 `.squeeze()` 可能會變成 1D 向量 `(D,)`，導致後續矩陣運算 (dot product) 維度錯誤。
    * **正確讀法**: 始終確保轉型為 `(N, D)` 2D 陣列。

## 2. 匹配檔案 (`.h5` Matches)
包含兩張影像之間的特徵匹配索引。

### 結構特徵
* **Key Naming (最容易出錯處)**: HLOC 會將影像路徑中的特殊字元編碼。
    * 原始路徑: `db/image.jpg`
    * H5 Key 可能為: `db-image.jpg` (斜線 `/` 被換成 `-`)
    * 或是層級式: `query_img/db_img`
* **Datasets**:
    * `matches0`: Shape `(N,)`，存的是對應點的 Index (-1 代表無匹配)。
    * `matching_scores0`: Shape `(N,)`，匹配信心分數。

### 讀取策略
不要只用字串 `in` 搜尋。應實作「邊界感知」搜尋，同時嘗試原始路徑 (`/`) 與編碼路徑 (`-`)。

## 3. 定位 Log 檔 (`.pkl` Logs)
`localize_sfm` 輸出的詳細 Log。

### 結構特徵
這不是一個扁平的 Dictionary，而是有特定 Root Key。

```python
{
    "features": Path(...),
    "matches": Path(...),
    "retrieval": Path(...),
    "loc": {  # <--- 關鍵資料在這裡
        "query_image_name.jpg": {
            "PnP_ret": {
                "success": bool,
                "num_inliers": int,  # 我們要找的分數
                "qvec": [...],
                "tvec": [...]
            }
        },
        ...
    }
}
```
⚠️ 常見陷阱: 直接 iterate root keys 會讀到 features 路徑而不是結果。必須先進入 logs['loc']。