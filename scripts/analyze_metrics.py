import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

# ==============================================================================
# [Core Function] Ground Truth Check
# 用途: 判斷定位結果是否與真實情況相符 (Ground Truth)
# ==============================================================================
def check_ground_truth(image_name, selected_block):
    """
    輸入:
      image_name (str): 查詢影像的檔名 (e.g., "brazil360_frame_00012.jpg")
      selected_block (str): 系統定位出的區塊名稱 (e.g., "brazil360")
    
    輸出:
      True: 表示定位結果正確
      False: 表示定位結果錯誤
    """
    # [防呆] 確保輸入是字串
    if not isinstance(image_name, str) or not isinstance(selected_block, str):
        return False
    
    # -----------------------------------------------------------
    # [策略 1] 字串包含法 (預設)
    # 假設: 測試影像的檔名中已經標記了正確的區塊名稱
    # 例如: 若檔名是 "brazil_360_query_01.jpg"，只要 selected_block 是 "brazil_360" 就算對
    # -----------------------------------------------------------
    if selected_block in image_name:
        return True
        
    # -----------------------------------------------------------
    # [策略 2] 外部對照表 (範例，若有需要可啟用)
    # -----------------------------------------------------------
    # gt_map = {
    #     "frame_001.jpg": "block_A",
    #     "frame_002.jpg": "block_B"
    # }
    # if gt_map.get(image_name) == selected_block:
    #     return True
    
    return False

# ==============================================================================
# Analysis Logic
# ==============================================================================
def analyze(csv_path):
    # 讀取資料
    df = pd.read_csv(csv_path)
    print(f"原始資料筆數: {len(df)}")

    # 1. 資料清洗與填補
    df['PnP_Inliers'] = df['PnP_Inliers'].fillna(0)
    df['Second_Inliers'] = df['Second_Inliers'].fillna(0)
    df['Top1_Score'] = df['Top1_Score'].fillna(0)
    df['Top2_Score'] = df['Top2_Score'].fillna(0)
    df['Selected_Block'] = df['Selected_Block'].fillna("None")

    # 2. 建立 Ground Truth 欄位 (Is_Correct)
    # 使用 apply 逐行呼叫 check_ground_truth
    df['Is_Correct'] = df.apply(
        lambda row: check_ground_truth(row['ImageName'], row['Selected_Block']), 
        axis=1
    )
    
    num_correct = df['Is_Correct'].sum()
    num_wrong = len(df) - num_correct
    print(f"Ground Truth 統計: 正確={num_correct}, 錯誤={num_wrong}")
    
    # 若沒有任何錯誤數據，分析無法進行
    if num_wrong == 0:
        print("[警告] 資料中沒有「錯誤」案例，無法計算鑑別度！")
        print("建議: 請混入一些非該場景的圖片，或手動修改 csv 製造假失敗數據以測試腳本。")

    # 3. 計算各種 Metrics (候選指標)
    
    # (A) Inlier Gap
    df['Inlier_Gap'] = df['PnP_Inliers'] - df['Second_Inliers']
    
    # (B) Inlier Ratio (處理除以零)
    denom_inlier = np.maximum(df['Second_Inliers'], 0.1)
    df['Inlier_Ratio'] = df['PnP_Inliers'] / denom_inlier
    
    # (C) Score Gap
    df['Score_Gap'] = df['Top1_Score'] - df['Top2_Score']
    
    # (D) Score Ratio (處理除以零)
    denom_score = np.maximum(df['Top2_Score'], 0.0001)
    df['Score_Ratio'] = df['Top1_Score'] / denom_score

    print("-" * 65)

    # 4. 評估鑑別度 (Discrimination Power)
    # 比較 Correct 群組與 Wrong 群組在各指標上的差異倍數
    metrics = ['Inlier_Gap', 'Inlier_Ratio', 'Score_Gap', 'Score_Ratio']
    
    print(f"{'Metric':<15} | {'Correct Mean':<13} | {'Wrong Mean':<13} | {'Power (Ratio)'}")
    print("-" * 65)
    
    for m in metrics:
        valid_series = df[m].replace([np.inf, -np.inf], np.nan).dropna()
        
        # 根據 Is_Correct 分組計算平均
        if num_correct > 0:
            mean_good = valid_series[df['Is_Correct']].mean()
        else:
            mean_good = 0
            
        if num_wrong > 0:
            mean_bad = valid_series[~df['Is_Correct']].mean()
        else:
            mean_bad = 0 # 避免稍後除以零
        
        # Power 計算
        if mean_bad <= 0.0001: 
            # 處理分母為 0 的情況
            power_str = "Inf" if mean_good > 0 else "N/A"
        else:
            power = mean_good / mean_bad
            power_str = f"{power:.2f}x"
        
        print(f"{m:<15} | {mean_good:<13.4f} | {mean_bad:<13.4f} | {power_str}")

    # 5. 視覺化 (Boxplot)
    if num_correct > 0 and num_wrong > 0:
        plot_df = df[['Is_Correct'] + metrics].copy()
        
        # 正規化數據以便畫在同一張圖 (Min-Max + Clipping)
        for m in metrics:
            upper = plot_df[m].quantile(0.95) # 去除極端值影響視覺
            plot_df[m] = plot_df[m].clip(upper=upper)
            
            min_v, max_v = plot_df[m].min(), plot_df[m].max()
            if max_v - min_v > 0:
                plot_df[m] = (plot_df[m] - min_v) / (max_v - min_v)
            else:
                plot_df[m] = 0

        melted = plot_df.melt(id_vars='Is_Correct', value_vars=metrics, 
                              var_name='Metric', value_name='Normalized_Value')
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=melted, x='Metric', y='Normalized_Value', hue='Is_Correct', palette={True: "g", False: "r"})
        
        plt.title('Metrics Discrimination Power (True=Correct, False=Wrong)')
        plt.ylabel('Normalized Score')
        plt.grid(True, axis='y', alpha=0.5)
        
        output_img = Path(csv_path).with_suffix('.png')
        plt.savefig(output_img, dpi=150)
        print("-" * 65)
        print(f"[Viz] Boxplot saved to: {output_img}")
    else:
        print("\n[Skip] 跳過繪圖，因為資料樣本不足 (缺乏 Correct 或 Wrong 樣本)。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("report_csv", help="Path to diagnosis_report.csv")
    args = parser.parse_args()
    
    if Path(args.report_csv).exists():
        analyze(args.report_csv)
    else:
        print(f"[Error] 找不到檔案: {args.report_csv}")