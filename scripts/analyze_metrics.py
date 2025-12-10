# scripts/analyze_metrics.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

def check_ground_truth(image_name, selected_block):
    """
    判斷定位結果是否正確 (Ground Truth Check)
    """
    if not isinstance(image_name, str) or not isinstance(selected_block, str):
        return False
    # 策略: 檔名包含 Block 名稱即視為正確 (e.g. "brazil_frame01" contains "brazil")
    if selected_block in image_name:
        return True
    return False

def get_optimal_threshold(df, metric_col):
    """
    [New] 自動搜尋最佳分類門檻值 (Maximize Accuracy)
    假設: Metric 數值越高代表越可能是 Correct (例如 Inliers 越多越好)
    回傳: (best_threshold, best_accuracy)
    """
    # 取得該指標的有效數據
    valid_data = df[[metric_col, 'Is_Correct']].copy()
    valid_data = valid_data.replace([np.inf, -np.inf], np.nan).dropna()
    
    if valid_data.empty:
        return None, 0.0
    
    y_true = valid_data['Is_Correct'].values
    scores = valid_data[metric_col].values
    
    # 候選門檻值: 使用數據中的數值 (排序)
    # 若數據量太大，可改用 np.linspace 取樣以加速，這邊假設量級不大直接全搜
    thresholds = np.unique(scores)
    thresholds = np.sort(thresholds)
    
    # 若候選值過多 (>500)，則進行降取樣以加速計算
    if len(thresholds) > 500:
        thresholds = np.linspace(thresholds[0], thresholds[-1], 500)
    
    best_acc = 0.0
    best_th = thresholds[0]
    N = len(y_true)
    
    # 暴力搜尋: 找出讓 Accuracy 最高的切分點
    for th in thresholds:
        # 預測規則: 分數 >= 門檻 則預測為 Correct (True)
        y_pred = scores >= th
        acc = np.sum(y_pred == y_true) / N
        
        if acc > best_acc:
            best_acc = acc
            best_th = th
            
    return best_th, best_acc

def analyze(csv_path):
    df = pd.read_csv(csv_path)
    print(f"原始資料筆數: {len(df)}")

    # 1. 資料清洗與填補 (配合新欄位名稱)
    df['PnP_Top1_Inliers'] = df.get('PnP_Top1_Inliers', pd.Series(0)).fillna(0)
    df['PnP_Top2_Inliers'] = df.get('PnP_Top2_Inliers', pd.Series(0)).fillna(0)
    
    df['Retrieval_Score1'] = df.get('Retrieval_Score1', pd.Series(0)).fillna(0)
    df['Retrieval_Score2'] = df.get('Retrieval_Score2', pd.Series(0)).fillna(0)
    
    df['PnP_Top1_Block'] = df.get('PnP_Top1_Block', pd.Series("None")).fillna("None")

    # 2. 建立 Ground Truth 欄位 (Is_Correct)
    # 注意：這邊使用 PnP_Top1_Block 作為判斷依據
    df['Is_Correct'] = df.apply(
        lambda row: check_ground_truth(row['ImageName'], row['PnP_Top1_Block']), 
        axis=1
    )
    
    num_correct = df['Is_Correct'].sum()
    num_wrong = len(df) - num_correct
    print(f"Ground Truth 統計: 正確={num_correct}, 錯誤={num_wrong}")
    
    if num_wrong == 0:
        print("[警告] 資料中沒有「錯誤」案例，無法計算鑑別度與建議門檻！")

    # 3. 計算各種 Metrics (候選指標)
    
    # (A) Inlier Gap: 第一名 - 第二名
    df['Inlier_Gap'] = df['PnP_Top1_Inliers'] - df['PnP_Top2_Inliers']
    
    # (B) Inlier Ratio: 第一名 / 第二名 (避免除以零)
    denom_inlier = np.maximum(df['PnP_Top2_Inliers'], 0.1)
    df['Inlier_Ratio'] = df['PnP_Top1_Inliers'] / denom_inlier
    
    # (C) Score Gap: 檢索分數差
    df['Score_Gap'] = df['Retrieval_Score1'] - df['Retrieval_Score2']
    
    # (D) Score Ratio: 檢索分數比
    denom_score = np.maximum(df['Retrieval_Score2'], 0.0001)
    df['Score_Ratio'] = df['Retrieval_Score1'] / denom_score
    
    # (E) Raw Inliers: 直接用 Inliers 數量 (最直覺)
    df['Raw_Inliers'] = df['PnP_Top1_Inliers']

    print("-" * 95)

    # 4. 評估鑑別度與建議門檻
    # 加入 'Raw_Inliers' 一起比較
    metrics = ['Inlier_Gap', 'Inlier_Ratio', 'Raw_Inliers', 'Score_Gap', 'Score_Ratio']
    
    # 表格 Header
    print(f"{'Metric':<15} | {'Correct Mean':<12} | {'Wrong Mean':<12} | {'Power':<8} | {'Best Thresh':<12} | {'Acc':<6}")
    print("-" * 95)
    
    for m in metrics:
        valid_series = df[m].replace([np.inf, -np.inf], np.nan).dropna()
        
        # 計算平均值
        if num_correct > 0:
            mean_good = valid_series[df['Is_Correct']].mean()
        else:
            mean_good = 0
            
        if num_wrong > 0:
            mean_bad = valid_series[~df['Is_Correct']].mean()
        else:
            mean_bad = 0
        
        # 計算鑑別力 (Power)
        if mean_bad <= 0.0001: 
            power_str = "Inf" if mean_good > 0 else "N/A"
        else:
            power = mean_good / mean_bad
            power_str = f"{power:.1f}x"
            
        # [New] 計算建議門檻 (Threshold)
        best_th, best_acc = get_optimal_threshold(df, m)
        
        if best_th is not None:
            th_str = f"> {best_th:.2f}"
            acc_str = f"{best_acc*100:.1f}%"
        else:
            th_str = "N/A"
            acc_str = "N/A"
        
        print(f"{m:<15} | {mean_good:<12.2f} | {mean_bad:<12.2f} | {power_str:<8} | {th_str:<12} | {acc_str:<6}")

    # 5. 視覺化 (Boxplot) - 保持正規化以利顯示，但 Log 已提供原始門檻值
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
        
        plt.figure(figsize=(14, 6))
        sns.boxplot(data=melted, x='Metric', y='Normalized_Value', hue='Is_Correct', palette={True: "g", False: "r"})
        
        plt.title('Metrics Discrimination Power (Normalized for Viz)\nCheck console for raw thresholds')
        plt.ylabel('Normalized Score (0~1)')
        plt.grid(True, axis='y', alpha=0.5)
        
        output_img = Path(csv_path).with_suffix('.png')
        plt.savefig(output_img, dpi=150)
        print("-" * 95)
        print(f"[Viz] Boxplot saved to: {output_img}")
    else:
        print("\n[Skip] 跳過繪圖，因為資料樣本不足。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("report_csv", help="Path to diagnosis_report.csv")
    args = parser.parse_args()
    
    if Path(args.report_csv).exists():
        analyze(args.report_csv)
    else:
        print(f"[Error] 找不到檔案: {args.report_csv}")