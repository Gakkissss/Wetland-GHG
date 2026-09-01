# -*- coding: utf-8 -*-
# @Time    : 2025/9/30 14:29
# @Author  : 55050
# @File    : knn.py
# @Software: PyCharm

from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# === 读取数据（含 N2O 目标列） ===
PROJECT_ROOT = Path(__file__).resolve().parents[3]
path = PROJECT_ROOT / "data" / "model_inputs" / "N2O_model_input.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "model_comparison" / "N2O"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_excel(path)
print(f"加载数据: {df.shape}")

# === 已选特征子集 ===
selected_features = ['TN_re', 'CN_in', 'NH4_re', 'Depth', 'NH4_in', 'NO3_re', 'TN_in', 'NO3_in']
missing_feats = [f for f in selected_features if f not in df.columns]
if missing_feats:
    raise ValueError(f"以下所需特征在数据集中缺失: {missing_feats}")
X = df[selected_features].copy()
y = df['N2O'].copy()

# === 目标列 1%~99% 分位裁剪 ===
q_low, q_high = y.quantile([0.01, 0.99])
mask = y.between(q_low, q_high)
removed = (~mask).sum()
if removed > 0:
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
print(f"CH4 1%-99% 分位裁剪完成: 保留 {mask.sum()} 条, 删除 {removed} 条, 下界={q_low:.4f}, 上界={q_high:.4f}")

# === 50次基准测试循环 (KNN) ===
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}

n_runs = 50
records = []
print(f"开始进行 {n_runs} 次 KNN 基准测试 (含每次CV网格搜索)...")

for run_seed in range(n_runs):
    # 划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=run_seed
    )

    # 预处理（训练集fit -> 测试集transform）
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy='mean')
    scaler = MinMaxScaler()
    X_train.loc[:, num_cols] = imputer.fit_transform(X_train[num_cols])
    X_test.loc[:, num_cols] = imputer.transform(X_test[num_cols])
    X_train.loc[:, num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test.loc[:, num_cols] = scaler.transform(X_test[num_cols])

    # 网格搜索
    model_knn = KNeighborsRegressor()
    grid_search_knn = GridSearchCV(model_knn, param_grid_knn, scoring='r2', cv=5, n_jobs=-1, verbose=0)
    grid_search_knn.fit(X_train, y_train)

    best_params = grid_search_knn.best_params_
    cv_best_r2 = grid_search_knn.best_score_
    best_model = grid_search_knn.best_estimator_

    # 测试表现
    y_pred = best_model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)
    try:
        test_rmse = mean_squared_error(y_test, y_pred, squared=False)
    except TypeError:
        test_rmse = mean_squared_error(y_test, y_pred) ** 0.5

    records.append({
        'run': run_seed + 1,
        'random_state': run_seed,
        'cv_best_R2': cv_best_r2,
        'test_R2': test_r2,
        'test_MAE': test_mae,
        'test_RMSE': test_rmse,
        'best_n_neighbors': best_params['n_neighbors'],
        'best_weights': best_params['weights'],
        'best_p': best_params['p']
    })

    if (run_seed + 1) % 5 == 0 or run_seed == 0:
        print(f"完成第 {run_seed + 1}/{n_runs} 次: CV_R2={cv_best_r2:.4f} Test_R2={test_r2:.4f}")

# 汇总输出
results_df = pd.DataFrame(records)
summary_df = results_df[['test_R2','test_MAE','test_RMSE']].agg(['mean','std','min','max']).reset_index().rename(columns={'index':'stat'})

output_path = OUTPUT_DIR / 'KNN_benchmark_results.xlsx'
with pd.ExcelWriter(output_path) as writer:
    results_df.to_excel(writer, sheet_name='runs', index=False)
    summary_df.to_excel(writer, sheet_name='summary', index=False)

print('全部完成。结果已写入:', output_path)
