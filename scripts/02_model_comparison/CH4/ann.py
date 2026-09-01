# -*- coding: utf-8 -*-
# @Time    : 2025/9/30 13:35
# @Author  : 55050
# @File    : ann.py
# @Software: PyCharm

from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# === 读取已完成特征筛选后的数据（含 CH4 目标列） ===
PROJECT_ROOT = Path(__file__).resolve().parents[3]
path = PROJECT_ROOT / "data" / "model_inputs" / "CH4_model_input.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "model_comparison" / "CH4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_excel(path)
print(f"加载数据: {df.shape}")

# === 若需要可在此加入基础清洗（此处假设数据已清洗且特征已选） ===
# 直接使用外部已确定的特征子集（与原脚本后续建模一致）
selected_features = ['Water_Temperature', 'Depth', 'TN_re', 'Ratio_CN_inout', 'NH4_re', 'NH4_rl', 'HRT', 'NH4_in', 'TN_in', 'HLR', 'COD_re', 'COD_in']
missing_feats = [f for f in selected_features if f not in df.columns]
if missing_feats:
    raise ValueError(f"以下所需特征在数据集中缺失: {missing_feats}")

X = df[selected_features].copy()
y = df['CH4'].copy()

# === 对目标变量执行 1%~99% 分位裁剪（同时过滤特征行） ===
q_low, q_high = y.quantile([0.01, 0.99])
mask = y.between(q_low, q_high)
removed = (~mask).sum()
if removed > 0:
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
print(f"N2O 1%-99% 分位裁剪完成: 保留 {mask.sum()} 条, 删除 {removed} 条, 下界={q_low:.4f}, 上界={q_high:.4f}")

# ================= 50次基准测试循环 =================
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

param_grid_ann = {
    'hidden_layer_sizes': [(64,), (128,), (128, 64), (256, 128)],
    'learning_rate_init': [0.001, 0.005, 0.01],
    'alpha': [1e-5, 1e-4, 1e-3]
}

n_runs = 50
records = []
print(f"开始进行 {n_runs} 次 ANN 基准测试 (含每次CV网格搜索)...")

for run_seed in range(n_runs):
    # 数据划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=run_seed
    )

    # 预处理（均值填充 + MinMax）
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy='mean')
    scaler = MinMaxScaler()
    X_train.loc[:, num_cols] = imputer.fit_transform(X_train[num_cols])
    X_test.loc[:, num_cols] = imputer.transform(X_test[num_cols])
    X_train.loc[:, num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test.loc[:, num_cols] = scaler.transform(X_test[num_cols])

    # 模型与网格搜索
    params_ann_base = {
        'random_state': run_seed,
        'max_iter': 1000,
        'early_stopping': True,
        'n_iter_no_change': 20,
    }
    model_ann = MLPRegressor(**params_ann_base)
    grid_search_ann = GridSearchCV(model_ann, param_grid_ann, scoring='r2', cv=5, n_jobs=-1, verbose=0)
    grid_search_ann.fit(X_train, y_train)

    best_params = grid_search_ann.best_params_
    cv_best_r2 = grid_search_ann.best_score_
    ann_best = grid_search_ann.best_estimator_

    y_pred = ann_best.predict(X_test)
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
        'best_hidden_layer_sizes': best_params['hidden_layer_sizes'],
        'best_learning_rate_init': best_params['learning_rate_init'],
        'best_alpha': best_params['alpha']
    })

    if (run_seed + 1) % 5 == 0 or run_seed == 0:
        print(f"完成第 {run_seed + 1}/{n_runs} 次: CV_R2={cv_best_r2:.4f} Test_R2={test_r2:.4f}")

# 结果汇总与导出
results_df = pd.DataFrame(records)
summary_df = results_df[['test_R2','test_MAE','test_RMSE']].agg(['mean','std','min','max']).reset_index().rename(columns={'index':'stat'})

output_path = OUTPUT_DIR / 'ANN_benchmark_results.xlsx'
with pd.ExcelWriter(output_path) as writer:
    results_df.to_excel(writer, sheet_name='runs', index=False)
    summary_df.to_excel(writer, sheet_name='summary', index=False)

print("全部完成。结果已写入:", output_path)
