# -*- coding: utf-8 -*-
# @Time    : 2025/9/30 14:06
# @Author  : 55050
# @File    : extra_trees.py
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

# === 目标变量 1%~99% 分位裁剪 ===
q_low, q_high = y.quantile([0.01, 0.99])
mask = y.between(q_low, q_high)
removed = (~mask).sum()
if removed:
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
print(f"CH4 分位裁剪: 保留 {mask.sum()} 删 {removed} 下界={q_low:.4f} 上界={q_high:.4f}")

# === 50次基准测试 (ExtraTrees) ===
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

param_grid_et = {
    'n_estimators': [100, 200, 400],
    'criterion': ['squared_error', 'friedman_mse'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'max_features': ['auto', 'sqrt', 0.8]
}

n_runs = 50
records = []
print(f"开始进行 {n_runs} 次 ExtraTrees 基准测试 (含每次CV网格搜索)...")

for run_seed in range(n_runs):
    # 数据划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=run_seed
    )

    # 预处理：均值填充 + MinMax 归一化（仅对数值列）
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy='mean')
    scaler = MinMaxScaler()
    X_train.loc[:, num_cols] = imputer.fit_transform(X_train[num_cols])
    X_test.loc[:, num_cols] = imputer.transform(X_test[num_cols])
    X_train.loc[:, num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test.loc[:, num_cols] = scaler.transform(X_test[num_cols])

    # 模型 + 网格搜索
    base_model = ExtraTreesRegressor(random_state=run_seed, n_jobs=-1)
    grid = GridSearchCV(base_model, param_grid_et, scoring='r2', cv=5, n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)

    best_params = grid.best_params_
    cv_best_r2 = grid.best_score_
    best_model = grid.best_estimator_

    # 测试集评估
    y_pred = best_model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)
    try:
        test_rmse = mean_squared_error(y_test, y_pred, squared=False)
    except TypeError:
        test_rmse = mean_squared_error(y_test, y_pred) ** 0.5

    rec = {
        'run': run_seed + 1,
        'random_state': run_seed,
        'cv_best_R2': cv_best_r2,
        'test_R2': test_r2,
        'test_MAE': test_mae,
        'test_RMSE': test_rmse,
        'best_n_estimators': best_params['n_estimators'],
        'best_criterion': best_params['criterion'],
        'best_max_depth': best_params['max_depth'],
        'best_min_samples_split': best_params['min_samples_split'],
        'best_min_samples_leaf': best_params['min_samples_leaf'],
        'best_max_features': best_params['max_features']
    }

    # 可选：保存本次特征重要性（后续聚合）
    if hasattr(best_model, 'feature_importances_'):
        for f, val in zip(num_cols, best_model.feature_importances_):
            rec[f'FI_{f}'] = val

    records.append(rec)

    if (run_seed + 1) % 5 == 0 or run_seed == 0:
        print(f"完成第 {run_seed + 1}/{n_runs} 次: CV_R2={cv_best_r2:.4f} Test_R2={test_r2:.4f}")

# 汇总结果
results_df = pd.DataFrame(records)
metric_cols = ['test_R2','test_MAE','test_RMSE']
summary_df = results_df[metric_cols].agg(['mean','std','min','max']).reset_index().rename(columns={'index':'stat'})

# 若包含特征重要性，计算其均值排序
fi_cols = [c for c in results_df.columns if c.startswith('FI_')]
if fi_cols:
    fi_mean = results_df[fi_cols].mean().sort_values(ascending=False)
    fi_mean_df = fi_mean.reset_index()
    fi_mean_df.columns = ['feature','mean_importance']
else:
    fi_mean_df = pd.DataFrame(columns=['feature','mean_importance'])

output_path = OUTPUT_DIR / 'ET_benchmark_results.xlsx'
with pd.ExcelWriter(output_path) as writer:
    results_df.to_excel(writer, sheet_name='runs', index=False)
    summary_df.to_excel(writer, sheet_name='summary', index=False)
    if not fi_mean_df.empty:
        fi_mean_df.to_excel(writer, sheet_name='feature_importance_mean', index=False)

print('全部完成。结果已写入:', output_path)
# （已移除单次绘图与即时Top5打印，改为聚合均值FI）
