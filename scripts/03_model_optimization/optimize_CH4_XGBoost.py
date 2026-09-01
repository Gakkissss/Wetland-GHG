# -*- coding: utf-8 -*-
# @Time    : 2025/10/5
# @Author  : 55050
# @File    : optimize_CH4_XGBoost.py
# @Software: PyCharm
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


import json
import joblib
from sklearn.pipeline import Pipeline

# === 读取已完成特征筛选后的数据（含 CH4 目标列） ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]
path = PROJECT_ROOT / "data" / "model_inputs" / "CH4_model_input.xlsx"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "model_optimization"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_excel(path)
print(f"加载数据: {df.shape}")

# === 若需要可在此加入基础清洗（此处假设数据已清洗且特征已选） ===
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
print(f"CH4 1%-99% 分位裁剪完成: 保留 {mask.sum()} 条, 删除 {removed} 条, 下界={q_low:.4f}, 上界={q_high:.4f}")

# ================= 500次优化实验循环 =================
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

param_grid_xgb = {
    'n_estimators': [300, 600, 900],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

n_runs = 500
records = []
print(f"开始进行 {n_runs} 次 XGBoost 优化实验 (每次10折CV网格搜索)...")

# 新增：存储每次实验中已拟合的预处理与模型，用于最终保存最佳模型
_models_storage = []

for run_seed in range(n_runs):
    # 数据划分 70%训练 30%测试
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
    params_xgb_base = {
        'random_state': run_seed,
        'objective': 'reg:squarederror',
        'n_jobs': -1
    }
    model_xgb = XGBRegressor(**params_xgb_base)
    grid_search_xgb = GridSearchCV(model_xgb, param_grid_xgb, scoring='r2', cv=10, n_jobs=-1, verbose=0)
    grid_search_xgb.fit(X_train, y_train)

    best_params = grid_search_xgb.best_params_
    cv_best_r2 = grid_search_xgb.best_score_
    xgb_best = grid_search_xgb.best_estimator_

    y_pred = xgb_best.predict(X_test)
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
        'best_n_estimators': best_params['n_estimators'],
        'best_learning_rate': best_params['learning_rate'],
        'best_max_depth': best_params['max_depth'],
        'best_subsample': best_params['subsample'],
        'best_colsample_bytree': best_params['colsample_bytree']
    })

    # 存储当前轮次的拟合对象与关键信息，后续用于保存最佳模型
    _models_storage.append({
        'run': run_seed + 1,
        'random_state': run_seed,
        'imputer': imputer,
        'scaler': scaler,
        'model': xgb_best
    })

    if (run_seed + 1) % 10 == 0 or run_seed == 0:
        print(f"完成第 {run_seed + 1}/{n_runs} 次: CV_R2={cv_best_r2:.4f} Test_R2={test_r2:.4f}")

# 结果汇总与导出
results_df = pd.DataFrame(records)
summary_df = results_df[['test_R2','test_MAE','test_RMSE']].agg(['mean','std','min','max']).reset_index().rename(columns={'index':'stat'})

# 选择测试R2最接近平均值的那次实验的best params
mean_r2 = summary_df.loc[summary_df['stat']=='mean', 'test_R2'].values[0]
results_df['r2_diff'] = (results_df['test_R2'] - mean_r2).abs()
closest_idx = results_df['r2_diff'].idxmin()

# 从存储中取出该次已拟合的预处理与模型
chosen_run = int(results_df.loc[closest_idx, 'run'])
chosen_row = results_df.loc[closest_idx]
_store = _models_storage[chosen_run - 1]

# 组装可复用Pipeline（预处理 + 模型）并保存
best_pipeline = Pipeline([
    ('imputer', _store['imputer']),
    ('scaler', _store['scaler']),
    ('model', _store['model'])
])

model_path = MODEL_DIR / "CH4_XGBoost_model.joblib"
joblib.dump(best_pipeline, model_path)

# 保存元数据，便于复现与下游使用
meta = {
    'chosen_run': int(chosen_run),
    'random_state': int(_store['random_state']),
    'split': {'train': 0.7, 'test': 0.3},
    'cv': 10,
    'metrics': {
        'test_R2': float(chosen_row['test_R2']),
        'test_MAE': float(chosen_row['test_MAE']),
        'test_RMSE': float(chosen_row['test_RMSE'])
    },
    'best_params': {
        'n_estimators': int(chosen_row['best_n_estimators']),
        'learning_rate': float(chosen_row['best_learning_rate']),
        'max_depth': int(chosen_row['best_max_depth']),
        'subsample': float(chosen_row['best_subsample']),
        'colsample_bytree': float(chosen_row['best_colsample_bytree'])
    },
    'selected_features': selected_features,
    'target': 'CH4',
    'clip_quantiles': {'low_0.01': float(q_low), 'high_0.99': float(q_high)}
}
with open(MODEL_DIR / "CH4_XGBoost_metadata.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

# 导出Excel（runs/summary/best_params）
best_params_row = results_df.loc[closest_idx, [
    'run', 'random_state', 'test_R2', 'test_MAE', 'test_RMSE',
    'best_n_estimators', 'best_learning_rate', 'best_max_depth', 'best_subsample', 'best_colsample_bytree'
]]
best_params_df = pd.DataFrame([best_params_row])

output_path = OUTPUT_DIR / "CH4_XGBoost_optimization.xlsx"
with pd.ExcelWriter(output_path) as writer:
    results_df.drop(columns=['r2_diff']).to_excel(writer, sheet_name='runs', index=False)
    summary_df.to_excel(writer, sheet_name='summary', index=False)
    best_params_df.to_excel(writer, sheet_name='best_params', index=False)

print("全部完成。结果已写入:", output_path)
print(f"最佳模型Pipeline已保存: {model_path}")
print(f"元数据文件: {metadata_path}")
