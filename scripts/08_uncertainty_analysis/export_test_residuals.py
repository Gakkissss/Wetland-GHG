# -*- coding: utf-8 -*-
# @Time    : 2026/8/12 14:04
# @Author  : 55050
# @File    : export_test_residuals.py
# @Software: PyCharm


from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# =========================================================
# 1. 文件位置
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "model_inputs"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "scenario_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CH4_DATA_FILE = DATA_DIR / "CH4_model_input.xlsx"
N2O_DATA_FILE = DATA_DIR / "N2O_model_input.xlsx"
CH4_MODEL_FILE = MODEL_DIR / "CH4_XGBoost_model.joblib"
CH4_META_FILE = MODEL_DIR / "CH4_XGBoost_metadata.json"
N2O_MODEL_FILE = MODEL_DIR / "N2O_ET_model.joblib"
N2O_META_FILE = MODEL_DIR / "N2O_ET_metadata.json"
CH4_OUTPUT_FILE = OUTPUT_DIR / "CH4_test_residuals.csv"
N2O_OUTPUT_FILE = OUTPUT_DIR / "N2O_test_residuals.csv"


# =========================================================
# 2. 通用残差导出函数
# =========================================================

def export_test_residuals(
    data_file,
    model_file,
    meta_file,
    output_file,
    target_name
):
    print("\n" + "=" * 70)
    print(f"正在处理 {target_name}")
    print("=" * 70)

    # 读取元数据
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    selected_features = meta["selected_features"]
    random_state = meta["random_state"]
    test_size = meta["split"]["test"]

    expected_r2 = meta["metrics"]["test_R2"]
    expected_mae = meta["metrics"]["test_MAE"]
    expected_rmse = meta["metrics"]["test_RMSE"]

    # 读取原始建模数据
    df = pd.read_excel(data_file)

    missing_columns = [
        col for col in selected_features + [target_name]
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{target_name}数据缺少以下列：{missing_columns}"
        )

    X = df[selected_features].copy()
    y = df[target_name].copy()

    # 与原训练代码完全相同的1%–99%筛选
    q_low, q_high = y.quantile([0.01, 0.99])
    mask = y.between(q_low, q_high)

    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    # 保留筛选后的记录编号，便于核查
    record_id = np.arange(len(y))

    # 必须与原模型使用相同的random_state和70%/30%划分
    (
        X_train,
        X_test,
        y_train,
        y_test,
        id_train,
        id_test
    ) = train_test_split(
        X,
        y,
        record_id,
        test_size=test_size,
        random_state=random_state
    )

    # 加载已经训练好的Pipeline
    pipeline = joblib.load(model_file)

    # 注意：这里直接输入未缩放的X_test
    # Pipeline会自动执行缺失值填补和MinMax缩放
    y_pred = pipeline.predict(X_test)

    # 残差定义：真实值 - 预测值
    residuals = y_test.to_numpy() - y_pred

    # 重新计算测试指标
    calculated_r2 = r2_score(y_test, y_pred)
    calculated_mae = mean_absolute_error(y_test, y_pred)
    calculated_rmse = np.sqrt(
        mean_squared_error(y_test, y_pred)
    )

    print(f"筛选后总样本量：{len(y)}")
    print(f"训练集样本量：{len(y_train)}")
    print(f"测试集样本量：{len(y_test)}")
    print()
    print("重新计算的测试指标：")
    print(f"R2   = {calculated_r2:.10f}")
    print(f"MAE  = {calculated_mae:.10f}")
    print(f"RMSE = {calculated_rmse:.10f}")
    print()
    print("JSON中保存的测试指标：")
    print(f"R2   = {expected_r2:.10f}")
    print(f"MAE  = {expected_mae:.10f}")
    print(f"RMSE = {expected_rmse:.10f}")

    # 检查是不是同一个模型、同一个测试集
    tolerance = 1e-6

    metrics_match = (
        abs(calculated_r2 - expected_r2) < tolerance
        and abs(calculated_mae - expected_mae) < tolerance
        and abs(calculated_rmse - expected_rmse) < tolerance
    )

    if metrics_match:
        print("\n检查通过：模型和测试集与原训练结果一致。")
    else:
        print("\n警告：指标与JSON记录不完全一致。")
        print("暂时不要使用导出的残差进行正式分析。")
        print("请检查joblib和JSON是否来自同一次训练。")

    # 导出测试集真实值、预测值和残差
    residual_df = pd.DataFrame({
        "filtered_record_id": id_test,
        "observed_log10": y_test.to_numpy(),
        "predicted_log10": y_pred,
        "residual_log10": residuals
    })

    # 一并保存测试集特征，便于日后核查
    X_test_export = X_test.reset_index(drop=True)

    residual_df = pd.concat(
        [residual_df.reset_index(drop=True), X_test_export],
        axis=1
    )

    residual_df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\n残差平均值：{residuals.mean():.6f}")
    print(
        "残差标准差："
        f"{residuals.std(ddof=1):.6f}"
    )
    print(
        "残差范围："
        f"{residuals.min():.6f} 至 {residuals.max():.6f}"
    )
    print(f"残差文件已保存：{output_file}")

    return residual_df, metrics_match


# =========================================================
# 3. 导出CH4测试残差
# =========================================================

ch4_residual_df, ch4_match = export_test_residuals(
    data_file=CH4_DATA_FILE,
    model_file=CH4_MODEL_FILE,
    meta_file=CH4_META_FILE,
    output_file=CH4_OUTPUT_FILE,
    target_name="CH4"
)


# =========================================================
# 4. 导出N2O测试残差
# =========================================================

n2o_residual_df, n2o_match = export_test_residuals(
    data_file=N2O_DATA_FILE,
    model_file=N2O_MODEL_FILE,
    meta_file=N2O_META_FILE,
    output_file=N2O_OUTPUT_FILE,
    target_name="N2O"
)


# =========================================================
# 5. 最终检查
# =========================================================

print("\n" + "=" * 70)

if ch4_match and n2o_match:
    print("全部完成：CH4和N2O残差均可用于蒙特卡洛分析。")
else:
    print("导出完成，但至少一个模型未通过一致性检查。")
    print("请先解决模型与元数据不一致问题。")

print("=" * 70)
