# -*- coding: utf-8 -*-
# @Time    : 2026/4/16 15:07
# @Author  : 55050
# @File    : model_performance_CH4.py
# @Software: PyCharm
import warnings

warnings.filterwarnings("ignore")

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# ================= 1. 路径与全局变量配置 =================
# 这里使用的是你代码中提供的真实路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / 'data' / 'model_inputs' / 'CH4_model_input.xlsx'
BEST_MODEL_PATH = PROJECT_ROOT / 'models' / 'CH4_XGBoost_model.joblib'
OUTPUT_PATH = PROJECT_ROOT / 'outputs' / 'figures' / 'CH4_model_performance.png'
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

SELECTED_FEATURES = ['Water_Temperature', 'Depth', 'TN_re', 'Ratio_CN_inout', 'NH4_re', 'NH4_rl', 'HRT', 'NH4_in',
                     'TN_in', 'HLR', 'COD_re', 'COD_in']
RANDOM_STATE_TARGET = 3


def main():
    # ================= 2. 数据加载与预处理 (保留你的原始严谨逻辑) =================
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"未找到数据文件: {DATA_PATH}")
    df = pd.read_excel(DATA_PATH)

    missing_feats = [f for f in SELECTED_FEATURES if f not in df.columns]
    if missing_feats:
        raise ValueError(f"以下所需特征在数据集中缺失: {missing_feats}")

    X = df[SELECTED_FEATURES].copy()
    y = df['CH4'].copy()

    # 1%~99% 分位裁剪 CH4 (去除极端异常值)
    q_low, q_high = y.quantile([0.01, 0.99])
    mask = y.between(q_low, q_high)
    if (~mask).sum() > 0:
        X = X.loc[mask].reset_index(drop=True)
        y = y.loc[mask].reset_index(drop=True)

    # 固定 random_state=3 划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE_TARGET
    )

    # ================= 3. 模型加载与预测 =================
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"未找到最佳模型文件: {BEST_MODEL_PATH}")
    best_model = joblib.load(BEST_MODEL_PATH)

    # 区分 Pipeline 还是 裸模型
    if isinstance(best_model, Pipeline):
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
    else:
        # 与原流程一致：均值填充 + MinMaxScaler 在训练集拟合
        num_cols = X_train.select_dtypes(include=[np.number]).columns
        imputer = SimpleImputer(strategy='mean')
        scaler = MinMaxScaler()

        X_train_copy = X_train.copy()
        X_test_copy = X_test.copy()

        X_train_copy.loc[:, num_cols] = imputer.fit_transform(X_train[num_cols])
        X_test_copy.loc[:, num_cols] = imputer.transform(X_test[num_cols])
        X_train_copy.loc[:, num_cols] = scaler.fit_transform(X_train_copy[num_cols])
        X_test_copy.loc[:, num_cols] = scaler.transform(X_test_copy[num_cols])

        if not isinstance(best_model, XGBRegressor):
            raise TypeError("加载的模型不是 Pipeline，也不是 XGBRegressor。")
        y_train_pred = best_model.predict(X_train_copy)
        y_test_pred = best_model.predict(X_test_copy)

    # 计算测试集指标
    test_r2 = r2_score(y_test, y_test_pred)
    try:
        test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
    except TypeError:
        test_rmse = mean_squared_error(y_test, y_test_pred) ** 0.5

    # ================= 4. 绘制顶刊散点图 (Nature 审美规范) =================

    # [全局配置] 应用 Arial 字体，轴线加粗，刻度向外
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['xtick.major.width'] = 1.2
    plt.rcParams['ytick.major.width'] = 1.2
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11

    # [配色方案] NPG 经典科研色板
    COLOR_TRAIN = "#3C5488"  # 深蓝
    COLOR_TEST = "#F39B7F"  # 珊瑚红

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300, facecolor='white')

    # 计算坐标轴范围 (留白 5%)
    y_min = float(min(y.min(), np.min(np.concatenate([y_train_pred, y_test_pred]))))
    y_max = float(max(y.max(), np.max(np.concatenate([y_train_pred, y_test_pred]))))
    pad = (y_max - y_min) * 0.05
    axis_range = [y_min - pad, y_max + pad]

    # 画 1:1 参考线 (浅灰色虚线，置于底层 zorder=1)
    ax.plot(axis_range, axis_range, color='#bdbdbd', linestyle='--', linewidth=1.5, zorder=1, label='1:1 Reference')

    # 画散点图
    # 训练集：深蓝，无描边，半透明展示数据密集度
    ax.scatter(y_train, y_train_pred, s=35, color=COLOR_TRAIN, alpha=0.6, edgecolor='none', zorder=2,
               label='Train Data')
    # 测试集：珊瑚红，带极细黑色描边，突出展示
    ax.scatter(y_test, y_test_pred, s=45, color=COLOR_TEST, alpha=0.85, edgecolor='k', linewidth=0.4, zorder=3,
               label='Test Data')

    # 坐标轴标签 (加粗，标准科学单位格式)
    ax.set_xlabel('Measured CH$_4$ Emission', fontsize=13, fontweight='bold')
    ax.set_ylabel('Predicted CH$_4$ Emission', fontsize=13, fontweight='bold')
    ax.set_xlim(axis_range)
    ax.set_ylim(axis_range)

    # 图表标题
    ax.set_title('XGBoost Performance: Predicted vs. Measured CH$_4$', fontsize=14, fontweight='bold', pad=15)

    # 图例设置 (去除边框 frameon=False，极其干净)
    ax.legend(loc='upper left', fontsize=11, frameon=False)
    ax.grid(True, linestyle=':', alpha=0.5, zorder=0)

    # 在图内右下角添加关键指标 (纯文本，无边框)
    text_str = f"Test $R^2$: {test_r2:.3f}\nTest RMSE: {test_rmse:.3f}"
    ax.text(0.95, 0.05, text_str, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='bottom', ha='right',
            color='#333333')  # 略带灰度的黑色，更显高级

    # Despine: 去掉右边框和上边框（极简学术排版核心）
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ================= 5. 整理并保存 =================
    out_path = OUTPUT_PATH
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')  # bbox_inches 防止标签被截断
    plt.close()

    print(f"==================================================")
    print(f"🎯 运行成功！")
    print(f"指标核对 -> Test R2: {test_r2:.3f} | Test RMSE: {test_rmse:.3f}")
    print(f"绝美顶刊散点图已保存至：\n{out_path}")
    print(f"==================================================")


if __name__ == '__main__':
    main()
