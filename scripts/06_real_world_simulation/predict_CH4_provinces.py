# -*- coding: utf-8 -*-
# @Time    : 2025/12/30 15:45
# @Author  : 55050
# @File    : predict_CH4_provinces.py
# @Software: PyCharm


import pandas as pd
import numpy as np
import json
import joblib
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

# ============ 配置区 ============
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / 'data' / 'scenario_inputs' / 'province_secondary_effluent.xlsx'
MODEL_FILE = PROJECT_ROOT / 'models' / 'CH4_XGBoost_model.joblib'
META_FILE = PROJECT_ROOT / 'models' / 'CH4_XGBoost_metadata.json'
OUTPUT_FILE = PROJECT_ROOT / 'outputs' / 'scenario_analysis' / 'CH4_province_predictions.xlsx'
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
FIGURE_OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'figures' / 'province_driver_analysis' / 'CH4'
FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 必需特征列表（从元数据获取）
REQUIRED_FEATURES = [
    'Water_Temperature', 'Depth', 'TN_re', 'Ratio_CN_inout', 
    'NH4_re', 'NH4_rl', 'HRT', 'NH4_in', 'TN_in', 
    'HLR', 'COD_re', 'COD_in'
]
# ================================

print("=" * 60)
print("CH4排放速率预测系统")
print("=" * 60)

# 1. 加载模型和元数据
print("\n[1/4] 加载模型...")
try:
    pipeline = joblib.load(MODEL_FILE)
    print(f"✓ 模型加载成功: {MODEL_FILE.name}")
except Exception as e:
    raise FileNotFoundError(f"模型文件加载失败: {e}")

try:
    with open(META_FILE, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    print(f"✓ 元数据加载成功: {META_FILE.name}")
    print(f"  - 模型R²: {meta.get('metrics', {}).get('test_R2', 'N/A'):.4f}")
    print(f"  - 训练参数: n_estimators={meta.get('best_params', {}).get('n_estimators', 'N/A')}")
except Exception as e:
    raise FileNotFoundError(f"元数据文件加载失败: {e}")

# 2. 读取输入数据
print("\n[2/4] 读取输入数据...")
try:
    df = pd.read_excel(INPUT_FILE)
    print(f"✓ 数据加载成功: {INPUT_FILE.name}")
    print(f"  - 数据形状: {df.shape}")
    print(f"  - 列名: {df.columns.tolist()}")
except Exception as e:
    raise FileNotFoundError(f"输入文件读取失败: {e}")

# 获取省份列（假设第一列是省份名）
province_col = df.columns[0]
provinces = df[province_col].copy()
print(f"  - 省份列: '{province_col}'")
print(f"  - 省份数量: {len(provinces)}")

# 3. 准备特征数据
print("\n[3/4] 准备特征数据...")

# 检查必需特征是否存在
missing_features = [f for f in REQUIRED_FEATURES if f not in df.columns]
if missing_features:
    raise ValueError(f"缺少必需特征: {missing_features}")

# 提取特征（按模型训练时的顺序）
X = df[REQUIRED_FEATURES].copy()
print(f"✓ 特征提取完成")
print(f"  - 特征数量: {len(REQUIRED_FEATURES)}")
print(f"  - 特征列表: {REQUIRED_FEATURES}")

# 检查缺失值
missing_count = X.isnull().sum().sum()
if missing_count > 0:
    print(f"⚠ 警告: 数据中存在 {missing_count} 个缺失值，将由模型pipeline自动处理")
    print("  缺失值分布:")
    for col in X.columns:
        n_missing = X[col].isnull().sum()
        if n_missing > 0:
            print(f"    - {col}: {n_missing} 个缺失值")

# 显示数据统计
print("\n特征数据统计摘要:")
print(X.describe().T[['mean', 'std', 'min', 'max']])

# 4. 使用Pipeline进行预测（自动完成imputer + scaler + model）
print("\n[4/4] 执行预测...")
try:
    predictions = pipeline.predict(X)
    print(f"✓ 预测完成")
    print(f"  - 预测样本数: {len(predictions)}")
    print(f"  - CH4排放速率范围: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"  - CH4排放速率均值: {predictions.mean():.4f}")
except Exception as e:
    raise RuntimeError(f"预测过程失败: {e}")

# 5. 生成结果表格
result_df = pd.DataFrame({
    province_col: provinces,
    'CH4排放速率': predictions
})

# 按CH4排放速率降序排列
result_df = result_df.sort_values('CH4排放速率', ascending=False).reset_index(drop=True)

# 添加排名列
result_df.insert(0, '排名', range(1, len(result_df) + 1))

# 6. 保存结果
print("\n[5/5] 保存结果...")
try:
    result_df.to_excel(OUTPUT_FILE, index=False, sheet_name='CH4预测结果')
    print(f"✓ 结果已保存: {OUTPUT_FILE}")
except Exception as e:
    raise IOError(f"结果保存失败: {e}")

# 7. 显示结果摘要
print("\n" + "=" * 60)
print("预测结果摘要（前10名）")
print("=" * 60)
print(result_df.head(10).to_string(index=False))

print("\n" + "=" * 60)
print("预测结果摘要（后10名）")
print("=" * 60)
print(result_df.tail(10).to_string(index=False))

print("\n" + "=" * 60)
print("统计信息")
print("=" * 60)
print(f"总省份数: {len(result_df)}")
print(f"CH4排放速率最大值: {result_df['CH4排放速率'].max():.4f} ({result_df.loc[result_df['CH4排放速率'].idxmax(), province_col]})")
print(f"CH4排放速率最小值: {result_df['CH4排放速率'].min():.4f} ({result_df.loc[result_df['CH4排放速率'].idxmin(), province_col]})")
print(f"CH4排放速率平均值: {result_df['CH4排放速率'].mean():.4f}")
print(f"CH4排放速率标准差: {result_df['CH4排放速率'].std():.4f}")
print(f"CH4排放速率中位数: {result_df['CH4排放速率'].median():.4f}")

print("\n" + "=" * 60)
print("预测完成！")
print("=" * 60)

# ============ 追加：全局 SHAP 特征重要性分析（揭示省际差异主导因素） ============
print("\n[6/6] 执行全局 SHAP 特征重要性分析...")
import shap
import matplotlib.pyplot as plt

# 设置中文字体（根据你的系统选择，这里以常见设置为例）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

try:
    # 提取预处理和模型
    preprocessor = pipeline[:-1]
    xgb_model = pipeline.steps[-1][1]

    # 对31个省的数据进行预处理
    X_processed = preprocessor.transform(X)
    X_processed_df = pd.DataFrame(X_processed, columns=REQUIRED_FEATURES)

    # 计算 SHAP 值
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_processed_df)

    # ---------------- 第一张图：平均绝对 SHAP (Mean |SHAP|) ----------------
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_processed_df, plot_type="bar", show=False, color="#2c7bb6")
    plt.title("驱动31省CH4排放空间差异的核心因素排名 (Mean |SHAP|)", fontsize=14, fontweight='bold')
    plt.xlabel("平均绝对 SHAP 值 (特征对预测差异的平均贡献度)")
    plt.tight_layout()
    plt.savefig(FIGURE_OUTPUT_DIR / 'SHAP_Global_Importance_Bar.png', dpi=300)
    plt.close()
    print("  ✓ 全局重要性柱状图已保存: SHAP_Global_Importance_Bar.png")

    # ---------------- 第二张图：SHAP 标准差 (真正的差异制造者 - 棒棒糖图，精准高亮因果因子) ----------------
    print("  [计算驱动省际差异的核心特征...]")
    import seaborn as sns

    # 1. 计算标准差并排序
    shap_std = np.std(shap_values, axis=0)

    variance_df = pd.DataFrame({
        'Feature': REQUIRED_FEATURES,
        'SHAP_Std': shap_std
    }).sort_values(by='SHAP_Std', ascending=True)

    # 2. 绘制棒棒糖图
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.figure(figsize=(10, 6), facecolor='white')

    # 动态颜色配置：精准锁定 DAG 因果特征 (TN_in 和 COD_in)
    target_features = ['TN_in', 'COD_in']

    # 利用列表推导式，如果在目标列表中就用红色，否则用灰色
    line_colors = ['#d73027' if feat in target_features else '#e0e0e0' for feat in variance_df['Feature']]
    dot_colors = ['#a50026' if feat in target_features else '#bdbdbd' for feat in variance_df['Feature']]
    # 让非重点的数据标签颜色稍微暗一点，进一步凸显重点
    text_colors = ['#a50026' if feat in target_features else '#888888' for feat in variance_df['Feature']]

    # 画“棍子”
    plt.hlines(y=variance_df['Feature'], xmin=0, xmax=variance_df['SHAP_Std'],
               color=line_colors, linewidth=3.5, alpha=0.9)

    # 画“糖果”（圆点）
    plt.scatter(x=variance_df['SHAP_Std'], y=variance_df['Feature'],
                color=dot_colors, s=180, alpha=1, zorder=3, edgecolor='white', linewidth=1.5)

    # 添加数据标签
    for index, (feat, value) in enumerate(zip(variance_df['Feature'], variance_df['SHAP_Std'])):
        plt.text(value + 0.015, index, f'{value:.3f}', va='center', fontsize=10,
                 color=text_colors[index], fontweight='bold')

    # 美化图表
    plt.title("省域 CH$_4$ 排放差异的底层驱动者鉴别 (SHAP Std)",
              fontsize=15, fontweight='bold', pad=20, loc='left')
    plt.xlabel("SHAP 值的空间波动幅度 (Standard Deviation)", fontsize=13, fontweight='bold')
    plt.ylabel("")

    sns.despine(left=True, bottom=False)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.grid(axis='y', visible=False)

    plt.tight_layout()
    plt.savefig(FIGURE_OUTPUT_DIR / 'SHAP_Variance_Lollipop_Custom.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 自定义高亮棒棒糖图已保存: SHAP_Variance_Lollipop_Custom.png")

except Exception as e:
    print(f"⚠ SHAP 分析失败: {e}")
# =====================================================================
