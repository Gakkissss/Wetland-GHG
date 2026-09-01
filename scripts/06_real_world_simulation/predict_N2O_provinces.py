# -*- coding: utf-8 -*-
# @Time    : 2025/12/30 15:45
# @Author  : 55050
# @File    : predict_N2O_provinces.py
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
MODEL_FILE = PROJECT_ROOT / 'models' / 'N2O_ET_model.joblib'
META_FILE = PROJECT_ROOT / 'models' / 'N2O_ET_metadata.json'
OUTPUT_FILE = PROJECT_ROOT / 'outputs' / 'scenario_analysis' / 'N2O_province_predictions.xlsx'
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
FIGURE_OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'figures' / 'province_driver_analysis' / 'N2O'
FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 必需特征列表（从元数据获取）
REQUIRED_FEATURES = [
    'TN_re', 'CN_in', 'NH4_re', 'Depth', 'NH4_in', 'NO3_re', 'TN_in', 'NO3_in'
]
# ================================

print("=" * 60)
print("N2O排放速率预测系统")
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
    print(f"  - N2O排放速率范围: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"  - N2O排放速率均值: {predictions.mean():.4f}")
except Exception as e:
    raise RuntimeError(f"预测过程失败: {e}")

# 5. 生成结果表格
result_df = pd.DataFrame({
    province_col: provinces,
    'N2O排放速率': predictions
})

# 按N2O排放速率降序排列
result_df = result_df.sort_values('N2O排放速率', ascending=False).reset_index(drop=True)

# 添加排名列
result_df.insert(0, '排名', range(1, len(result_df) + 1))

# 6. 保存结果
print("\n[5/5] 保存结果...")
try:
    result_df.to_excel(OUTPUT_FILE, index=False, sheet_name='N2O预测结果')
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
print(f"N2O排放速率最大值: {result_df['N2O排放速率'].max():.4f} ({result_df.loc[result_df['N2O排放速率'].idxmax(), province_col]})")
print(f"N2O排放速率最小值: {result_df['N2O排放速率'].min():.4f} ({result_df.loc[result_df['N2O排放速率'].idxmin(), province_col]})")
print(f"N2O排放速率平均值: {result_df['N2O排放速率'].mean():.4f}")
print(f"N2O排放速率标准差: {result_df['N2O排放速率'].std():.4f}")
print(f"N2O排放速率中位数: {result_df['N2O排放速率'].median():.4f}")

print("\n" + "=" * 60)
print("预测完成！")
print("=" * 60)

# ============ 追加：N2O 全局 SHAP 特征重要性与差异驱动分析 ============
print("\n[6/6] 执行 N2O 专属 SHAP 差异驱动归因分析...")
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图环境
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

try:
    # 1. 提取预处理和 Extra Trees 模型
    preprocessor = pipeline[:-1]
    et_model = pipeline.steps[-1][1]  # 这里对应你的 ET 模型

    # 2. 准备 31 省数据
    X_processed = preprocessor.transform(X)
    X_processed_df = pd.DataFrame(X_processed, columns=REQUIRED_FEATURES)

    # 3. 计算 SHAP 值 (TreeExplainer 完美支持 Extra Trees)
    explainer = shap.TreeExplainer(et_model)
    shap_values = explainer.shap_values(X_processed_df)

    # ---------------- 视图 A：N2O 平均绝对重要性 (Mean |SHAP|) ----------------
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_processed_df, plot_type="bar", show=False, color="#7b3294")  # N2O 使用紫色调
    plt.title("驱动31省 N$_2$O 排放差异的全局因子排名 (Mean |SHAP|)", fontsize=14, fontweight='bold')
    plt.xlabel("平均绝对 SHAP 值 (特征对预测值的平均贡献度)")
    plt.tight_layout()
    plt.savefig(FIGURE_OUTPUT_DIR / 'N2O_SHAP_Global_Bar.png', dpi=300)
    plt.close()
    print("  ✓ 全局重要性图已保存: N2O_SHAP_Global_Bar.png")

    # ---------------- 视图 B：N2O 差异制造者 (SHAP Std 棒棒糖图) ----------------
    print("  [计算驱动 N2O 省际差异的核心特征...]")
    shap_std = np.std(shap_values, axis=0)

    variance_df = pd.DataFrame({
        'Feature': REQUIRED_FEATURES,
        'SHAP_Std': shap_std
    }).sort_values(by='SHAP_Std', ascending=True)

    # 绘制绝美棒棒糖图
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.figure(figsize=(10, 6), facecolor='white')

    # 高亮 N2O 的因果核心特征：进水 C/N 比 (CN_in) 、 总氮 (TN_in)和NH4_in
    target_features = ['CN_in', 'TN_in', 'NH4_in']

    line_colors = ['#7b3294' if feat in target_features else '#e0e0e0' for feat in variance_df['Feature']]
    dot_colors = ['#49006a' if feat in target_features else '#bdbdbd' for feat in variance_df['Feature']]
    text_colors = ['#49006a' if feat in target_features else '#888888' for feat in variance_df['Feature']]

    # 画“棍子”
    plt.hlines(y=variance_df['Feature'], xmin=0, xmax=variance_df['SHAP_Std'],
               color=line_colors, linewidth=3.5, alpha=0.9)

    # 画“糖果”
    plt.scatter(x=variance_df['SHAP_Std'], y=variance_df['Feature'],
                color=dot_colors, s=180, alpha=1, zorder=3, edgecolor='white', linewidth=1.5)

    # 添加数值标签
    for index, (feat, value) in enumerate(zip(variance_df['Feature'], variance_df['SHAP_Std'])):
        plt.text(value + 0.005, index, f'{value:.3f}', va='center', fontsize=10,
                 color=text_colors[index], fontweight='bold')

    # 美化
    plt.title("省域 N$_2$O 排放差异的底层驱动者鉴别 (SHAP Std)",
              fontsize=15, fontweight='bold', pad=20, loc='left')
    plt.xlabel("SHAP 值的空间波动幅度 (Standard Deviation)", fontsize=13, fontweight='bold')
    plt.ylabel("")

    sns.despine(left=True, bottom=False)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.grid(axis='y', visible=False)

    plt.tight_layout()
    plt.savefig(FIGURE_OUTPUT_DIR / 'N2O_SHAP_Variance_Lollipop.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ N2O 差异驱动棒棒糖图已保存: N2O_SHAP_Variance_Lollipop.png")

except Exception as e:
    print(f"⚠ N2O SHAP 分析失败: {e}")
# =====================================================================
