# -*- coding: utf-8 -*-
# @Time    : 2025/9/28 13:38
# @Author  : 55050
# @File    : feature_selection_CH4.py
# @Software: PyCharm


from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
import warnings
# 忽略所有警告
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
path = PROJECT_ROOT / "data" / "model_inputs" / "CH4_model_input.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "feature_selection" / "CH4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_excel(path)
df

df.info() # 数据基本信息

# 划分特征和目标变量
X = df.drop(['CH4'], axis=1)
y = df['CH4']

# ====== 裁剪目标变量极端值 (1%~99% 分位) ======
original_len = len(y)
lower, upper = y.quantile([0.01, 0.99])
mask = (y >= lower) & (y <= upper)
X = X.loc[mask].reset_index(drop=True)
y = y.loc[mask].reset_index(drop=True)
print(f"CH4分位裁剪: 下界={lower:.4f}, 上界={upper:.4f}, 保留{len(y)}条, 移除{original_len - len(y)}条")
# ====== 结束 ======

# ====== 先划分，再在训练集上拟合填补与缩放，避免泄漏 ======
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
)
numeric_cols = X_train.select_dtypes(include=[np.number]).columns
imputer_lgb = SimpleImputer(strategy='mean')
scaler_lgb = MinMaxScaler()
# 训练集拟合并变换
X_train.loc[:, numeric_cols] = imputer_lgb.fit_transform(X_train[numeric_cols])
X_train.loc[:, numeric_cols] = scaler_lgb.fit_transform(X_train[numeric_cols])
# 测试集仅变换
X_test.loc[:, numeric_cols] = imputer_lgb.transform(X_test[numeric_cols])
X_test.loc[:, numeric_cols] = scaler_lgb.transform(X_test[numeric_cols])
print("LightGBM阶段：训练集均值填充+0-1归一化已完成（无数据泄漏）")
# ====== 结束 ======

import lightgbm as lgb
# 创建LGBM回归器（连续目标）
lgbm_reg = lgb.LGBMRegressor(random_state=42, verbose=-1)
# 训练模型（使用处理后的训练集）
lgbm_reg.fit(X_train, y_train)

# 获取特征重要性
feature_importances = lgbm_reg.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)


# 只取前20个重要特征
top_n = 20
top_features = feature_importance_df.head(top_n)

# 调整字体大小
plt.figure(figsize=(12, 8), dpi=300)
plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.title(f'Top {top_n} Feature Importance', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.gca().invert_yaxis()
plt.savefig("1.pdf", format='pdf', bbox_inches='tight')
plt.show()

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

# 初始化存储结果的DataFrame
selection_results = pd.DataFrame(columns=['Feature', 'Importance', 'Mean_R2'])

# 初始化用于训练的特征列表
selected_features = []

# K折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)
n_splits = kf.get_n_splits()  # 获取折数

# 动态创建列名（R2）
fold_columns = [f'Fold_{i+1}_R2' for i in range(n_splits)]

# 依次添加特征
for i in range(len(top_features)):
    # 当前特征
    current_feature = top_features.iloc[i]['Feature']
    selected_features.append(current_feature)

    fold_r2_scores = []  # 每折R2

    # K折交叉验证
    for train_idx, val_idx in kf.split(X_train):
        # 划分训练集和验证集
        X_train_fold, X_val_fold = X_train.iloc[train_idx][selected_features], X_train.iloc[val_idx][selected_features]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # 创建并训练LGB模型
        lgbm_reg = lgb.LGBMRegressor(random_state=42, verbose=-1)
        lgbm_reg.fit(X_train_fold, y_train_fold)

        # 预测并计算R2
        y_val_pred = lgbm_reg.predict(X_val_fold)
        fold_r2_score = r2_score(y_val_fold, y_val_pred)
        fold_r2_scores.append(fold_r2_score)

    mean_r2_score = np.mean(fold_r2_scores)

    # 保存结果
    row_data = {
        'Feature': current_feature,
        'Importance': top_features.iloc[i]['Importance'],
        'Mean_R2': mean_r2_score,
    }
    for j, score in enumerate(fold_r2_scores):
        row_data[fold_columns[j]] = score
    row_df = pd.DataFrame([row_data])
    selection_results = pd.concat([selection_results, row_df], ignore_index=True)

selection_results

# 将 Importance 列百分比化并归一化
selection_results['Importance'] = (
    selection_results['Importance'] / selection_results['Importance'].sum()
)
selection_results

import scipy.stats as stats

# 动态获取折列名称
fold_columns = [col for col in selection_results.columns if 'Fold_' in col]

# 添加置信区间列到 selection_results
selection_results['CI_Lower'] = None
selection_results['CI_Upper'] = None

# 遍历每行计算置信区间
for index, row in selection_results.iterrows():
    fold_scores = [row[fold] for fold in fold_columns]
    n_folds = len(fold_scores)
    mean_metric = row['Mean_R2']
    std_err = stats.sem(fold_scores)
    t_value = stats.t.ppf(0.975, df=n_folds - 1)
    ci_lower = mean_metric - t_value * std_err
    ci_upper = mean_metric + t_value * std_err
    selection_results.at[index, 'CI_Lower'] = ci_lower
    selection_results.at[index, 'CI_Upper'] = ci_upper

# 输出结果
selection_results

import matplotlib.colors as mcolors

# 确保 Importance 列为数值类型
selection_results['Importance'] = pd.to_numeric(selection_results['Importance'], errors='coerce')
# 确保置信区间的列为数值类型
selection_results['CI_Lower'] = pd.to_numeric(selection_results['CI_Lower'], errors='coerce')
selection_results['CI_Upper'] = pd.to_numeric(selection_results['CI_Upper'], errors='coerce')

# 参数：选择前 n 个特征
n_features = 12

fig, ax1 = plt.subplots(figsize=(16, 6), dpi=300)

# 渐变柱状图：特征贡献度
norm = plt.Normalize(selection_results['Importance'].min(), selection_results['Importance'].max())
colors = plt.cm.Blues(norm(selection_results['Importance']))

ax1.bar(selection_results['Feature'], selection_results['Importance'], color=colors, label='Feature Importance')
ax1.set_xlabel("Features", fontsize=18, fontweight='bold')
ax1.set_ylabel("Feature Importance", fontsize=18, fontweight='bold')
ax1.tick_params(axis='y', labelsize=12, width=1.5)

# 修改 x 轴特征颜色，前 n_features 用红色，其他用黑色
x_labels = selection_results['Feature']
x_colors = ['red' if i < n_features else 'black' for i in range(len(x_labels))]
for tick_label, color in zip(ax1.get_xticklabels(), x_colors):
    tick_label.set_color(color)

ax1.tick_params(axis='x', rotation=45, labelsize=12, width=1.5)

# 折线图：R2成绩
ax2 = ax1.twinx()

# 红点和红线：前 n_features 个特征
ax2.plot(
    selection_results['Feature'][:n_features + 1],
    selection_results['Mean_R2'][:n_features + 1],
    color="red", marker='o', linestyle='-', label="Mean R2 (Top Features)"
)
ax2.plot(
    selection_results['Feature'][n_features:],
    selection_results['Mean_R2'][n_features:],
    color="black", marker='o', linestyle='-', label="Mean R2 (Other Features)"
)

# 添加置信区间阴影
ax2.fill_between(
    selection_results['Feature'],
    selection_results['CI_Lower'],  # 置信区间下限
    selection_results['CI_Upper'],  # 置信区间上限
    color='red',
    alpha=0.2,  # 设置透明度
)

ax2.set_ylabel("Mean R2", fontsize=18, fontweight='bold')
ax2.tick_params(axis='y', labelsize=12, width=1.5)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))  # 保留三位小数

# 添加标题
plt.title(f"Feature Contribution and R2 Performance (Top {n_features} Features Highlighted)", fontsize=18, fontweight='bold')
fig.tight_layout()
plt.savefig("2.pdf", format='pdf', bbox_inches='tight')
plt.show()

list(selection_results['Feature'][0:8])
