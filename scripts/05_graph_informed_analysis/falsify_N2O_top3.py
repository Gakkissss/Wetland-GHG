# -*- coding: utf-8 -*-
# @Time    : 2025/10/22 16:37
# @Author  : 55050
# @File    : falsify_N2O_top3.py
# @Software: PyCharm


from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings("ignore")

# 数据加载（与 XGBoost_Optimize 保持一致的数据源与预处理思路）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
path = PROJECT_ROOT / "data" / "model_inputs" / "N2O_model_input.xlsx"
df = pd.read_excel(path)
print("数据加载完成，数据形状:", df.shape)

# === 目标列分位裁剪（1%~99%），并按掩码过滤整表 ===
y = df['N2O'].copy()
q_low, q_high = y.quantile([0.01, 0.99])
mask = y.between(q_low, q_high)
removed = (~mask).sum()
if removed > 0:
    df = df.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
print(f"N2O 1%-99% 分位裁剪完成: 保留 {mask.sum()} 条, 删除 {removed} 条, 下界={q_low:.4f}, 上界={q_high:.4f}")

# === 仅对因果分析使用的5个字段进行：数值化 -> 均值填充（不做归一化） ===
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

_causal_cols = ['Depth', 'CN_in', 'NH4_in', 'N2O']
missing_cols = [c for c in _causal_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"以下因果字段在数据集中缺失: {missing_cols}")

# 数值化
for c in _causal_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# 均值填充 + MinMax归一化
_imputer = SimpleImputer(strategy='mean')
_vals = _imputer.fit_transform(df[_causal_cols])
scaler = MinMaxScaler()
_vals = scaler.fit_transform(_vals)
for i, c in enumerate(_causal_cols):
    df[c] = _vals[:, i]


df_causally = df[['Depth', 'CN_in', 'NH4_in', 'N2O']]
df_causally

# 使用数据的列名来生成标签
labels = [f'{col}' for i, col in enumerate(df_causally.columns)]
# 将数据转换为numpy数组
data = df_causally.to_numpy()

SEED = 1332
import os, warnings
os.environ["PYTHONWARNINGS"] = "ignore"; warnings.simplefilter("ignore")
from dowhy.gcm.util.general import set_random_seed
from dowhy.gcm.falsify import falsify_graph
from dowhy.gcm.independence_test.generalised_cov_measure import generalised_cov_based
from dowhy.gcm.ml import SklearnRegressionModel
from sklearn.ensemble import GradientBoostingRegressor
set_random_seed(SEED)

def create_gradient_boost_regressor(**kwargs):
    return SklearnRegressionModel(GradientBoostingRegressor(**kwargs))

def gcm(X, Y, Z=None):
    return generalised_cov_based(X, Y, Z=Z,
                                 prediction_model_X=create_gradient_boost_regressor,
                                 prediction_model_Y=create_gradient_boost_regressor)

# ================ 3. 直接 falsify ================
import networkx as nx

# 手动定义因果图的节点和边
nodes = ['Depth', 'CN_in', 'NH4_in', 'N2O']
edges = [
    ('NH4_in', 'CN_in'),
    ('NH4_in', 'N2O'),
    ('CN_in', 'N2O'),
    ('Depth', 'N2O'),

]

# 创建一个有向图
g_lingam = nx.DiGraph()
g_lingam.add_nodes_from(nodes)
g_lingam.add_edges_from(edges)

# ====== 自查代码 ======
print('节点：', g_lingam.nodes)
print('边  ：', g_lingam.edges)
print('样本行数：', len(df_causally))
# ======================

result = falsify_graph(
    g_lingam,
    df_causally,                # 就是你前面用过的数据
    n_permutations=500,
    independence_test=gcm,
    conditional_independence_test=gcm,
    plot_histogram=True         # Windows 已装 pygraphviz，不会报错
)

print(result)
