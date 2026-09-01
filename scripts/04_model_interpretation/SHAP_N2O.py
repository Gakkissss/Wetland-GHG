# -*- coding: utf-8 -*-
# @Time    : 2025/10/5
# @Author  : 55050
# @File    : SHAP_N2O.py
# @Software: PyCharm

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# 非交互后端，便于直接保存图片
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 依赖 SHAP
try:
    import shap
except Exception as e:
    raise SystemExit(
        "未检测到 shap，请先安装: pip install shap\n原始错误: %s" % str(e)
    )

from sklearn.pipeline import Pipeline

# ---------------- 配置区 ----------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / 'models' / 'N2O_ET_model.joblib'
META_PATH = PROJECT_ROOT / 'models' / 'N2O_ET_metadata.json'
DATA_PATH = PROJECT_ROOT / 'data' / 'model_inputs' / 'N2O_model_input.xlsx'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'figures' / 'SHAP' / 'N2O'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TOP_K = 5              # 依赖图展示前K个重要特征
# ---------------------------------------

def load_model_and_meta():
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError(
            f"未找到模型或元数据文件:\n  {MODEL_PATH}\n  {META_PATH}\n请先运行 Causal_ET_Optimize.py 生成这些文件。"
        )
    pipe = joblib.load(MODEL_PATH)
    with open(META_PATH, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    if not isinstance(pipe, Pipeline) or 'model' not in pipe.named_steps:
        raise ValueError("加载的对象不是带有 'model' 步的Pipeline，请检查模型文件。")
    return pipe, meta


def load_data(features, target, clip_quantiles=None):
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"未找到数据文件: {DATA_PATH}")
    df = pd.read_excel(DATA_PATH)

    # 确保特征列齐全与顺序一致，不存在的列以 NaN 填充，后续 imputer 处理
    for col in features:
        if col not in df.columns:
            df[col] = np.nan
    X = df[features].copy()
    y = df[target].copy() if target in df.columns else None

    # 与训练一致的分位裁剪（若 y 存在时生效）
    if y is not None and isinstance(clip_quantiles, dict):
        low = clip_quantiles.get('low_0.01', None)
        high = clip_quantiles.get('high_0.99', None)
        if low is not None and high is not None:
            mask = y.between(low, high)
            X = X.loc[mask].reset_index(drop=True)
            y = y.loc[mask].reset_index(drop=True)
            print(f"依据 meta 分位阈值过滤: 保留 {mask.sum()} 条, 删除 {(~mask).sum()} 条")
    return X, y


def compute_shap(pipe: Pipeline, X: pd.DataFrame, feature_names: list):
    # 拆分出预处理与模型
    pre = pipe[:-1]
    model = pipe.named_steps['model']

    # 预处理到与训练一致的空间
    X_proc = pre.transform(X)

    # 优先使用 TreeExplainer
    try:
        explainer = shap.TreeExplainer(model)
        # 新版 API，调用 explainer(...) 返回 Explanation
        try:
            explanation = explainer(X_proc)
            shap_values = explanation.values
        except Exception:
            shap_values = explainer.shap_values(X_proc)
    except Exception:
        # 兜底：通用解释器
        explainer = shap.Explainer(model)
        shap_values = explainer(X_proc)
        # 若返回 Explanation，取 values
        try:
            shap_values = shap_values.values
        except Exception:
            pass

    # 统一为 numpy 数组形状 [n_samples, n_features]
    if hasattr(shap_values, 'toarray'):
        shap_values = shap_values.toarray()
    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        # 对于某些多输出情形，取第一维
        shap_values = shap_values[..., 0]

    # 重要性（全局平均绝对 SHAP）
    global_importance = np.mean(np.abs(shap_values), axis=0)

    return explainer, shap_values, X_proc, global_importance


def save_figures(shap_values, X_proc, feature_names, global_importance):
    # summary dot
    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, X_proc, feature_names=feature_names, show=False)
    plt.tight_layout()
    path_summary = OUTPUT_DIR / 'ET_shap_summary.pdf'
    plt.savefig(path_summary, dpi=300, bbox_inches='tight')
    plt.close()

    # summary bar
    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, X_proc, feature_names=feature_names, plot_type='bar', show=False)
    plt.tight_layout()
    path_bar = OUTPUT_DIR / 'ET_shap_bar.pdf'
    plt.savefig(path_bar, dpi=200)
    plt.close()

    # top-K dependence
    order = np.argsort(global_importance)[::-1]
    top_idx = order[:min(len(order), TOP_K)]
    dep_paths = []
    for idx in top_idx:
        feat = feature_names[idx]
        # 注意：由于已缩放，依赖图中数值为缩放后的特征值
        plt.figure(figsize=(7, 5))
        shap.dependence_plot(idx, shap_values, X_proc, feature_names=feature_names, show=False)
        plt.tight_layout()
        path_dep = OUTPUT_DIR / f"ET_shap_dependence_{feat}.pdf"
        plt.savefig(path_dep, dpi=300, bbox_inches='tight')
        plt.close()
        dep_paths.append(str(path_dep))

    return str(path_summary), str(path_bar), dep_paths


def export_shap_csv(shap_values, X_proc, feature_names, pipe: Pipeline, X_raw: pd.DataFrame, y=None):
    # 预测值（在预处理后一致空间，直接使用 pipe.predict）
    y_pred = pipe.predict(X_raw)

    df_shap = pd.DataFrame(shap_values, columns=[f"shap_{c}" for c in feature_names])
    df_pred = pd.DataFrame({
        'prediction': y_pred
    })
    df_out = pd.concat([df_pred, df_shap], axis=1)
    if y is not None:
        df_out.insert(0, 'N2O', y.values)

    out_path = OUTPUT_DIR / 'ET_shap_values.csv'
    df_out.to_csv(out_path, index=False, encoding='utf-8-sig')
    return str(out_path)


def main():
    pipe, meta = load_model_and_meta()
    features = meta.get('selected_features', [])
    target = meta.get('target', 'N2O')

    X, y = load_data(features, target, clip_quantiles=meta.get('clip_quantiles', None))
    print(f"数据加载后用于解释的样本: {X.shape}")

    explainer, shap_values, X_proc, global_imp = compute_shap(pipe, X, features)

    p_sum, p_bar, p_deps = save_figures(shap_values, X_proc, features, global_imp)
    p_csv = export_shap_csv(shap_values, X_proc, features, pipe, X, y)

    # 另导出全局重要性
    imp_df = pd.DataFrame({
        'feature': features,
        'mean_abs_shap': global_imp
    }).sort_values('mean_abs_shap', ascending=False)
    imp_path = OUTPUT_DIR / 'ET_shap_importance.csv'
    imp_df.to_csv(imp_path, index=False, encoding='utf-8-sig')

    print("SHAP 分析完成，输出如下：")
    print("- summary(dot):", p_sum)
    print("- summary(bar):", p_bar)
    print("- dependence top-K:")
    for p in p_deps:
        print("  ", p)
    print("- 行级 SHAP 与预测:", p_csv)
    print("- 全局重要性:", str(imp_path))


if __name__ == '__main__':
    main()
