# -*- coding: utf-8 -*-
# @Time    : 2025/10/11 15:42
# @Author  : 55050
# @File    : estimate_CH4_effects.py
# @Software: PyCharm

import os
import warnings
from datetime import datetime
import argparse

import numpy as np
from pathlib import Path
import pandas as pd

warnings.filterwarnings("ignore")

# ----------------------- 用户配置 -----------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "model_inputs" / "CH4_model_input.xlsx"
TREATMENTS = ["TN_in", "COD_in", "HRT", "Depth"]
OUTCOME = "CH4"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "causal_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_BASENAME = "Causal_Effect_Estimation_results"
# --------------------------------------------------------


def load_and_prepare_data():
    """读取数据，数值化，X均值填补，去除y缺失。

    返回
    ------
    df_ready: pd.DataFrame 仅包含 treatments + [outcome]
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"找不到数据文件: {DATA_PATH}")

    df = pd.read_excel(DATA_PATH)
    required_cols = TREATMENTS + [OUTCOME]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"数据集中缺少必要列: {missing}")

    # 数值化
    for c in required_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 只保留所需列
    df = df[required_cols].copy()

    # 先分离X和y，按要求：X用均值填补，y不填补，直接删除缺失行
    X = df[TREATMENTS].copy()
    y = df[OUTCOME].copy()

    # 删除y缺失
    non_null_mask = y.notna()
    X = X.loc[non_null_mask].reset_index(drop=True)
    y = y.loc[non_null_mask].reset_index(drop=True)

    # X均值填补
    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy="mean")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    df_ready = pd.concat([X_imputed, y.rename(OUTCOME)], axis=1)
    return df_ready


def build_graph_dot():
    """构造基于领域知识的DAG(dot字符串)。"""
    # 用户指定的因果关系:
    #nodes = ['TN_in', 'COD_in', 'HRT', 'Depth', 'CH4']
    #Wedges = [
    #    ('COD_in', 'TN_in'),
    #    ('COD_in', 'CH4'),
    #    ('TN_in', 'CH4'),
    #    ('HRT', 'CH4'),
    #    ('Depth', 'HRT'),
    #]
    edges_str = "; ".join([
        'COD_in -> TN_in',
        'COD_in -> CH4',
        'TN_in -> CH4',
        'HRT -> CH4',
        'Depth -> HRT'
     ])
    graph = f"digraph {{ {edges_str}; }}"
    return graph


def estimate_and_refute_for_treatment(df_ready: pd.DataFrame, treatment: str, graph_dot: str):
    """对单一treatment进行因果估计与三种反事实/稳健性检验。

    打印与Fuxian风格一致的输出；返回一个dict，包含估计值、置信区间、p值以及三种refuter的核心输出。
    """
    from dowhy import CausalModel

    model = CausalModel(
        data=df_ready,
        treatment=treatment,
        outcome=OUTCOME,
        graph=graph_dot,
    )

    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    # 打印识别的估计量（Fuxian风格）
    print(identified_estimand)

    # 对连续处理变量，使用线性回归backdoor。将1 vs 0视为单位变化的效应。
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression",
        control_value=0,
        treatment_value=1,
        confidence_intervals=True,
        test_significance=True,
    )

    # 打印估计值（Fuxian风格）
    print("Causal Estimate is " + str(estimate.value))

    res = {
        "treatment": treatment,
        "estimand": str(identified_estimand),
        "effect_value": float(np.nan if estimate.value is None else estimate.value),
        "confidence_intervals": getattr(estimate, "confidence_intervals", None),
        "p_value": getattr(estimate, "p_value", None),
    }

    # 三种反事实/稳健性检验（并打印，Fuxian风格）
    try:
        refute_random = model.refute_estimate(identified_estimand, estimate, method_name="random_common_cause")
        print(refute_random)
        res["refute_random_common_cause"] = str(refute_random)
    except Exception as e:
        msg = f"ERROR in random_common_cause: {e}"
        print(msg)
        res["refute_random_common_cause"] = msg

    try:
        refute_placebo = model.refute_estimate(identified_estimand, estimate, method_name="placebo_treatment_refuter")
        print(refute_placebo)
        res["refute_placebo_treatment"] = str(refute_placebo)
    except Exception as e:
        msg = f"ERROR in placebo_treatment_refuter: {e}"
        print(msg)
        res["refute_placebo_treatment"] = msg

    try:
        refute_subset = model.refute_estimate(identified_estimand, estimate, method_name="data_subset_refuter")
        print(refute_subset)
        res["refute_data_subset"] = str(refute_subset)
    except Exception as e:
        msg = f"ERROR in data_subset_refuter: {e}"
        print(msg)
        res["refute_data_subset"] = msg

    return res


def save_results_txt_csv(results: list, selected_treatments: list):
    """保存详细结果为txt，同时导出一个简要csv。"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = os.path.join(str(OUTPUT_DIR), f"{RESULT_BASENAME}_{timestamp}.txt")
    csv_path = os.path.join(str(OUTPUT_DIR), f"{RESULT_BASENAME}_{timestamp}.csv")

    # 写入txt（包含完整refuter字符串）
    lines = []
    lines.append(f"数据文件: {DATA_PATH}")
    lines.append(f"处理变量: {selected_treatments}")
    lines.append(f"结果变量: {OUTCOME}")
    lines.append("")

    for r in results:
        lines.append("=" * 80)
        lines.append(f"Treatment: {r['treatment']}")
        lines.append("- 识别的估计量 (estimand):")
        lines.append(str(r["estimand"]))
        lines.append(f"- 因果效应估计值: {r['effect_value']}")
        lines.append(f"- 置信区间: {r.get('confidence_intervals', None)}")
        lines.append(f"- p值: {r.get('p_value', None)}")
        lines.append("- 反事实/稳健性检验:")
        lines.append("  [random_common_cause]")
        lines.append(str(r.get("refute_random_common_cause", "")))
        lines.append("  [placebo_treatment_refuter]")
        lines.append(str(r.get("refute_placebo_treatment", "")))
        lines.append("  [data_subset_refuter]")
        lines.append(str(r.get("refute_data_subset", "")))
        lines.append("")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # 导出简表为csv
    rows = []
    for r in results:
        rows.append({
            "treatment": r["treatment"],
            "effect_value": r["effect_value"],
            "p_value": r.get("p_value", None),
            "confidence_intervals": r.get("confidence_intervals", None),
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    return txt_path, csv_path


def parse_args():
    parser = argparse.ArgumentParser(description="Causal effect estimation with DoWhy.")
    parser.add_argument(
        "--treatment",
        type=str,
        default=None,
        help="指定一个或多个处理变量，逗号分隔。如: HRT 或 HRT,NH4_in。不传则默认跑全部。",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 解析选择的 treatment
    if args.treatment is None or args.treatment.strip() == "":
        selected_treatments = TREATMENTS
    else:
        selected_treatments = [t.strip() for t in args.treatment.split(",") if t.strip()]
        # 合法性检查
        invalid = [t for t in selected_treatments if t not in TREATMENTS]
        if invalid:
            raise ValueError(f"未知的处理变量: {invalid}；可选: {TREATMENTS}")

    print("[1/4] 读取并准备数据 ...")
    df_ready = load_and_prepare_data()
    print(f"数据形状: {df_ready.shape}; 缺失情况:\n{df_ready.isna().sum()}")

    print("[2/4] 构建DAG (基于领域知识) ...")
    graph_dot = build_graph_dot()
    print("DAG:", graph_dot)

    # 可选：保存模型图（若系统已安装 pygraphviz）
    try:
        from dowhy import CausalModel
        tmp_model = CausalModel(data=df_ready, treatment=selected_treatments[0], outcome=OUTCOME, graph=graph_dot)
        tmp_model.view_model(layout='dot')
        print("已尝试输出 causal_model.png（若依赖齐全则会生成）。")
    except Exception:
        pass

    print("[3/4] 逐个处理变量进行估计与稳健性检验 ...")
    results = []
    for t in selected_treatments:
        print(f"\n================ Treatment: {t} ================")
        r = estimate_and_refute_for_treatment(df_ready, t, graph_dot)
        results.append(r)

    print("\n[4/4] 保存结果 ...")
    txt_path, csv_path = save_results_txt_csv(results, selected_treatments)
    print(f"结果已保存:\n- {txt_path}\n- {csv_path}")


if __name__ == "__main__":
    main()
