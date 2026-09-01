# -*- coding: utf-8 -*-
# @Time    : 2026/8/12 14:17
# @Author  : 55050
# @File    : residual_monte_carlo.py
# @Software: PyCharm


from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd


# ============================================================
# 1. 配置：一般只需检查 BASE_DIR
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "scenario_inputs" / "statutory_effluent_scenarios.xlsx"
CH4_MODEL_FILE = PROJECT_ROOT / "models" / "CH4_XGBoost_model.joblib"
N2O_MODEL_FILE = PROJECT_ROOT / "models" / "N2O_ET_model.joblib"
CH4_RESIDUAL_FILE = PROJECT_ROOT / "outputs" / "scenario_analysis" / "CH4_test_residuals.csv"
N2O_RESIDUAL_FILE = PROJECT_ROOT / "outputs" / "scenario_analysis" / "N2O_test_residuals.csv"
OUTPUT_FILE = PROJECT_ROOT / "outputs" / "scenario_analysis" / "GB_GHG_MonteCarlo_results.xlsx"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

N_SIMULATIONS = 10_000
RANDOM_SEED = 20260812

# 与原 Excel “CW计算log值转化为速率.xlsx”一致
TOTAL_WATER_VOLUME_M3_PER_YEAR = 67_828_164_300
DEFAULT_HLR_M3_M2_D = 0.2
GWP_CH4 = 28
GWP_N2O = 273

STANDARDS = ["Class I-A", "Class I-B", "Class II", "Class III"]

CH4_FEATURES = [
    "Water_Temperature", "Depth", "TN_re", "Ratio_CN_inout",
    "NH4_re", "NH4_rl", "HRT", "NH4_in", "TN_in",
    "HLR", "COD_re", "COD_in"
]

N2O_FEATURES = [
    "TN_re", "CN_in", "NH4_re", "Depth",
    "NH4_in", "NO3_re", "TN_in", "NO3_in"
]

# 用于确认固定模型预测与原结果一致；允许极小浮点误差。
EXPECTED_POINT_PREDICTIONS = {
    "Class I-A": {"CH4": 2.392892360687256, "N2O": 1.685372700549187},
    "Class I-B": {"CH4": 1.904382348060608, "N2O": 1.573195643866429},
    "Class II":  {"CH4": 2.187045097351074, "N2O": 1.551201816479810},
    "Class III":  {"CH4": 3.316656351089478, "N2O": 1.544445423663422},
}


# ============================================================
# 2. 辅助函数
# ============================================================

def require_files(paths):
    """运行前检查所有文件是否存在。"""
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "以下文件不存在，请检查路径：\n" + "\n".join(missing)
        )


def read_excel_with_wrong_extension(path):
    """读取实际为 xlsx、但扩展名误写为 csv 的输入文件。"""
    try:
        return pd.read_excel(path)
    except Exception as exc:
        raise RuntimeError(
            f"无法读取输入文件：{path}\n"
            "该文件实际应为 Excel 工作簿。可把它复制并改名为 "
            "tert_Country.xlsx，然后同步修改 INPUT_FILE。"
        ) from exc


def flux_to_t_co2eq(flux_ug_m2_h, hlr, gwp, water_volume):
    """
    完全复现原 Excel 换算：
    ug/m2/h -> mg/m3 -> g CO2-eq/m3 -> t CO2-eq/yr
    """
    return (
        flux_ug_m2_h
        * 24.0
        / hlr
        / 1000.0
        * gwp
        / 1000.0
        * water_volume
        / 1_000_000.0
    )


def summarize(simulations, analysis_name):
    """汇总点估计、中位数和 95% 不确定性区间。"""
    rows = []
    for standard in STANDARDS:
        part = simulations[simulations["standard"] == standard]
        row = {"analysis": analysis_name, "standard": standard}

        for variable in ["CH4_tCO2eq", "N2O_tCO2eq", "Total_tCO2eq"]:
            short = variable.replace("_tCO2eq", "")
            values = part[variable].to_numpy()
            row[f"{short}_median"] = np.quantile(values, 0.500)
            row[f"{short}_lower_2.5%"] = np.quantile(values, 0.025)
            row[f"{short}_upper_97.5%"] = np.quantile(values, 0.975)

        rows.append(row)

    return pd.DataFrame(rows)


def compare_1a_1b(simulations, analysis_name):
    """逐次比较一级 A 和一级 B 的总排放。"""
    wide = simulations.pivot(
        index="simulation",
        columns="standard",
        values="Total_tCO2eq"
    )

    result = pd.DataFrame({
        "simulation": wide.index,
        "Class_I-A_tCO2eq": wide["Class I-A"].to_numpy(),
        "Class_I-B_tCO2eq": wide["Class I-B"].to_numpy(),
    })

    result["difference_1A_minus_1B_tCO2eq"] = (
        result["Class_I-A_tCO2eq"] - result["Class_I-B_tCO2eq"]
    )
    result["relative_increase_1A_vs_1B"] = (
        result["difference_1A_minus_1B_tCO2eq"]
        / result["Class_I-B_tCO2eq"]
    )
    result.insert(0, "analysis", analysis_name)
    return result


def comparison_summary(comparison, analysis_name):
    """汇总一级 A 高于一级 B 的概率和增幅区间。"""
    difference = comparison["difference_1A_minus_1B_tCO2eq"].to_numpy()
    increase = comparison["relative_increase_1A_vs_1B"].to_numpy()

    return pd.DataFrame([{
        "analysis": analysis_name,
        "P_Level1A_higher_than_Level1B": np.mean(difference > 0),
        "difference_median_tCO2eq": np.quantile(difference, 0.500),
        "difference_lower_2.5%_tCO2eq": np.quantile(difference, 0.025),
        "difference_upper_97.5%_tCO2eq": np.quantile(difference, 0.975),
        "relative_increase_median": np.quantile(increase, 0.500),
        "relative_increase_lower_2.5%": np.quantile(increase, 0.025),
        "relative_increase_upper_97.5%": np.quantile(increase, 0.975),
    }])


# ============================================================
# 3. 主程序
# ============================================================

def main():
    print("=" * 72)
    print("GB 标准全国 GHG 排放 Monte Carlo 不确定性分析")
    print("=" * 72)

    require_files([
        INPUT_FILE,
        CH4_MODEL_FILE,
        N2O_MODEL_FILE,
        CH4_RESIDUAL_FILE,
        N2O_RESIDUAL_FILE,
    ])

    # ---------- 读取四种标准 ----------
    source_df = pd.read_excel(INPUT_FILE)

    required_columns = set(
        ["Province"] + CH4_FEATURES + N2O_FEATURES
    )
    missing_columns = sorted(required_columns - set(source_df.columns))
    if missing_columns:
        raise ValueError(f"输入表缺少列：{missing_columns}")

    scenario_df = (
        source_df[source_df["Province"].isin(STANDARDS)]
        .copy()
        .set_index("Province")
        .loc[STANDARDS]
        .reset_index()
    )

    if len(scenario_df) != 4:
        raise ValueError("没有完整读取四种标准，请检查 Province 列。")

    # ---------- 加载训练好的模型 ----------
    try:
        ch4_model = joblib.load(CH4_MODEL_FILE)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "CH4模型加载失败。请在PyCharm中选择当时训练XGBoost模型所用的"
            "Python解释器，并确保其中已安装xgboost。"
        ) from exc

    n2o_model = joblib.load(N2O_MODEL_FILE)

    # ---------- 固定模型的点预测 ----------
    scenario_df["log10_CH4_point"] = ch4_model.predict(
        scenario_df[CH4_FEATURES]
    )
    scenario_df["log10_N2O_point"] = n2o_model.predict(
        scenario_df[N2O_FEATURES]
    )

    print("\n[1] 固定模型点预测核验")
    for _, row in scenario_df.iterrows():
        standard = row["Province"]
        ch4_diff = abs(
            row["log10_CH4_point"]
            - EXPECTED_POINT_PREDICTIONS[standard]["CH4"]
        )
        n2o_diff = abs(
            row["log10_N2O_point"]
            - EXPECTED_POINT_PREDICTIONS[standard]["N2O"]
        )
        status = "通过" if ch4_diff < 1e-6 and n2o_diff < 1e-6 else "不一致"
        print(
            f"  {standard:9s} | CH4={row['log10_CH4_point']:.6f} | "
            f"N2O={row['log10_N2O_point']:.6f} | {status}"
        )
        if status == "不一致":
            raise RuntimeError(
                "点预测与原结果不一致，请先检查输入表和模型文件。"
            )

    # 模型输入表中的 HLR 单位为 L/m2/d（当前为 200）；全国排放换算公式
    # 需要 m3/m2/d，因此必须除以 1000，得到 0.2 m3/m2/d。
    # 这里只转换排放换算所用的 HLR_used；模型特征 HLR 仍保持原来的 200。
    scenario_df["HLR_used"] = (
        pd.to_numeric(scenario_df["HLR"], errors="coerce") / 1000.0
    ).fillna(DEFAULT_HLR_M3_M2_D)

    scenario_df["CH4_point_tCO2eq"] = flux_to_t_co2eq(
        10.0 ** scenario_df["log10_CH4_point"].to_numpy(),
        scenario_df["HLR_used"].to_numpy(),
        GWP_CH4,
        TOTAL_WATER_VOLUME_M3_PER_YEAR,
    )
    scenario_df["N2O_point_tCO2eq"] = flux_to_t_co2eq(
        10.0 ** scenario_df["log10_N2O_point"].to_numpy(),
        scenario_df["HLR_used"].to_numpy(),
        GWP_N2O,
        TOTAL_WATER_VOLUME_M3_PER_YEAR,
    )
    scenario_df["Total_point_tCO2eq"] = (
        scenario_df["CH4_point_tCO2eq"]
        + scenario_df["N2O_point_tCO2eq"]
    )

    # ---------- 读取并核验残差 ----------
    ch4_residual_df = pd.read_csv(CH4_RESIDUAL_FILE)
    n2o_residual_df = pd.read_csv(N2O_RESIDUAL_FILE)

    residual_column = "residual_log10"
    if residual_column not in ch4_residual_df.columns:
        raise ValueError("CH4残差文件缺少 residual_log10 列。")
    if residual_column not in n2o_residual_df.columns:
        raise ValueError("N2O残差文件缺少 residual_log10 列。")

    ch4_residuals = (
        pd.to_numeric(ch4_residual_df[residual_column], errors="coerce")
        .dropna()
        .to_numpy()
    )
    n2o_residuals = (
        pd.to_numeric(n2o_residual_df[residual_column], errors="coerce")
        .dropna()
        .to_numpy()
    )

    if len(ch4_residuals) != 72:
        raise ValueError(f"CH4残差应有72条，当前为{len(ch4_residuals)}条。")
    if len(n2o_residuals) != 87:
        raise ValueError(f"N2O残差应有87条，当前为{len(n2o_residuals)}条。")

    print("\n[2] 残差核验")
    print(
        f"  CH4: n={len(ch4_residuals)}, "
        f"mean={ch4_residuals.mean():.6f}, "
        f"RMSE={np.sqrt(np.mean(ch4_residuals ** 2)):.6f}"
    )
    print(
        f"  N2O: n={len(n2o_residuals)}, "
        f"mean={n2o_residuals.mean():.6f}, "
        f"RMSE={np.sqrt(np.mean(n2o_residuals ** 2)):.6f}"
    )

    rng = np.random.default_rng(RANDOM_SEED)
    n_standards = len(STANDARDS)

    ch4_point = scenario_df["log10_CH4_point"].to_numpy()
    n2o_point = scenario_df["log10_N2O_point"].to_numpy()
    hlr = scenario_df["HLR_used"].to_numpy()

    # ========================================================
    # 4. 主分析：配对共同残差
    # ========================================================
    # 每一次模拟抽取一个CH4残差、一个N2O残差；同一次中四种标准共享。
    common_ch4_errors = rng.choice(
        ch4_residuals, size=(N_SIMULATIONS, 1), replace=True
    )
    common_n2o_errors = rng.choice(
        n2o_residuals, size=(N_SIMULATIONS, 1), replace=True
    )

    common_log_ch4 = ch4_point[None, :] + common_ch4_errors
    common_log_n2o = n2o_point[None, :] + common_n2o_errors

    common_ch4_total = flux_to_t_co2eq(
        10.0 ** common_log_ch4,
        hlr[None, :],
        GWP_CH4,
        TOTAL_WATER_VOLUME_M3_PER_YEAR,
    )
    common_n2o_total = flux_to_t_co2eq(
        10.0 ** common_log_n2o,
        hlr[None, :],
        GWP_N2O,
        TOTAL_WATER_VOLUME_M3_PER_YEAR,
    )

    common_records = []
    for standard_index, standard in enumerate(STANDARDS):
        common_records.append(pd.DataFrame({
            "simulation": np.arange(1, N_SIMULATIONS + 1),
            "standard": standard,
            "sampled_CH4_residual_log10": common_ch4_errors[:, 0],
            "sampled_N2O_residual_log10": common_n2o_errors[:, 0],
            "CH4_tCO2eq": common_ch4_total[:, standard_index],
            "N2O_tCO2eq": common_n2o_total[:, standard_index],
            "Total_tCO2eq": (
                common_ch4_total[:, standard_index]
                + common_n2o_total[:, standard_index]
            ),
        }))
    common_sim = pd.concat(common_records, ignore_index=True)

    common_summary = summarize(common_sim, "paired_common_residual")
    common_comparison = compare_1a_1b(
        common_sim, "paired_common_residual"
    )
    common_comparison_summary = comparison_summary(
        common_comparison, "paired_common_residual"
    )

    # ========================================================
    # 5. 敏感性分析：每个标准独立抽取残差
    # ========================================================
    independent_ch4_errors = rng.choice(
        ch4_residuals,
        size=(N_SIMULATIONS, n_standards),
        replace=True,
    )
    independent_n2o_errors = rng.choice(
        n2o_residuals,
        size=(N_SIMULATIONS, n_standards),
        replace=True,
    )

    independent_log_ch4 = ch4_point[None, :] + independent_ch4_errors
    independent_log_n2o = n2o_point[None, :] + independent_n2o_errors

    independent_ch4_total = flux_to_t_co2eq(
        10.0 ** independent_log_ch4,
        hlr[None, :],
        GWP_CH4,
        TOTAL_WATER_VOLUME_M3_PER_YEAR,
    )
    independent_n2o_total = flux_to_t_co2eq(
        10.0 ** independent_log_n2o,
        hlr[None, :],
        GWP_N2O,
        TOTAL_WATER_VOLUME_M3_PER_YEAR,
    )

    independent_records = []
    for standard_index, standard in enumerate(STANDARDS):
        independent_records.append(pd.DataFrame({
            "simulation": np.arange(1, N_SIMULATIONS + 1),
            "standard": standard,
            "sampled_CH4_residual_log10": (
                independent_ch4_errors[:, standard_index]
            ),
            "sampled_N2O_residual_log10": (
                independent_n2o_errors[:, standard_index]
            ),
            "CH4_tCO2eq": independent_ch4_total[:, standard_index],
            "N2O_tCO2eq": independent_n2o_total[:, standard_index],
            "Total_tCO2eq": (
                independent_ch4_total[:, standard_index]
                + independent_n2o_total[:, standard_index]
            ),
        }))
    independent_sim = pd.concat(independent_records, ignore_index=True)

    independent_summary = summarize(
        independent_sim, "independent_residual_sensitivity"
    )
    independent_comparison = compare_1a_1b(
        independent_sim, "independent_residual_sensitivity"
    )
    independent_comparison_summary = comparison_summary(
        independent_comparison, "independent_residual_sensitivity"
    )

    # ---------- 合并汇总 ----------
    summary_all = pd.concat(
        [common_summary, independent_summary], ignore_index=True
    )
    comparison_summary_all = pd.concat(
        [common_comparison_summary, independent_comparison_summary],
        ignore_index=True,
    )

    assumptions = pd.DataFrame({
        "item": [
            "Monte Carlo simulations",
            "Random seed",
            "Annual treated-water volume (m3/yr)",
            "Default HLR (m3/m2/d)",
            "CH4 GWP100",
            "N2O GWP100",
            "Uncertainty propagated",
            "Inputs held fixed",
            "Main comparison",
            "Sensitivity comparison",
        ],
        "value": [
            N_SIMULATIONS,
            RANDOM_SEED,
            TOTAL_WATER_VOLUME_M3_PER_YEAR,
            DEFAULT_HLR_M3_M2_D,
            GWP_CH4,
            GWP_N2O,
            "Empirical out-of-sample residuals on the log10 scale",
            "GB concentrations, water volume, temperature, depth, HRT, HLR, removal efficiencies",
            "Paired common residuals across standards within each simulation",
            "Residuals sampled independently for each standard",
        ],
    })

    # ---------- 保存 Excel ----------
    print("\n[3] 保存结果")
    try:
        with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
            scenario_df.to_excel(
                writer, sheet_name="Point_estimates", index=False
            )
            summary_all.to_excel(
                writer, sheet_name="Uncertainty_summary", index=False
            )
            comparison_summary_all.to_excel(
                writer, sheet_name="1A_vs_1B_summary", index=False
            )
            common_comparison.to_excel(
                writer, sheet_name="Paired_1A_vs_1B", index=False
            )
            independent_comparison.to_excel(
                writer, sheet_name="Sensitivity_1A_vs_1B", index=False
            )
            common_sim.to_excel(
                writer, sheet_name="Paired_all_simulations", index=False
            )
            independent_sim.to_excel(
                writer, sheet_name="Sensitivity_all_sims", index=False
            )
            assumptions.to_excel(
                writer, sheet_name="Assumptions", index=False
            )
    except PermissionError as exc:
        raise PermissionError(
            f"无法写入 {OUTPUT_FILE.name}。请先关闭Excel中的同名文件。"
        ) from exc

    print(f"  已保存：{OUTPUT_FILE}")

    # ---------- 控制台输出最重要结果 ----------
    print("\n[4] 主分析结果：配对共同残差")
    display_columns = [
        "standard",
        "Total_median",
        "Total_lower_2.5%",
        "Total_upper_97.5%",
    ]
    print(common_summary[display_columns].to_string(index=False))

    main_result = common_comparison_summary.iloc[0]
    print("\n一级A与一级B比较：")
    print(
        "  P(一级A > 一级B) = "
        f"{main_result['P_Level1A_higher_than_Level1B']:.2%}"
    )
    print(
        "  一级A相对一级B增幅中位数 = "
        f"{main_result['relative_increase_median']:.2%}"
    )
    print(
        "  增幅95%不确定性区间 = "
        f"[{main_result['relative_increase_lower_2.5%']:.2%}, "
        f"{main_result['relative_increase_upper_97.5%']:.2%}]"
    )

    print("\n分析完成。")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\n运行失败：")
        print(error)
        sys.exit(1)
