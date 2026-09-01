# Outputs

This directory contains compact reference outputs from the analysis and receives newly generated files when the scripts are rerun.

- `model_comparison/`: reference summaries and per-algorithm rerun outputs
- `model_optimization/`: final-model optimization summaries
- `causal_analysis/`: graph-informed effect estimates and refutation outputs
- `scenario_analysis/`: deterministic predictions, held-out residuals, and Monte Carlo results
- `figures/`: generated model, SHAP, driver, and map figures

The large per-iteration Monte Carlo sheets are included in the reference workbook but can be regenerated with `scripts/08_uncertainty_analysis/residual_monte_carlo.py`.

