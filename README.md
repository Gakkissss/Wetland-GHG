# Cleaner upstream effluents do not consistently reduce greenhouse gas emissions from tertiary subsurface-flow constructed wetlands
This repository contains the data inputs, trained models, and analysis code used to evaluate methane (CH4) and nitrous oxide (N2O) emissions from subsurface-flow constructed wetlands (SSFCWs). It covers model development, interpretation, graph-informed effect estimation, province-level simulations, statutory discharge scenarios, and residual-based predictive uncertainty.

## Repository structure

```text
data/
  model_inputs/          Model-ready global empirical datasets
  scenario_inputs/       Province-level and statutory-scenario inputs
  figure_inputs/         Derived values used by province map scripts
models/                  Final trained CH4 XGBoost and N2O Extra Trees models
scripts/
  01_feature_selection/
  02_model_comparison/
  03_model_optimization/
  04_model_interpretation/
  05_graph_informed_analysis/
  06_real_world_simulation/
  07_statutory_scenarios/
  08_uncertainty_analysis/
  09_figures/
outputs/                 Reference outputs and newly generated results
```

The original working directories are not required. All scripts resolve files relative to the repository root and can therefore be run after the repository has been moved to another computer.

## Environment

The original analysis environment used Python 3.9.25. Create an isolated environment and install the pinned package versions:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The supplied models were created with scikit-learn 1.6.1 and XGBoost 2.1.4. The package versions in `requirements.txt` were recovered from the original `causal` Conda environment and are pinned for reproducibility.

## Data

- `data/model_inputs/CH4_model_input.xlsx`: 243 CH4 observations before 1st-99th percentile filtering.
- `data/model_inputs/N2O_model_input.xlsx`: 295 N2O observations before 1st-99th percentile filtering.
- `data/scenario_inputs/province_secondary_effluent.xlsx`: 31 province-level secondary-effluent profiles.
- `data/scenario_inputs/statutory_effluent_scenarios.xlsx`: Class I-A, Class I-B, Class II, and Class III standardized scenarios.

The `CH4` and `N2O` response columns are log10-transformed areal emission fluxes. Additional definitions and units are provided in `data/README.md`.

## Recommended run order

### 1. Validate the repository

```bash
python scripts/validate_repository.py
```

This checks the required files, input-table dimensions and columns, scenario labels, model metadata, and SHA-256 hashes of the supplied final models.

### 2. Feature selection

```bash
python scripts/01_feature_selection/feature_selection_CH4.py
python scripts/01_feature_selection/feature_selection_N2O.py
```

### 3. Compare the 13 candidate algorithms

Run all model-comparison scripts for one gas:

```bash
python scripts/run_model_comparison.py --gas CH4
python scripts/run_model_comparison.py --gas N2O
```

Each candidate algorithm repeats the train-test split over 50 random seeds. Some algorithms additionally perform grid-search cross-validation and may require substantial computation time.

### 4. Optimize the final models

```bash
python scripts/03_model_optimization/optimize_CH4_XGBoost.py
python scripts/03_model_optimization/optimize_N2O_ET.py
```

Each optimization script evaluates 500 train-test splits, with 10-fold cross-validation and grid search within each split. These are the most computationally intensive analyses in the repository. The supplied models in `models/` allow downstream analyses to be run without repeating optimization.

### 5. SHAP interpretation

```bash
python scripts/04_model_interpretation/SHAP_CH4.py
python scripts/04_model_interpretation/SHAP_N2O.py
```

### 6. Candidate-DAG falsification and effect estimation

Candidate graphs are evaluated separately for CH4 and N2O in `scripts/05_graph_informed_analysis/`. The final graph-informed estimates are generated with:

```bash
python scripts/05_graph_informed_analysis/estimate_CH4_effects.py
python scripts/05_graph_informed_analysis/estimate_N2O_effects.py
```

These estimates are conditional on the specified graph structures and measured covariates and should not be interpreted as substitutes for randomized interventions.

### 7. Province-level secondary-effluent simulations

```bash
python scripts/06_real_world_simulation/predict_CH4_provinces.py
python scripts/06_real_world_simulation/predict_N2O_provinces.py
```

These simulations apply a standardized SSFCW configuration to the observed range of province-level secondary-effluent characteristics. They are not an inventory of existing wetland facilities.

### 8. Statutory discharge scenarios and uncertainty

The complete scenario sequence can be run with:

```bash
python scripts/run_scenario_analysis.py
```

This performs deterministic CH4 and N2O predictions, reconstructs held-out residuals, and runs 10,000 residual-based Monte Carlo iterations under paired-common-error and independent-error assumptions.

The uncertainty intervals represent out-of-sample model predictive uncertainty. They do not include variation in actual wastewater flow, effluent composition, wetland configuration, temperature, removal performance, or long-term microbial adaptation.

## Important unit conventions

- Model input HLR is expressed as L m-2 d-1. The standardized value of 200 L m-2 d-1 is equivalent to 0.2 m3 m-2 d-1.
- CH4 and N2O areal fluxes are expressed as micrograms of gas m-2 h-1 before log10 transformation.
- GWP100 values are 28 for CH4 and 273 for N2O.
- National statutory-scenario calculations use an annual treated-water volume of 67,828,164,300 m3 yr-1, corresponding to the rounded value of 6.783 x 10^10 m3 yr-1 reported in the manuscript.

## Reproducibility notes

Random seeds, selected features, final hyperparameters, train-test splits, and clipping thresholds are stored in the model metadata files. The supplied reference outputs can be used to compare rerun results. Small numerical differences may occur across operating systems or library builds.

## Citation and data sources

Please cite the associated manuscript when using this repository. The literature-derived empirical database, the nationwide wastewater-treatment dataset, the Chinese discharge standard GB 18918-2002, climate inputs, GWP values, and national treated-water volume should be cited using the sources listed in the manuscript and Supplementary Information.
