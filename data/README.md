# Data dictionary and processing notes

## Model-input datasets

`CH4_model_input.xlsx` and `N2O_model_input.xlsx` are the exact model-ready datasets used by the supplied scripts.

### Naming conventions

- `_in`: influent value
- `_out`: effluent value
- `_re`: removal efficiency (%)
- `_rl`: areal removal load (mg m-2 d-1)
- `Ratio_CN_inout`: influent C/N ratio divided by effluent C/N ratio
- `Ratio_CNH4_inout`: influent COD/NH4-N ratio divided by effluent COD/NH4-N ratio
- `Ratio_Do_inout`: influent DO divided by effluent DO

### Core variables and units

| Variable | Definition | Unit |
|---|---|---|
| `Depth` | Effective wetland depth | m |
| `HLR` | Hydraulic loading rate | L m-2 d-1 |
| `HRT` | Hydraulic retention time | d |
| `Water_Temperature` | Wetland water temperature | degrees C |
| `COD_in` | Influent chemical oxygen demand | mg L-1 |
| `TN_in` | Influent total nitrogen | mg L-1 |
| `NH4_in` | Influent NH4+-N | mg L-1 |
| `NO3_in` | Influent NO3--N | mg L-1 |
| `CN_in` | Influent COD/TN ratio | dimensionless |
| `CNH4_in` | Influent COD/NH4+-N ratio | dimensionless |
| `CH4` | log10-transformed CH4 areal flux | log10(ug CH4 m-2 h-1) |
| `N2O` | log10-transformed N2O areal flux | log10(ug N2O m-2 h-1) |

The response-variable filtering retains observations between the empirical 1st and 99th percentiles. This gives 238 CH4 observations and 289 N2O observations for final model fitting and held-out evaluation.

## Province-level scenario input

`province_secondary_effluent.xlsx` contains 31 province-level secondary-effluent profiles. Wetland design and operating inputs were standardized for cross-province comparison. These profiles represent real-world upstream wastewater conditions; they do not identify existing SSFCW facilities.

## Statutory-scenario input

`statutory_effluent_scenarios.xlsx` contains the four scenarios evaluated in the manuscript: Class III, Class II, Class I-B, and Class I-A. The original working file had an incorrect `.csv` extension despite containing an Excel workbook; this release uses a valid `.xlsx` extension. An exploratory `Ex-N` row from the working file was excluded because it was not part of the four reported statutory scenarios.

## Figure inputs

The files in `figure_inputs/` contain the province-level values used by the original map scripts. They are retained to reproduce the map panels without requiring manual intermediate-file reconstruction.

