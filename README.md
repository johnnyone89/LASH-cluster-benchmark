# LASH: Final 10-Seed Reproducibility Package

This repository provides the reproducibility workflow for the final LASH study on hourly rolling direct 24-step electricity-demand forecasting.

The repository is portable: it uses only paths relative to the repository root. There are no user-specific absolute paths and no external `frozen_protocol` directory is required.

## Final evaluation contract

All stochastic benchmark models use the same ten fixed final-refit seeds:

`42, 142, 242, 342, 442, 542, 642, 742, 842, 942`

Stochastic models: RF, GBM, XGBoost, LightGBM, CatBoost, MLP, LSTM, GRU, CNN-LSTM, TCN, Transformer, and LASH.

Deterministic references (Seasonal-24, Seasonal-168, Anchor, and Ridge) are evaluated once.

The final analysis uses:

- direct t+1 through t+24 rolling forecasts;
- chronological Train / Validation / Test partitions;
- validation-only model selection and routing;
- 10 final refits for every stochastic benchmark family;
- 5,000-replicate hierarchical circular-block bootstrap;
- dependence scales of 24, 72, 168, and 336 h, with 168 h primary;
- Holm adjustment across the six predefined focal contrasts at each block scale;
- a 1.0% relative-NMAE practical-magnitude reference evaluated separately from statistical significance;
- matched structural, reduced-architecture, retained-component, and annual-component analyses;
- constrained storage-dispatch sensitivity under small, medium, and large scenarios.

The exact machine-readable settings are stored in `configs/final_protocol.json`.

## Data contract

All four model-ready datasets use the common schema:

`Date, Holi, Temp, Humi, WS, Consumption`

Expected files under `data/`:

- `Cluster_1_Harmonized.csv`
- `Cluster_2_Harmonized.csv`
- `BDG_Edu_Harmonized.csv`
- `BDG_Dorm_Harmonized.csv`

The main benchmark uses historical-only weather. Derived calendar, thermal, and demand-history variables are rebuilt causally from information available at or before each forecast origin.

## Fixed chronological evaluation

| Dataset | Train | Validation | Test |
|---|---|---|---|
| Cluster 1 | 2015-03-01 to 2017-02-28 | 2017-03-01 to 2018-08-31 | 2018-09-01 to 2020-02-29 |
| Cluster 2 | 2015-09-01 to 2016-12-31 | 2017-01-01 to 2017-12-31 | 2018-01-01 to 2018-12-31 |
| BDG-Edu | 2015-01-01 to 2015-06-30 | 2015-07-01 to 2015-09-30 | 2015-10-01 to 2015-12-31 |
| BDG-Dorm | 2015-01-01 to 2015-06-30 | 2015-07-01 to 2015-09-30 | 2015-10-01 to 2015-12-31 |

Validation is divided chronologically into tuning and calibration segments with a purge between them. After all adaptive decisions are fixed, selected models are refit on Train + complete Validation and evaluated on Test.

## Repository structure

```text
.
├── configs/
│   └── final_protocol.json
├── data/
│   ├── Cluster_1_Harmonized.csv
│   ├── Cluster_2_Harmonized.csv
│   ├── BDG_Edu_Harmonized.csv
│   └── BDG_Dorm_Harmonized.csv
├── notebooks/
│   ├── 00_Protocol_and_Data_Audit.ipynb
│   ├── 01_Final_10Seed_Benchmark.ipynb
│   ├── 02_Matched_Ablation_Router_YoY.ipynb
│   ├── 03_Posthoc_Statistics_XAI_Visualization.ipynb
│   ├── 04_Final_Robustness_Architecture_Storage.ipynb
│   └── 05_Build_Supplementary_Files.ipynb
├── src/
│   ├── lash_revision_core.py
│   ├── lash_revision_analysis.py
│   ├── lash_revision_ablation.py
│   ├── lash_deadline48.py
│   ├── lash_hardware_optimized.py
│   └── lash_per_dataset_hpo.py
├── outputs/
├── RUN_ORDER.txt
├── requirements.txt
└── .gitignore
```

## Installation

```bash
pip install -r requirements.txt
```

Install the PyTorch build appropriate for the local CPU/CUDA environment separately.

The reported experiments used an AMD Ryzen 7 7800X3D workstation with 32 GB RAM and an NVIDIA GeForce RTX 5070 Ti (16 GB). CPU execution is supported but substantially slower.

## Run order

Run the notebooks in numerical order:

1. `00_Protocol_and_Data_Audit.ipynb`
2. `01_Final_10Seed_Benchmark.ipynb`
3. `02_Matched_Ablation_Router_YoY.ipynb`
4. `03_Posthoc_Statistics_XAI_Visualization.ipynb`
5. `04_Final_Robustness_Architecture_Storage.ipynb`
6. `05_Build_Supplementary_Files.ipynb`

All generated artifacts are written under `outputs/`. The final supplementary builder produces S1, S2, and a ZIP bundle from the generated results.

## Interpretation boundaries

The code reproduces the manuscript evidence without extending claims beyond the evaluated design:

- numerical rank is not automatically interpreted as statistically supported superiority;
- BDG-Edu and BDG-Dorm are public external-source checks with dataset-local development, not frozen cross-site or zero-shot transfer tests;
- added nonlinear complexity is retained only where validation evidence warrants it;
- statistical support and the 1.0% relative-NMAE practical reference are interpreted separately;
- constrained storage dispatch is an offline downstream scheduling sensitivity analysis, not evidence of realized tariff savings, HVAC control benefit, comfort improvement, or production BEMS performance.
