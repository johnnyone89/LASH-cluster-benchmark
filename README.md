# LASH Revision Experiment Pipeline

This repository implements the reproducible experiment pipeline for the revised LASH study on hourly updated direct 24-step electricity-demand forecasting.

The repository contains **code and frozen model-ready inputs**. Generated predictions, tables, figures, Optuna databases, and reviewer-analysis outputs are produced locally under `outputs/` and are intentionally not versioned.

## Data contract

All four model-ready datasets use the same six-column schema:

`Date, Holi, Temp, Humi, WS, Consumption`

- `Holi=1` denotes weekends or holidays under the harmonized source calendar.
- Temperature is in degrees Celsius, humidity in percent, and wind speed in m/s.
- The main benchmark uses **historical-only weather**. Realized target-period weather is not used as a headline input.
- Derived calendar, thermal, and demand-history variables are rebuilt by one shared function for all datasets.

## Fixed chronological evaluation

| Dataset | Train | Validation | Test |
|---|---|---|---|
| Cluster 1 | 2015-03-01 to 2017-02-28 | 2017-03-01 to 2018-08-31 | 2018-09-01 to 2020-02-29 |
| Cluster 2 | 2015-09-01 to 2016-12-31 | 2017-01-01 to 2017-12-31 | 2018-01-01 to 2018-12-31 |
| BDG-Edu | 2015-01-01 to 2015-06-30 | 2015-07-01 to 2015-09-30 | 2015-10-01 to 2015-12-31 |
| BDG-Dorm | 2015-01-01 to 2015-06-30 | 2015-07-01 to 2015-09-30 | 2015-10-01 to 2015-12-31 |

Validation is divided chronologically into tuning and calibration segments using a 2:1 rule, with a 23-origin purge between them. After all adaptive choices are frozen, preprocessing and final models are refit on Train + complete Validation and the Test partition is evaluated once.

## Benchmark suite

The benchmark includes persistence baselines, the leakage-safe anchor, Ridge, five tree-ensemble families (RF, GBM, XGBoost, LightGBM, CatBoost), six neural comparators (MLP, LSTM, GRU, CNN-LSTM, TCN, Transformer), and LASH.

Tree methods are implemented as **24 direct horizon-specific regressors** using the same causal origin-level information available to the neural models. This avoids recursive decoding and prevents future demand from entering the forecasting contract.

## Dataset-local hyperparameter optimization

The manuscript results use the dataset-local protocol implemented in `src/lash_per_dataset_hpo.py`.

For each dataset independently:

- Ridge uses the same eight-value chronological validation grid.
- Every stochastic model uses two TPE searches with seeds `42` and `142`.
- Search budgets are dimension-adaptive: **3–5 finite COMPLETE trials per repeat**.
- Neural pruning may attempt at most twice the declared COMPLETE-trial budget; if necessary, a deterministic no-pruning fallback trains only the missing COMPLETE trials.
- PRUNED and FAILED trials remain auditable but are never eligible for selection.
- Repeat finalists are confirmed on the complete validation-tuning segment.
- The selected setting is frozen before final refitting.
- Final stochastic refits use seeds `42`, `142`, and `242`.
- No prior result directory and no Test target are used for hyperparameter, gain, epoch, or routing selection.

This is the protocol used to generate the first-revision benchmark reported in the manuscript.

## Repository structure

```text
.
├── data/
│   ├── Cluster_1_Harmonized.csv
│   ├── Cluster_2_Harmonized.csv
│   ├── BDG_Edu_Harmonized.csv
│   ├── BDG_Dorm_Harmonized.csv
│   └── README_DATA.md
├── notebooks/
│   ├── 00_Protocol_and_Data_Audit.ipynb
│   ├── 01_LASH_and_Fair_Benchmark.ipynb
│   ├── 02_Matched_Ablation_Router_YoY.ipynb
│   ├── 03_Posthoc_Statistics_XAI_Visualization.ipynb
│   └── 04_SecondRound_Confirmatory.ipynb
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

## Notebook order

1. `00_Protocol_and_Data_Audit.ipynb`  
   Verifies the four harmonized files, the common causal feature contract, fixed chronological partitions, and search spaces.

2. `01_LASH_and_Fair_Benchmark.ipynb`  
   Runs the exact dataset-local HPO protocol used for the revised benchmark and performs the frozen final refits.

3. `02_Matched_Ablation_Router_YoY.ipynb`  
   Evaluates router sensitivity on all four datasets. Structural/feature mechanism tests freeze each dataset's selected full-model setting. Year-over-year components are evaluated only on the two multiyear cluster datasets.

4. `03_Posthoc_Statistics_XAI_Visualization.ipynb`  
   Performs dependence-aware inference, temporal/peak analyses, XAI, HPO-convergence analysis, and publication-oriented visualization. BDG nonlinear diagnostics refer to candidate sequential experts when the deployed router is Ridge-only.

5. `04_SecondRound_Confirmatory.ipynb`  
   Reproduces the targeted second-round reviewer analyses: ten-seed focal confirmation, effect sizes and practical-magnitude interpretation, validation-only reduced-architecture selection with a pre-test lock, retained-component tests, BDG claim-scope audit, and common constrained storage-dispatch sensitivity.

## Installation

Create and activate a Python environment, then install the repository dependencies:

```bash
pip install -r requirements.txt
```

Install the PyTorch build appropriate for your CPU/CUDA environment separately using the official PyTorch installation selector.

The reported experiments were executed on an AMD Ryzen 7 7800X3D workstation with 32 GB RAM and an NVIDIA GeForce RTX 5070 Ti (16 GB). The code permits CPU fallback, but the full paper workflow is substantially slower without CUDA.

## Reproduction

Run the notebooks in the order listed in `RUN_ORDER.txt`.

Notebook 01 uses resumable Optuna SQLite studies. If an interrupted run is restarted with the same repository and `outputs/` directory, completed work is reused according to the frozen protocol; it does not import settings or predictions from a different result directory.

Notebook 04 reads the frozen settings produced by Notebook 01 and writes only to:

```text
outputs/second_round_confirmatory/
```

It does not overwrite the original benchmark outputs.

## Generated outputs

All generated files stay under `outputs/`, including:

- benchmark predictions and aggregate metrics;
- horizon-level and origin-level loss files;
- dataset-local HPO audits and selected settings;
- router-calibration and mechanism-test outputs;
- XAI and posthoc tables/figures;
- second-round confirmatory results.

These generated artifacts are intentionally excluded from Git version control. A user can regenerate them by running the supplied workflow.

## Interpretation boundaries

The code is organized to reproduce the evidence reported in the manuscript, including its limitations:

- numerical rank is not automatically interpreted as statistically supported superiority;
- BDG-Edu and BDG-Dorm are same-site public external-source checks with dataset-local development, not frozen cross-site or zero-shot transfer tests;
- internal architectural components are evaluated empirically rather than assumed to be universally necessary;
- the second-round 1% relative-NMAE threshold is a revision-stage interpretation rule, not an original preregistration or a universal BEMS threshold;
- the storage-dispatch experiment is a constrained scenario analysis, not evidence of realized tariff, HVAC, comfort, rebound, or production-deployment benefits.
