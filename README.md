# LASH Revision Experiment Pipeline

This repository implements a frozen, chronology-preserving experiment pipeline for hourly updated direct 24-step electricity-demand forecasting.

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

Validation is divided chronologically into tuning and calibration segments using a 2:1 rule, with a 23-origin purge between them. After every setting is frozen, preprocessing and final models are refit on train + full validation, and the test partition is evaluated once.

## Benchmark suite

The benchmark includes persistence baselines, the leakage-safe anchor, Ridge, five tree-ensemble families (RF, GBM, XGBoost, LightGBM, CatBoost), six neural comparators (MLP, LSTM, GRU, CNN-LSTM, TCN, Transformer), and LASH.

Tree methods are implemented as **24 direct horizon-specific regressors** using the same causal origin-level information available to the neural models. This avoids recursive decoding and avoids relying on experimental vector-leaf multi-output implementations.

## Hyperparameter optimization

Search spaces are model-specific rather than forcing an equal trial count on search spaces of different dimensionality. The paper profile uses a transparent dimension-adaptive budget:

`trials = clip(6 × number_of_tuned_dimensions, 30, 60)`

and repeats each HPO search with seeds `42, 142, 242`. Optuna optimization is sequential (`n_jobs=1`) for reproducibility. All trials, convergence histories, selected settings, training times, inference latency, and model-size diagnostics are saved.

The tree search ranges retain the settings used in the preceding ensemble-learning studies as interior or boundary candidates, while broadening the spaces to avoid favoring any one implementation. The earlier residential ensemble study used GridSearchCV with five-fold cross-validation; this repository deliberately replaces that validation mechanism with chronological validation-tuning plus a purged calibration segment because the present task is an hourly rolling time-series forecast. The change is applied to every model family and is therefore part of the common fairness protocol.

## Notebook order

1. `00_Protocol_and_Data_Audit.ipynb` — verifies the four harmonized files, common feature contract, fixed chronological partitions, and search-space manifest.
2. `01_LASH_and_Fair_Benchmark.ipynb` — trains LASH and the full benchmark suite, performs repeated HPO, refits on train+validation, and exports Excel workbooks plus prediction/model artifacts.
3. `02_Matched_Ablation_Router_YoY.ipynb` — retunes structural/feature ablations, evaluates router sensitivity and NNLS alternatives, and introduces annual-information components separately.
4. `03_Posthoc_Statistics_XAI_Visualization.ipynb` — automatically discovers benchmark Excel workbooks and companion prediction/model artifacts, then performs dependence-aware inference, temporal/peak analyses, XAI, HPO-convergence analysis, and publication-ready visualization.

## Publication run

Use `RUN_PROFILE = "paper"` in Notebook 01. `smoke` is only an end-to-end wiring test and intentionally caps expensive tree lengths and neural epochs. Do not report smoke-mode values.

Because the full paper profile is computationally intensive, Optuna studies use SQLite storage and can resume interrupted searches. Keep the `outputs/` directory outside Git version control; release final compact result tables, origin-level loss files, and scripts as supplementary artifacts as appropriate.
