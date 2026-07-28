# LASH: Leakage-Aware Sequential Hybrid Learning

A reproducible, validation-gated selective hybrid pipeline for hourly direct 24-step electricity-demand forecasting.

## Highlights

- 168-hour historical lookback and 24-hour direct forecasting horizon
- Leakage-safe daily, weekly, and routine-aware anchor construction
- Causal TCN, feature gating, and compact GRN sequential expert
- Ridge residual expert trained on the same information set
- Validation-only residual-gain selection and horizon-aware routing
- Chronological train, validation, and test protocol with a 23-origin purge gap
- No year-over-year demand features
- Automatic CPU/GPU selection with optional mixed-precision training

## Repository structure

```text
.
├── LASH_Cluster_Benchmark_GPU.ipynb
├── Cluster 1.csv
├── Cluster 2.csv
├── README.md
├── requirements.txt
└── .gitignore
```

## Dataset mapping

| Dataset key | File | Timestamp format | Train | Validation | Test |
|---|---|---|---|---|---|
| `CLUSTER_1` | `Cluster 1.csv` | `Date` string | 2015-03 to 2017-02 | 2017-03 to 2018-08 | 2018-09 to 2020-02 |
| `CLUSTER_2` | `Cluster 2.csv` | `Year/Month/Day/Hour` | 2015-09 to 2016-12 | 2017-01 to 2017-12 | 2018-01 to 2018-12 |

The notebook converts both files to one hourly schema. Source-provided lag columns are audited but are not used directly as model inputs.

## Quick start

1. Clone or download the repository.
2. Keep the notebook and both CSV files in the same directory.
3. Create and activate a Python environment.
4. Install the dependencies:

```bash
pip install -r requirements.txt
```

5. Install the PyTorch build appropriate for your CPU or CUDA environment.
6. Open `LASH_Cluster_Benchmark_GPU.ipynb` and run all cells.

The notebook defaults to:

```python
RUN_MODE = "smoke"
DATASETS_TO_RUN = ["CLUSTER_1", "CLUSTER_2"]
```

Smoke mode samples up to 512 forecast origins per split and is intended for an end-to-end wiring check. Change `RUN_MODE` to `"paper"` to process every valid origin.

## Configuration

Key settings are defined near the top of the notebook:

```python
RUN_MODE = "smoke"                    # "smoke" or "paper"
WEATHER_MODE = "benchmark_observed"   # or "historical_only"
DATASETS_TO_RUN = ["CLUSTER_1", "CLUSTER_2"]
REQUIRE_CUDA = False
USE_AMP = True
```

Set `LASH_DATA_ROOT` when the CSV files are stored outside the repository directory.

## Outputs

Generated artifacts are written to:

```text
outputs/LASH/<run-tag>/
```

The output folder contains:

- test predictions and aggregate metrics;
- horizon-level metrics;
- Optuna and Ridge search tables;
- router-calibration results;
- trained PyTorch and Ridge models;
- preprocessing objects;
- run configuration and package versions;
- 600-dpi PNG and vector PDF figures; and
- leakage-audit results.

## Reproducibility notes

- All random seeds are fixed.
- Preprocessing is fitted only on the permitted training partition.
- Router calibration uses only the held-out validation-calibration partition.
- Test targets are excluded from tuning, epoch selection, residual-gain selection, and router selection.
- The `benchmark_observed` weather mode is a benchmark condition. Deployment studies should use weather forecasts available at prediction time or report the `historical_only` configuration.

## Citation

When using this implementation in academic work, cite the associated LASH manuscript and describe the selected run mode, weather mode, data split, and hardware environment.
