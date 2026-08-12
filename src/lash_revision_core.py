"""Core utilities for the reviewer-ready LASH revision experiments.

The module is intentionally explicit rather than clever.  It enforces one causal
information contract across four harmonized university-load datasets, supports
hourly rolling 168->24 forecasting, implements the LASH residual hybrid and a
broad benchmark suite, and saves enough intermediate artifacts for independent
statistical/XAI analysis.

Main design principles
----------------------
1. Test data are never used for preprocessing, HPO, epoch selection, residual
   gain selection, router selection, or model-family selection.
2. All learned benchmark models receive the same information set. Neural models
   receive it as sequences; tabular models receive an origin-level flattened
   representation of the same causal information.
3. Main experiments use historical-only weather.  Realized target-period
   weather is not a primary input.
4. Tree ensembles are direct: one regressor is trained for each horizon h=1..24.
5. Final models are refit on train+validation only after all choices are frozen.
6. Origin-level losses and predictions are persisted so dependence-aware tests
   can be rerun without retraining.
"""
from __future__ import annotations

import copy
import gc
import json
import math
import os
import platform
import random
import re
import time
import warnings
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed
from scipy.optimize import nnls
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------------------------------------------------------
# Global forecasting constants
# -----------------------------------------------------------------------------
LOOKBACK = 168
HORIZON = 24
ORIGIN_STRIDE_HOURS = 1
VALIDATION_TUNE_FRACTION = 2.0 / 3.0
VALIDATION_PURGE_ORIGINS = HORIZON - 1
FINAL_REFIT_SEEDS = (42, 142, 242)
HPO_SEEDS = (42, 142, 242)

# The original LASH paper used this alpha grid.  It is retained for continuity.
RIDGE_ALPHA_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)
RESIDUAL_GAIN_GRID = tuple(np.unique(np.r_[0.0, 1.0, np.linspace(0.25, 1.25, 21)]).tolist())

# Router defaults are retained only as one pre-specified candidate.  Notebook 02
# evaluates broad sensitivity and NNLS alternatives.
ROUTER_WEIGHT_GRID = tuple(np.linspace(0.0, 1.0, 21).tolist())
ROUTER_SMOOTH_WINDOW = 3
ROUTER_SHRINKAGE = 0.5
ROUTER_MIN_IMPROVEMENT = 0.005

# -----------------------------------------------------------------------------
# Dataset metadata
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetSpec:
    key: str
    name: str
    filename: str
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str
    external_validation: bool = False
    demand_unit: str = "original dataset unit"


DEFAULT_SPECS: Dict[str, DatasetSpec] = {
    "CLUSTER_1": DatasetSpec(
        key="CLUSTER_1", name="Cluster 1", filename="Cluster_1_Harmonized.csv",
        train_start="2015-03-01", train_end="2017-03-01",
        val_start="2017-03-01", val_end="2018-09-01",
        test_start="2018-09-01", test_end="2020-03-01",
    ),
    "CLUSTER_2": DatasetSpec(
        key="CLUSTER_2", name="Cluster 2", filename="Cluster_2_Harmonized.csv",
        train_start="2015-09-01", train_end="2017-01-01",
        val_start="2017-01-01", val_end="2018-01-01",
        test_start="2018-01-01", test_end="2019-01-01",
    ),
    "BDG_EDU": DatasetSpec(
        key="BDG_EDU", name="BDG-Edu Aggregate", filename="BDG_Edu_Harmonized.csv",
        train_start="2015-01-01", train_end="2015-07-01",
        val_start="2015-07-01", val_end="2015-10-01",
        test_start="2015-10-01", test_end="2016-01-01",
        external_validation=True,
    ),
    "BDG_DORM": DatasetSpec(
        key="BDG_DORM", name="BDG-Dorm Aggregate", filename="BDG_Dorm_Harmonized.csv",
        train_start="2015-01-01", train_end="2015-07-01",
        val_start="2015-07-01", val_end="2015-10-01",
        test_start="2015-10-01", test_end="2016-01-01",
        external_validation=True,
    ),
}


@dataclass
class ExperimentConfig:
    data_root: Path
    output_root: Path
    dataset_keys: Tuple[str, ...] = ("CLUSTER_1", "CLUSTER_2", "BDG_EDU", "BDG_DORM")
    weather_mode: str = "historical_only"
    run_profile: str = "paper"  # paper | smoke
    use_amp: bool = True
    require_cuda: bool = False
    tree_horizon_jobs: int = 4
    tree_threads_per_model: int = 1
    resume: bool = True
    final_refit_seeds: Tuple[int, ...] = FINAL_REFIT_SEEDS
    hpo_seeds: Tuple[int, ...] = HPO_SEEDS
    hpo_trials_per_dimension: int = 6
    hpo_min_trials: int = 30
    hpo_max_trials: int = 60
    max_epochs: int = 80
    early_stopping_patience: int = 10
    smoke_max_origins_per_split: int = 384
    primary_hpo_repeats: int = 3
    external_hpo_repeats: int = 3
    xai_sample_origins: int = 256
    save_models: bool = True
    specs: Dict[str, DatasetSpec] = field(default_factory=lambda: dict(DEFAULT_SPECS))

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root).expanduser().resolve()
        self.output_root = Path(self.output_root).expanduser().resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        if self.weather_mode != "historical_only":
            raise ValueError(
                "The primary revision pipeline is intentionally fixed to historical_only weather. "
                "Run oracle weather only in a separately labeled sensitivity experiment."
            )
        if self.run_profile not in {"paper", "smoke"}:
            raise ValueError("run_profile must be 'paper' or 'smoke'.")
        if self.run_profile == "smoke":
            self.max_epochs = min(self.max_epochs, 4)
            self.early_stopping_patience = min(self.early_stopping_patience, 2)

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def amp_enabled(self) -> bool:
        return bool(self.use_amp and self.device.type == "cuda")

    def hpo_repeat_count(self, dataset_key: str) -> int:
        if self.run_profile == "smoke":
            return 1
        return self.external_hpo_repeats if self.specs[dataset_key].external_validation else self.primary_hpo_repeats

    def hpo_trials(self, model_name: str) -> int:
        if self.run_profile == "smoke":
            return 2
        dims = MODEL_SEARCH_DIMENSIONS[model_name]
        return int(np.clip(
            dims * self.hpo_trials_per_dimension,
            self.hpo_min_trials,
            self.hpo_max_trials,
        ))


# -----------------------------------------------------------------------------
# Reproducibility and file discovery
# -----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def runtime_environment() -> Dict[str, Any]:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_build": torch.version.cuda,
    }
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        info.update({
            "gpu": torch.cuda.get_device_name(0),
            "gpu_vram_gb": round(prop.total_memory / 1024**3, 3),
        })
    return info


def _canonical_file_name(name: str) -> str:
    p = Path(name)
    stem = re.sub(r"\s*\(\d+\)$", "", p.stem).replace(" ", "_")
    return f"{stem}{p.suffix}".casefold()


def resolve_data_file(root: Path, expected: str) -> Path:
    root = Path(root)
    exact = root / expected
    if exact.exists():
        return exact
    canonical = _canonical_file_name(expected)
    matches = [p for p in root.glob("*.csv") if _canonical_file_name(p.name) == canonical]
    if not matches:
        raise FileNotFoundError(f"Could not find {expected} under {root}.")
    return sorted(matches, key=lambda p: (len(p.name), p.name.casefold()))[0]


def load_base_frame(spec: DatasetSpec, data_root: Path) -> pd.DataFrame:
    path = resolve_data_file(data_root, spec.filename)
    df = pd.read_csv(path)
    required = ["Date", "Holi", "Temp", "Humi", "WS", "Consumption"]
    missing = sorted(set(required).difference(df.columns))
    if missing:
        raise ValueError(f"{spec.name}: missing columns {missing}")
    df = df[required].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    for col in required[1:]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    df = df.set_index("Date").sort_index()
    if df.index.duplicated().any():
        raise ValueError(f"{spec.name}: duplicate timestamps")
    expected_index = pd.date_range(df.index.min(), df.index.max(), freq="h")
    if not df.index.equals(expected_index):
        raise ValueError(f"{spec.name}: non-hourly gap detected")
    if df.isna().any().any():
        raise ValueError(f"{spec.name}: missing base values remain; repair them in preprocessing, not here")
    if not set(df["Holi"].astype(int).unique()).issubset({0, 1}):
        raise ValueError(f"{spec.name}: Holi must be binary")
    weekend = df.index.dayofweek >= 5
    if not np.all(df.loc[weekend, "Holi"].to_numpy() == 1):
        raise ValueError(f"{spec.name}: weekend hours must have Holi=1")
    return df.astype({"Holi": int})


# -----------------------------------------------------------------------------
# Shared leakage-safe feature engineering
# -----------------------------------------------------------------------------
CALENDAR_COLS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "month_sin", "month_cos", "doy_sin", "doy_cos",
    "hour_shift8_sin", "hour_shift8_cos", "Holi", "Weekend",
    "holi_hour_sin", "holi_hour_cos",
]
RAW_WEATHER_COLS = ["Temp", "Humi", "WS"]
NONLINEAR_WEATHER_COLS = [
    "THI", "WCT", "HDD18", "CDD18", "HDD18_sq", "CDD18_sq", "Temp_sq", "Temp_x_Humi"
]
WEATHER_COLS = RAW_WEATHER_COLS + NONLINEAR_WEATHER_COLS
SAFE_DEMAND_COLS = [
    "cons_lag24_safe", "cons_lag48", "cons_lag168_safe",
    "Cons_avg_same_type_7", "recent_mean_24", "recent_mean_168", "recent_std_168", "anchor",
]
HISTORICAL_WEATHER_PROXY_COLS = [f"{c}_lag{lag}" for c in WEATHER_COLS for lag in (24, 168)]


def _calendar_matched_previous_year(series: pd.Series) -> pd.Series:
    idx = series.index
    prev = idx - pd.DateOffset(years=1)
    return pd.Series(series.reindex(prev).to_numpy(), index=idx, dtype=float)


def add_causal_features(base: pd.DataFrame, yoy_components: Sequence[str] = ()) -> pd.DataFrame:
    """Create the common feature set from the six-column harmonized base schema.

    `yoy_components` is only used by the dedicated component-wise sensitivity notebook.
    Valid component names: annual_lag, holiday_matched_proxy, annual_scaling, annual_anchor.
    The main benchmark passes an empty tuple.
    """
    x = base.copy()
    idx = x.index
    hour = idx.hour.to_numpy(float)
    dow = idx.dayofweek.to_numpy(float)
    month = idx.month.to_numpy(float)
    doy = idx.dayofyear.to_numpy(float)

    x["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    x["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    x["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    x["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    x["month_sin"] = np.sin(2 * np.pi * (month - 1.0) / 12.0)
    x["month_cos"] = np.cos(2 * np.pi * (month - 1.0) / 12.0)
    x["doy_sin"] = np.sin(2 * np.pi * (doy - 1.0) / 365.2425)
    x["doy_cos"] = np.cos(2 * np.pi * (doy - 1.0) / 365.2425)
    shifted_hour = (hour - 8.0) % 24.0
    x["hour_shift8_sin"] = np.sin(2 * np.pi * shifted_hour / 24.0)
    x["hour_shift8_cos"] = np.cos(2 * np.pi * shifted_hour / 24.0)
    x["Weekend"] = (idx.dayofweek >= 5).astype(int)
    x["holi_hour_sin"] = x["Holi"] * x["hour_sin"]
    x["holi_hour_cos"] = x["Holi"] * x["hour_cos"]

    T = x["Temp"].astype(float)
    H = x["Humi"].astype(float)
    V = x["WS"].astype(float).clip(lower=0)
    x["THI"] = (1.8 * T + 32.0) - ((0.55 - 0.0055 * H) * (1.8 * T - 26.0))
    x["WCT"] = 13.12 + 0.6215 * T - 11.37 * np.power(V, 0.16) + 0.3965 * T * np.power(V, 0.16)
    x["HDD18"] = np.maximum(18.0 - T, 0.0)
    x["CDD18"] = np.maximum(T - 18.0, 0.0)
    x["HDD18_sq"] = x["HDD18"] ** 2
    x["CDD18_sq"] = x["CDD18"] ** 2
    x["Temp_sq"] = T ** 2
    x["Temp_x_Humi"] = T * H

    y = x["Consumption"].astype(float)
    x["cons_lag24"] = y.shift(24)
    x["holi_lag24"] = x["Holi"].shift(24)
    x["cons_lag48"] = y.shift(48)
    x["cons_lag168"] = y.shift(168)
    x["holi_lag168"] = x["Holi"].shift(168)
    x["cons_lag336"] = y.shift(336)

    x["recent_mean_24"] = y.shift(24).rolling(24, min_periods=12).mean()
    x["recent_mean_168"] = y.shift(24).rolling(168, min_periods=48).mean()
    x["recent_std_168"] = y.shift(24).rolling(168, min_periods=48).std()

    tmp = x.assign(_Hour=idx.hour)
    routine_raw = tmp.groupby(["_Hour", "Holi"], sort=False)["Consumption"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).mean()
    )
    # Early in a series there may be fewer than seven same-hour observations in
    # the same holiday regime.  Use only earlier observed demand for a causal
    # fallback rather than model-stage imputation.  This preserves the declared
    # 168-hour lookback warm-up for every dataset.
    x["Cons_avg_same_type_7"] = routine_raw.fillna(x["recent_mean_168"]).fillna(x["recent_mean_24"])

    fallback = x["Cons_avg_same_type_7"].fillna(x["recent_mean_168"])
    x["cons_lag24_safe"] = x["cons_lag24"].where(x["holi_lag24"] == x["Holi"], fallback)
    x["cons_lag168_safe"] = x["cons_lag168"].where(x["holi_lag168"] == x["Holi"], fallback)

    # Main no-YoY anchor. Availability-normalized weights preserve the established
    # short-term component ratios without silently introducing annual information.
    candidates = [
        ("cons_lag24_safe", 0.35),
        ("cons_lag168_safe", 0.30),
        ("Cons_avg_same_type_7", 0.20),
    ]
    numerator = np.zeros(len(x), dtype=float)
    denominator = np.zeros(len(x), dtype=float)
    for col, weight in candidates:
        values = x[col].to_numpy(float)
        valid = np.isfinite(values)
        numerator += np.where(valid, values * weight, 0.0)
        denominator += valid * weight
    x["anchor"] = np.divide(
        numerator, denominator,
        out=x["recent_mean_168"].to_numpy(float).copy(),
        where=denominator > 0,
    )

    # Only historical weather values are available to future target hours in the
    # main revision benchmark. For s=t+h, s-24 <= t for all h in 1..24.
    for col in WEATHER_COLS:
        x[f"{col}_lag24"] = x[col].shift(24)
        x[f"{col}_lag168"] = x[col].shift(168)

    # Component-wise annual sensitivity. These fields are never generated in the
    # main benchmark, preventing accidental annual-feature use on BDG.
    allowed_yoy = {"annual_lag", "holiday_matched_proxy", "annual_scaling", "annual_anchor"}
    unknown = set(yoy_components).difference(allowed_yoy)
    if unknown:
        raise ValueError(f"Unknown YoY components: {sorted(unknown)}")
    if yoy_components:
        x["cons_yoy"] = _calendar_matched_previous_year(y)
        prev_holi = _calendar_matched_previous_year(x["Holi"].astype(float))
        x["holi_yoy"] = prev_holi
        x["cons_yoy_holiday_matched"] = x["cons_yoy"].where(prev_holi == x["Holi"], fallback)
        prev_recent = _calendar_matched_previous_year(x["recent_mean_168"])
        scale = x["recent_mean_168"] / prev_recent.replace(0, np.nan)
        x["annual_scale"] = scale.clip(0.5, 1.5)
        x["cons_yoy_scaled"] = x["cons_yoy"] * x["annual_scale"]
        if "annual_anchor" in yoy_components:
            # The annual component receives a conservative 0.15 share; all anchor
            # weights are then availability-normalized. This component is evaluated
            # separately rather than bundled with the other annual representations.
            vals = x["cons_yoy_holiday_matched"].to_numpy(float)
            valid = np.isfinite(vals)
            num2 = numerator + np.where(valid, vals * 0.15, 0.0)
            den2 = denominator + valid * 0.15
            x["anchor"] = np.divide(num2, den2, out=x["anchor"].to_numpy(float).copy(), where=den2 > 0)

    return x


def feature_contract(yoy_components: Sequence[str] = (), include_nonlinear_weather: bool = True,
                     include_phase_shift_calendar: bool = True) -> Tuple[List[str], List[str]]:
    calendar = list(CALENDAR_COLS)
    if not include_phase_shift_calendar:
        calendar = [c for c in calendar if not c.startswith("hour_shift8_")]
    weather = list(WEATHER_COLS if include_nonlinear_weather else RAW_WEATHER_COLS)
    past = ["Consumption", *calendar, *weather]
    future_weather = [f"{c}_lag{lag}" for c in weather for lag in (24, 168)]
    future = [*calendar, *SAFE_DEMAND_COLS, *future_weather]
    if yoy_components:
        if "annual_lag" in yoy_components:
            future.append("cons_yoy")
        if "holiday_matched_proxy" in yoy_components:
            future.append("cons_yoy_holiday_matched")
        if "annual_scaling" in yoy_components:
            future.extend(["annual_scale", "cons_yoy_scaled"])
        # annual_anchor changes anchor itself, so no extra future column is needed.
    return past, future


# -----------------------------------------------------------------------------
# Window construction and chronological partitioning
# -----------------------------------------------------------------------------
@dataclass
class WindowBundle:
    past: np.ndarray
    future: np.ndarray
    anchor: np.ndarray
    y: np.ndarray
    naive24: np.ndarray
    naive168: np.ndarray
    forecast_origin: np.ndarray
    target_time: np.ndarray
    split: str
    past_cols: List[str]
    future_cols: List[str]

    def __len__(self) -> int:
        return int(self.y.shape[0])


def _split_bounds(spec: DatasetSpec, split: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    if split == "train":
        return pd.Timestamp(spec.train_start), pd.Timestamp(spec.train_end)
    if split == "val":
        return pd.Timestamp(spec.val_start), pd.Timestamp(spec.val_end)
    if split == "test":
        return pd.Timestamp(spec.test_start), pd.Timestamp(spec.test_end)
    raise ValueError(split)


def build_windows(frame: pd.DataFrame, spec: DatasetSpec, split: str,
                  past_cols: Sequence[str], future_cols_base: Sequence[str],
                  smoke_max_origins: Optional[int] = None) -> WindowBundle:
    start, end = _split_bounds(spec, split)
    target_starts = pd.date_range(start, end - pd.Timedelta(hours=HORIZON), freq="h")
    target_pos = frame.index.get_indexer(target_starts)
    target_pos = target_pos[(target_pos >= LOOKBACK) & (target_pos + HORIZON <= len(frame))]
    if len(target_pos) == 0:
        raise ValueError(f"{spec.name}-{split}: no windows")
    if smoke_max_origins and len(target_pos) > smoke_max_origins:
        keep = np.linspace(0, len(target_pos) - 1, smoke_max_origins, dtype=int)
        target_pos = target_pos[keep]

    past_idx = target_pos[:, None] + np.arange(-LOOKBACK, 0)[None, :]
    future_idx = target_pos[:, None] + np.arange(0, HORIZON)[None, :]
    past_values = frame[list(past_cols)].to_numpy(np.float32)
    future_values = frame[list(future_cols_base)].to_numpy(np.float32)
    anchor_values = frame["anchor"].to_numpy(np.float32)
    y_values = frame["Consumption"].to_numpy(np.float32)
    naive24_values = frame["cons_lag24"].to_numpy(np.float32)
    naive168_values = frame["cons_lag168"].to_numpy(np.float32)

    past = past_values[past_idx]
    future = future_values[future_idx]
    lead_hour = np.broadcast_to(
        np.arange(1, HORIZON + 1, dtype=np.float32)[None, :, None],
        (len(target_pos), HORIZON, 1),
    )
    future = np.concatenate([future, lead_hour], axis=-1)
    anchor = anchor_values[future_idx]
    y = y_values[future_idx]
    naive24 = naive24_values[future_idx]
    naive168 = naive168_values[future_idx]

    valid = (
        np.isfinite(past).all(axis=(1, 2))
        & np.isfinite(future).all(axis=(1, 2))
        & np.isfinite(anchor).all(axis=1)
        & np.isfinite(y).all(axis=1)
        & np.isfinite(naive24).all(axis=1)
        & np.isfinite(naive168).all(axis=1)
    )
    past, future = past[valid], future[valid]
    anchor, y = anchor[valid], y[valid]
    naive24, naive168 = naive24[valid], naive168[valid]
    target_pos, future_idx = target_pos[valid], future_idx[valid]
    if len(target_pos) == 0:
        raise ValueError(f"{spec.name}-{split}: all windows invalid")

    bundle = WindowBundle(
        past=past, future=future, anchor=anchor, y=y,
        naive24=naive24, naive168=naive168,
        forecast_origin=frame.index.to_numpy()[target_pos - 1],
        target_time=frame.index.to_numpy()[future_idx],
        split=split, past_cols=list(past_cols), future_cols=[*future_cols_base, "lead_hour"],
    )
    origins = pd.to_datetime(bundle.forecast_origin)
    assert np.all(pd.to_datetime(bundle.target_time[:, 0]) == origins + pd.Timedelta(hours=1))
    assert np.all(pd.to_datetime(bundle.target_time[:, -1]) == origins + pd.Timedelta(hours=HORIZON))
    return bundle


def take_bundle(bundle: WindowBundle, indices: Sequence[int], split: Optional[str] = None) -> WindowBundle:
    idx = np.asarray(indices, dtype=int)
    return WindowBundle(
        past=bundle.past[idx], future=bundle.future[idx], anchor=bundle.anchor[idx], y=bundle.y[idx],
        naive24=bundle.naive24[idx], naive168=bundle.naive168[idx],
        forecast_origin=bundle.forecast_origin[idx], target_time=bundle.target_time[idx],
        split=split or bundle.split, past_cols=bundle.past_cols, future_cols=bundle.future_cols,
    )


def concat_bundles(*bundles: WindowBundle, split: str) -> WindowBundle:
    first = bundles[0]
    merged = WindowBundle(
        past=np.concatenate([b.past for b in bundles]),
        future=np.concatenate([b.future for b in bundles]),
        anchor=np.concatenate([b.anchor for b in bundles]),
        y=np.concatenate([b.y for b in bundles]),
        naive24=np.concatenate([b.naive24 for b in bundles]),
        naive168=np.concatenate([b.naive168 for b in bundles]),
        forecast_origin=np.concatenate([b.forecast_origin for b in bundles]),
        target_time=np.concatenate([b.target_time for b in bundles]),
        split=split, past_cols=first.past_cols, future_cols=first.future_cols,
    )
    order = np.argsort(merged.forecast_origin)
    return take_bundle(merged, order, split=split)


def split_validation_bundle(val: WindowBundle) -> Tuple[WindowBundle, WindowBundle]:
    cut = int(math.floor(len(val) * VALIDATION_TUNE_FRACTION))
    calibration_start = cut + VALIDATION_PURGE_ORIGINS
    if cut < 2 * HORIZON or len(val) - calibration_start < 2 * HORIZON:
        raise ValueError("Validation period too short for tune/purge/calibration split")
    tune = take_bundle(val, np.arange(0, cut), split="val_tune")
    cal = take_bundle(val, np.arange(calibration_start, len(val)), split="val_calibration")
    assert pd.Timestamp(tune.target_time.max()) < pd.Timestamp(cal.target_time.min())
    return tune, cal


def build_dataset_bundles(config: ExperimentConfig, dataset_key: str,
                          yoy_components: Sequence[str] = (),
                          include_nonlinear_weather: bool = True,
                          include_phase_shift_calendar: bool = True) -> Tuple[pd.DataFrame, Dict[str, WindowBundle]]:
    spec = config.specs[dataset_key]
    base = load_base_frame(spec, config.data_root)
    frame = add_causal_features(base, yoy_components=yoy_components)
    past_cols, future_cols = feature_contract(
        yoy_components=yoy_components,
        include_nonlinear_weather=include_nonlinear_weather,
        include_phase_shift_calendar=include_phase_shift_calendar,
    )
    smoke_n = config.smoke_max_origins_per_split if config.run_profile == "smoke" else None
    bundles = {
        split: build_windows(frame, spec, split, past_cols, future_cols, smoke_max_origins=smoke_n)
        for split in ("train", "val", "test")
    }
    assert bundles["train"].target_time.max() < bundles["val"].target_time.min()
    assert bundles["val"].target_time.max() < bundles["test"].target_time.min()
    return frame, bundles


# -----------------------------------------------------------------------------
# Metrics and origin-level evaluation
# -----------------------------------------------------------------------------
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt = np.asarray(y_true, float).reshape(-1)
    yp = np.asarray(y_pred, float).reshape(-1)
    err = yt - yp
    nz = np.abs(yt) > 1e-12
    mape = float(np.mean(np.abs(err[nz] / yt[nz])) * 100.0) if nz.any() else np.nan
    mean_y = float(np.mean(yt))
    cvrmse = float(np.sqrt(np.mean(err**2)) / mean_y * 100.0)
    nmae = float(np.mean(np.abs(err)) / mean_y * 100.0)
    return {
        "MAPE": mape,
        "CVRMSE": cvrmse,
        "NMAE": nmae,
        "selection_score": float(np.nanmean([mape, cvrmse, nmae])),
        "MAPE_excluded_zero_count": int((~nz).sum()),
    }


def per_horizon_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame([{"horizon": h + 1, **regression_metrics(y_true[:, h], y_pred[:, h])}
                         for h in range(HORIZON)])


def origin_loss_frame(bundle: WindowBundle, prediction: np.ndarray, model: str, seed: int) -> pd.DataFrame:
    actual = bundle.y.astype(float)
    pred = np.asarray(prediction, float)
    mean_demand = float(np.mean(actual))
    abs_err = np.abs(actual - pred)
    sq_err = (actual - pred) ** 2
    return pd.DataFrame({
        "forecast_origin": pd.to_datetime(bundle.forecast_origin),
        "model": model,
        "seed": int(seed),
        "origin_MAE": abs_err.mean(axis=1),
        "origin_NMAE_pct": abs_err.mean(axis=1) / mean_demand * 100.0,
        "origin_MSE": sq_err.mean(axis=1),
        "origin_RMSE": np.sqrt(sq_err.mean(axis=1)),
    })


def prediction_matrix_path(dataset_dir: Path, model: str, seed: int) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", model)
    return dataset_dir / "predictions" / f"{safe}__seed{seed}.npz"


def save_prediction_matrix(dataset_dir: Path, bundle: WindowBundle, prediction: np.ndarray,
                           model: str, seed: int) -> Path:
    p = prediction_matrix_path(dataset_dir, model, seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p,
        forecast_origin=bundle.forecast_origin,
        target_time=bundle.target_time,
        actual=bundle.y.astype(np.float32),
        prediction=np.asarray(prediction, np.float32),
    )
    return p


# -----------------------------------------------------------------------------
# Scaling and tabular design
# -----------------------------------------------------------------------------
@dataclass
class NeuralScaler:
    past_scaler: StandardScaler
    future_scaler: StandardScaler
    target_scaler: StandardScaler


def fit_neural_scaler(bundle: WindowBundle) -> NeuralScaler:
    if not (np.isfinite(bundle.past).all() and np.isfinite(bundle.future).all()):
        raise ValueError("Non-finite neural input reached model preprocessing")
    p = bundle.past.reshape(-1, bundle.past.shape[-1])
    f = bundle.future.reshape(-1, bundle.future.shape[-1])
    r = (bundle.y - bundle.anchor).reshape(-1, 1)
    return NeuralScaler(
        past_scaler=StandardScaler().fit(p),
        future_scaler=StandardScaler().fit(f),
        target_scaler=StandardScaler().fit(r),
    )


def transform_neural_inputs(bundle: WindowBundle, scaler: NeuralScaler) -> Tuple[np.ndarray, np.ndarray]:
    n = len(bundle)
    p = scaler.past_scaler.transform(bundle.past.reshape(-1, bundle.past.shape[-1])).reshape(n, LOOKBACK, -1)
    f = scaler.future_scaler.transform(bundle.future.reshape(-1, bundle.future.shape[-1])).reshape(n, HORIZON, -1)
    return p.astype(np.float32), f.astype(np.float32)


def transform_neural_train(bundle: WindowBundle, scaler: NeuralScaler) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p, f = transform_neural_inputs(bundle, scaler)
    r = scaler.target_scaler.transform((bundle.y - bundle.anchor).reshape(-1, 1)).reshape(len(bundle), HORIZON)
    return p, f, r.astype(np.float32)


def tabular_feature_names(bundle: WindowBundle) -> List[str]:
    names: List[str] = []
    names += [f"past_consumption_lag_{LOOKBACK-i}" for i in range(LOOKBACK)]
    for prefix in ("last", "mean168", "std168"):
        names += [f"{prefix}__{c}" for c in bundle.past_cols]
    for h in range(HORIZON):
        names += [f"h{h+1:02d}__{c}" for c in bundle.future_cols]
    names += [f"anchor_h{h+1:02d}" for h in range(HORIZON)]
    return names


def make_tabular_origin_design(bundle: WindowBundle) -> np.ndarray:
    # Every tabular model receives exactly the same origin-level information.
    cidx = bundle.past_cols.index("Consumption")
    design = np.concatenate([
        bundle.past[:, :, cidx],
        bundle.past[:, -1, :],
        bundle.past.mean(axis=1),
        bundle.past.std(axis=1),
        bundle.future.reshape(len(bundle), -1),
        bundle.anchor,
    ], axis=1)
    if not np.isfinite(design).all():
        raise ValueError("Non-finite tabular design")
    return np.ascontiguousarray(design, dtype=np.float32)


@dataclass
class RidgeFit:
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    model: Ridge
    train_seconds: float
    parameter_count: int


def fit_ridge(bundle: WindowBundle, alpha: float) -> RidgeFit:
    started = time.perf_counter()
    X = make_tabular_origin_design(bundle)
    R = bundle.y - bundle.anchor
    x_scaler = StandardScaler().fit(X)
    y_scaler = StandardScaler().fit(R)
    Xs = x_scaler.transform(X)
    Rs = y_scaler.transform(R)
    model = Ridge(alpha=float(alpha), solver="lsqr", tol=1e-4, max_iter=5000)
    model.fit(Xs, Rs)
    return RidgeFit(
        x_scaler=x_scaler, y_scaler=y_scaler, model=model,
        train_seconds=time.perf_counter() - started,
        parameter_count=int(model.coef_.size + model.intercept_.size),
    )


def predict_ridge_raw(result: RidgeFit, bundle: WindowBundle) -> np.ndarray:
    Xs = result.x_scaler.transform(make_tabular_origin_design(bundle))
    residual = result.y_scaler.inverse_transform(result.model.predict(Xs))
    return bundle.anchor + residual


def apply_residual_gain(bundle: WindowBundle, raw_prediction: np.ndarray, gain: float) -> np.ndarray:
    return np.maximum(0.0, bundle.anchor + float(gain) * (raw_prediction - bundle.anchor))


def select_residual_gain(bundle: WindowBundle, raw_prediction: np.ndarray) -> Tuple[float, pd.DataFrame]:
    rows = []
    for gain in RESIDUAL_GAIN_GRID:
        pred = apply_residual_gain(bundle, raw_prediction, gain)
        rows.append({"residual_gain": gain, **regression_metrics(bundle.y, pred)})
    table = pd.DataFrame(rows)
    table["distance_to_one"] = np.abs(table["residual_gain"] - 1.0)
    best = table.sort_values(["selection_score", "distance_to_one"]).iloc[0]
    return float(best["residual_gain"]), table.drop(columns="distance_to_one")


def tune_ridge(train: WindowBundle, val_tune: WindowBundle) -> Tuple[Dict[str, Any], pd.DataFrame]:
    rows = []
    for alpha in RIDGE_ALPHA_GRID:
        fit = fit_ridge(train, alpha)
        raw = predict_ridge_raw(fit, val_tune)
        gain, _ = select_residual_gain(val_tune, raw)
        pred = apply_residual_gain(val_tune, raw, gain)
        rows.append({"alpha": alpha, "gain": gain, **regression_metrics(val_tune.y, pred),
                     "train_seconds": fit.train_seconds})
    table = pd.DataFrame(rows).sort_values(["selection_score", "alpha"]).reset_index(drop=True)
    best = table.iloc[0]
    return {"alpha": float(best.alpha), "gain": float(best.gain)}, table


# -----------------------------------------------------------------------------
# Neural model definitions
# -----------------------------------------------------------------------------
class NumpySequenceDataset(Dataset):
    def __init__(self, past: np.ndarray, future: np.ndarray, target: np.ndarray):
        self.past = torch.from_numpy(past)
        self.future = torch.from_numpy(future)
        self.target = torch.from_numpy(target)
    def __len__(self) -> int:
        return len(self.target)
    def __getitem__(self, idx: int):
        return self.past[idx], self.future[idx], self.target[idx]


class Chomp1d(nn.Module):
    def __init__(self, size: int):
        super().__init__(); self.size = int(size)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.size].contiguous() if self.size > 0 else x


class LiteTemporalBlock(nn.Module):
    """Depthwise causal convolution + pointwise GLU + residual LayerNorm."""
    def __init__(self, d_model: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.depthwise = nn.Conv1d(d_model, d_model, kernel_size, padding=padding,
                                   dilation=dilation, groups=d_model)
        self.chomp = Chomp1d(padding)
        self.pointwise = nn.Conv1d(d_model, 2 * d_model, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.transpose(1, 2)
        z = self.chomp(self.depthwise(z))
        z = F.glu(self.pointwise(z), dim=1).transpose(1, 2)
        return self.norm(x + self.dropout(z))


class FullTemporalBlock(nn.Module):
    """Conventional full-channel causal TCN block used by the benchmark TCN."""
    def __init__(self, d_model: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.transpose(1, 2)
        z = self.dropout(F.gelu(self.chomp1(self.conv1(z))))
        z = self.dropout(F.gelu(self.chomp2(self.conv2(z)))).transpose(1, 2)
        return self.norm(x + z)


class GatedResidualBlock(nn.Module):
    """Gated Residual Block (GRB), distinct from a gated recurrent unit (GRU)."""
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.candidate = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.gate = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + torch.sigmoid(self.gate(x)) * self.candidate(x))


class HorizonDecoder(nn.Module):
    def __init__(self, d_model: int, n_future: int, dropout: float, use_grb: bool = False,
                 use_horizon_embedding: bool = True):
        super().__init__()
        self.future_projection = nn.Sequential(nn.Linear(n_future, d_model), nn.GELU(), nn.Dropout(dropout))
        self.horizon_embedding = nn.Parameter(torch.zeros(1, HORIZON, d_model), requires_grad=use_horizon_embedding)
        nn.init.normal_(self.horizon_embedding, 0.0, 0.02)
        self.use_horizon_embedding = use_horizon_embedding
        self.block = GatedResidualBlock(d_model, dropout) if use_grb else nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(d_model)
        )
        self.head = nn.Sequential(nn.Linear(d_model, max(16, d_model // 2)), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(max(16, d_model // 2), 1))
    def forward(self, context: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        z = self.future_projection(future) + context.unsqueeze(1)
        if self.use_horizon_embedding:
            z = z + self.horizon_embedding
        return self.head(self.block(z)).squeeze(-1)


class MLPResidualNet(nn.Module):
    def __init__(self, n_past: int, n_future: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        dims = [LOOKBACK * n_past] + [hidden] * layers
        blocks = []
        for a, b in zip(dims[:-1], dims[1:]):
            blocks += [nn.Linear(a, b), nn.GELU(), nn.Dropout(dropout)]
        self.encoder = nn.Sequential(*blocks)
        self.decoder = HorizonDecoder(hidden, n_future, dropout)
    def forward(self, past, future, return_explanation: bool = False):
        context = self.encoder(past.flatten(1))
        out = self.decoder(context, future)
        return (out, {}) if return_explanation else out


class RecurrentResidualNet(nn.Module):
    def __init__(self, cell: str, n_past: int, n_future: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.cell = cell
        self.rnn = cls(n_past, hidden, num_layers=layers, batch_first=True,
                       dropout=dropout if layers > 1 else 0.0)
        self.decoder = HorizonDecoder(hidden, n_future, dropout)
    def forward(self, past, future, return_explanation: bool = False):
        _, state = self.rnn(past)
        h = state[0] if self.cell == "lstm" else state
        context = h[-1]
        out = self.decoder(context, future)
        return (out, {}) if return_explanation else out


class CNNLSTMResidualNet(nn.Module):
    def __init__(self, n_past: int, n_future: int, conv_channels: int, kernel_size: int,
                 hidden: int, layers: int, dropout: float):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(n_past, conv_channels, kernel_size, padding=0)
        self.rnn = nn.LSTM(conv_channels, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.decoder = HorizonDecoder(hidden, n_future, dropout)
    def forward(self, past, future, return_explanation: bool = False):
        z = F.pad(past.transpose(1, 2), (self.kernel_size - 1, 0))
        z = F.gelu(self.conv(z)).transpose(1, 2)
        _, (h, _) = self.rnn(z)
        out = self.decoder(h[-1], future)
        return (out, {}) if return_explanation else out


class TCNResidualNet(nn.Module):
    def __init__(self, n_past: int, n_future: int, d_model: int, layers: int, kernel_size: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(n_past, d_model)
        self.blocks = nn.ModuleList([FullTemporalBlock(d_model, kernel_size, 2**i, dropout) for i in range(layers)])
        self.decoder = HorizonDecoder(d_model, n_future, dropout)
    def forward(self, past, future, return_explanation: bool = False):
        z = self.proj(past)
        for b in self.blocks: z = b(z)
        context = 0.5 * z[:, -1] + 0.5 * z.mean(dim=1)
        out = self.decoder(context, future)
        return (out, {}) if return_explanation else out


class TransformerResidualNet(nn.Module):
    def __init__(self, n_past: int, n_future: int, d_model: int, layers: int, nhead: int,
                 ff_mult: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(n_past, d_model)
        self.pos = nn.Parameter(torch.zeros(1, LOOKBACK, d_model))
        nn.init.normal_(self.pos, 0.0, 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * ff_mult,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.decoder = HorizonDecoder(d_model, n_future, dropout)
    def forward(self, past, future, return_explanation: bool = False):
        z = self.encoder(self.proj(past) + self.pos)
        context = 0.5 * z[:, -1] + 0.5 * z.mean(dim=1)
        out = self.decoder(context, future)
        return (out, {}) if return_explanation else out


class LASHSequentialNet(nn.Module):
    def __init__(self, n_past: int, n_future: int, d_model: int, tcn_layers: int,
                 kernel_size: int, dropout: float, use_feature_gates: bool = True,
                 use_tcn: bool = True, learned_pooling: bool = True, use_grb: bool = True,
                 use_horizon_embedding: bool = True):
        super().__init__()
        self.n_past = n_past; self.n_future = n_future
        self.use_feature_gates = use_feature_gates
        self.use_tcn = use_tcn
        self.learned_pooling = learned_pooling
        gate_hidden = max(16, n_past)
        self.past_feature_gate = nn.Sequential(nn.Linear(n_past, gate_hidden), nn.GELU(), nn.Linear(gate_hidden, n_past))
        self.future_feature_gate = nn.Sequential(nn.LayerNorm(n_future), nn.Linear(n_future, n_future))
        self.past_projection = nn.Linear(n_past, d_model)
        self.temporal_blocks = nn.ModuleList([
            LiteTemporalBlock(d_model, kernel_size, 2**i, dropout) for i in range(tcn_layers)
        ])
        self.temporal_score = nn.Sequential(nn.Linear(d_model, max(8, d_model // 2)), nn.Tanh(),
                                            nn.Linear(max(8, d_model // 2), 1))
        self.context_mix = nn.Sequential(nn.Linear(2*d_model, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.future_projection = nn.Sequential(nn.Linear(n_future, d_model), nn.GELU(), nn.Dropout(dropout))
        self.horizon_embedding = nn.Parameter(torch.zeros(1, HORIZON, d_model), requires_grad=use_horizon_embedding)
        nn.init.normal_(self.horizon_embedding, 0.0, 0.02)
        self.use_horizon_embedding = use_horizon_embedding
        self.decoder = GatedResidualBlock(d_model, dropout) if use_grb else nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(d_model)
        )
        self.output_head = nn.Sequential(nn.Linear(d_model, max(16, d_model // 2)), nn.GELU(), nn.Dropout(dropout),
                                         nn.Linear(max(16, d_model // 2), 1))

    @staticmethod
    def _normalized_softmax_gate(logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=-1) * logits.shape[-1]

    def forward(self, past: torch.Tensor, future: torch.Tensor, return_explanation: bool = False):
        if self.use_feature_gates:
            past_weights = self._normalized_softmax_gate(self.past_feature_gate(past.mean(dim=1)))
        else:
            past_weights = torch.ones(past.shape[0], past.shape[-1], device=past.device)
        z = self.past_projection(past * past_weights.unsqueeze(1))
        if self.use_tcn:
            for block in self.temporal_blocks: z = block(z)
        if self.learned_pooling:
            temporal_weights = torch.softmax(self.temporal_score(z).squeeze(-1), dim=-1)
        else:
            temporal_weights = torch.full((z.shape[0], z.shape[1]), 1.0 / z.shape[1], device=z.device)
        pooled = torch.sum(z * temporal_weights.unsqueeze(-1), dim=1)
        context = self.context_mix(torch.cat([pooled, z[:, -1]], dim=-1))

        if self.use_feature_gates:
            future_weights = self._normalized_softmax_gate(self.future_feature_gate(future))
        else:
            future_weights = torch.ones_like(future)
        decoded = self.future_projection(future * future_weights) + context.unsqueeze(1)
        if self.use_horizon_embedding:
            decoded = decoded + self.horizon_embedding
        decoded = self.decoder(decoded)
        out = self.output_head(decoded).squeeze(-1)
        if return_explanation:
            return out, {
                "temporal_weights": temporal_weights,
                "past_feature_weights": past_weights,
                "future_feature_weights": future_weights,
            }
        return out


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def make_neural_model(model_name: str, params: Mapping[str, Any], n_past: int, n_future: int,
                      ablation: Optional[Mapping[str, Any]] = None) -> nn.Module:
    p = dict(params); a = dict(ablation or {})
    if model_name == "MLP":
        return MLPResidualNet(n_past, n_future, int(p["hidden"]), int(p["layers"]), float(p["dropout"]))
    if model_name in {"LSTM", "GRU"}:
        return RecurrentResidualNet(model_name.lower(), n_past, n_future, int(p["hidden"]), int(p["layers"]), float(p["dropout"]))
    if model_name == "CNN_LSTM":
        return CNNLSTMResidualNet(n_past, n_future, int(p["conv_channels"]), int(p["kernel_size"]),
                                  int(p["hidden"]), int(p["layers"]), float(p["dropout"]))
    if model_name == "TCN":
        return TCNResidualNet(n_past, n_future, int(p["d_model"]), int(p["tcn_layers"]),
                              int(p["kernel_size"]), float(p["dropout"]))
    if model_name == "TRANSFORMER":
        return TransformerResidualNet(n_past, n_future, int(p["d_model"]), int(p["layers"]),
                                      int(p["nhead"]), int(p["ff_mult"]), float(p["dropout"]))
    if model_name == "LASH_SEQ":
        return LASHSequentialNet(
            n_past, n_future, int(p["d_model"]), int(p["tcn_layers"]), int(p["kernel_size"]), float(p["dropout"]),
            use_feature_gates=bool(a.get("use_feature_gates", True)),
            use_tcn=bool(a.get("use_tcn", True)),
            learned_pooling=bool(a.get("learned_pooling", True)),
            use_grb=bool(a.get("use_grb", True)),
            use_horizon_embedding=bool(a.get("use_horizon_embedding", True)),
        )
    raise ValueError(model_name)


# -----------------------------------------------------------------------------
# Neural training / HPO
# -----------------------------------------------------------------------------
@dataclass
class NeuralFit:
    model: nn.Module
    scaler: NeuralScaler
    best_epoch: int
    history: List[Dict[str, float]]
    train_seconds: float
    parameter_count: int
    peak_gpu_memory_mb: float


def _autocast(config: ExperimentConfig):
    return torch.autocast(device_type=config.device.type, dtype=torch.float16,
                          enabled=config.amp_enabled)


def _grad_scaler(config: ExperimentConfig):
    try:
        return torch.amp.GradScaler("cuda", enabled=config.amp_enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=config.amp_enabled)


def trajectory_loss(prediction: torch.Tensor, target: torch.Tensor, huber_beta: float,
                    shape_loss_weight: float) -> torch.Tensor:
    point = F.smooth_l1_loss(prediction, target, beta=huber_beta)
    if shape_loss_weight <= 0:
        return point
    shape = F.smooth_l1_loss(prediction[:, 1:] - prediction[:, :-1],
                             target[:, 1:] - target[:, :-1], beta=huber_beta)
    return point + shape_loss_weight * shape


@torch.no_grad()
def _predict_neural_scaled(model: nn.Module, past: np.ndarray, future: np.ndarray,
                           batch_size: int, config: ExperimentConfig,
                           return_explanation: bool = False):
    model.eval()
    dummy = np.zeros((len(past), HORIZON), np.float32)
    loader = DataLoader(NumpySequenceDataset(past, future, dummy), batch_size=batch_size,
                        shuffle=False, num_workers=0, pin_memory=config.device.type == "cuda")
    outs = []; aux_acc: Dict[str, List[np.ndarray]] = {}
    for p, f, _ in loader:
        p = p.to(config.device, non_blocking=True); f = f.to(config.device, non_blocking=True)
        with _autocast(config):
            if return_explanation:
                out, aux = model(p, f, return_explanation=True)
            else:
                out = model(p, f)
        outs.append(out.float().cpu().numpy())
        if return_explanation:
            for k, v in aux.items():
                aux_acc.setdefault(k, []).append(v.float().cpu().numpy())
    pred = np.concatenate(outs, axis=0)
    if return_explanation:
        return pred, {k: np.concatenate(v, axis=0) for k, v in aux_acc.items()}
    return pred


def predict_neural_raw(fit: NeuralFit, bundle: WindowBundle, params: Mapping[str, Any],
                       config: ExperimentConfig, return_explanation: bool = False):
    p, f = transform_neural_inputs(bundle, fit.scaler)
    output = _predict_neural_scaled(fit.model, p, f, int(params["batch_size"]), config,
                                    return_explanation=return_explanation)
    if return_explanation:
        scaled, aux = output
    else:
        scaled = output
    residual = fit.scaler.target_scaler.inverse_transform(scaled.reshape(-1, 1)).reshape(len(bundle), HORIZON)
    raw = bundle.anchor + residual
    return (raw, aux) if return_explanation else raw


def fit_neural(model_name: str, train_bundle: WindowBundle, params: Mapping[str, Any],
               config: ExperimentConfig, validation_bundle: Optional[WindowBundle] = None,
               fixed_epochs: Optional[int] = None, trial: Optional[optuna.Trial] = None,
               seed: int = 42, ablation: Optional[Mapping[str, Any]] = None) -> NeuralFit:
    if (validation_bundle is None) == (fixed_epochs is None):
        raise ValueError("Use validation_bundle for HPO/early stopping or fixed_epochs for frozen refit")
    set_seed(seed)
    if config.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    scaler = fit_neural_scaler(train_bundle)
    tr_p, tr_f, tr_r = transform_neural_train(train_bundle, scaler)
    model = make_neural_model(model_name, params, tr_p.shape[-1], tr_f.shape[-1], ablation=ablation).to(config.device)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        NumpySequenceDataset(tr_p, tr_f, tr_r), batch_size=int(params["batch_size"]),
        shuffle=True, generator=generator, num_workers=0,
        pin_memory=config.device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(params["learning_rate"]),
                                  weight_decay=float(params["weight_decay"]))
    epochs = int(fixed_epochs if fixed_epochs is not None else config.max_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    grad_scaler = _grad_scaler(config)
    if validation_bundle is not None:
        va_p, va_f = transform_neural_inputs(validation_bundle, scaler)

    best_score = np.inf; best_state = None; best_epoch = 0; stale = 0; history = []
    for epoch in range(epochs):
        model.train(); losses = []
        for p, f, target in loader:
            p = p.to(config.device, non_blocking=True); f = f.to(config.device, non_blocking=True)
            target = target.to(config.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(config):
                pred = model(p, f)
                loss = trajectory_loss(pred, target, float(params["huber_beta"]), float(params["shape_loss_weight"]))
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_scaler.step(optimizer); grad_scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        row: Dict[str, float] = {"epoch": epoch + 1, "train_loss": float(np.mean(losses))}
        if validation_bundle is not None:
            scaled = _predict_neural_scaled(model, va_p, va_f, int(params["batch_size"]), config)
            residual = scaler.target_scaler.inverse_transform(scaled.reshape(-1, 1)).reshape(len(validation_bundle), HORIZON)
            raw = validation_bundle.anchor + residual
            gain, _ = select_residual_gain(validation_bundle, raw)
            val_pred = apply_residual_gain(validation_bundle, raw, gain)
            metrics = regression_metrics(validation_bundle.y, val_pred)
            score = metrics["selection_score"]
            row.update({f"val_{k}": float(v) for k, v in metrics.items() if np.isscalar(v)})
            row["val_gain"] = gain
            if score < best_score - 1e-8:
                best_score = score; best_epoch = epoch + 1; stale = 0
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            else:
                stale += 1
            if trial is not None:
                trial.report(score, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            if stale >= config.early_stopping_patience:
                history.append(row); break
        else:
            best_epoch = epoch + 1
        history.append(row)
    if validation_bundle is not None:
        if best_state is None:
            raise RuntimeError("No best neural state")
        model.load_state_dict(best_state)
    peak_mb = 0.0
    if config.device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    return NeuralFit(model=model.eval(), scaler=scaler, best_epoch=int(best_epoch), history=history,
                     train_seconds=time.perf_counter() - started,
                     parameter_count=count_parameters(model), peak_gpu_memory_mb=float(peak_mb))


def suggest_neural_params(model_name: str, trial: optuna.Trial) -> Dict[str, Any]:
    common = {
        "dropout": trial.suggest_float("dropout", 0.05, 0.30),
        "learning_rate": trial.suggest_float("learning_rate", 2e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-8, 5e-3, log=True),
        "huber_beta": trial.suggest_float("huber_beta", 0.5, 1.5),
        "shape_loss_weight": trial.suggest_categorical("shape_loss_weight", [0.0, 0.05, 0.10, 0.20]),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256]),
    }
    if model_name == "MLP":
        common.update(hidden=trial.suggest_categorical("hidden", [64, 128, 256]),
                      layers=trial.suggest_int("layers", 1, 3))
    elif model_name in {"LSTM", "GRU"}:
        common.update(hidden=trial.suggest_categorical("hidden", [32, 64, 96, 128]),
                      layers=trial.suggest_int("layers", 1, 3))
    elif model_name == "CNN_LSTM":
        common.update(conv_channels=trial.suggest_categorical("conv_channels", [32, 64, 96]),
                      kernel_size=trial.suggest_categorical("kernel_size", [3, 5, 7]),
                      hidden=trial.suggest_categorical("hidden", [32, 64, 96]),
                      layers=trial.suggest_int("layers", 1, 2))
    elif model_name == "TCN":
        common.update(d_model=trial.suggest_categorical("d_model", [32, 48, 64, 96]),
                      tcn_layers=trial.suggest_int("tcn_layers", 2, 4),
                      kernel_size=trial.suggest_categorical("kernel_size", [3, 5, 7]))
    elif model_name == "TRANSFORMER":
        d_model = trial.suggest_categorical("d_model", [32, 48, 64, 96])
        valid_heads = [h for h in (2, 4, 8) if d_model % h == 0]
        common.update(d_model=d_model, layers=trial.suggest_int("layers", 1, 3),
                      nhead=trial.suggest_categorical("nhead", valid_heads),
                      ff_mult=trial.suggest_categorical("ff_mult", [2, 4]))
    elif model_name == "LASH_SEQ":
        # Retains and slightly broadens the manuscript Appendix-A search space.
        common["dropout"] = trial.suggest_float("dropout", 0.05, 0.25)
        common["learning_rate"] = trial.suggest_float("learning_rate", 3e-4, 3e-3, log=True)
        common["weight_decay"] = trial.suggest_float("weight_decay", 1e-7, 3e-3, log=True)
        common.update(d_model=trial.suggest_categorical("d_model", [32, 48, 64]),
                      tcn_layers=trial.suggest_int("tcn_layers", 2, 3),
                      kernel_size=trial.suggest_categorical("kernel_size", [3, 5, 7]))
    else:
        raise ValueError(model_name)
    return common


NEURAL_MODELS = ("MLP", "LSTM", "GRU", "CNN_LSTM", "TCN", "TRANSFORMER", "LASH_SEQ")
MODEL_SEARCH_DIMENSIONS: Dict[str, int] = {
    "MLP": 8, "LSTM": 8, "GRU": 8, "CNN_LSTM": 10, "TCN": 9,
    "TRANSFORMER": 10, "LASH_SEQ": 9,
    "RF": 6, "GBM": 6, "XGBOOST": 10, "LIGHTGBM": 10, "CATBOOST": 7,
}


def search_space_manifest() -> pd.DataFrame:
    """Return the prespecified search spaces as a reporting-ready table."""
    rows = []
    def add(model, parameter, search, scale="categorical_or_linear", rationale=""):
        rows.append({"model": model, "parameter": parameter, "search_space": search,
                     "scale": scale, "rationale": rationale})

    common = [
        ("dropout", "[0.05, 0.30]", "linear"),
        ("learning_rate", "[2e-4, 3e-3]", "log"),
        ("weight_decay", "[1e-8, 5e-3]", "log"),
        ("huber_beta", "[0.5, 1.5]", "linear"),
        ("shape_loss_weight", "{0, 0.05, 0.10, 0.20}", "categorical"),
        ("batch_size", "{128, 256}", "categorical"),
    ]
    for model in ["MLP", "LSTM", "GRU", "CNN_LSTM", "TCN", "TRANSFORMER"]:
        for p, r, sc in common: add(model, p, r, sc, "common neural optimization control")
    for p, r, sc in [
        ("dropout", "[0.05, 0.25]", "linear"),
        ("learning_rate", "[3e-4, 3e-3]", "log"),
        ("weight_decay", "[1e-7, 3e-3]", "log"),
        ("huber_beta", "[0.5, 1.5]", "linear"),
        ("shape_loss_weight", "{0, 0.05, 0.10, 0.20}", "categorical"),
        ("batch_size", "{128, 256}", "categorical"),
        ("d_model", "{32, 48, 64}", "categorical"),
        ("tcn_layers", "{2, 3}", "integer"),
        ("kernel_size", "{3, 5, 7}", "categorical"),
    ]: add("LASH_SEQ", p, r, sc, "manuscript Appendix-A continuity")

    extras = {
        "MLP": [("hidden", "{64,128,256}"), ("layers", "1..3")],
        "LSTM": [("hidden", "{32,64,96,128}"), ("layers", "1..3")],
        "GRU": [("hidden", "{32,64,96,128}"), ("layers", "1..3")],
        "CNN_LSTM": [("conv_channels", "{32,64,96}"), ("kernel_size", "{3,5,7}"), ("hidden", "{32,64,96}"), ("layers", "1..2")],
        "TCN": [("d_model", "{32,48,64,96}"), ("tcn_layers", "2..4"), ("kernel_size", "{3,5,7}")],
        "TRANSFORMER": [("d_model", "{32,48,64,96}"), ("layers", "1..3"), ("nhead", "{2,4,8} subject to divisibility"), ("ff_mult", "{2,4}")],
        "RF": [("n_estimators", "{128,256,512,768,1024,1280}"), ("max_depth", "{None,6,10,14,18}"), ("min_samples_leaf", "{1,2,4,8,16}"), ("min_samples_split", "{2,5,10,20}"), ("max_features", "{0.5,0.7,1.0}"), ("bootstrap", "{True,False}")],
        "GBM": [("n_estimators", "100..1200 step 50"), ("learning_rate", "[0.01,0.15] log"), ("max_depth", "2..6"), ("min_samples_leaf", "{2,5,10,20,40}"), ("subsample", "[0.6,1.0]"), ("max_features", "[0.6,1.0]")],
        "XGBOOST": [("n_estimators", "250..1750 step 50"), ("learning_rate", "[0.01,0.15] log"), ("max_depth", "3..10"), ("min_child_weight", "1..20"), ("subsample", "[0.6,1.0]"), ("colsample_bytree", "[0.6,1.0]"), ("reg_alpha", "[1e-8,10] log"), ("reg_lambda", "[1e-3,100] log"), ("gamma", "[0,5]"), ("booster", "{gbtree,dart}")],
        "LIGHTGBM": [("n_estimators", "500..2000 step 100"), ("learning_rate", "[0.01,0.15] log"), ("num_leaves", "15..127"), ("max_depth", "4..12"), ("min_child_samples", "10..200 step 10"), ("subsample", "[0.6,1.0]"), ("colsample_bytree", "[0.6,1.0]"), ("reg_alpha", "[1e-8,10] log"), ("reg_lambda", "[1e-3,100] log"), ("boosting_type", "{gbdt,dart}")],
        "CATBOOST": [("iterations", "300..1600 step 100"), ("learning_rate", "[0.01,0.15] log"), ("depth", "4..10"), ("l2_leaf_reg", "[1e-2,100] log"), ("random_strength", "[0,2]"), ("bagging_temperature", "[0,5]"), ("border_count", "{64,128,254}")],
    }
    for model, items in extras.items():
        for p, r in items: add(model, p, r, "model_specific", "prespecified before test evaluation")
    add("RIDGE", "alpha", "{1e-3,1e-2,1e-1,1,10,100,1e3,1e4}", "grid", "original LASH continuity")
    return pd.DataFrame(rows)


def _study_storage(root: Path, dataset_key: str, model_name: str, hpo_seed: int) -> optuna.storages.RDBStorage:
    d = root / "optuna" / dataset_key
    d.mkdir(parents=True, exist_ok=True)
    db = (d / f"{model_name.lower()}__hposeed{hpo_seed}.sqlite3").resolve()
    return optuna.storages.RDBStorage(url="sqlite:///" + db.as_posix(),
                                      engine_kwargs={"connect_args": {"timeout": 60}})


def _finished_trial_count(study: optuna.Study) -> int:
    finished = {optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED, optuna.trial.TrialState.FAIL}
    return sum(t.state in finished for t in study.trials)


def tune_neural_repeated(config: ExperimentConfig, dataset_key: str, model_name: str,
                         train: WindowBundle, val_tune: WindowBundle,
                         ablation: Optional[Mapping[str, Any]] = None,
                         study_suffix: str = "") -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    repeats = config.hpo_repeat_count(dataset_key)
    trial_budget = config.hpo_trials(model_name)
    repeat_rows = []; all_trials = []
    for repeat_idx, hpo_seed in enumerate(config.hpo_seeds[:repeats]):
        study_name = f"LASH_revision_{dataset_key}_{model_name}{study_suffix}_hposeed{hpo_seed}"
        study = optuna.create_study(
            study_name=study_name, direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=hpo_seed, multivariate=True),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=max(5, trial_budget // 5), n_warmup_steps=4),
            storage=_study_storage(config.output_root, dataset_key, model_name + study_suffix, hpo_seed),
            load_if_exists=True,
        )
        def objective(trial: optuna.Trial) -> float:
            params = suggest_neural_params(model_name, trial)
            fit = fit_neural(model_name, train, params, config, validation_bundle=val_tune,
                             trial=trial, seed=hpo_seed, ablation=ablation)
            raw = predict_neural_raw(fit, val_tune, params, config)
            gain, _ = select_residual_gain(val_tune, raw)
            score = regression_metrics(val_tune.y, apply_residual_gain(val_tune, raw, gain))["selection_score"]
            trial.set_user_attr("best_epoch", int(fit.best_epoch))
            trial.set_user_attr("gain", float(gain))
            trial.set_user_attr("parameter_count", int(fit.parameter_count))
            trial.set_user_attr("train_seconds", float(fit.train_seconds))
            del fit; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            return float(score)
        remaining = max(0, trial_budget - _finished_trial_count(study))
        if remaining:
            study.optimize(objective, n_trials=remaining, n_jobs=1, show_progress_bar=True, gc_after_trial=True)
        best = study.best_trial
        row = {
            "repeat": repeat_idx + 1, "hpo_seed": hpo_seed, "best_score": best.value,
            "best_epoch": best.user_attrs.get("best_epoch"), "gain": best.user_attrs.get("gain"),
            "parameter_count": best.user_attrs.get("parameter_count"), "n_trials": len(study.trials),
            "params_json": json.dumps(best.params, sort_keys=True),
        }
        repeat_rows.append(row)
        tdf = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
        tdf.insert(0, "hpo_seed", hpo_seed); tdf.insert(0, "model", model_name); tdf.insert(0, "dataset", dataset_key)
        train_col = "user_attrs_train_seconds"
        repeat_rows[-1]["sum_trial_train_seconds"] = float(pd.to_numeric(tdf[train_col], errors="coerce").sum()) if train_col in tdf else np.nan
        all_trials.append(tdf)
    repeat_df = pd.DataFrame(repeat_rows).sort_values("best_score").reset_index(drop=True)
    selected = repeat_df.iloc[0]
    selected_params = json.loads(selected["params_json"])
    selected_config = {
        "params": selected_params,
        "best_epoch": int(selected["best_epoch"]),
        "gain": float(selected["gain"]),
        "selected_hpo_seed": int(selected["hpo_seed"]),
        "validation_score": float(selected["best_score"]),
    }
    return selected_config, pd.concat(all_trials, ignore_index=True), repeat_df


# -----------------------------------------------------------------------------
# Direct horizon-specific tree ensembles
# -----------------------------------------------------------------------------
TREE_MODELS = ("RF", "GBM", "XGBOOST", "LIGHTGBM", "CATBOOST")


def suggest_tree_params(model_name: str, trial: optuna.Trial) -> Dict[str, Any]:
    if model_name == "RF":
        return {
            "n_estimators": trial.suggest_categorical("n_estimators", [128, 256, 512, 768, 1024, 1280]),
            "max_depth": trial.suggest_categorical("max_depth", [None, 6, 10, 14, 18]),
            "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [1, 2, 4, 8, 16]),
            "min_samples_split": trial.suggest_categorical("min_samples_split", [2, 5, 10, 20]),
            "max_features": trial.suggest_categorical("max_features", [0.5, 0.7, 1.0]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        }
    if model_name == "GBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1200, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [2, 5, 10, 20, 40]),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "max_features": trial.suggest_float("max_features", 0.6, 1.0),
        }
    if model_name == "XGBOOST":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 250, 1750, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
        }
    if model_name == "LIGHTGBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 500, 2000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200, step=10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
            "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
        }
    if model_name == "CATBOOST":
        return {
            "iterations": trial.suggest_int("iterations", 300, 1600, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 100.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
            "border_count": trial.suggest_categorical("border_count", [64, 128, 254]),
        }
    raise ValueError(model_name)


def build_tree_estimator(model_name: str, params: Mapping[str, Any], seed: int,
                         threads: int = 1):
    p = dict(params)
    if model_name == "RF":
        return RandomForestRegressor(**p, random_state=seed, n_jobs=threads)
    if model_name == "GBM":
        return GradientBoostingRegressor(**p, random_state=seed, loss="huber")
    if model_name == "XGBOOST":
        from xgboost import XGBRegressor
        return XGBRegressor(**p, objective="reg:squarederror", tree_method="hist", device="cpu",
                            random_state=seed, n_jobs=threads, verbosity=0)
    if model_name == "LIGHTGBM":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**p, objective="regression", random_state=seed, n_jobs=threads,
                             subsample_freq=1, verbosity=-1)
    if model_name == "CATBOOST":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(**p, loss_function="RMSE", random_seed=seed,
                                 thread_count=threads, verbose=False, allow_writing_files=False,
                                 bootstrap_type="Bayesian")
    raise ValueError(model_name)


@dataclass
class TreeFit:
    model_name: str
    models: List[Any]
    params: Dict[str, Any]
    train_seconds: float
    parameter_proxy: int


def _fit_one_tree_horizon(model_name: str, params: Mapping[str, Any], seed: int,
                          threads: int, X: np.ndarray, y: np.ndarray):
    m = build_tree_estimator(model_name, params, seed, threads=threads)
    m.fit(X, y)
    return m


def fit_direct_tree(model_name: str, bundle: WindowBundle, params: Mapping[str, Any],
                    seed: int, config: ExperimentConfig) -> TreeFit:
    X = make_tabular_origin_design(bundle)
    residual = bundle.y - bundle.anchor
    # Smoke mode is a wiring test, not a paper experiment.  Cap boosting / forest
    # length aggressively so users can verify the full 24-horizon pipeline quickly.
    effective_params = dict(params)
    if config.run_profile == "smoke":
        if "n_estimators" in effective_params:
            effective_params["n_estimators"] = min(int(effective_params["n_estimators"]), 5)
        if "iterations" in effective_params:
            effective_params["iterations"] = min(int(effective_params["iterations"]), 5)
    started = time.perf_counter()
    jobs = max(1, int(config.tree_horizon_jobs))
    models = Parallel(n_jobs=jobs, prefer="threads")([
        delayed(_fit_one_tree_horizon)(model_name, effective_params, seed + h,
                                       config.tree_threads_per_model, X, residual[:, h])
        for h in range(HORIZON)
    ])
    # Parameter count is not uniformly exposed across libraries.  We save a transparent
    # tree-count proxy and model-file size later rather than inventing a coefficient count.
    proxy = 0
    for m in models:
        proxy += int(getattr(m, "n_estimators", getattr(m, "tree_count_", 0)) or 0)
    return TreeFit(model_name=model_name, models=models, params=dict(effective_params),
                   train_seconds=time.perf_counter() - started, parameter_proxy=proxy)


def predict_direct_tree_raw(fit: TreeFit, bundle: WindowBundle) -> np.ndarray:
    X = make_tabular_origin_design(bundle)
    residual = np.column_stack([m.predict(X) for m in fit.models])
    return bundle.anchor + residual


def tune_tree_repeated(config: ExperimentConfig, dataset_key: str, model_name: str,
                       train: WindowBundle, val_tune: WindowBundle) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    repeats = config.hpo_repeat_count(dataset_key)
    budget = config.hpo_trials(model_name)
    repeat_rows = []; trial_frames = []
    for repeat_idx, hpo_seed in enumerate(config.hpo_seeds[:repeats]):
        study = optuna.create_study(
            study_name=f"LASH_revision_{dataset_key}_{model_name}_hposeed{hpo_seed}",
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=hpo_seed, multivariate=True),
            storage=_study_storage(config.output_root, dataset_key, model_name, hpo_seed),
            load_if_exists=True,
        )
        def objective(trial: optuna.Trial) -> float:
            params = suggest_tree_params(model_name, trial)
            fit = fit_direct_tree(model_name, train, params, hpo_seed, config)
            raw = predict_direct_tree_raw(fit, val_tune)
            gain, _ = select_residual_gain(val_tune, raw)
            pred = apply_residual_gain(val_tune, raw, gain)
            score = regression_metrics(val_tune.y, pred)["selection_score"]
            trial.set_user_attr("gain", gain)
            trial.set_user_attr("train_seconds", fit.train_seconds)
            trial.set_user_attr("tree_count_proxy", fit.parameter_proxy)
            del fit; gc.collect()
            return float(score)
        remaining = max(0, budget - _finished_trial_count(study))
        if remaining:
            study.optimize(objective, n_trials=remaining, n_jobs=1, show_progress_bar=True, gc_after_trial=True)
        best = study.best_trial
        repeat_rows.append({
            "repeat": repeat_idx + 1, "hpo_seed": hpo_seed, "best_score": best.value,
            "gain": best.user_attrs.get("gain"), "n_trials": len(study.trials),
            "params_json": json.dumps(best.params, sort_keys=True),
        })
        tdf = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
        tdf.insert(0, "hpo_seed", hpo_seed); tdf.insert(0, "model", model_name); tdf.insert(0, "dataset", dataset_key)
        train_col = "user_attrs_train_seconds"
        repeat_rows[-1]["sum_trial_train_seconds"] = float(pd.to_numeric(tdf[train_col], errors="coerce").sum()) if train_col in tdf else np.nan
        trial_frames.append(tdf)
    repeat_df = pd.DataFrame(repeat_rows).sort_values("best_score").reset_index(drop=True)
    selected = repeat_df.iloc[0]
    config_out = {
        "params": json.loads(selected.params_json),
        "gain": float(selected.gain),
        "selected_hpo_seed": int(selected.hpo_seed),
        "validation_score": float(selected.best_score),
    }
    return config_out, pd.concat(trial_frames, ignore_index=True), repeat_df


# -----------------------------------------------------------------------------
# Router and NNLS alternatives
# -----------------------------------------------------------------------------
@dataclass
class RouterSelection:
    mode: str
    sequential_weights: np.ndarray
    calibration_score: float
    best_single_score: float
    improvement_over_single: float
    table: pd.DataFrame


def blend_predictions(seq: np.ndarray, ridge: np.ndarray, seq_weights: np.ndarray) -> np.ndarray:
    w = np.asarray(seq_weights, float).reshape(1, HORIZON)
    if np.any((w < -1e-12) | (w > 1 + 1e-12)):
        raise ValueError("router weights outside [0,1]")
    return np.maximum(0.0, w * seq + (1.0 - w) * ridge)


def select_router(cal: WindowBundle, seq_pred: np.ndarray, ridge_pred: np.ndarray,
                  weight_step: float = 0.05, smooth_window: int = 3,
                  shrinkage: float = 0.5, min_improvement: float = 0.005) -> RouterSelection:
    grid = np.round(np.arange(0.0, 1.0 + weight_step / 2, weight_step), 10)
    global_rows = []
    for w in grid:
        pred = blend_predictions(seq_pred, ridge_pred, np.full(HORIZON, w))
        global_rows.append({"weight": w, "score": regression_metrics(cal.y, pred)["selection_score"]})
    global_table = pd.DataFrame(global_rows)
    global_weight = float(global_table.sort_values(["score", "weight"]).iloc[0].weight)

    raw_weights = []
    for h in range(HORIZON):
        candidates = []
        for w in grid:
            pred = w * seq_pred[:, h] + (1 - w) * ridge_pred[:, h]
            score = regression_metrics(cal.y[:, h], pred)["selection_score"]
            candidates.append((score, abs(w - global_weight), w))
        raw_weights.append(min(candidates)[2])
    raw_weights = np.asarray(raw_weights, float)
    if smooth_window <= 1:
        smooth = raw_weights.copy()
    else:
        smooth = pd.Series(raw_weights).rolling(smooth_window, center=True, min_periods=1).median().to_numpy()
    shrunk = shrinkage * global_weight + (1 - shrinkage) * smooth

    candidate_vectors = {
        "ridge_only": np.zeros(HORIZON),
        "sequential_only": np.ones(HORIZON),
        "fixed_0.5": np.full(HORIZON, 0.5),
        f"global_{global_weight:.3f}": np.full(HORIZON, global_weight),
        "horizon_smoothed_shrunk": shrunk,
    }
    rows = []
    for mode, weights in candidate_vectors.items():
        pred = blend_predictions(seq_pred, ridge_pred, weights)
        rows.append({
            "mode": mode, "mean_seq_weight": float(np.mean(weights)),
            "min_seq_weight": float(np.min(weights)), "max_seq_weight": float(np.max(weights)),
            "weights_json": json.dumps(np.round(weights, 6).tolist()),
            **regression_metrics(cal.y, pred),
        })
    table = pd.DataFrame(rows).sort_values("selection_score").reset_index(drop=True)
    single = table[table["mode"].isin(["ridge_only", "sequential_only"])].sort_values("selection_score").iloc[0]
    best = table.iloc[0]
    improvement = float(single.selection_score - best.selection_score)
    if best["mode"] not in {"ridge_only", "sequential_only"} and improvement < min_improvement:
        best = single; improvement = 0.0
    weights = np.asarray(json.loads(best["weights_json"]), float)
    return RouterSelection(
        mode=str(best["mode"]), sequential_weights=weights,
        calibration_score=float(best.selection_score), best_single_score=float(single.selection_score),
        improvement_over_single=improvement, table=table,
    )


def nnls_router(cal: WindowBundle, seq_pred: np.ndarray, ridge_pred: np.ndarray,
                horizon_specific: bool = True) -> np.ndarray:
    if horizon_specific:
        weights = []
        for h in range(HORIZON):
            A = np.column_stack([seq_pred[:, h], ridge_pred[:, h]])
            coef, _ = nnls(A, cal.y[:, h])
            if coef.sum() <= 0: coef = np.array([0.5, 0.5])
            coef = coef / coef.sum()
            weights.append(coef[0])
        return np.asarray(weights)
    A = np.column_stack([seq_pred.reshape(-1), ridge_pred.reshape(-1)])
    coef, _ = nnls(A, cal.y.reshape(-1))
    if coef.sum() <= 0: coef = np.array([0.5, 0.5])
    coef = coef / coef.sum()
    return np.full(HORIZON, coef[0])


# -----------------------------------------------------------------------------
# Artifact helpers and benchmark orchestration
# -----------------------------------------------------------------------------
BENCHMARK_MODELS = (
    "SEASONAL_24", "SEASONAL_168", "ANCHOR", "RIDGE",
    "RF", "GBM", "XGBOOST", "LIGHTGBM", "CATBOOST",
    "MLP", "LSTM", "GRU", "CNN_LSTM", "TCN", "TRANSFORMER", "LASH",
)


def dataset_output_dir(config: ExperimentConfig, dataset_key: str) -> Path:
    d = config.output_root / "benchmark" / dataset_key
    for sub in ("models", "predictions", "tables", "configs", "calibration"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


def _latency_ms(predict_fn: Callable[[], Any], repeats: int = 5) -> float:
    # Measures batch prediction time per forecast origin, not per scalar horizon.
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter(); predict_fn(); times.append(time.perf_counter() - t0)
    return float(np.median(times) * 1000.0)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def _save_model_size(path: Path) -> float:
    return float(path.stat().st_size / 1024**2) if path.exists() else np.nan


def _collect_model_result(dataset_dir: Path, model: str, seed: int, bundle: WindowBundle,
                          pred: np.ndarray, train_seconds: float = 0.0,
                          latency_total_ms: float = np.nan, parameter_count: float = np.nan,
                          peak_memory_mb: float = np.nan, model_file_mb: float = np.nan) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric = pd.DataFrame([{"model": model, "seed": seed, **regression_metrics(bundle.y, pred)}])
    horizon = per_horizon_metrics(bundle.y, pred); horizon.insert(0, "seed", seed); horizon.insert(0, "model", model)
    origin = origin_loss_frame(bundle, pred, model, seed)
    metric["train_seconds"] = train_seconds
    metric["inference_total_ms"] = latency_total_ms
    metric["inference_ms_per_origin"] = latency_total_ms / max(len(bundle), 1)
    metric["parameter_count_or_tree_proxy"] = parameter_count
    metric["peak_gpu_memory_mb"] = peak_memory_mb
    metric["model_file_mb"] = model_file_mb
    save_prediction_matrix(dataset_dir, bundle, pred, model, seed)
    return metric, horizon, origin


def _save_neural_artifact(dataset_dir: Path, model_name: str, seed: int, fit: NeuralFit,
                          params: Mapping[str, Any], gain: float, ablation: Optional[Mapping[str, Any]] = None) -> float:
    safe = model_name.lower()
    state_path = dataset_dir / "models" / f"{safe}__seed{seed}.pt"
    torch.save({
        "state_dict": fit.model.state_dict(), "params": dict(params), "gain": float(gain),
        "best_epoch": int(fit.best_epoch), "ablation": dict(ablation or {}),
    }, state_path)
    joblib.dump(fit.scaler, dataset_dir / "models" / f"{safe}__seed{seed}__scaler.joblib")
    return _save_model_size(state_path)


def _save_tree_artifact(dataset_dir: Path, model_name: str, seed: int, fit: TreeFit, gain: float) -> float:
    p = dataset_dir / "models" / f"{model_name.lower()}__seed{seed}.joblib"
    joblib.dump({"fit": fit, "gain": gain}, p, compress=3)
    return _save_model_size(p)


def _save_ridge_artifact(dataset_dir: Path, fit: RidgeFit, tuning: Mapping[str, Any]) -> float:
    p = dataset_dir / "models" / "ridge.joblib"
    joblib.dump({"fit": fit, "tuning": dict(tuning)}, p, compress=3)
    return _save_model_size(p)


def run_dataset_benchmark(config: ExperimentConfig, dataset_key: str) -> Dict[str, pd.DataFrame]:
    """Run the complete primary benchmark for one dataset and save incremental artifacts."""
    set_seed(42)
    spec = config.specs[dataset_key]
    dataset_dir = dataset_output_dir(config, dataset_key)
    frame, bundles = build_dataset_bundles(config, dataset_key)
    train, val, test = bundles["train"], bundles["val"], bundles["test"]
    val_tune, val_cal = split_validation_bundle(val)
    pretest = concat_bundles(train, val, split="train_plus_validation")

    # Save immutable data/audit metadata before model fitting.
    audit = {
        "spec": asdict(spec), "environment": runtime_environment(), "weather_mode": config.weather_mode,
        "lookback": LOOKBACK, "horizon": HORIZON, "validation_tune_fraction": VALIDATION_TUNE_FRACTION,
        "purge_origins": VALIDATION_PURGE_ORIGINS, "past_cols": train.past_cols,
        "future_cols": train.future_cols, "n_origins": {k: len(v) for k, v in bundles.items()},
        "val_tune_origins": len(val_tune), "val_calibration_origins": len(val_cal),
    }
    save_json(dataset_dir / "configs" / "data_and_protocol.json", audit)

    metrics_parts = []; horizon_parts = []; origin_parts = []; hpo_parts = []; repeat_parts = []
    component_metric_parts = []
    selected_rows = []; router_tables = []

    # Deterministic baselines.
    for model, pred in [("SEASONAL_24", test.naive24), ("SEASONAL_168", test.naive168), ("ANCHOR", test.anchor)]:
        m, h, o = _collect_model_result(dataset_dir, model, 0, test, pred, 0.0, 0.0, 0, 0.0, 0.0)
        metrics_parts.append(m); horizon_parts.append(h); origin_parts.append(o)

    # Ridge tuning and frozen final refit.
    ridge_tuning, ridge_search = tune_ridge(train, val_tune)
    ridge_search.to_csv(dataset_dir / "tables" / "ridge_search.csv", index=False)
    ridge_final = fit_ridge(pretest, ridge_tuning["alpha"])
    ridge_raw = predict_ridge_raw(ridge_final, test)
    ridge_pred = apply_residual_gain(test, ridge_raw, ridge_tuning["gain"])
    ridge_latency = _latency_ms(lambda: predict_ridge_raw(ridge_final, test), repeats=3)
    ridge_size = _save_ridge_artifact(dataset_dir, ridge_final, ridge_tuning) if config.save_models else np.nan
    m, h, o = _collect_model_result(dataset_dir, "RIDGE", 0, test, ridge_pred,
                                     ridge_final.train_seconds, ridge_latency,
                                     ridge_final.parameter_count, 0.0, ridge_size)
    metrics_parts.append(m); horizon_parts.append(h); origin_parts.append(o)
    selected_rows.append({"model": "RIDGE", **ridge_tuning})

    # Tree ensembles.
    for model_name in TREE_MODELS:
        selected, trial_df, repeat_df = tune_tree_repeated(config, dataset_key, model_name, train, val_tune)
        trial_df.to_csv(dataset_dir / "tables" / f"{model_name.lower()}_hpo_trials.csv", index=False)
        repeat_df.to_csv(dataset_dir / "tables" / f"{model_name.lower()}_hpo_repeats.csv", index=False)
        hpo_parts.append(trial_df); repeat_parts.append(repeat_df.assign(model=model_name))
        selected_rows.append({"model": model_name, **selected, "params_json": json.dumps(selected["params"], sort_keys=True)})
        for seed in config.final_refit_seeds:
            fit = fit_direct_tree(model_name, pretest, selected["params"], seed, config)
            raw = predict_direct_tree_raw(fit, test)
            pred = apply_residual_gain(test, raw, selected["gain"])
            latency = _latency_ms(lambda f=fit: predict_direct_tree_raw(f, test), repeats=2)
            size = _save_tree_artifact(dataset_dir, model_name, seed, fit, selected["gain"]) if config.save_models else np.nan
            m, h, o = _collect_model_result(dataset_dir, model_name, seed, test, pred, fit.train_seconds,
                                             latency, fit.parameter_proxy, 0.0, size)
            metrics_parts.append(m); horizon_parts.append(h); origin_parts.append(o)
            del fit; gc.collect()

    # Neural comparator families (excluding LASH sequential branch).
    for model_name in ("MLP", "LSTM", "GRU", "CNN_LSTM", "TCN", "TRANSFORMER"):
        selected, trial_df, repeat_df = tune_neural_repeated(config, dataset_key, model_name, train, val_tune)
        trial_df.to_csv(dataset_dir / "tables" / f"{model_name.lower()}_hpo_trials.csv", index=False)
        repeat_df.to_csv(dataset_dir / "tables" / f"{model_name.lower()}_hpo_repeats.csv", index=False)
        hpo_parts.append(trial_df); repeat_parts.append(repeat_df.assign(model=model_name))
        selected_rows.append({"model": model_name, **selected, "params_json": json.dumps(selected["params"], sort_keys=True)})
        for seed in config.final_refit_seeds:
            fit = fit_neural(model_name, pretest, selected["params"], config,
                             validation_bundle=None, fixed_epochs=selected["best_epoch"], seed=seed)
            raw = predict_neural_raw(fit, test, selected["params"], config)
            pred = apply_residual_gain(test, raw, selected["gain"])
            latency = _latency_ms(lambda f=fit: predict_neural_raw(f, test, selected["params"], config), repeats=2)
            size = _save_neural_artifact(dataset_dir, model_name, seed, fit, selected["params"], selected["gain"]) if config.save_models else np.nan
            m, h, o = _collect_model_result(dataset_dir, model_name, seed, test, pred, fit.train_seconds,
                                             latency, fit.parameter_count, fit.peak_gpu_memory_mb, size)
            metrics_parts.append(m); horizon_parts.append(h); origin_parts.append(o)
            del fit; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    # LASH sequential expert HPO, held-out router calibration, and final hybrid refits.
    lash_sel, lash_trials, lash_repeats = tune_neural_repeated(config, dataset_key, "LASH_SEQ", train, val_tune)
    lash_trials.to_csv(dataset_dir / "tables" / "lash_seq_hpo_trials.csv", index=False)
    lash_repeats.to_csv(dataset_dir / "tables" / "lash_seq_hpo_repeats.csv", index=False)
    hpo_parts.append(lash_trials); repeat_parts.append(lash_repeats.assign(model="LASH_SEQ"))
    selected_rows.append({"model": "LASH_SEQ", **lash_sel, "params_json": json.dumps(lash_sel["params"], sort_keys=True)})

    pre_cal = concat_bundles(train, val_tune, split="train_plus_val_tune")
    cal_seq_fit = fit_neural("LASH_SEQ", pre_cal, lash_sel["params"], config,
                             validation_bundle=None, fixed_epochs=lash_sel["best_epoch"], seed=42)
    cal_seq_raw = predict_neural_raw(cal_seq_fit, val_cal, lash_sel["params"], config)
    cal_seq = apply_residual_gain(val_cal, cal_seq_raw, lash_sel["gain"])
    cal_ridge_fit = fit_ridge(pre_cal, ridge_tuning["alpha"])
    cal_ridge_raw = predict_ridge_raw(cal_ridge_fit, val_cal)
    cal_ridge = apply_residual_gain(val_cal, cal_ridge_raw, ridge_tuning["gain"])
    router = select_router(val_cal, cal_seq, cal_ridge)
    router.table.to_csv(dataset_dir / "calibration" / "router_default_candidates.csv", index=False)
    np.savez_compressed(dataset_dir / "calibration" / "expert_predictions.npz",
                        actual=val_cal.y, seq=cal_seq, ridge=cal_ridge,
                        forecast_origin=val_cal.forecast_origin, target_time=val_cal.target_time)
    save_json(dataset_dir / "configs" / "router_default.json", {
        "mode": router.mode, "weights": router.sequential_weights.tolist(),
        "calibration_score": router.calibration_score,
        "best_single_score": router.best_single_score,
        "improvement": router.improvement_over_single,
    })
    del cal_seq_fit, cal_ridge_fit; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Final LASH hybrid: Ridge is deterministic and shared; sequential branch varies by seed.
    for seed in config.final_refit_seeds:
        fit = fit_neural("LASH_SEQ", pretest, lash_sel["params"], config,
                         validation_bundle=None, fixed_epochs=lash_sel["best_epoch"], seed=seed)
        seq_raw = predict_neural_raw(fit, test, lash_sel["params"], config)
        seq_pred = apply_residual_gain(test, seq_raw, lash_sel["gain"])
        lash_pred = blend_predictions(seq_pred, ridge_pred, router.sequential_weights)
        latency = _latency_ms(lambda f=fit: predict_neural_raw(f, test, lash_sel["params"], config), repeats=2)
        size = _save_neural_artifact(dataset_dir, "LASH_SEQ", seed, fit, lash_sel["params"], lash_sel["gain"]) if config.save_models else np.nan
        m, h, o = _collect_model_result(dataset_dir, "LASH", seed, test, lash_pred, fit.train_seconds,
                                         latency, fit.parameter_count + ridge_final.parameter_count,
                                         fit.peak_gpu_memory_mb, size + ridge_size)
        metrics_parts.append(m); horizon_parts.append(h); origin_parts.append(o)
        component_metric_parts.append(pd.DataFrame([{
            "model": "LASH_SEQ_COMPONENT", "seed": seed,
            **regression_metrics(test.y, seq_pred),
            "train_seconds": fit.train_seconds,
            "parameter_count": fit.parameter_count,
        }]))
        # Save sequential component separately for router/ablation diagnostics.
        save_prediction_matrix(dataset_dir, test, seq_pred, "LASH_SEQ_COMPONENT", seed)
        del fit; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    metrics = pd.concat(metrics_parts, ignore_index=True)
    horizons = pd.concat(horizon_parts, ignore_index=True)
    origins = pd.concat(origin_parts, ignore_index=True)
    selected_df = pd.DataFrame(selected_rows)
    hpo_df = pd.concat(hpo_parts, ignore_index=True) if hpo_parts else pd.DataFrame()
    repeat_df = pd.concat(repeat_parts, ignore_index=True) if repeat_parts else pd.DataFrame()
    router_df = router.table.copy()
    component_metrics = pd.concat(component_metric_parts, ignore_index=True) if component_metric_parts else pd.DataFrame()

    # Seed-level summary is the main result table; deterministic models naturally have SD=0.
    agg_cols = ["MAPE", "CVRMSE", "NMAE", "selection_score", "train_seconds",
                "inference_ms_per_origin", "parameter_count_or_tree_proxy", "peak_gpu_memory_mb", "model_file_mb"]
    summary = metrics.groupby("model", as_index=False)[agg_cols].agg(["mean", "std"])
    summary.columns = ["model"] + [f"{a}_{b}" for a, b in summary.columns.tolist()[1:]]
    summary = summary.sort_values("selection_score_mean").reset_index(drop=True)

    tables_dir = dataset_dir / "tables"
    metrics.to_csv(tables_dir / "metrics_by_seed.csv", index=False)
    horizons.to_csv(tables_dir / "horizon_metrics.csv", index=False)
    origins.to_csv(tables_dir / "origin_losses.csv.gz", index=False, compression="gzip")
    selected_df.to_csv(tables_dir / "selected_hyperparameters.csv", index=False)
    summary.to_csv(tables_dir / "benchmark_summary.csv", index=False)
    component_metrics.to_csv(tables_dir / "lash_component_metrics.csv", index=False)

    workbook = dataset_dir / f"{dataset_key}_benchmark_results.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Benchmark_Summary", index=False)
        metrics.to_excel(writer, sheet_name="Metrics_By_Seed", index=False)
        horizons.to_excel(writer, sheet_name="Horizon_Metrics", index=False)
        origins.to_excel(writer, sheet_name="Origin_Losses", index=False)
        selected_df.to_excel(writer, sheet_name="Selected_Params", index=False)
        repeat_df.to_excel(writer, sheet_name="HPO_Repeats", index=False)
        router_df.to_excel(writer, sheet_name="Router_Default", index=False)
        component_metrics.to_excel(writer, sheet_name="LASH_Components", index=False)
        search_space_manifest().to_excel(writer, sheet_name="Search_Spaces", index=False)
        pd.DataFrame([audit]).astype(str).to_excel(writer, sheet_name="Protocol", index=False)
    return {
        "summary": summary, "metrics": metrics, "horizons": horizons, "origins": origins,
        "selected": selected_df, "hpo_repeats": repeat_df, "router": router_df,
        "component_metrics": component_metrics,
    }


def run_full_benchmark(config: ExperimentConfig) -> Dict[str, Dict[str, pd.DataFrame]]:
    if config.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA required but unavailable")
    out = {}
    for key in config.dataset_keys:
        print("\n" + "=" * 80)
        print(f"Running {key}: {config.specs[key].name}")
        print("=" * 80)
        out[key] = run_dataset_benchmark(config, key)
    return out


# -----------------------------------------------------------------------------
# Reload helpers used by later notebooks
# -----------------------------------------------------------------------------
def find_benchmark_workbooks(output_root: Path) -> List[Path]:
    return sorted(Path(output_root).glob("benchmark/*/*_benchmark_results.xlsx"))


def read_benchmark_workbooks(output_root: Path) -> Dict[str, Dict[str, pd.DataFrame]]:
    out = {}
    for path in find_benchmark_workbooks(output_root):
        key = path.parent.name
        xls = pd.ExcelFile(path)
        out[key] = {sheet: pd.read_excel(path, sheet_name=sheet) for sheet in xls.sheet_names}
    return out


def load_prediction(dataset_dir: Path, model: str, seed: int) -> Dict[str, np.ndarray]:
    p = prediction_matrix_path(dataset_dir, model, seed)
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def load_selected_params(dataset_dir: Path) -> pd.DataFrame:
    return pd.read_csv(dataset_dir / "tables" / "selected_hyperparameters.csv")


# -----------------------------------------------------------------------------
# Router sensitivity utilities used by Notebook 02
# -----------------------------------------------------------------------------
def router_sensitivity_table(cal: WindowBundle, seq: np.ndarray, ridge: np.ndarray,
                             steps=(0.01, 0.05, 0.10), smooth_windows=(1, 3, 5),
                             shrinkages=(0.0, 0.5, 1.0), thresholds=(0.0, 0.0025, 0.005, 0.01)) -> pd.DataFrame:
    rows = []
    for step in steps:
        for sw in smooth_windows:
            for shrink in shrinkages:
                for threshold in thresholds:
                    r = select_router(cal, seq, ridge, step, sw, shrink, threshold)
                    rows.append({
                        "weight_step": step, "smooth_window": sw, "shrinkage_to_global": shrink,
                        "min_improvement": threshold, "mode": r.mode,
                        "calibration_score": r.calibration_score,
                        "mean_seq_weight": float(r.sequential_weights.mean()),
                        "min_seq_weight": float(r.sequential_weights.min()),
                        "max_seq_weight": float(r.sequential_weights.max()),
                        "weights_json": json.dumps(np.round(r.sequential_weights, 6).tolist()),
                    })
    return pd.DataFrame(rows).sort_values("calibration_score").reset_index(drop=True)


def router_temporal_stability(cal: WindowBundle, seq: np.ndarray, ridge: np.ndarray, n_blocks: int = 3) -> pd.DataFrame:
    rows = []
    blocks = np.array_split(np.arange(len(cal)), n_blocks)
    for i, idx in enumerate(blocks, start=1):
        b = take_bundle(cal, idx, split=f"cal_block_{i}")
        r = select_router(b, seq[idx], ridge[idx])
        for h, w in enumerate(r.sequential_weights, start=1):
            rows.append({"block": i, "horizon": h, "seq_weight": w, "mode": r.mode})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Peak / temporal robustness metrics used by Notebook 03
# -----------------------------------------------------------------------------
def daily_peak_metrics(target_time: np.ndarray, actual: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """Peak quality over each rolling 24-hour forecast trajectory.

    Each hourly origin is one operational decision opportunity.  Evaluating the
    peak inside that origin's own 24-hour trajectory avoids mixing overlapping
    forecasts of the same target timestamp and matches the hourly-updated service.
    """
    a = np.asarray(actual, float)
    p = np.asarray(pred, float)
    if a.shape != p.shape or a.ndim != 2 or a.shape[1] != HORIZON:
        raise ValueError((a.shape, p.shape))
    a_idx = np.argmax(a, axis=1)
    p_idx = np.argmax(p, axis=1)
    a_peak = np.max(a, axis=1)
    p_peak = np.max(p, axis=1)
    ape = np.abs(p_peak - a_peak) / np.maximum(np.abs(a_peak), 1e-12) * 100.0
    timing = np.abs(p_idx - a_idx).astype(float)
    top3_a = np.argpartition(a, -3, axis=1)[:, -3:]
    top3_p = np.argpartition(p, -3, axis=1)[:, -3:]
    top3_hit = np.asarray([len(set(x).intersection(set(y))) > 0 for x, y in zip(top3_a, top3_p)], float)
    return {
        "trajectory_peak_MAPE": float(np.mean(ape)),
        "trajectory_peak_timing_MAE_h": float(np.mean(timing)),
        "trajectory_top3_peak_hit_rate": float(np.mean(top3_hit)),
    }


def high_load_recall(actual: np.ndarray, pred: np.ndarray, quantile: float) -> float:
    a = actual.reshape(-1); p = pred.reshape(-1)
    at = np.quantile(a, quantile); pt = np.quantile(p, quantile)
    true = a >= at; called = p >= pt
    return float((true & called).sum() / max(true.sum(), 1))


# -----------------------------------------------------------------------------
# Model reloading for XAI
# -----------------------------------------------------------------------------
def load_neural_artifact(config: ExperimentConfig, dataset_key: str, model_name: str,
                         seed: int, bundle: WindowBundle, ablation: Optional[Mapping[str, Any]] = None) -> Tuple[NeuralFit, Dict[str, Any], float]:
    d = dataset_output_dir(config, dataset_key)
    safe = model_name.lower()
    payload = torch.load(d / "models" / f"{safe}__seed{seed}.pt", map_location=config.device)
    scaler = joblib.load(d / "models" / f"{safe}__seed{seed}__scaler.joblib")
    model = make_neural_model(model_name, payload["params"], bundle.past.shape[-1], bundle.future.shape[-1],
                              ablation=payload.get("ablation", ablation)).to(config.device)
    model.load_state_dict(payload["state_dict"]); model.eval()
    fit = NeuralFit(model=model, scaler=scaler, best_epoch=int(payload["best_epoch"]), history=[],
                    train_seconds=np.nan, parameter_count=count_parameters(model), peak_gpu_memory_mb=np.nan)
    return fit, payload["params"], float(payload["gain"])


def load_tree_artifact(config: ExperimentConfig, dataset_key: str, model_name: str, seed: int):
    d = dataset_output_dir(config, dataset_key)
    return joblib.load(d / "models" / f"{model_name.lower()}__seed{seed}.joblib")


def load_ridge_artifact(config: ExperimentConfig, dataset_key: str):
    d = dataset_output_dir(config, dataset_key)
    return joblib.load(d / "models" / "ridge.joblib")

