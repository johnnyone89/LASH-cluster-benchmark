"""Reviewer-defensible 48-hour execution policy for the LASH revision.

This module is intentionally separate from :mod:`lash_revision_core`.  It keeps
the scientific data construction, chronology, metrics, models, and causal
24-step targets unchanged while changing only the allocation of computation:

* HPO is performed on CLUSTER_1 (the development dataset) and frozen before
  the other three datasets are opened for model fitting.
* Newly optimized families use two independent HPO seeds, dimension-adaptive
  trial budgets, deterministic screening samples spanning the complete
  training interval, and full validation-tune confirmation of the finalists.
* The expensive 12-trial RF study already completed in ``20260821_results`` is
  reused read-only and complemented by one small independent HPO repeat.  Its
  candidates are confirmed under the common validation protocol before a
  parsimonious setting is frozen for matched one-seed final refits.
* Hyperparameters are transferred unchanged to CLUSTER_2, BDG_EDU, and
  BDG_DORM.  Model weights are refit on each dataset's train+validation period;
  test targets are never used for selection or calibration.
* Tree and neural comparators use one prespecified final seed.  LASH uses three
  final seeds so dependence-aware inference incorporates training variability.
* Mechanism tests use the full-model hyperparameters for every structural,
  feature, and YoY variant.  The sequential expert is the primary ablation
  estimand; rerouted performance is reported separately.

The policy is a compute-bounded protocol, not a smoke test.  Validation-
calibration, final refitting, and testing retain every eligible hourly origin.
"""
from __future__ import annotations

import gc
import json
import math
import shutil
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch

import lash_revision_core as core
import lash_revision_ablation as ablation


PROTOCOL_NAME = "reviewer_deadline48_frozen_transfer"
DEVELOPMENT_DATASET = "CLUSTER_1"
EXTERNAL_DATASETS = ("CLUSTER_2", "BDG_EDU", "BDG_DORM")
EXECUTION_ORDER = ("BDG_EDU", "BDG_DORM", "CLUSTER_2", "CLUSTER_1")
TREE_MODELS = ("RF", "GBM", "XGBOOST", "LIGHTGBM", "CATBOOST")
NEURAL_COMPARATORS = ("MLP", "LSTM", "GRU", "CNN_LSTM", "TCN", "TRANSFORMER")
HPO_MODELS = TREE_MODELS + NEURAL_COMPARATORS + ("LASH_SEQ",)
INFERENTIAL_COMPARATORS = (
    "SEASONAL_24", "RIDGE", "RF", "GBM", "XGBOOST", "LIGHTGBM",
    "CATBOOST", "TRANSFORMER",
)


@dataclass(frozen=True)
class Deadline48Policy:
    development_dataset: str = DEVELOPMENT_DATASET
    hpo_seeds: Tuple[int, ...] = (42, 142)
    tree_final_seeds: Tuple[int, ...] = (42,)
    neural_comparator_final_seeds: Tuple[int, ...] = (42,)
    lash_final_seeds: Tuple[int, ...] = (42, 142, 242)
    hpo_train_origins: int = 1024
    hpo_validation_origins: int = 2048
    max_epochs: int = 35
    patience: int = 5
    parsimony_margin_score_points: float = 0.02
    bootstrap_reps: int = 1000
    mechanism_seeds: Tuple[int, ...] = (42,)
    computational_budget_hours: float = 40.0
    save_tree_models: Tuple[str, ...] = ("LIGHTGBM",)
    save_neural_models: Tuple[str, ...] = ("LASH_SEQ",)

    def hpo_trials_per_repeat(self, model_name: str) -> int:
        # Dimension-adaptive rather than the criticized equal-20-trial rule.
        dimensions = int(core.MODEL_SEARCH_DIMENSIONS[model_name])
        if dimensions <= 6:
            return 3
        if dimensions <= 8:
            return 4
        return 5

    def hpo_timeout_seconds(self, model_name: str) -> int:
        # Timeout is per HPO repeat.  A prespecified default is always
        # enqueued, so a valid configuration exists even when the timeout wins.
        return {
            "GBM": 1800,
            "XGBOOST": 1200,
            "LIGHTGBM": 1200,
            "CATBOOST": 1200,
            "MLP": 900,
            "LSTM": 1200,
            "GRU": 1200,
            "CNN_LSTM": 1500,
            "TCN": 1500,
            "TRANSFORMER": 1500,
            "LASH_SEQ": 1800,
            "RF": 1800,
        }[model_name]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(v) for v in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _read_json(path: Path, default: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    if not path.exists():
        return dict(default or {})
    return json.loads(path.read_text(encoding="utf-8"))


def _systematic_sample(bundle: core.WindowBundle, maximum: int, split: str) -> core.WindowBundle:
    if len(bundle) <= maximum:
        return bundle
    idx = np.unique(np.linspace(0, len(bundle) - 1, maximum, dtype=int))
    return core.take_bundle(bundle, idx, split=split)


def _deadline_root(config: core.ExperimentConfig) -> Path:
    root = Path(config.output_root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _progress_path(config: core.ExperimentConfig) -> Path:
    return _deadline_root(config) / "experiment_progress.csv"


def _append_progress(config: core.ExperimentConfig, row: Mapping[str, Any]) -> None:
    path = _progress_path(config)
    frame = pd.DataFrame([_json_ready(row)])
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


@contextmanager
def progress_stage(config: core.ExperimentConfig, stage: str, **fields: Any):
    started = time.perf_counter()
    wall_started = pd.Timestamp.now(tz="UTC")
    print(f"\n[START] {stage} | {fields}", flush=True)
    _append_progress(config, {"stage": stage, "status": "started", "utc": wall_started, **fields})
    try:
        yield
    except Exception as exc:
        elapsed = time.perf_counter() - started
        _append_progress(config, {
            "stage": stage, "status": "failed", "seconds": elapsed,
            "utc": pd.Timestamp.now(tz="UTC"), "error": repr(exc), **fields,
        })
        print(f"[FAILED] {stage} after {elapsed/60:.1f} min: {exc}", flush=True)
        raise
    else:
        elapsed = time.perf_counter() - started
        _append_progress(config, {
            "stage": stage, "status": "completed", "seconds": elapsed,
            "utc": pd.Timestamp.now(tz="UTC"), **fields,
        })
        print(f"[DONE] {stage} | {elapsed/60:.1f} min", flush=True)


def _study_storage(
    config: core.ExperimentConfig,
    model_name: str,
    seed: int,
    dataset_key: str = DEVELOPMENT_DATASET,
):
    directory = _deadline_root(config) / "optuna_hpo" / dataset_key
    directory.mkdir(parents=True, exist_ok=True)
    database = (directory / f"{model_name.lower()}__hposeed{seed}.sqlite3").resolve()
    return core.optuna.storages.RDBStorage(
        url="sqlite:///" + database.as_posix(),
        engine_kwargs={"connect_args": {"timeout": 60}},
    )


def _default_neural_params(model_name: str) -> Dict[str, Any]:
    common: Dict[str, Any] = {
        "dropout": 0.10,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "huber_beta": 1.0,
        "shape_loss_weight": 0.05,
        "batch_size": 256,
    }
    if model_name == "MLP":
        common.update(hidden=128, layers=2)
    elif model_name in {"LSTM", "GRU"}:
        common.update(hidden=64, layers=2)
    elif model_name == "CNN_LSTM":
        common.update(conv_channels=64, kernel_size=5, hidden=64, layers=1)
    elif model_name == "TCN":
        common.update(d_model=48, tcn_layers=3, kernel_size=5)
    elif model_name == "TRANSFORMER":
        common.update(d_model=48, layers=2, nhead=4, ff_mult=2)
    elif model_name == "LASH_SEQ":
        common.update(d_model=48, tcn_layers=2, kernel_size=5)
    else:
        raise ValueError(model_name)
    return common


def _default_tree_params(model_name: str) -> Dict[str, Any]:
    defaults = {
        "RF": {
            "n_estimators": 256, "max_depth": 14, "min_samples_leaf": 4,
            "min_samples_split": 5, "max_features": 0.5, "bootstrap": True,
        },
        "GBM": {
            "n_estimators": 100, "learning_rate": 0.05, "max_depth": 2,
            "min_samples_leaf": 20, "subsample": 0.80, "max_features": 0.50,
        },
        "XGBOOST": {
            "n_estimators": 400, "learning_rate": 0.05, "max_depth": 6,
            "min_child_weight": 5, "subsample": 0.80, "colsample_bytree": 0.80,
            "reg_alpha": 1e-4, "reg_lambda": 1.0, "gamma": 0.0, "booster": "gbtree",
        },
        "LIGHTGBM": {
            "n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31,
            "max_depth": 8, "min_child_samples": 40, "subsample": 0.80,
            "colsample_bytree": 0.80, "reg_alpha": 1e-4, "reg_lambda": 1.0,
            "boosting_type": "gbdt",
        },
        "CATBOOST": {
            "iterations": 400, "learning_rate": 0.05, "depth": 6,
            "l2_leaf_reg": 3.0, "random_strength": 0.5,
            "bagging_temperature": 1.0, "border_count": 128,
        },
    }
    return dict(defaults[model_name])


def suggest_tree_params_deadline(model_name: str, trial: Any) -> Dict[str, Any]:
    """Prespecified workstation-bounded tree spaces used only in the 48-h study."""
    if model_name == "RF":
        return {
            "n_estimators": trial.suggest_categorical("n_estimators", [128, 256, 384, 512]),
            "max_depth": trial.suggest_categorical("max_depth", [6, 10, 14, 18]),
            "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [1, 2, 4, 8, 16]),
            "min_samples_split": trial.suggest_categorical("min_samples_split", [2, 5, 10, 20]),
            "max_features": trial.suggest_categorical("max_features", [0.35, 0.5, 0.7]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        }
    if model_name == "GBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 150, step=25),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 3),
            "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [10, 20, 40]),
            "subsample": trial.suggest_float("subsample", 0.70, 1.0),
            "max_features": trial.suggest_float("max_features", 0.35, 0.70),
        }
    if model_name == "XGBOOST":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 3.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 3.0),
            "booster": trial.suggest_categorical("booster", ["gbtree"]),
        }
    if model_name == "LIGHTGBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 700, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 95),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 160, step=20),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 3.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
            "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt"]),
        }
    if model_name == "CATBOOST":
        return {
            "iterations": trial.suggest_int("iterations", 200, 600, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 30.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 1.5),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 3.0),
            "border_count": trial.suggest_categorical("border_count", [64, 128]),
        }
    raise ValueError(model_name)


def deadline_search_space_manifest() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    def add(model: str, parameter: str, space: str, scale: str = "model_specific"):
        rows.append({
            "model": model, "parameter": parameter, "search_space": space,
            "scale": scale, "rationale": "prespecified compute-bounded development-only HPO",
        })
    for model, params in {
        "RF": {
            "n_estimators": "{128,256,384,512}", "max_depth": "{6,10,14,18}",
            "min_samples_leaf": "{1,2,4,8,16}", "min_samples_split": "{2,5,10,20}",
            "max_features": "{0.35,0.50,0.70}", "bootstrap": "{True,False}",
        },
        "GBM": {
            "n_estimators": "50..150 step 25", "learning_rate": "[0.02,0.12] log",
            "max_depth": "2..3", "min_samples_leaf": "{10,20,40}",
            "subsample": "[0.70,1.00]", "max_features": "[0.35,0.70]",
        },
        "XGBOOST": {
            "n_estimators": "200..600 step 100", "learning_rate": "[0.02,0.12] log",
            "max_depth": "3..8", "min_child_weight": "1..15", "subsample": "[0.65,1.00]",
            "colsample_bytree": "[0.65,1.00]", "reg_alpha": "[1e-8,3] log",
            "reg_lambda": "[1e-3,30] log", "gamma": "[0,3]", "booster": "{gbtree}",
        },
        "LIGHTGBM": {
            "n_estimators": "200..700 step 100", "learning_rate": "[0.02,0.12] log",
            "num_leaves": "15..95", "max_depth": "4..10", "min_child_samples": "20..160 step 20",
            "subsample": "[0.65,1.00]", "colsample_bytree": "[0.65,1.00]",
            "reg_alpha": "[1e-8,3] log", "reg_lambda": "[1e-3,30] log",
            "boosting_type": "{gbdt}",
        },
        "CATBOOST": {
            "iterations": "200..600 step 100", "learning_rate": "[0.02,0.12] log",
            "depth": "4..8", "l2_leaf_reg": "[1e-2,30] log", "random_strength": "[0,1.5]",
            "bagging_temperature": "[0,3]", "border_count": "{64,128}",
        },
    }.items():
        for parameter, space in params.items():
            add(model, parameter, space)
    neural = core.search_space_manifest()
    neural = neural.loc[neural["model"].isin(NEURAL_COMPARATORS + ("LASH_SEQ",))].copy()
    neural["rationale"] = "two-seed dimension-adaptive development-only HPO"
    return pd.concat([pd.DataFrame(rows), neural], ignore_index=True)


def _coerce_rf_params(row: Mapping[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for key, value in row.items():
        if not str(key).startswith("params_") or pd.isna(value):
            continue
        name = str(key)[len("params_"):]
        if name == "bootstrap":
            if isinstance(value, str):
                params[name] = value.strip().lower() == "true"
            else:
                params[name] = bool(value)
        elif name in {"n_estimators", "max_depth", "min_samples_leaf", "min_samples_split"}:
            params[name] = int(value)
        elif name == "max_features":
            params[name] = float(value)
        else:
            params[name] = _json_ready(value)
    return params


def _load_source_rf_setting(
    source_root: Path,
    parsimony_margin: float,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    table_dir = Path(source_root) / "benchmark" / DEVELOPMENT_DATASET / "tables"
    repeats_path = table_dir / "rf_hpo_repeats.csv"
    trials_path = table_dir / "rf_hpo_trials.csv"
    if not repeats_path.exists():
        raise FileNotFoundError(
            f"Completed RF HPO table not found: {repeats_path}. Let the current RF seed 242 finish first."
        )
    repeats = pd.read_csv(repeats_path)
    trials = pd.read_csv(trials_path) if trials_path.exists() else pd.DataFrame()
    candidate: Optional[pd.Series] = None
    if not trials.empty and "value" in trials:
        eligible_trials = trials.copy()
        if "state" in eligible_trials:
            eligible_trials = eligible_trials.loc[
                eligible_trials["state"].astype(str).str.upper().str.contains("COMPLETE")
            ]
        eligible_trials["_score"] = pd.to_numeric(eligible_trials["value"], errors="coerce")
        eligible_trials = eligible_trials.loc[np.isfinite(eligible_trials["_score"])]
        if not eligible_trials.empty:
            best_score = float(eligible_trials["_score"].min())
            eligible_trials = eligible_trials.loc[
                eligible_trials["_score"] <= best_score + float(parsimony_margin)
            ].copy()
            runtime_col = next(
                (name for name in ("user_attrs_train_seconds", "train_seconds")
                 if name in eligible_trials), None
            )
            if runtime_col is not None:
                eligible_trials["_runtime"] = pd.to_numeric(
                    eligible_trials[runtime_col], errors="coerce"
                )
            else:
                eligible_trials["_runtime"] = np.nan
            eligible_trials["_trees"] = pd.to_numeric(
                eligible_trials.get("params_n_estimators", np.nan), errors="coerce"
            )
            eligible_trials["_depth"] = pd.to_numeric(
                eligible_trials.get("params_max_depth", np.nan), errors="coerce"
            )
            eligible_trials["_complexity"] = (
                eligible_trials["_trees"].fillna(np.inf)
                * eligible_trials["_depth"].fillna(20.0)
            )
            # Measured runtime is primary; declared tree/depth complexity is a
            # deterministic fallback for legacy tables without timing fields.
            eligible_trials["_runtime_missing"] = ~np.isfinite(eligible_trials["_runtime"])
            candidate = eligible_trials.sort_values(
                ["_runtime_missing", "_runtime", "_complexity", "_score"]
            ).iloc[0]

    if candidate is not None:
        params = _coerce_rf_params(candidate.to_dict())
        if not params:
            params = json.loads(repeats.sort_values("best_score").iloc[0]["params_json"])
        gain_value = candidate.get("user_attrs_gain", np.nan)
        if pd.isna(gain_value):
            gain_value = repeats.sort_values("best_score").iloc[0].get("gain", 1.0)
        selected_seed_value = candidate.get("hpo_seed", 42)
        selected_seed = 42 if pd.isna(selected_seed_value) else int(selected_seed_value)
        return {
            "params": params,
            "gain": float(gain_value),
            "best_epoch": None,
            "validation_score": float(candidate["_score"]),
            "selected_hpo_seed": selected_seed,
            "selection_source": (
                "parsimonious candidate from completed 12-trial hardware RF study"
            ),
        }, trials, repeats

    best = repeats.sort_values("best_score").iloc[0]
    return {
        "params": json.loads(best["params_json"]),
        "gain": float(best["gain"]),
        "best_epoch": None,
        "validation_score": float(best["best_score"]),
        "selected_hpo_seed": int(best["hpo_seed"]),
        "selection_source": "best candidate from completed 12-trial hardware RF study",
    }, trials, repeats


def _complete_trials(study: Any) -> List[Any]:
    return [
        trial for trial in study.trials
        if trial.state == core.optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]


def _trial_state_counts(study: Any) -> Dict[str, int]:
    """Return auditable Optuna state counts without admitting non-complete trials."""
    counts: Dict[str, int] = {}
    for trial in study.trials:
        state = getattr(trial.state, "name", str(trial.state))
        counts[state] = counts.get(state, 0) + 1
    return counts


def _attempted_trial_count(study: Any) -> int:
    """Count started trials while excluding merely enqueued WAITING records."""
    return sum(
        count for state, count in _trial_state_counts(study).items()
        if state != "WAITING"
    )


def _optimize_to_completed_budget(
    study: Any,
    objective: Any,
    budget: int,
    *,
    timeout: Optional[int],
    catch: Tuple[type, ...],
    model_name: str,
    seed: int,
    max_new_trials: int,
    phase: str,
) -> bool:
    """Run bounded refill batches until the prespecified COMPLETE-trial budget is met.

    Optuna's ``n_trials`` counts PRUNED and FAILED attempts.  The reviewer
    protocol instead admits exactly ``budget`` finite COMPLETE trials.  This
    helper therefore refills missing completions while retaining every attempt
    in the RDB audit trail.  ``max_new_trials`` is a safety bound for numerical
    or resource failures; normal pruning is handled by the completion fallback
    in ``_tune_neural_development``.
    """
    start_attempted = _attempted_trial_count(study)
    while len(_complete_trials(study)) < budget:
        used = _attempted_trial_count(study) - start_attempted
        remaining_attempts = int(max_new_trials) - used
        if remaining_attempts <= 0:
            break
        missing = budget - len(_complete_trials(study))
        batch = min(max(1, missing), remaining_attempts)
        before_attempted = _attempted_trial_count(study)
        study.optimize(
            objective,
            n_trials=batch,
            timeout=timeout,
            n_jobs=1,
            show_progress_bar=True,
            gc_after_trial=True,
            catch=catch,
        )
        counts = _trial_state_counts(study)
        print(
            f"[HPO REFILL] {model_name} seed={seed} phase={phase}: "
            f"complete={len(_complete_trials(study))}/{budget}, states={counts}",
            flush=True,
        )
        # A timeout can return without creating a trial.  Avoid a tight loop;
        # the caller will issue the usual incomplete-budget diagnostic.
        if _attempted_trial_count(study) == before_attempted:
            break
    return len(_complete_trials(study)) >= budget


def _choose_parsimonious(finalists: pd.DataFrame, margin: float) -> pd.Series:
    finite = finalists[np.isfinite(finalists["full_validation_score"])].copy()
    if finite.empty:
        raise RuntimeError("No finite full-validation finalist")
    best = float(finite["full_validation_score"].min())
    eligible = finite[finite["full_validation_score"] <= best + float(margin)].copy()
    return eligible.sort_values(["confirmation_train_seconds", "full_validation_score"]).iloc[0]


def _tune_tree_development(
    config: core.ExperimentConfig,
    policy: Deadline48Policy,
    model_name: str,
    train: core.WindowBundle,
    val_tune: core.WindowBundle,
    dataset_key: str = DEVELOPMENT_DATASET,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_screen = _systematic_sample(train, policy.hpo_train_origins, "deadline48_hpo_train")
    val_screen = _systematic_sample(val_tune, policy.hpo_validation_origins, "deadline48_hpo_val")
    trial_frames: List[pd.DataFrame] = []
    repeat_rows: List[Dict[str, Any]] = []
    finalist_specs: List[Tuple[int, Dict[str, Any]]] = []

    for repeat, seed in enumerate(policy.hpo_seeds, start=1):
        name = f"LASH_hpo_{dataset_key}_{model_name}_hposeed{seed}"
        study = core.optuna.create_study(
            study_name=name,
            direction="minimize",
            sampler=core.optuna.samplers.TPESampler(seed=seed, multivariate=True),
            storage=_study_storage(config, model_name, seed, dataset_key),
            load_if_exists=True,
        )
        if not study.trials:
            study.enqueue_trial(
                _default_tree_params(model_name),
                user_attrs={"prespecified_default": True, "prior_result_reuse": False},
            )

        def objective(trial: Any) -> float:
            params = suggest_tree_params_deadline(model_name, trial)
            fit = core.fit_direct_tree(model_name, train_screen, params, seed, config)
            try:
                raw = core.predict_direct_tree_raw(fit, val_screen)
                gain, _ = core.select_residual_gain(val_screen, raw)
                score = core.regression_metrics(
                    val_screen.y, core.apply_residual_gain(val_screen, raw, gain)
                )["selection_score"]
                trial.set_user_attr("gain", float(gain))
                trial.set_user_attr("train_seconds", float(fit.train_seconds))
                trial.set_user_attr("parameter_proxy", int(fit.parameter_proxy))
                trial.set_user_attr("hpo_train_origins", len(train_screen))
                trial.set_user_attr("hpo_validation_origins", len(val_screen))
                return float(score)
            finally:
                del fit
                gc.collect()

        budget = policy.hpo_trials_per_repeat(model_name)
        _optimize_to_completed_budget(
            study,
            objective,
            budget,
            timeout=policy.hpo_timeout_seconds(model_name),
            catch=(MemoryError,),
            model_name=model_name,
            seed=seed,
            max_new_trials=max(8, budget * 4),
            phase="complete_budget",
        )
        completed = _complete_trials(study)
        if len(completed) < budget:
            state_counts = _trial_state_counts(study)
            raise RuntimeError(
                f"Only {len(completed)}/{budget} completed trials for {model_name}, "
                f"seed {seed} after the bounded completion safeguard; states={state_counts}. "
                "The incomplete repeat was not admitted. Check the preceding exception "
                "(usually memory or backend failure), correct it, and rerun to resume."
            )
        admitted = sorted(completed, key=lambda trial: trial.number)[:budget]
        best = min(admitted, key=lambda trial: float(trial.value))
        finalist_specs.append((seed, dict(best.params)))
        repeat_rows.append({
            "model": model_name, "repeat": repeat, "hpo_seed": seed,
            "budget": budget, "completed_trials": len(admitted),
            "total_attempted_trials": _attempted_trial_count(study),
            "pruned_trials": _trial_state_counts(study).get("PRUNED", 0),
            "failed_trials": _trial_state_counts(study).get("FAIL", 0),
            "completion_fallback_used": False,
            "budget_unit": "finite COMPLETE trials",
            "screening_best_score": float(best.value),
            "screening_best_params_json": json.dumps(best.params, sort_keys=True),
            "screening_train_origins": len(train_screen),
            "screening_validation_origins": len(val_screen),
        })
        frame = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
        frame.insert(0, "hpo_seed", seed)
        frame.insert(0, "model", model_name)
        frame = frame.loc[frame["number"].astype(int).isin({trial.number for trial in admitted})]
        trial_frames.append(frame)

    confirmation_rows: List[Dict[str, Any]] = []
    for seed, params in finalist_specs:
        key = json.dumps(params, sort_keys=True)
        fit = core.fit_direct_tree(model_name, train_screen, params, seed, config)
        try:
            raw = core.predict_direct_tree_raw(fit, val_tune)
            gain, _ = core.select_residual_gain(val_tune, raw)
            score = core.regression_metrics(
                val_tune.y, core.apply_residual_gain(val_tune, raw, gain)
            )["selection_score"]
            confirmation_rows.append({
                "model": model_name, "hpo_seed": seed,
                "params_json": key, "gain": float(gain),
                "full_validation_score": float(score),
                "confirmation_train_seconds": float(fit.train_seconds),
                "full_validation_origins": len(val_tune),
            })
        finally:
            del fit
            gc.collect()
    confirmations = pd.DataFrame(confirmation_rows)
    selected = _choose_parsimonious(confirmations, policy.parsimony_margin_score_points)
    setting = {
        "params": json.loads(selected["params_json"]),
        "gain": float(selected["gain"]),
        "best_epoch": None,
        "validation_score": float(selected["full_validation_score"]),
        "selected_hpo_seed": int(selected["hpo_seed"]),
        "selection_source": "two-seed development HPO plus full-validation confirmation",
    }
    return setting, pd.concat(trial_frames, ignore_index=True), pd.DataFrame(repeat_rows), confirmations


def _tune_neural_development(
    config: core.ExperimentConfig,
    policy: Deadline48Policy,
    model_name: str,
    train: core.WindowBundle,
    val_tune: core.WindowBundle,
    dataset_key: str = DEVELOPMENT_DATASET,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_screen = _systematic_sample(train, policy.hpo_train_origins, "deadline48_hpo_train")
    val_screen = _systematic_sample(val_tune, policy.hpo_validation_origins, "deadline48_hpo_val")
    trial_frames: List[pd.DataFrame] = []
    repeat_rows: List[Dict[str, Any]] = []
    finalist_specs: List[Tuple[int, Dict[str, Any], int]] = []

    for repeat, seed in enumerate(policy.hpo_seeds, start=1):
        name = f"LASH_hpo_{dataset_key}_{model_name}_hposeed{seed}"
        budget = policy.hpo_trials_per_repeat(model_name)
        storage = _study_storage(config, model_name, seed, dataset_key)
        sampler = core.optuna.samplers.TPESampler(seed=seed, multivariate=True)
        study = core.optuna.create_study(
            study_name=name,
            direction="minimize",
            sampler=sampler,
            pruner=core.optuna.pruners.MedianPruner(
                n_startup_trials=max(2, budget // 2), n_warmup_steps=3
            ),
            storage=storage,
            load_if_exists=True,
        )
        if not study.trials:
            study.enqueue_trial(
                _default_neural_params(model_name),
                user_attrs={"prespecified_default": True, "prior_result_reuse": False},
            )

        def objective(trial: Any) -> float:
            trial.set_user_attr("completion_phase", completion_phase["name"])
            params = core.suggest_neural_params(model_name, trial)
            fit = None
            try:
                fit = core.fit_neural(
                    model_name, train_screen, params, config,
                    validation_bundle=val_screen, trial=trial, seed=seed,
                )
                raw = core.predict_neural_raw(fit, val_screen, params, config)
                gain, _ = core.select_residual_gain(val_screen, raw)
                score = core.regression_metrics(
                    val_screen.y, core.apply_residual_gain(val_screen, raw, gain)
                )["selection_score"]
                trial.set_user_attr("gain", float(gain))
                trial.set_user_attr("best_epoch", int(fit.best_epoch))
                trial.set_user_attr("train_seconds", float(fit.train_seconds))
                trial.set_user_attr("parameter_count", int(fit.parameter_count))
                trial.set_user_attr("peak_gpu_memory_mb", float(fit.peak_gpu_memory_mb))
                trial.set_user_attr("hpo_train_origins", len(train_screen))
                trial.set_user_attr("hpo_validation_origins", len(val_screen))
                return float(score)
            finally:
                if fit is not None:
                    del fit
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        completion_phase = {"name": "median_pruner"}
        # The same deterministic rule is applied to every neural model and
        # dataset.  Median pruning may create more attempts than completions;
        # it is allowed up to 2x the declared completed-trial budget.
        median_attempt_cap = budget * 2
        median_remaining = max(
            0, median_attempt_cap - _attempted_trial_count(study)
        )
        if median_remaining and len(_complete_trials(study)) < budget:
            _optimize_to_completed_budget(
                study,
                objective,
                budget,
                timeout=policy.hpo_timeout_seconds(model_name),
                catch=(RuntimeError, MemoryError),
                model_name=model_name,
                seed=seed,
                max_new_trials=median_remaining,
                phase="median_pruner",
            )

        fallback_used = len(_complete_trials(study)) < budget
        if fallback_used:
            # With tiny 3--5-trial budgets, repeated median pruning can leave
            # fewer admissible trials even though training is healthy.  Reload
            # the same RDB study with a NopPruner and finish only the missing
            # COMPLETE trials.  Existing COMPLETE/PRUNED/FAILED records remain
            # intact, and test data are never accessed.
            completion_phase["name"] = "no_prune_completion_fallback"
            print(
                f"[HPO COMPLETION FALLBACK] {model_name} seed={seed}: "
                f"{len(_complete_trials(study))}/{budget} complete after "
                f"{_attempted_trial_count(study)} attempts. Finishing the missing "
                "complete trials "
                "without pruning; prior trials remain in the audit trail.",
                flush=True,
            )
            study = core.optuna.create_study(
                study_name=name,
                direction="minimize",
                sampler=core.optuna.samplers.TPESampler(seed=seed, multivariate=True),
                pruner=core.optuna.pruners.NopPruner(),
                storage=storage,
                load_if_exists=True,
            )
            _optimize_to_completed_budget(
                study,
                objective,
                budget,
                timeout=policy.hpo_timeout_seconds(model_name),
                catch=(RuntimeError, MemoryError),
                model_name=model_name,
                seed=seed,
                max_new_trials=max(8, budget * 4),
                phase="no_prune_completion_fallback",
            )
        completed = _complete_trials(study)
        if len(completed) < budget:
            state_counts = _trial_state_counts(study)
            raise RuntimeError(
                f"Only {len(completed)}/{budget} completed trials for {model_name}, "
                f"seed {seed} after the no-prune completion safeguard; "
                f"states={state_counts}. The incomplete repeat was not admitted. "
                "This now indicates a repeated runtime/memory failure rather than normal "
                "Optuna pruning; inspect the immediately preceding trial exception."
            )
        admitted = sorted(completed, key=lambda trial: trial.number)[:budget]
        best = min(admitted, key=lambda trial: float(trial.value))
        best_epoch = int(best.user_attrs.get("best_epoch", 10))
        finalist_specs.append((seed, dict(best.params), best_epoch))
        repeat_rows.append({
            "model": model_name, "repeat": repeat, "hpo_seed": seed,
            "budget": budget, "completed_trials": len(admitted),
            "total_attempted_trials": _attempted_trial_count(study),
            "pruned_trials": _trial_state_counts(study).get("PRUNED", 0),
            "failed_trials": _trial_state_counts(study).get("FAIL", 0),
            "completion_fallback_used": bool(fallback_used),
            "budget_unit": "finite COMPLETE trials",
            "screening_best_score": float(best.value),
            "screening_best_epoch": best_epoch,
            "screening_best_params_json": json.dumps(best.params, sort_keys=True),
            "screening_train_origins": len(train_screen),
            "screening_validation_origins": len(val_screen),
        })
        frame = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
        frame.insert(0, "hpo_seed", seed)
        frame.insert(0, "model", model_name)
        frame = frame.loc[frame["number"].astype(int).isin({trial.number for trial in admitted})]
        trial_frames.append(frame)

    confirmation_rows: List[Dict[str, Any]] = []
    for seed, params, best_epoch in finalist_specs:
        key = json.dumps(params, sort_keys=True)
        epochs = int(np.clip(best_epoch, 4, policy.max_epochs))
        fit = core.fit_neural(
            model_name, train_screen, params, config,
            validation_bundle=None, fixed_epochs=epochs, seed=seed,
        )
        try:
            raw = core.predict_neural_raw(fit, val_tune, params, config)
            gain, _ = core.select_residual_gain(val_tune, raw)
            score = core.regression_metrics(
                val_tune.y, core.apply_residual_gain(val_tune, raw, gain)
            )["selection_score"]
            confirmation_rows.append({
                "model": model_name, "hpo_seed": seed,
                "params_json": key, "gain": float(gain), "best_epoch": epochs,
                "full_validation_score": float(score),
                "confirmation_train_seconds": float(fit.train_seconds),
                "parameter_count": int(fit.parameter_count),
                "full_validation_origins": len(val_tune),
            })
        finally:
            del fit
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    confirmations = pd.DataFrame(confirmation_rows)
    selected = _choose_parsimonious(confirmations, policy.parsimony_margin_score_points)
    setting = {
        "params": json.loads(selected["params_json"]),
        "gain": float(selected["gain"]),
        "best_epoch": int(selected["best_epoch"]),
        "validation_score": float(selected["full_validation_score"]),
        "selected_hpo_seed": int(selected["hpo_seed"]),
        "selection_source": "two-seed development HPO plus full-validation confirmation",
    }
    return setting, pd.concat(trial_frames, ignore_index=True), pd.DataFrame(repeat_rows), confirmations


def _tune_rf_with_completed_repeat(
    config: core.ExperimentConfig,
    source_root: Path,
    policy: Deadline48Policy,
    train: core.WindowBundle,
    val_tune: core.WindowBundle,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Combine the completed RF search with one independent bounded repeat.

    The legacy study is never edited.  Its parsimonious admissible candidate is
    confirmed under the same deterministic screening/full-validation protocol
    used for the new seed-142 repeat, after which one frozen RF setting is used
    for all four one-seed final refits.
    """
    source_setting, source_trials, source_repeats = _load_source_rf_setting(
        Path(source_root), policy.parsimony_margin_score_points
    )
    fresh_seed = next(
        (seed for seed in policy.hpo_seeds if seed != source_setting["selected_hpo_seed"]),
        policy.hpo_seeds[-1],
    )
    fresh_policy = replace(policy, hpo_seeds=(int(fresh_seed),))
    fresh_setting, fresh_trials, fresh_repeats, fresh_confirmations = _tune_tree_development(
        config, fresh_policy, "RF", train, val_tune
    )

    train_screen = _systematic_sample(
        train, policy.hpo_train_origins, "deadline48_hpo_train"
    )
    fit = core.fit_direct_tree(
        "RF", train_screen, source_setting["params"],
        int(source_setting["selected_hpo_seed"]), config,
    )
    try:
        raw = core.predict_direct_tree_raw(fit, val_tune)
        gain, _ = core.select_residual_gain(val_tune, raw)
        score = core.regression_metrics(
            val_tune.y, core.apply_residual_gain(val_tune, raw, gain)
        )["selection_score"]
        source_confirmation = pd.DataFrame([{
            "model": "RF",
            "hpo_seed": int(source_setting["selected_hpo_seed"]),
            "params_json": json.dumps(source_setting["params"], sort_keys=True),
            "gain": float(gain),
            "full_validation_score": float(score),
            "confirmation_train_seconds": float(fit.train_seconds),
            "full_validation_origins": len(val_tune),
            "candidate_source": "completed_12_trial_repeat",
        }])
    finally:
        del fit
        gc.collect()

    confirmations = pd.concat(
        [source_confirmation, fresh_confirmations.assign(candidate_source="fresh_repeat")],
        ignore_index=True, sort=False,
    ).drop_duplicates("params_json", keep="first")
    selected = _choose_parsimonious(
        confirmations, policy.parsimony_margin_score_points
    )
    setting = {
        "params": json.loads(selected["params_json"]),
        "gain": float(selected["gain"]),
        "best_epoch": None,
        "validation_score": float(selected["full_validation_score"]),
        "selected_hpo_seed": int(selected["hpo_seed"]),
        "selection_source": (
            "completed 12-trial RF repeat plus independent bounded repeat and "
            "common full-validation confirmation"
        ),
    }

    legacy_repeats = source_repeats.copy()
    legacy_repeats["model"] = "RF"
    legacy_repeats["repeat_source"] = "completed_12_trial_hardware_study"
    if "repeat" not in legacy_repeats:
        legacy_repeats["repeat"] = np.arange(1, len(legacy_repeats) + 1)
    fresh_repeats = fresh_repeats.copy()
    fresh_repeats["repeat"] = np.arange(
        len(legacy_repeats) + 1, len(legacy_repeats) + len(fresh_repeats) + 1
    )
    fresh_repeats["repeat_source"] = "independent_deadline48_repeat"
    repeats = pd.concat([legacy_repeats, fresh_repeats], ignore_index=True, sort=False)

    legacy_trials = source_trials.copy()
    legacy_trials["model"] = "RF"
    legacy_trials["trial_source"] = "completed_12_trial_hardware_study"
    fresh_trials = fresh_trials.copy()
    fresh_trials["trial_source"] = "independent_deadline48_repeat"
    trials = pd.concat([legacy_trials, fresh_trials], ignore_index=True, sort=False)
    return setting, trials, repeats, confirmations


def build_or_load_frozen_settings(
    config: core.ExperimentConfig,
    source_root: Path,
    policy: Deadline48Policy = Deadline48Policy(),
    force: bool = False,
) -> Dict[str, Any]:
    """Select settings once on CLUSTER_1 and freeze them for all four datasets."""
    root = _deadline_root(config)
    frozen_path = root / "deadline48_frozen_settings.json"
    required = set(HPO_MODELS) | {"RIDGE"}
    if frozen_path.exists() and not force:
        payload = _read_json(frozen_path)
        if required.issubset(payload.get("models", {})):
            print("Frozen settings already complete; HPO is skipped.")
            return payload

    _, bundles = core.build_dataset_bundles(config, policy.development_dataset)
    train, val = bundles["train"], bundles["val"]
    val_tune, _ = core.split_validation_bundle(val)
    hpo_dir = root / "hpo_development"
    hpo_dir.mkdir(parents=True, exist_ok=True)

    models: Dict[str, Any] = {}
    trial_parts: List[pd.DataFrame] = []
    repeat_parts: List[pd.DataFrame] = []
    confirm_parts: List[pd.DataFrame] = []

    with progress_stage(config, "HPO_RIDGE", dataset=policy.development_dataset):
        ridge, ridge_table = core.tune_ridge(train, val_tune)
        ridge_table.to_csv(hpo_dir / "ridge_search.csv", index=False)
        models["RIDGE"] = {
            "params": {"alpha": float(ridge["alpha"])},
            "gain": float(ridge["gain"]), "best_epoch": None,
            "validation_score": float(ridge_table.iloc[0]["selection_score"]),
            "selected_hpo_seed": 0,
            "selection_source": "development validation grid",
        }

    # Preserve the completed 12-trial RF search and add one small independent
    # repeat.  Both finalists receive the same full-validation confirmation.
    with progress_stage(config, "HPO_RF_REUSE_PLUS_REPEAT", dataset=policy.development_dataset):
        rf_setting, rf_trials, rf_repeats, rf_confirmations = _tune_rf_with_completed_repeat(
            config, Path(source_root), policy, train, val_tune
        )
        models["RF"] = rf_setting
        if not rf_trials.empty:
            trial_parts.append(rf_trials)
            rf_trials.to_csv(hpo_dir / "rf_hpo_trials_reused_plus_repeat.csv", index=False)
        if not rf_repeats.empty:
            repeat_parts.append(rf_repeats)
            rf_repeats.to_csv(hpo_dir / "rf_hpo_repeats_reused_plus_repeat.csv", index=False)
        rf_confirmations.to_csv(
            hpo_dir / "rf_full_validation_confirmation.csv", index=False
        )
        confirm_parts.append(rf_confirmations)

    hpo_order = (
        "LASH_SEQ", "XGBOOST", "LIGHTGBM", "CATBOOST", "GBM",
        "MLP", "LSTM", "GRU", "CNN_LSTM", "TCN", "TRANSFORMER",
    )
    for model_name in hpo_order:
        with progress_stage(config, f"HPO_{model_name}", dataset=policy.development_dataset):
            if model_name in TREE_MODELS:
                setting, trials, repeats, confirmations = _tune_tree_development(
                    config, policy, model_name, train, val_tune
                )
            else:
                setting, trials, repeats, confirmations = _tune_neural_development(
                    config, policy, model_name, train, val_tune
                )
            models[model_name] = setting
            trials.to_csv(hpo_dir / f"{model_name.lower()}_hpo_trials.csv", index=False)
            repeats.to_csv(hpo_dir / f"{model_name.lower()}_hpo_repeats.csv", index=False)
            confirmations.to_csv(hpo_dir / f"{model_name.lower()}_full_validation_confirmation.csv", index=False)
            trial_parts.append(trials)
            repeat_parts.append(repeats)
            confirm_parts.append(confirmations)

    all_trials = pd.concat(trial_parts, ignore_index=True, sort=False)
    all_repeats = pd.concat(repeat_parts, ignore_index=True, sort=False)
    all_confirm = pd.concat(confirm_parts, ignore_index=True, sort=False)
    all_trials.to_csv(hpo_dir / "all_models_hpo_trials.csv", index=False)
    all_repeats.to_csv(hpo_dir / "all_models_hpo_repeats.csv", index=False)
    all_confirm.to_csv(hpo_dir / "all_models_full_validation_confirmation.csv", index=False)
    deadline_search_space_manifest().to_csv(hpo_dir / "deadline48_search_spaces.csv", index=False)

    payload = {
        "protocol": PROTOCOL_NAME,
        "development_dataset": policy.development_dataset,
        "external_datasets": list(EXTERNAL_DATASETS),
        "selection_rule": (
            "best full-validation finalist, with the fastest finalist selected when its score is "
            f"within {policy.parsimony_margin_score_points:.3f} points of the best"
        ),
        "test_used_for_selection": False,
        "hpo_train_sampling": "deterministic systematic coverage of complete development training interval",
        "hpo_train_origins": policy.hpo_train_origins,
        "hpo_validation_origins": policy.hpo_validation_origins,
        "full_validation_confirmation": True,
        "models": models,
    }
    _write_json(frozen_path, payload)
    return payload


def _prediction_path(dataset_dir: Path, model: str, seed: int) -> Path:
    return core.prediction_matrix_path(dataset_dir, model, seed)


def _meta_path(dataset_dir: Path, model: str, seed: int) -> Path:
    safe = core.re.sub(r"[^A-Za-z0-9_.-]+", "_", model)
    return dataset_dir / "configs" / f"{safe}__seed{seed}__runtime.json"


def _save_runtime_meta(dataset_dir: Path, model: str, seed: int, meta: Mapping[str, Any]) -> None:
    _write_json(_meta_path(dataset_dir, model, seed), dict(meta))


def _result_frames(
    dataset_dir: Path,
    model: str,
    seed: int,
    bundle: core.WindowBundle,
    prediction: np.ndarray,
    meta: Optional[Mapping[str, Any]] = None,
    save_prediction: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    details = dict(meta or {})
    metric = pd.DataFrame([{
        "model": model, "seed": seed, **core.regression_metrics(bundle.y, prediction),
        "train_seconds": details.get("train_seconds", np.nan),
        "inference_total_ms": details.get("inference_total_ms", np.nan),
        "inference_ms_per_origin": details.get("inference_ms_per_origin", np.nan),
        "parameter_count_or_tree_proxy": details.get("parameter_count_or_tree_proxy", np.nan),
        "peak_gpu_memory_mb": details.get("peak_gpu_memory_mb", np.nan),
        "model_file_mb": details.get("model_file_mb", np.nan),
        "execution_device": details.get("execution_device", "deterministic"),
        "selection_source": details.get("selection_source", "fixed"),
    }])
    horizon = core.per_horizon_metrics(bundle.y, prediction)
    horizon.insert(0, "seed", seed); horizon.insert(0, "model", model)
    origin = core.origin_loss_frame(bundle, prediction, model, seed)
    if save_prediction:
        core.save_prediction_matrix(dataset_dir, bundle, prediction, model, seed)
    return metric, horizon, origin


def _cached_result(
    dataset_dir: Path,
    model: str,
    seed: int,
    bundle: core.WindowBundle,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    path = _prediction_path(dataset_dir, model, seed)
    if not path.exists():
        return None
    payload = core.load_prediction(dataset_dir, model, seed)
    if payload["prediction"].shape != bundle.y.shape:
        return None
    meta = _read_json(_meta_path(dataset_dir, model, seed))
    print(f"[RESUME] {dataset_dir.name} {model} seed={seed}", flush=True)
    return _result_frames(
        dataset_dir, model, seed, bundle, payload["prediction"], meta, save_prediction=False
    )


def _infer_reused_rf_seconds(source_root: Path, seed: int) -> float:
    d = Path(source_root) / "benchmark" / DEVELOPMENT_DATASET
    current = d / "predictions" / f"RF__seed{seed}.npz"
    if not current.exists():
        return float("nan")
    if seed == 42:
        previous = d / "tables" / "rf_hpo_repeats.csv"
    else:
        seeds = [42, 142, 242]
        previous = d / "predictions" / f"RF__seed{seeds[seeds.index(seed)-1]}.npz"
    if not previous.exists():
        return float("nan")
    return max(0.0, current.stat().st_mtime - previous.stat().st_mtime)


def _reuse_source_rf_predictions(
    config: core.ExperimentConfig,
    source_root: Path,
    dataset_dir: Path,
    test: core.WindowBundle,
    setting: Mapping[str, Any],
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    source_dir = Path(source_root) / "benchmark" / DEVELOPMENT_DATASET / "predictions"
    results = []
    for seed in (42, 142, 242):
        source = source_dir / f"RF__seed{seed}.npz"
        if not source.exists():
            continue
        target = _prediction_path(dataset_dir, "RF", seed)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        payload = core.load_prediction(dataset_dir, "RF", seed)
        meta = {
            "train_seconds": _infer_reused_rf_seconds(source_root, seed),
            "inference_total_ms": np.nan,
            "inference_ms_per_origin": np.nan,
            "parameter_count_or_tree_proxy": 24 * int(setting["params"].get("n_estimators", 0)),
            "peak_gpu_memory_mb": 0.0,
            "model_file_mb": np.nan,
            "execution_device": "CPU",
            "selection_source": "reused completed 20260821_results RF",
        }
        _save_runtime_meta(dataset_dir, "RF", seed, meta)
        results.append(_result_frames(
            dataset_dir, "RF", seed, test, payload["prediction"], meta, save_prediction=False
        ))
    if not results:
        raise FileNotFoundError(
            "No completed CLUSTER_1 RF predictions were found in the source results. "
            "Let the current RF run finish before starting this notebook."
        )
    return results


def _fit_tree_final(
    config: core.ExperimentConfig,
    policy: Deadline48Policy,
    dataset_key: str,
    dataset_dir: Path,
    pretest: core.WindowBundle,
    test: core.WindowBundle,
    model_name: str,
    seed: int,
    setting: Mapping[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cached = _cached_result(dataset_dir, model_name, seed, test)
    if cached is not None:
        return cached
    with progress_stage(config, "FINAL_TREE", dataset=dataset_key, model=model_name, seed=seed):
        fit = core.fit_direct_tree(model_name, pretest, setting["params"], seed, config)
        started = time.perf_counter()
        raw = core.predict_direct_tree_raw(fit, test)
        inference_ms = (time.perf_counter() - started) * 1000.0
        prediction = core.apply_residual_gain(test, raw, setting["gain"])
        model_mb = np.nan
        if model_name in policy.save_tree_models:
            model_mb = core._save_tree_artifact(
                dataset_dir, model_name, seed, fit, float(setting["gain"])
            )
        meta = {
            "train_seconds": float(fit.train_seconds),
            "inference_total_ms": float(inference_ms),
            "inference_ms_per_origin": float(inference_ms / max(len(test), 1)),
            "parameter_count_or_tree_proxy": int(fit.parameter_proxy),
            "peak_gpu_memory_mb": 0.0,
            "model_file_mb": model_mb,
            "execution_device": "GPU" if model_name in {"XGBOOST", "CATBOOST"} else "CPU",
            "selection_source": setting["selection_source"],
        }
        _save_runtime_meta(dataset_dir, model_name, seed, meta)
        result = _result_frames(dataset_dir, model_name, seed, test, prediction, meta)
        del fit
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result


def _fit_neural_final(
    config: core.ExperimentConfig,
    policy: Deadline48Policy,
    dataset_key: str,
    dataset_dir: Path,
    pretest: core.WindowBundle,
    test: core.WindowBundle,
    model_name: str,
    output_name: str,
    seed: int,
    setting: Mapping[str, Any],
    ablation_flags: Optional[Mapping[str, Any]] = None,
) -> Tuple[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], np.ndarray, Any]:
    cached = _cached_result(dataset_dir, output_name, seed, test)
    if cached is not None and output_name != "LASH_SEQ_COMPONENT":
        payload = core.load_prediction(dataset_dir, output_name, seed)
        return cached, payload["prediction"], None
    with progress_stage(config, "FINAL_NEURAL", dataset=dataset_key, model=output_name, seed=seed):
        fit = core.fit_neural(
            model_name, pretest, setting["params"], config,
            validation_bundle=None, fixed_epochs=int(setting["best_epoch"]),
            seed=seed, ablation=dict(ablation_flags or {}),
        )
        started = time.perf_counter()
        raw = core.predict_neural_raw(fit, test, setting["params"], config)
        inference_ms = (time.perf_counter() - started) * 1000.0
        prediction = core.apply_residual_gain(test, raw, setting["gain"])
        model_mb = np.nan
        if model_name in policy.save_neural_models and seed == 42:
            model_mb = core._save_neural_artifact(
                dataset_dir, model_name, seed, fit, setting["params"],
                float(setting["gain"]), ablation=ablation_flags,
            )
        meta = {
            "train_seconds": float(fit.train_seconds),
            "inference_total_ms": float(inference_ms),
            "inference_ms_per_origin": float(inference_ms / max(len(test), 1)),
            "parameter_count_or_tree_proxy": int(fit.parameter_count),
            "peak_gpu_memory_mb": float(fit.peak_gpu_memory_mb),
            "model_file_mb": model_mb,
            "execution_device": "GPU_AMP" if config.amp_enabled else str(config.device),
            "selection_source": setting["selection_source"],
        }
        _save_runtime_meta(dataset_dir, output_name, seed, meta)
        result = _result_frames(dataset_dir, output_name, seed, test, prediction, meta)
        return result, prediction, fit


def _finalize_dataset_tables(
    config: core.ExperimentConfig,
    dataset_key: str,
    audit: Mapping[str, Any],
    metrics_parts: Sequence[pd.DataFrame],
    horizon_parts: Sequence[pd.DataFrame],
    origin_parts: Sequence[pd.DataFrame],
    selected_rows: Sequence[Mapping[str, Any]],
    router_table: pd.DataFrame,
    component_metrics: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    dataset_dir = core.dataset_output_dir(config, dataset_key)
    metrics = pd.concat(metrics_parts, ignore_index=True)
    horizons = pd.concat(horizon_parts, ignore_index=True)
    origins = pd.concat(origin_parts, ignore_index=True)
    selected = pd.DataFrame(selected_rows)
    agg_cols = [
        "MAPE", "CVRMSE", "NMAE", "selection_score", "train_seconds",
        "inference_ms_per_origin", "parameter_count_or_tree_proxy",
        "peak_gpu_memory_mb", "model_file_mb",
    ]
    summary = metrics.groupby("model", as_index=False)[agg_cols].agg(["mean", "std", "count"])
    summary.columns = ["model"] + [f"{a}_{b}" for a, b in summary.columns.tolist()[1:]]
    summary = summary.sort_values("selection_score_mean").reset_index(drop=True)

    tables = dataset_dir / "tables"
    metrics.to_csv(tables / "metrics_by_seed.csv", index=False)
    horizons.to_csv(tables / "horizon_metrics.csv", index=False)
    origins.to_csv(tables / "origin_losses.csv.gz", index=False, compression="gzip")
    selected.to_csv(tables / "selected_hyperparameters.csv", index=False)
    summary.to_csv(tables / "benchmark_summary.csv", index=False)
    component_metrics.to_csv(tables / "lash_component_metrics.csv", index=False)

    local_repeat_path = (
        _deadline_root(config) / "hpo_by_dataset" / dataset_key
        / "all_models_hpo_repeats.csv"
    )
    legacy_repeat_path = (
        _deadline_root(config) / "hpo_development" / "all_models_hpo_repeats.csv"
    )
    repeat_path = local_repeat_path if local_repeat_path.exists() else legacy_repeat_path
    repeats = pd.read_csv(repeat_path) if repeat_path.exists() else pd.DataFrame()
    workbook = dataset_dir / f"{dataset_key}_benchmark_results.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Benchmark_Summary", index=False)
        metrics.to_excel(writer, sheet_name="Metrics_By_Seed", index=False)
        horizons.to_excel(writer, sheet_name="Horizon_Metrics", index=False)
        origins.to_excel(writer, sheet_name="Origin_Losses", index=False)
        selected.to_excel(writer, sheet_name="Selected_Params", index=False)
        repeats.to_excel(writer, sheet_name="HPO_Repeats_Local", index=False)
        router_table.to_excel(writer, sheet_name="Router_Default", index=False)
        component_metrics.to_excel(writer, sheet_name="LASH_Components", index=False)
        deadline_search_space_manifest().to_excel(writer, sheet_name="Search_Spaces", index=False)
        pd.DataFrame([_json_ready(audit)]).astype(str).to_excel(writer, sheet_name="Protocol", index=False)
    return {
        "summary": summary, "metrics": metrics, "horizons": horizons,
        "origins": origins, "selected": selected, "router": router_table,
        "component_metrics": component_metrics,
    }


def run_deadline48_dataset(
    config: core.ExperimentConfig,
    dataset_key: str,
    frozen: Mapping[str, Any],
    policy: Deadline48Policy = Deadline48Policy(),
) -> Dict[str, pd.DataFrame]:
    core.set_seed(42)
    dataset_dir = core.dataset_output_dir(config, dataset_key)
    _, bundles = core.build_dataset_bundles(config, dataset_key)
    train, val, test = bundles["train"], bundles["val"], bundles["test"]
    val_tune, val_cal = core.split_validation_bundle(val)
    pre_cal = core.concat_bundles(train, val_tune, split="train_plus_val_tune")
    pretest = core.concat_bundles(train, val, split="train_plus_validation")
    settings = frozen["models"]

    selection_dataset = str(
        frozen.get("selection_dataset", frozen.get("development_dataset", DEVELOPMENT_DATASET))
    )
    audit = {
        "protocol": frozen.get("protocol", PROTOCOL_NAME),
        "spec": asdict(config.specs[dataset_key]),
        "weather_mode": config.weather_mode,
        "lookback": core.LOOKBACK, "horizon": core.HORIZON,
        "hyperparameter_selection_dataset": selection_dataset,
        "hyperparameters_selected_on_this_dataset": selection_dataset == dataset_key,
        "hyperparameters_frozen_before_test": True,
        "test_used_for_selection": False,
        "full_origin_final_refit_and_test": True,
        "tree_final_seeds": list(policy.tree_final_seeds),
        "neural_comparator_final_seeds": list(policy.neural_comparator_final_seeds),
        "lash_final_seeds": list(policy.lash_final_seeds),
        "n_origins": {key: len(value) for key, value in bundles.items()},
        "val_tune_origins": len(val_tune), "val_calibration_origins": len(val_cal),
    }
    _write_json(dataset_dir / "configs" / "data_and_protocol.json", audit)

    metrics_parts: List[pd.DataFrame] = []
    horizon_parts: List[pd.DataFrame] = []
    origin_parts: List[pd.DataFrame] = []
    selected_rows: List[Dict[str, Any]] = []
    component_rows: List[Dict[str, Any]] = []

    def add(result: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]):
        metric, horizon, origin = result
        metrics_parts.append(metric); horizon_parts.append(horizon); origin_parts.append(origin)

    # Deterministic baselines.
    for model, prediction in (
        ("SEASONAL_24", test.naive24),
        ("SEASONAL_168", test.naive168),
        ("ANCHOR", test.anchor),
    ):
        cached = _cached_result(dataset_dir, model, 0, test)
        add(cached or _result_frames(dataset_dir, model, 0, test, prediction, {
            "train_seconds": 0.0, "inference_total_ms": 0.0,
            "inference_ms_per_origin": 0.0, "parameter_count_or_tree_proxy": 0,
            "peak_gpu_memory_mb": 0.0, "model_file_mb": 0.0,
            "execution_device": "deterministic", "selection_source": "prespecified",
        }))

    # Frozen Ridge setting, local weight refit only.
    ridge_setting = settings["RIDGE"]
    cached_ridge = _cached_result(dataset_dir, "RIDGE", 0, test)
    if cached_ridge is None:
        with progress_stage(config, "FINAL_RIDGE", dataset=dataset_key, model="RIDGE", seed=0):
            ridge_fit = core.fit_ridge(pretest, float(ridge_setting["params"]["alpha"]))
            started = time.perf_counter(); ridge_raw = core.predict_ridge_raw(ridge_fit, test)
            inference_ms = (time.perf_counter() - started) * 1000.0
            ridge_pred = core.apply_residual_gain(test, ridge_raw, float(ridge_setting["gain"]))
            size = core._save_ridge_artifact(dataset_dir, ridge_fit, {
                "alpha": ridge_setting["params"]["alpha"], "gain": ridge_setting["gain"],
            })
            meta = {
                "train_seconds": ridge_fit.train_seconds, "inference_total_ms": inference_ms,
                "inference_ms_per_origin": inference_ms / max(len(test), 1),
                "parameter_count_or_tree_proxy": ridge_fit.parameter_count,
                "peak_gpu_memory_mb": 0.0, "model_file_mb": size,
                "execution_device": "CPU", "selection_source": ridge_setting["selection_source"],
            }
            _save_runtime_meta(dataset_dir, "RIDGE", 0, meta)
            cached_ridge = _result_frames(dataset_dir, "RIDGE", 0, test, ridge_pred, meta)
    else:
        ridge_pred = core.load_prediction(dataset_dir, "RIDGE", 0)["prediction"]
        ridge_fit = joblib.load(dataset_dir / "models" / "ridge.joblib")["fit"]
    add(cached_ridge)
    selected_rows.append({
        "model": "RIDGE", "params_json": json.dumps(ridge_setting["params"], sort_keys=True),
        "gain": ridge_setting["gain"], "selection_source": ridge_setting["selection_source"],
    })

    # LASH first so the central and external-validation claims finish before
    # optional comparator families consume the remaining wall-clock budget.
    lash_setting = settings["LASH_SEQ"]
    router_path = dataset_dir / "configs" / "router_default.json"
    expert_path = dataset_dir / "calibration" / "expert_predictions.npz"
    if router_path.exists() and expert_path.exists():
        router_payload = _read_json(router_path)
        router_weights = np.asarray(router_payload["weights"], dtype=float)
        router_table = pd.read_csv(dataset_dir / "calibration" / "router_default_candidates.csv")
    else:
        with progress_stage(config, "ROUTER_CALIBRATION", dataset=dataset_key, model="LASH"):
            cal_seq_fit = core.fit_neural(
                "LASH_SEQ", pre_cal, lash_setting["params"], config,
                validation_bundle=None, fixed_epochs=int(lash_setting["best_epoch"]), seed=42,
            )
            cal_seq_raw = core.predict_neural_raw(cal_seq_fit, val_cal, lash_setting["params"], config)
            cal_seq = core.apply_residual_gain(val_cal, cal_seq_raw, lash_setting["gain"])
            cal_ridge_fit = core.fit_ridge(pre_cal, float(ridge_setting["params"]["alpha"]))
            cal_ridge_raw = core.predict_ridge_raw(cal_ridge_fit, val_cal)
            cal_ridge = core.apply_residual_gain(val_cal, cal_ridge_raw, ridge_setting["gain"])
            router = core.select_router(val_cal, cal_seq, cal_ridge)
            router_weights = router.sequential_weights
            router_table = router.table
            router_table.to_csv(dataset_dir / "calibration" / "router_default_candidates.csv", index=False)
            np.savez_compressed(
                expert_path, actual=val_cal.y.astype(np.float32),
                seq=cal_seq.astype(np.float32), ridge=cal_ridge.astype(np.float32),
                forecast_origin=val_cal.forecast_origin, target_time=val_cal.target_time,
            )
            _write_json(router_path, {
                "mode": router.mode, "weights": router_weights,
                "calibration_score": router.calibration_score,
                "best_single_score": router.best_single_score,
                "improvement": router.improvement_over_single,
                "test_used_for_selection": False,
            })
            del cal_seq_fit, cal_ridge_fit
            gc.collect(); torch.cuda.empty_cache()

    for seed in policy.lash_final_seeds:
        cached = _cached_result(dataset_dir, "LASH", seed, test)
        if cached is not None:
            add(cached)
            component_payload = core.load_prediction(dataset_dir, "LASH_SEQ_COMPONENT", seed)
            component_rows.append({
                "model": "LASH_SEQ_COMPONENT", "seed": seed,
                **core.regression_metrics(test.y, component_payload["prediction"]),
            })
            continue
        (seq_result, seq_prediction, seq_fit) = _fit_neural_final(
            config, policy, dataset_key, dataset_dir, pretest, test,
            "LASH_SEQ", "LASH_SEQ_COMPONENT", seed, lash_setting,
        )
        lash_prediction = core.blend_predictions(seq_prediction, ridge_pred, router_weights)
        seq_meta = _read_json(_meta_path(dataset_dir, "LASH_SEQ_COMPONENT", seed))
        lash_meta = dict(seq_meta)
        lash_meta["parameter_count_or_tree_proxy"] = (
            float(seq_meta.get("parameter_count_or_tree_proxy", 0)) + float(ridge_fit.parameter_count)
        )
        _save_runtime_meta(dataset_dir, "LASH", seed, lash_meta)
        add(_result_frames(dataset_dir, "LASH", seed, test, lash_prediction, lash_meta))
        component_rows.append({
            "model": "LASH_SEQ_COMPONENT", "seed": seed,
            **core.regression_metrics(test.y, seq_prediction),
            "train_seconds": seq_meta.get("train_seconds", np.nan),
            "parameter_count": seq_meta.get("parameter_count_or_tree_proxy", np.nan),
        })
        if seq_fit is not None:
            del seq_fit
        gc.collect(); torch.cuda.empty_cache()
    selected_rows.append({
        "model": "LASH_SEQ", "params_json": json.dumps(lash_setting["params"], sort_keys=True),
        "gain": lash_setting["gain"], "best_epoch": lash_setting["best_epoch"],
        "selection_source": lash_setting["selection_source"],
        "router_weights_json": json.dumps(router_weights.tolist()),
    })

    # Reviewer-requested tree ensembles all use one fixed final seed and the
    # same frozen development setting on every dataset.  Legacy RF predictions
    # are retained as provenance but are not mixed into this matched benchmark.
    for model_name in TREE_MODELS:
        setting = settings[model_name]
        selected_rows.append({
            "model": model_name, "params_json": json.dumps(setting["params"], sort_keys=True),
            "gain": setting["gain"], "selection_source": setting["selection_source"],
        })
        for seed in policy.tree_final_seeds:
            add(_fit_tree_final(
                config, policy, dataset_key, dataset_dir, pretest, test,
                model_name, seed, setting,
            ))

    # Neural comparators use the same frozen development settings and one fixed
    # seed.  Their repeated-HPO variability is retained in the HPO tables.
    for model_name in NEURAL_COMPARATORS:
        setting = settings[model_name]
        selected_rows.append({
            "model": model_name, "params_json": json.dumps(setting["params"], sort_keys=True),
            "gain": setting["gain"], "best_epoch": setting["best_epoch"],
            "selection_source": setting["selection_source"],
        })
        for seed in policy.neural_comparator_final_seeds:
            result, _, fit = _fit_neural_final(
                config, policy, dataset_key, dataset_dir, pretest, test,
                model_name, model_name, seed, setting,
            )
            add(result)
            if fit is not None:
                del fit
            gc.collect(); torch.cuda.empty_cache()

    return _finalize_dataset_tables(
        config, dataset_key, audit, metrics_parts, horizon_parts, origin_parts,
        selected_rows, router_table, pd.DataFrame(component_rows),
    )


# Neutral public alias used by protocols that supply their own already-frozen
# per-dataset settings.  The legacy function name is retained for compatibility.
run_dataset_with_selected_settings = run_deadline48_dataset


def run_deadline48_benchmark(
    config: core.ExperimentConfig,
    source_root: Path,
    policy: Deadline48Policy = Deadline48Policy(),
) -> Dict[str, Dict[str, pd.DataFrame]]:
    frozen = build_or_load_frozen_settings(config, source_root, policy)
    results: Dict[str, Dict[str, pd.DataFrame]] = {}
    for dataset_key in EXECUTION_ORDER:
        if dataset_key not in config.dataset_keys:
            continue
        print("\n" + "=" * 80)
        print(f"Deadline-48 benchmark: {dataset_key}")
        print("=" * 80, flush=True)
        results[dataset_key] = run_deadline48_dataset(
            config, dataset_key, frozen, policy
        )
    return results


def _batched_circular_block_bootstrap_mean(
    diff: np.ndarray,
    block_length: int = 168,
    reps: int = 1000,
    seed: int = 2026,
    batch_size: int = 64,
) -> Dict[str, float]:
    """Numerically equivalent CBB means with batched index generation."""
    values = np.asarray(diff, float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return {"boot_mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_boot": np.nan}
    rng = np.random.default_rng(seed)
    blocks = int(math.ceil(n / block_length))
    offsets = np.arange(block_length, dtype=np.int64)[None, None, :]
    replicate_means = np.empty(int(reps), dtype=float)
    for first in range(0, int(reps), int(batch_size)):
        count = min(int(batch_size), int(reps) - first)
        starts = rng.integers(0, n, size=(count, blocks), dtype=np.int64)
        indices = (starts[:, :, None] + offsets) % n
        sampled = values[indices.reshape(count, -1)[:, :n]]
        replicate_means[first:first + count] = sampled.mean(axis=1)
    low, high = np.quantile(replicate_means, [0.025, 0.975])
    p_value = 2 * min(
        float(((replicate_means <= 0).sum() + 1) / (len(replicate_means) + 1)),
        float(((replicate_means >= 0).sum() + 1) / (len(replicate_means) + 1)),
    )
    return {
        "boot_mean": float(replicate_means.mean()),
        "ci_low": float(low), "ci_high": float(high),
        "p_boot": float(min(p_value, 1.0)),
    }


def _batched_hierarchical_seed_block_bootstrap(
    seed_diffs: Mapping[int, np.ndarray],
    block_length: int = 168,
    reps: int = 1000,
    seed: int = 2026,
    batch_size: int = 64,
) -> Dict[str, float]:
    keys = sorted(seed_diffs)
    if not keys:
        return {"hier_mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_hier": np.nan}
    arrays = {
        key: np.asarray(seed_diffs[key], float)[
            np.isfinite(np.asarray(seed_diffs[key], float))
        ]
        for key in keys
    }
    rng = np.random.default_rng(seed)
    replicate_means = np.empty(int(reps), dtype=float)
    key_count = len(keys)
    for first in range(0, int(reps), int(batch_size)):
        count = min(int(batch_size), int(reps) - first)
        sampled_key_positions = rng.integers(0, key_count, size=(count, key_count))
        seed_means = np.empty((count, key_count), dtype=float)
        for slot in range(key_count):
            for key_position, key in enumerate(keys):
                rows = np.flatnonzero(sampled_key_positions[:, slot] == key_position)
                if len(rows) == 0:
                    continue
                values = arrays[key]
                n = len(values)
                blocks = int(math.ceil(n / block_length))
                starts = rng.integers(0, n, size=(len(rows), blocks), dtype=np.int64)
                offsets = np.arange(block_length, dtype=np.int64)[None, None, :]
                indices = (starts[:, :, None] + offsets) % n
                sampled = values[indices.reshape(len(rows), -1)[:, :n]]
                seed_means[rows, slot] = sampled.mean(axis=1)
        replicate_means[first:first + count] = seed_means.mean(axis=1)
    low, high = np.quantile(replicate_means, [0.025, 0.975])
    p_value = 2 * min(
        float(((replicate_means <= 0).sum() + 1) / (len(replicate_means) + 1)),
        float(((replicate_means >= 0).sum() + 1) / (len(replicate_means) + 1)),
    )
    return {
        "hier_mean": float(replicate_means.mean()),
        "ci_low": float(low), "ci_high": float(high),
        "p_hier": float(min(p_value, 1.0)),
    }


def run_priority_statistical_tests(
    config: core.ExperimentConfig,
    policy: Deadline48Policy = Deadline48Policy(),
    comparators: Sequence[str] = INFERENTIAL_COMPARATORS,
) -> Dict[str, pd.DataFrame]:
    """Run prespecified dependence-aware contrasts without test-driven selection.

    The comparator set covers the operational seasonal/Ridge references, every
    requested tree-ensemble family, and the strongest architectural neural
    reference.  Other benchmark models remain in the descriptive/horizon and
    operational tables.  Prespecification reduces both multiplicity and the
    deadline cost; no comparator is chosen from test performance.
    """
    import lash_revision_analysis as analysis
    from statsmodels.stats.multitest import multipletests

    # Same estimands and random resampling design as the supplied analysis
    # module, but batched NumPy indexing avoids millions of Python-level loops.
    analysis.circular_block_bootstrap_mean = _batched_circular_block_bootstrap_mean
    analysis.hierarchical_seed_block_bootstrap = _batched_hierarchical_seed_block_bootstrap

    outputs: Dict[str, pd.DataFrame] = {}
    for dataset_key in config.dataset_keys:
        dataset_dir = core.dataset_output_dir(config, dataset_key)
        origin_path = dataset_dir / "tables" / "origin_losses.csv.gz"
        if not origin_path.exists():
            continue
        origin_df = pd.read_csv(origin_path, parse_dates=["forecast_origin"])
        available = set(origin_df["model"].astype(str).unique())
        tables: List[pd.DataFrame] = []
        acfs: List[pd.DataFrame] = []
        for comparator in comparators:
            if comparator not in available:
                continue
            table, acf_frame = analysis.dependence_aware_comparison(
                origin_df,
                comparator,
                hac_lags=(24, 72, 168, 336),
                block_lengths=(24, 72, 168, 336),
                bootstrap_reps=policy.bootstrap_reps,
            )
            tables.append(table)
            acf_frame.insert(0, "comparator", comparator)
            acfs.append(acf_frame)
        if not tables:
            continue
        tests = pd.concat(tables, ignore_index=True)
        tests["p_raw"] = tests["p_hac"].where(
            tests["method"].eq("HAC"),
            tests["p_boot"].where(tests["method"].eq("CBB"), tests["p_hier"]),
        )
        tests["p_holm"] = np.nan
        for _, indices in tests.groupby(
            ["method", "setting", "seed"], dropna=False
        ).groups.items():
            index = list(indices)
            p_values = tests.loc[index, "p_raw"].fillna(1.0).to_numpy(float)
            tests.loc[index, "p_holm"] = multipletests(p_values, method="holm")[1]
        acf_all = pd.concat(acfs, ignore_index=True)
        tests.to_csv(dataset_dir / "tables" / "dependence_aware_tests.csv", index=False)
        acf_all.to_csv(dataset_dir / "tables" / "paired_loss_acf.csv", index=False)
        _write_json(dataset_dir / "configs" / "inferential_comparators.json", {
            "comparators_prespecified_before_test_analysis": list(comparators),
            "available_and_tested": [c for c in comparators if c in available],
            "bootstrap_reps": policy.bootstrap_reps,
            "hac_lags": [24, 72, 168, 336],
            "block_lengths": [24, 72, 168, 336],
            "resampling_implementation": "batched circular-block indexing; estimand unchanged",
            "monte_carlo_p_correction": "plus-one correction; minimum two-sided p is 2/(B+1)",
            "remaining_benchmark_models": "descriptive and operational analyses only",
        })
        outputs[dataset_key] = tests
    return outputs


def _frozen_variant_dir(config: core.ExperimentConfig, dataset_key: str, variant: str) -> Path:
    directory = Path(config.output_root) / "controlled_mechanism" / dataset_key / variant
    for sub in ("tables", "predictions", "configs"):
        (directory / sub).mkdir(parents=True, exist_ok=True)
    return directory


def run_one_frozen_variant(
    config: core.ExperimentConfig,
    frozen: Mapping[str, Any],
    dataset_key: str,
    variant_name: str,
    policy: Deadline48Policy,
    *,
    ablation_flags: Optional[Mapping[str, Any]] = None,
    include_nonlinear_weather: bool = True,
    include_phase_shift_calendar: bool = True,
    yoy_components: Sequence[str] = (),
) -> Dict[str, pd.DataFrame]:
    directory = _frozen_variant_dir(config, dataset_key, variant_name)
    cached_summary = directory / "tables" / "summary.csv"
    if cached_summary.exists():
        return {
            name: pd.read_csv(directory / "tables" / f"{name}.csv")
            for name in ("summary", "metrics", "horizons")
        }
    _, bundles = core.build_dataset_bundles(
        config, dataset_key, yoy_components=yoy_components,
        include_nonlinear_weather=include_nonlinear_weather,
        include_phase_shift_calendar=include_phase_shift_calendar,
    )
    train, val, test = bundles["train"], bundles["val"], bundles["test"]
    val_tune, val_cal = core.split_validation_bundle(val)
    pre_cal = core.concat_bundles(train, val_tune, split="train_plus_val_tune")
    pretest = core.concat_bundles(train, val, split="train_plus_validation")
    ridge_setting = frozen["models"]["RIDGE"]
    seq_setting = frozen["models"]["LASH_SEQ"]

    cal_seq_fit = core.fit_neural(
        "LASH_SEQ", pre_cal, seq_setting["params"], config,
        validation_bundle=None, fixed_epochs=int(seq_setting["best_epoch"]), seed=42,
        ablation=dict(ablation_flags or {}),
    )
    cal_seq = core.apply_residual_gain(
        val_cal,
        core.predict_neural_raw(cal_seq_fit, val_cal, seq_setting["params"], config),
        seq_setting["gain"],
    )
    cal_ridge_fit = core.fit_ridge(pre_cal, float(ridge_setting["params"]["alpha"]))
    cal_ridge = core.apply_residual_gain(
        val_cal, core.predict_ridge_raw(cal_ridge_fit, val_cal), ridge_setting["gain"]
    )
    router = core.select_router(val_cal, cal_seq, cal_ridge)
    del cal_seq_fit, cal_ridge_fit
    gc.collect(); torch.cuda.empty_cache()

    final_ridge = core.fit_ridge(pretest, float(ridge_setting["params"]["alpha"]))
    ridge_test = core.apply_residual_gain(
        test, core.predict_ridge_raw(final_ridge, test), ridge_setting["gain"]
    )
    metric_rows: List[Dict[str, Any]] = []
    horizons: List[pd.DataFrame] = []
    origins: List[pd.DataFrame] = []
    for seed in policy.mechanism_seeds:
        fit = core.fit_neural(
            "LASH_SEQ", pretest, seq_setting["params"], config,
            validation_bundle=None, fixed_epochs=int(seq_setting["best_epoch"]), seed=seed,
            ablation=dict(ablation_flags or {}),
        )
        seq_test = core.apply_residual_gain(
            test, core.predict_neural_raw(fit, test, seq_setting["params"], config),
            seq_setting["gain"],
        )
        routed = core.blend_predictions(seq_test, ridge_test, router.sequential_weights)
        for component, prediction in (
            ("SEQUENTIAL_ONLY_PRIMARY", seq_test),
            ("REROUTED_PIPELINE_SECONDARY", routed),
        ):
            metric_rows.append({
                "dataset": dataset_key, "variant": variant_name,
                "component": component, "seed": seed,
                **core.regression_metrics(test.y, prediction),
            })
            h = core.per_horizon_metrics(test.y, prediction)
            h.insert(0, "seed", seed); h.insert(0, "component", component)
            h.insert(0, "variant", variant_name); horizons.append(h)
            o = core.origin_loss_frame(test, prediction, f"{variant_name}::{component}", seed)
            o.insert(0, "variant", variant_name); o.insert(1, "component", component)
            origins.append(o)
            np.savez_compressed(
                directory / "predictions" / f"{component.lower()}__seed{seed}.npz",
                actual=test.y.astype(np.float32), prediction=prediction.astype(np.float32),
                forecast_origin=test.forecast_origin, target_time=test.target_time,
            )
        del fit
        gc.collect(); torch.cuda.empty_cache()
    metrics = pd.DataFrame(metric_rows)
    horizon_df = pd.concat(horizons, ignore_index=True)
    origin_df = pd.concat(origins, ignore_index=True)
    summary = metrics.copy()
    metrics.to_csv(directory / "tables" / "metrics.csv", index=False)
    horizon_df.to_csv(directory / "tables" / "horizons.csv", index=False)
    origin_df.to_csv(directory / "tables" / "origins.csv.gz", index=False, compression="gzip")
    summary.to_csv(cached_summary, index=False)
    router.table.to_csv(directory / "tables" / "router_calibration.csv", index=False)
    _write_json(directory / "configs" / "controlled_ablation_protocol.json", {
        "dataset": dataset_key, "variant": variant_name,
        "primary_estimand": "sequential expert with full-model hyperparameters frozen",
        "secondary_estimand": "pipeline after variant-specific held-out router recalibration",
        "full_model_params": seq_setting["params"],
        "full_model_best_epoch": seq_setting["best_epoch"],
        "full_model_gain": seq_setting["gain"],
        "ablation_flags": dict(ablation_flags or {}),
        "include_nonlinear_weather": include_nonlinear_weather,
        "include_phase_shift_calendar": include_phase_shift_calendar,
        "yoy_components": list(yoy_components),
        "test_used_for_selection": False,
    })
    return {"summary": summary, "metrics": metrics, "horizons": horizon_df}


def run_controlled_mechanism_suite(
    config: core.ExperimentConfig,
    policy: Deadline48Policy = Deadline48Policy(),
    dataset_keys: Sequence[str] = ("CLUSTER_1", "CLUSTER_2"),
) -> Dict[str, pd.DataFrame]:
    frozen = _read_json(Path(config.output_root) / "deadline48_frozen_settings.json")
    if not frozen:
        raise FileNotFoundError("deadline48_frozen_settings.json")
    structural = dict(ablation.STRUCTURAL_VARIANTS)
    feature = dict(ablation.FEATURE_VARIANTS)
    yoy = dict(ablation.YOY_VARIANTS)
    summary_parts: List[pd.DataFrame] = []
    for dataset_key in dataset_keys:
        benchmark_dir = core.dataset_output_dir(config, dataset_key)
        reference_rows: List[Dict[str, Any]] = []
        for component, prediction_model in (
            ("SEQUENTIAL_ONLY_PRIMARY", "LASH_SEQ_COMPONENT"),
            ("REROUTED_PIPELINE_SECONDARY", "LASH"),
        ):
            payload = core.load_prediction(benchmark_dir, prediction_model, 42)
            reference_rows.append({
                "dataset": dataset_key,
                "variant": "FULL_MODEL_REFERENCE",
                "component": component,
                "seed": 42,
                **core.regression_metrics(payload["actual"], payload["prediction"]),
            })
        summary_parts.append(pd.DataFrame(reference_rows))
        for name, flags in structural.items():
            with progress_stage(config, "CONTROLLED_ABLATION", dataset=dataset_key, variant=name):
                result = run_one_frozen_variant(
                    config, frozen, dataset_key, name, policy, ablation_flags=flags
                )
            summary_parts.append(result["summary"])
        for name, options in feature.items():
            with progress_stage(config, "CONTROLLED_ABLATION", dataset=dataset_key, variant=name):
                result = run_one_frozen_variant(
                    config, frozen, dataset_key, name, policy,
                    include_nonlinear_weather=options.get("include_nonlinear_weather", True),
                    include_phase_shift_calendar=options.get("include_phase_shift_calendar", True),
                )
            summary_parts.append(result["summary"])
        for name, components in yoy.items():
            with progress_stage(config, "CONTROLLED_YOY", dataset=dataset_key, variant=name):
                result = run_one_frozen_variant(
                    config, frozen, dataset_key, name, policy, yoy_components=components
                )
            summary_parts.append(result["summary"])
    summary = pd.concat(summary_parts, ignore_index=True)
    reference = summary.loc[
        summary["variant"].eq("FULL_MODEL_REFERENCE"),
        ["dataset", "component", "selection_score"],
    ].rename(columns={"selection_score": "full_model_selection_score"})
    summary = summary.merge(reference, on=["dataset", "component"], how="left")
    summary["delta_selection_score_vs_full"] = (
        summary["selection_score"] - summary["full_model_selection_score"]
    )
    output = Path(config.output_root) / "02_controlled_mechanism_tests.xlsx"
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Controlled_Effects", index=False)
        pd.DataFrame([{
            "optimization_control": "all variants use frozen full-model hyperparameters",
            "primary_estimand": "SEQUENTIAL_ONLY_PRIMARY",
            "secondary_estimand": "REROUTED_PIPELINE_SECONDARY",
            "mechanism_seeds": str(policy.mechanism_seeds),
            "test_used_for_selection": False,
        }]).to_excel(writer, sheet_name="Protocol", index=False)
    return {key: group.reset_index(drop=True) for key, group in summary.groupby("dataset")}


def write_reviewer_evidence_map(
    config: core.ExperimentConfig,
    policy: Deadline48Policy = Deadline48Policy(),
) -> pd.DataFrame:
    rows = [
        {
            "reviewer_comment": "R1-C4 / R3-C2 external generalizability",
            "protocol_response": "Hyperparameters selected only on CLUSTER_1 and frozen before CLUSTER_2, BDG_EDU, and BDG_DORM refits.",
            "primary_outputs": "benchmark/BDG_EDU; benchmark/BDG_DORM; deadline48_frozen_settings.json",
        },
        {
            "reviewer_comment": "R3-C3 operational weather availability",
            "protocol_response": "Historical-only weather is enforced for every model and every dataset.",
            "primary_outputs": "benchmark/*/configs/data_and_protocol.json",
        },
        {
            "reviewer_comment": "R3-C5 omitted tree ensembles",
            "protocol_response": "RF, GBM, XGBoost, LightGBM, and CatBoost use causally valid 24 horizon-specific direct models.",
            "primary_outputs": "benchmark/*/*_benchmark_results.xlsx",
        },
        {
            "reviewer_comment": "R3-C6 tuning fairness and computation",
            "protocol_response": "Two independent HPO seeds and dimension-adaptive budgets are used on the development dataset; finalists receive full-validation confirmation. RF combines the completed 12-trial repeat with one bounded independent repeat and fully discloses both.",
            "primary_outputs": "hpo_development/all_models_hpo_trials.csv; all_models_hpo_repeats.csv; all_models_full_validation_confirmation.csv",
        },
        {
            "reviewer_comment": "R3-C7 router justification",
            "protocol_response": "Grid, smoothing, shrinkage, threshold, temporal-block, and NNLS sensitivity use frozen expert predictions without retraining.",
            "primary_outputs": "router_sensitivity/*; benchmark/*/calibration",
        },
        {
            "reviewer_comment": "R3-C8 matched ablation",
            "protocol_response": "Every structural and feature variant uses identical frozen full-model hyperparameters; sequential-only effects are primary and rerouted effects secondary.",
            "primary_outputs": "02_controlled_mechanism_tests.xlsx; controlled_mechanism/*",
        },
        {
            "reviewer_comment": "R3-C9 YoY components",
            "protocol_response": "Annual lag, holiday-matched proxy, annual scaling, annual anchor, and their combination are introduced separately under the same frozen protocol.",
            "primary_outputs": "02_controlled_mechanism_tests.xlsx; controlled_mechanism/*/YOY_*",
        },
        {
            "reviewer_comment": "R3-C10 dependence and seed variability",
            "protocol_response": "HAC/CBB sensitivity covers 24, 72, 168, and 336 hours; hierarchical block bootstrap resamples the three LASH training seeds and time blocks.",
            "primary_outputs": "benchmark/*/tables/dependence_aware_tests.csv; paired_loss_acf.csv; origin predictions",
        },
        {
            "reviewer_comment": "R1-C5 engineering relevance",
            "protocol_response": "Peak timing/magnitude, high-load recall, and demand-response curtailment sensitivity are computed from every saved origin trajectory.",
            "primary_outputs": "benchmark/*/tables/operational_peak_metrics.csv; demand_response_sensitivity.csv",
        },
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(Path(config.output_root) / "reviewer_evidence_map.csv", index=False)
    return frame


def write_manuscript_protocol_text(
    config: core.ExperimentConfig,
    hardware: Mapping[str, Any],
    policy: Deadline48Policy = Deadline48Policy(),
) -> Path:
    text = f"""METHODS - Compute-bounded selection and external validation
Hyperparameter selection was confined to CLUSTER_1. For each model family, two independent TPE seeds ({policy.hpo_seeds[0]} and {policy.hpo_seeds[1]}) were represented and dimension-adaptive budgets replaced an equal trial count. Screening windows were sampled deterministically across the complete development training period (maximum {policy.hpo_train_origins:,} origins), and the best configuration from each repeat was re-evaluated against the complete validation-tune segment. A prespecified parsimony rule selected the fastest finalist whose composite validation score was within {policy.parsimony_margin_score_points:.2f} percentage points of the best finalist. For RF, the completed 12-trial seed-42 study was reused read-only and complemented by one bounded seed-142 repeat; candidates from both were subjected to the same full-validation confirmation. Legacy RF prediction files were retained as provenance but were not mixed into the matched final benchmark. After selection, all hyperparameters were frozen before CLUSTER_2, BDG_EDU, and BDG_DORM were processed. Model weights were refit separately on each dataset's complete hourly train-plus-validation origins, whereas all test targets remained untouched until final evaluation.

METHODS - Seed allocation and inference
Tree ensembles and neural comparators used the prespecified final seed 42, while LASH was independently refit with seeds 42, 142, and 242. This allocation concentrates stochastic replication on the proposed estimator while retaining fixed-seed reproducibility for comparators. Dependence-aware inference resamples LASH training seeds hierarchically and circular time blocks, and sensitivity is reported for HAC lags and block lengths of 24, 72, 168, and 336 hours. Inferential contrasts were prespecified for Seasonal-24, Ridge, all five requested tree ensembles, and Transformer; the remaining benchmark families were reported descriptively and operationally. This restriction was fixed before test analysis to control multiplicity and computation, not chosen from test performance. Comparator predictions are held fixed inside resampling and are not represented as independent replicated fits.

METHODS - Controlled mechanism tests
Controlled mechanism tests were evaluated on CLUSTER_2, a confirmatory dataset not used for HPO. All structural, feature, and year-over-year variants were trained with the frozen full-model hyperparameters, epoch count, residual gain, data splits, and seed. The sequential-expert result is the primary ablation estimand because it isolates the altered mechanism before routing. A separately labeled secondary result recalibrates the router only on the held-out validation-calibration segment. Therefore, a Ridge-only routing outcome is interpreted as a deployment decision and not as direct evidence that the removed neural component is ineffective.

COMPUTATIONAL ENVIRONMENT
Experiments were executed on {hardware.get('cpu', 'the reported CPU')} with {hardware.get('ram_gb', 'reported')} GB RAM and {hardware.get('gpu', 'the reported GPU')} ({hardware.get('gpu_vram_gb', 'reported')} GB VRAM). Neural models used CUDA automatic mixed precision; XGBoost and CatBoost used their validated CUDA backends, whereas RF, GBM, LightGBM, Ridge, and deterministic baselines used the CPU. Timing is reported descriptively and was not used to select a test result.
"""
    path = Path(config.output_root) / "deadline48_manuscript_protocol_text.txt"
    path.write_text(text, encoding="utf-8")
    return path


def protocol_manifest(policy: Deadline48Policy = Deadline48Policy()) -> pd.DataFrame:
    rows = []
    for model in HPO_MODELS:
        rows.append({
            "model": model,
            "development_hpo": True,
            "hpo_reused": model == "RF",
            "hpo_repeats": len(policy.hpo_seeds),
            "trials_per_new_repeat": policy.hpo_trials_per_repeat(model),
            "legacy_trial_count": 12 if model == "RF" else 0,
            "hpo_train_origins_max": policy.hpo_train_origins,
            "full_validation_confirmation": True,
            "final_seeds": (
                str(policy.lash_final_seeds) if model == "LASH_SEQ"
                else str(policy.tree_final_seeds) if model in TREE_MODELS
                else str(policy.neural_comparator_final_seeds)
            ),
            "external_hpo": False,
        })
    return pd.DataFrame(rows)
