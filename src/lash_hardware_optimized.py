"""Hardware-specific, resume-safe policy for Ryzen 7 7800X3D + RTX 5070 Ti.

Scientific preprocessing, models, metrics, splits, and search spaces remain in
``lash_revision_core``.  This policy reduces redundant HPO training origins,
uses dedicated studies, warm-starts them from compatible legacy best settings,
and routes supported boosters to CUDA with a transparent CPU fallback.
"""
from __future__ import annotations

import gc
import json
import sys
import threading
import warnings
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch

import lash_revision_core as core


HARDWARE_TAG = "ryzen7800x3d_rtx5070ti"
HPO_ORIGIN_STRIDE = 3
GPU_BOOSTER_MODELS = {"XGBOOST", "CATBOOST"}
GPU_SUCCESS_MODELS = set()
GPU_FALLBACK_MODELS = set()
GPU_DISABLED_MODELS = set()
HPO_SAMPLE_LOG = []
_BACKEND_LOCK = threading.Lock()

_ORIGINAL_BUILD_TREE_ESTIMATOR = core.build_tree_estimator
_ORIGINAL_SEARCH_SPACE_MANIFEST = core.search_space_manifest

FINISHED_STATES = {
    core.optuna.trial.TrialState.COMPLETE,
    core.optuna.trial.TrialState.PRUNED,
    core.optuna.trial.TrialState.FAIL,
}


def _hardware_storage(root: Path, dataset_key: str, model_name: str, hpo_seed: int):
    directory = Path(root) / "optuna_hw5070ti" / dataset_key
    directory.mkdir(parents=True, exist_ok=True)
    safe = core.re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).lower()
    database = (directory / f"{safe}__hposeed{hpo_seed}.sqlite3").resolve()
    return core.optuna.storages.RDBStorage(
        url="sqlite:///" + database.as_posix(),
        engine_kwargs={"connect_args": {"timeout": 60}},
    )


def _legacy_best_params(config, dataset_key, model_name, hpo_seed, study_suffix=""):
    legacy_model = model_name + study_suffix
    study_name = f"LASH_revision_{dataset_key}_{model_name}{study_suffix}_hposeed{hpo_seed}"
    try:
        legacy = core.optuna.load_study(
            study_name=study_name,
            storage=core._study_storage(
                config.output_root, dataset_key, legacy_model, hpo_seed
            ),
        )
        completed = [
            trial for trial in legacy.trials
            if trial.state == core.optuna.trial.TrialState.COMPLETE and trial.value is not None
        ]
        if completed:
            params = dict(min(completed, key=lambda trial: float(trial.value)).params)
            return _sanitize_legacy_tree_params(model_name, params)
    except Exception:
        return None
    return None


def _nearest(value, choices):
    return min(choices, key=lambda item: abs(float(item) - float(value)))


def _sanitize_legacy_tree_params(model_name, params):
    """Project a legacy best setting into the declared bounded hardware space."""
    p = dict(params)
    if model_name == "RF":
        p["n_estimators"] = _nearest(p.get("n_estimators", 256), [128, 256, 384, 512])
        p["max_depth"] = _nearest(p.get("max_depth") or 18, [6, 10, 14, 18])
        p["max_features"] = _nearest(p.get("max_features", 0.5), [0.35, 0.5, 0.7])
    elif model_name == "GBM":
        p["n_estimators"] = int(np.clip(round(p.get("n_estimators", 300) / 50) * 50, 100, 600))
    elif model_name == "XGBOOST":
        p["n_estimators"] = int(np.clip(round(p.get("n_estimators", 500) / 100) * 100, 200, 1000))
    elif model_name == "LIGHTGBM":
        p["n_estimators"] = int(np.clip(round(p.get("n_estimators", 600) / 100) * 100, 300, 1200))
    elif model_name == "CATBOOST":
        p["iterations"] = int(np.clip(round(p.get("iterations", 500) / 100) * 100, 300, 1000))
    return p


def suggest_tree_params_hardware(model_name, trial):
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
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [2, 5, 10, 20, 40]),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "max_features": trial.suggest_float("max_features", 0.6, 1.0),
        }
    if model_name == "XGBOOST":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
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
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200, step=100),
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
            "iterations": trial.suggest_int("iterations", 300, 1000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 100.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
            "border_count": trial.suggest_categorical("border_count", [64, 128, 254]),
        }
    raise ValueError(model_name)


def hardware_search_space_manifest() -> pd.DataFrame:
    frame = _ORIGINAL_SEARCH_SPACE_MANIFEST().copy()
    replacements = {
        ("RF", "n_estimators"): "{128,256,384,512}",
        ("RF", "max_depth"): "{6,10,14,18}",
        ("RF", "max_features"): "{0.35,0.5,0.7}",
        ("GBM", "n_estimators"): "100..600 step 50",
        ("XGBOOST", "n_estimators"): "200..1000 step 100",
        ("LIGHTGBM", "n_estimators"): "300..1200 step 100",
        ("CATBOOST", "iterations"): "300..1000 step 100",
    }
    for (model, parameter), value in replacements.items():
        mask = frame["model"].eq(model) & frame["parameter"].eq(parameter)
        frame.loc[mask, "search_space"] = value
        frame.loc[mask, "rationale"] = "workstation-bounded prespecified range"
    return frame


def _enqueue_legacy_once(study, params):
    if not params or study.trials:
        return
    try:
        study.enqueue_trial(params, user_attrs={"warm_start": "legacy_best_re_evaluated"}, skip_if_exists=True)
    except TypeError:
        study.enqueue_trial(params)


def _hpo_training_bundle(bundle, dataset_key: str, model_name: str):
    indices = np.arange(0, len(bundle), HPO_ORIGIN_STRIDE, dtype=int)
    if len(indices) == 0 or indices[-1] != len(bundle) - 1:
        indices = np.unique(np.r_[indices, len(bundle) - 1])
    reduced = core.take_bundle(bundle, indices, split=f"{bundle.split}_hpo_stride{HPO_ORIGIN_STRIDE}")
    HPO_SAMPLE_LOG.append({
        "dataset": dataset_key,
        "model": model_name,
        "full_training_origins": len(bundle),
        "hpo_training_origins": len(reduced),
        "origin_stride_hours": HPO_ORIGIN_STRIDE,
        "validation_origin_stride": 1,
        "final_refit_origin_stride": 1,
        "test_origin_stride": 1,
    })
    return reduced


def _admitted_trials(study, budget: int):
    finished = sorted(
        (trial for trial in study.trials if trial.state in FINISHED_STATES),
        key=lambda trial: trial.number,
    )
    return finished[:budget], len(finished)


def _best_complete(admitted):
    complete = [
        trial for trial in admitted
        if trial.state == core.optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    if not complete:
        raise RuntimeError("No complete Optuna trial exists inside the hardware budget.")
    return min(complete, key=lambda trial: float(trial.value))


def _bounded_frame(study, admitted):
    frame = study.trials_dataframe(
        attrs=("number", "value", "state", "params", "user_attrs")
    )
    allowed = {int(trial.number) for trial in admitted}
    frame = frame.loc[frame["number"].astype(int).isin(allowed)].copy()
    frame["admitted_under_hardware_budget"] = True
    frame["hpo_origin_stride"] = HPO_ORIGIN_STRIDE
    return frame.sort_values("number").reset_index(drop=True)


def build_tree_estimator_hardware(model_name, params, seed, threads=1):
    if model_name in GPU_DISABLED_MODELS or not torch.cuda.is_available():
        return _ORIGINAL_BUILD_TREE_ESTIMATOR(model_name, params, seed, threads)
    p = dict(params)
    if model_name == "XGBOOST":
        from xgboost import XGBRegressor
        return XGBRegressor(
            **p,
            objective="reg:squarederror",
            tree_method="hist",
            device="cuda",
            random_state=seed,
            n_jobs=max(1, threads),
            verbosity=0,
        )
    if model_name == "CATBOOST":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(
            **p,
            loss_function="RMSE",
            random_seed=seed,
            task_type="GPU",
            devices="0",
            gpu_ram_part=0.80,
            thread_count=max(1, threads),
            verbose=False,
            allow_writing_files=False,
            bootstrap_type="Bayesian",
        )
    return _ORIGINAL_BUILD_TREE_ESTIMATOR(model_name, params, seed, threads)


def _fit_one_tree_horizon_hardware(model_name, params, seed, threads, X, y):
    estimator = build_tree_estimator_hardware(model_name, params, seed, threads)
    try:
        estimator.fit(X, y)
        if model_name in GPU_BOOSTER_MODELS and model_name not in GPU_DISABLED_MODELS:
            with _BACKEND_LOCK:
                GPU_SUCCESS_MODELS.add(model_name)
        return estimator
    except Exception as exc:
        if model_name not in GPU_BOOSTER_MODELS:
            raise
        with _BACKEND_LOCK:
            first_failure = model_name not in GPU_DISABLED_MODELS
            GPU_DISABLED_MODELS.add(model_name)
            GPU_FALLBACK_MODELS.add(model_name)
        if first_failure:
            warnings.warn(
                f"{model_name} CUDA backend failed ({type(exc).__name__}: {exc}). "
                "All subsequent fits for this family use the CPU backend.",
                RuntimeWarning,
            )
        estimator = _ORIGINAL_BUILD_TREE_ESTIMATOR(model_name, params, seed, threads)
        estimator.fit(X, y)
        return estimator


def fit_direct_tree_hardware(model_name, bundle, params, seed, config):
    X = core.make_tabular_origin_design(bundle)
    residual = bundle.y - bundle.anchor
    effective_params = dict(params)
    started = core.time.perf_counter()
    # One GPU fit at a time avoids 16-GB VRAM contention. CPU families use all
    # eight physical cores through horizon-level jobs.
    jobs = 1 if model_name in GPU_BOOSTER_MODELS and model_name not in GPU_DISABLED_MODELS else max(
        1, int(config.tree_horizon_jobs)
    )
    models = core.Parallel(n_jobs=jobs, prefer="threads")([
        core.delayed(_fit_one_tree_horizon_hardware)(
            model_name,
            effective_params,
            seed + horizon,
            config.tree_threads_per_model,
            X,
            residual[:, horizon],
        )
        for horizon in range(core.HORIZON)
    ])
    proxy = sum(
        int(getattr(model, "n_estimators", getattr(model, "tree_count_", 0)) or 0)
        for model in models
    )
    return core.TreeFit(
        model_name=model_name,
        models=models,
        params=effective_params,
        train_seconds=core.time.perf_counter() - started,
        parameter_proxy=proxy,
    )


def tune_tree_hardware(config, dataset_key, model_name, train, val_tune):
    budget = config.hpo_trials(model_name)
    train_hpo = _hpo_training_bundle(train, dataset_key, model_name)
    rows, frames = [], []
    for repeat_idx, hpo_seed in enumerate(config.hpo_seeds[:config.hpo_repeat_count(dataset_key)]):
        study_name = f"LASH_hw5070ti_{dataset_key}_{model_name}_hposeed{hpo_seed}"
        study = core.optuna.create_study(
            study_name=study_name,
            direction="minimize",
            sampler=core.optuna.samplers.TPESampler(seed=hpo_seed, multivariate=True),
            storage=_hardware_storage(config.output_root, dataset_key, model_name, hpo_seed),
            load_if_exists=True,
        )
        _enqueue_legacy_once(
            study, _legacy_best_params(config, dataset_key, model_name, hpo_seed)
        )

        def objective(trial):
            parameters = core.suggest_tree_params(model_name, trial)
            fit = fit_direct_tree_hardware(
                model_name, train_hpo, parameters, hpo_seed, config
            )
            raw = core.predict_direct_tree_raw(fit, val_tune)
            gain, _ = core.select_residual_gain(val_tune, raw)
            prediction = core.apply_residual_gain(val_tune, raw, gain)
            score = core.regression_metrics(val_tune.y, prediction)["selection_score"]
            trial.set_user_attr("gain", gain)
            trial.set_user_attr("train_seconds", fit.train_seconds)
            trial.set_user_attr("tree_count_proxy", fit.parameter_proxy)
            trial.set_user_attr("hpo_training_origins", len(train_hpo))
            trial.set_user_attr("hpo_origin_stride", HPO_ORIGIN_STRIDE)
            del fit
            gc.collect()
            return float(score)

        admitted, _ = _admitted_trials(study, budget)
        if len(admitted) < budget:
            study.optimize(
                objective,
                n_trials=budget - len(admitted),
                n_jobs=1,
                show_progress_bar=True,
                gc_after_trial=True,
            )
        admitted, total_finished = _admitted_trials(study, budget)
        best = _best_complete(admitted)
        rows.append({
            "repeat": repeat_idx + 1,
            "hpo_seed": hpo_seed,
            "best_score": float(best.value),
            "gain": best.user_attrs.get("gain"),
            "n_trials": len(admitted),
            "available_finished_trials": total_finished,
            "hardware_budget": budget,
            "hpo_origin_stride": HPO_ORIGIN_STRIDE,
            "params_json": json.dumps(best.params, sort_keys=True),
        })
        frame = _bounded_frame(study, admitted)
        frame.insert(0, "hpo_seed", hpo_seed)
        frame.insert(0, "model", model_name)
        frame.insert(0, "dataset", dataset_key)
        train_col = "user_attrs_train_seconds"
        rows[-1]["sum_trial_train_seconds"] = (
            float(pd.to_numeric(frame[train_col], errors="coerce").sum())
            if train_col in frame else np.nan
        )
        frames.append(frame)

    repeat_df = pd.DataFrame(rows).sort_values("best_score").reset_index(drop=True)
    selected = repeat_df.iloc[0]
    return {
        "params": json.loads(selected["params_json"]),
        "gain": float(selected["gain"]),
        "selected_hpo_seed": int(selected["hpo_seed"]),
        "validation_score": float(selected["best_score"]),
    }, pd.concat(frames, ignore_index=True), repeat_df


def tune_neural_hardware(
    config,
    dataset_key,
    model_name,
    train,
    val_tune,
    ablation: Optional[Mapping[str, Any]] = None,
    study_suffix: str = "",
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    budget = config.hpo_trials(model_name)
    train_hpo = _hpo_training_bundle(train, dataset_key, model_name + study_suffix)
    rows, frames = [], []
    for repeat_idx, hpo_seed in enumerate(config.hpo_seeds[:config.hpo_repeat_count(dataset_key)]):
        storage_name = model_name + study_suffix
        study_name = f"LASH_hw5070ti_{dataset_key}_{storage_name}_hposeed{hpo_seed}"
        study = core.optuna.create_study(
            study_name=study_name,
            direction="minimize",
            sampler=core.optuna.samplers.TPESampler(seed=hpo_seed, multivariate=True),
            pruner=core.optuna.pruners.MedianPruner(
                n_startup_trials=max(4, budget // 4), n_warmup_steps=4
            ),
            storage=_hardware_storage(config.output_root, dataset_key, storage_name, hpo_seed),
            load_if_exists=True,
        )
        _enqueue_legacy_once(
            study,
            _legacy_best_params(
                config, dataset_key, model_name, hpo_seed, study_suffix
            ),
        )

        def objective(trial):
            parameters = core.suggest_neural_params(model_name, trial)
            fit = core.fit_neural(
                model_name,
                train_hpo,
                parameters,
                config,
                validation_bundle=val_tune,
                trial=trial,
                seed=hpo_seed,
                ablation=ablation,
            )
            raw = core.predict_neural_raw(fit, val_tune, parameters, config)
            gain, _ = core.select_residual_gain(val_tune, raw)
            score = core.regression_metrics(
                val_tune.y, core.apply_residual_gain(val_tune, raw, gain)
            )["selection_score"]
            trial.set_user_attr("best_epoch", int(fit.best_epoch))
            trial.set_user_attr("gain", float(gain))
            trial.set_user_attr("parameter_count", int(fit.parameter_count))
            trial.set_user_attr("train_seconds", float(fit.train_seconds))
            trial.set_user_attr("hpo_training_origins", len(train_hpo))
            trial.set_user_attr("hpo_origin_stride", HPO_ORIGIN_STRIDE)
            del fit
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return float(score)

        admitted, _ = _admitted_trials(study, budget)
        if len(admitted) < budget:
            study.optimize(
                objective,
                n_trials=budget - len(admitted),
                n_jobs=1,
                show_progress_bar=True,
                gc_after_trial=True,
            )
        admitted, total_finished = _admitted_trials(study, budget)
        best = _best_complete(admitted)
        rows.append({
            "repeat": repeat_idx + 1,
            "hpo_seed": hpo_seed,
            "best_score": float(best.value),
            "best_epoch": best.user_attrs.get("best_epoch"),
            "gain": best.user_attrs.get("gain"),
            "parameter_count": best.user_attrs.get("parameter_count"),
            "n_trials": len(admitted),
            "available_finished_trials": total_finished,
            "hardware_budget": budget,
            "hpo_origin_stride": HPO_ORIGIN_STRIDE,
            "params_json": json.dumps(best.params, sort_keys=True),
        })
        frame = _bounded_frame(study, admitted)
        frame.insert(0, "hpo_seed", hpo_seed)
        frame.insert(0, "model", model_name)
        frame.insert(0, "dataset", dataset_key)
        train_col = "user_attrs_train_seconds"
        rows[-1]["sum_trial_train_seconds"] = (
            float(pd.to_numeric(frame[train_col], errors="coerce").sum())
            if train_col in frame else np.nan
        )
        frames.append(frame)

    repeat_df = pd.DataFrame(rows).sort_values("best_score").reset_index(drop=True)
    selected = repeat_df.iloc[0]
    return {
        "params": json.loads(selected["params_json"]),
        "best_epoch": int(selected["best_epoch"]),
        "gain": float(selected["gain"]),
        "selected_hpo_seed": int(selected["hpo_seed"]),
        "validation_score": float(selected["best_score"]),
    }, pd.concat(frames, ignore_index=True), repeat_df


def backend_summary() -> Dict[str, Any]:
    return {
        "gpu_booster_success_models": sorted(GPU_SUCCESS_MODELS - GPU_FALLBACK_MODELS),
        "gpu_booster_fallback_models": sorted(GPU_FALLBACK_MODELS),
        "gpu_booster_disabled_models": sorted(GPU_DISABLED_MODELS),
    }


def probe_gpu_boosters() -> Dict[str, Any]:
    """Run tiny CUDA fits before the expensive benchmark and enable safe fallback."""
    if not torch.cuda.is_available():
        GPU_DISABLED_MODELS.update(GPU_BOOSTER_MODELS)
        GPU_FALLBACK_MODELS.update(GPU_BOOSTER_MODELS)
        return backend_summary()
    rng = np.random.default_rng(42)
    X = rng.normal(size=(96, 12)).astype(np.float32)
    y = rng.normal(size=96).astype(np.float32)
    probes = {
        "XGBOOST": {
            "n_estimators": 2, "learning_rate": 0.1, "max_depth": 2,
            "min_child_weight": 1, "subsample": 1.0, "colsample_bytree": 1.0,
            "reg_alpha": 1e-8, "reg_lambda": 1.0, "gamma": 0.0, "booster": "gbtree",
        },
        "CATBOOST": {
            "iterations": 2, "learning_rate": 0.1, "depth": 4,
            "l2_leaf_reg": 1.0, "random_strength": 0.0,
            "bagging_temperature": 1.0, "border_count": 64,
        },
    }
    for model_name, params in probes.items():
        try:
            estimator = build_tree_estimator_hardware(model_name, params, 42, threads=1)
            estimator.fit(X, y)
            GPU_SUCCESS_MODELS.add(model_name)
        except Exception as exc:
            GPU_DISABLED_MODELS.add(model_name)
            GPU_FALLBACK_MODELS.add(model_name)
            warnings.warn(
                f"{model_name} CUDA probe failed ({type(exc).__name__}: {exc}); CPU fallback enabled.",
                RuntimeWarning,
            )
    return backend_summary()


def hpo_sample_manifest() -> pd.DataFrame:
    return pd.DataFrame(HPO_SAMPLE_LOG).drop_duplicates().reset_index(drop=True)


def apply_hardware_patch() -> None:
    core.suggest_tree_params = suggest_tree_params_hardware
    core.search_space_manifest = hardware_search_space_manifest
    core.build_tree_estimator = build_tree_estimator_hardware
    core._fit_one_tree_horizon = _fit_one_tree_horizon_hardware
    core.fit_direct_tree = fit_direct_tree_hardware
    core.tune_tree_repeated = tune_tree_hardware
    core.tune_neural_repeated = tune_neural_hardware
    ablation = sys.modules.get("lash_revision_ablation")
    if ablation is not None:
        ablation.tune_neural_repeated = tune_neural_hardware
