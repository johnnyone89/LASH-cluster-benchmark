"""Reviewer-consistent per-dataset HPO protocol for the 2026-08-23 rerun.

Every model family, including Random Forest, is optimized from scratch under
the same prespecified rule on each of the four datasets.  No Optuna database,
hyperparameter setting, fitted model, or prediction from an earlier results
directory is read.  Within each dataset, HPO uses only chronological training
and validation-tune origins; the selected setting is frozen before the test
segment is evaluated.

The module reuses the audited causal feature construction, direct 24-horizon
estimators, neural architectures, metrics, checkpoint formats, hardware
backends, and posthoc methods from the supplied revision package.  It changes
only the selection scope from development-transfer to dataset-local HPO.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import lash_deadline48 as engine
import lash_revision_ablation as ablation
import lash_revision_core as core


PROTOCOL_NAME = "reviewer_20260823_per_dataset_hpo_v1"
EXECUTION_ORDER = ("BDG_EDU", "BDG_DORM", "CLUSTER_2", "CLUSTER_1")
TREE_MODELS = engine.TREE_MODELS
NEURAL_COMPARATORS = engine.NEURAL_COMPARATORS
HPO_MODELS = TREE_MODELS + NEURAL_COMPARATORS + ("LASH_SEQ",)
REQUIRED_SETTINGS = set(HPO_MODELS) | {"RIDGE"}
BENCHMARK_MODELS = core.BENCHMARK_MODELS


def _prior_result_reuse_forbidden(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError(
        "Prior-result reuse is forbidden by reviewer_20260823_per_dataset_hpo_v1."
    )


# Fail closed if a future edit accidentally reaches a legacy transfer/reuse API
# in the shared execution engine.  The local-HPO pipeline never calls these.
for _legacy_api in (
    "_load_source_rf_setting",
    "_tune_rf_with_completed_repeat",
    "_reuse_source_rf_predictions",
    "build_or_load_frozen_settings",
    "run_deadline48_benchmark",
):
    if hasattr(engine, _legacy_api):
        setattr(engine, _legacy_api, _prior_result_reuse_forbidden)


@dataclass(frozen=True)
class PerDatasetHPOPolicy(engine.Deadline48Policy):
    """One compute-bounded but identical HPO contract for every dataset."""

    computational_budget_hours: float = 48.0
    bootstrap_reps: int = 1000
    tree_final_seeds: Tuple[int, ...] = (42, 142, 242)
    neural_comparator_final_seeds: Tuple[int, ...] = (42, 142, 242)
    lash_final_seeds: Tuple[int, ...] = (42, 142, 242)
    mechanism_seeds: Tuple[int, ...] = (42, 142, 242)

    def hpo_timeout_seconds(self, model_name: str) -> Optional[int]:
        # Consistency takes precedence over a per-repeat wall-clock cutoff in
        # this rerun.  Every declared trial must finish or the study remains
        # incomplete and resumes on the next execution.
        return None


def _root(config: core.ExperimentConfig) -> Path:
    root = Path(config.output_root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _hpo_dir(config: core.ExperimentConfig, dataset_key: str) -> Path:
    path = _root(config) / "hpo_by_dataset" / dataset_key
    path.mkdir(parents=True, exist_ok=True)
    return path


def _settings_path(config: core.ExperimentConfig, dataset_key: str) -> Path:
    return _hpo_dir(config, dataset_key) / "local_hpo_settings.json"


def search_space_manifest() -> pd.DataFrame:
    frame = engine.deadline_search_space_manifest().copy()
    frame["selection_scope"] = "identical dataset-local chronological HPO"
    frame["prior_result_reuse"] = False
    return frame


def protocol_manifest(
    policy: PerDatasetHPOPolicy = PerDatasetHPOPolicy(),
    dataset_keys: Sequence[str] = EXECUTION_ORDER,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for dataset_key in dataset_keys:
        rows.append({
            "dataset": dataset_key,
            "model": "RIDGE",
            "selection_method": "8-value chronological validation grid",
            "hpo_repeats": 1,
            "trials_per_repeat": 8,
            "hpo_seeds": "deterministic",
            "prior_result_reuse": False,
            "full_validation_confirmation": True,
            "final_seeds": "(0,)",
            "test_used_for_selection": False,
            "trial_budget_unit": "complete grid evaluations",
            "pruning_policy": "not applicable",
            "pruned_trials_admitted": False,
        })
        for model in HPO_MODELS:
            is_neural = model not in TREE_MODELS
            rows.append({
                "dataset": dataset_key,
                "model": model,
                "selection_method": "TPE with dimension-adaptive budget",
                "hpo_repeats": len(policy.hpo_seeds),
                "trials_per_repeat": policy.hpo_trials_per_repeat(model),
                "hpo_seeds": str(policy.hpo_seeds),
                "prior_result_reuse": False,
                "hpo_train_origins_max": policy.hpo_train_origins,
                "hpo_validation_origins_max": policy.hpo_validation_origins,
                "full_validation_confirmation": True,
                "final_seeds": (
                    str(policy.lash_final_seeds) if model == "LASH_SEQ"
                    else str(policy.tree_final_seeds) if model in TREE_MODELS
                    else str(policy.neural_comparator_final_seeds)
                ),
                "test_used_for_selection": False,
                "trial_budget_unit": "finite COMPLETE trials",
                "pruning_policy": (
                    "MedianPruner up to 2x budget attempts; then deterministic "
                    "NopPruner completion fallback"
                    if is_neural else "not applicable"
                ),
                "pruned_trials_admitted": False,
            })
    return pd.DataFrame(rows)


def _protocol_signature(policy: PerDatasetHPOPolicy) -> str:
    payload = {
        "protocol": PROTOCOL_NAME,
        "policy": asdict(policy),
        "search_spaces": search_space_manifest().to_dict(orient="records"),
        "hpo_models": list(HPO_MODELS),
    }
    raw = json.dumps(engine._json_ready(payload), sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _checkpoint_settings(
    config: core.ExperimentConfig,
    dataset_key: str,
    policy: PerDatasetHPOPolicy,
    models: Mapping[str, Any],
    status: str,
) -> Dict[str, Any]:
    payload = {
        "protocol": PROTOCOL_NAME,
        "protocol_signature": _protocol_signature(policy),
        "status": status,
        "selection_dataset": dataset_key,
        "selection_scope": "dataset-local train and validation-tune only",
        "test_used_for_selection": False,
        "prior_result_reuse": False,
        "hpo_seeds": list(policy.hpo_seeds),
        "hpo_train_sampling": (
            "deterministic systematic coverage of the complete local training interval"
        ),
        "hpo_train_origins_max": policy.hpo_train_origins,
        "hpo_validation_origins_max": policy.hpo_validation_origins,
        "full_validation_confirmation": True,
        "trial_budget_unit": "finite COMPLETE trials",
        "neural_pruning_policy": (
            "MedianPruner for at most twice the declared budget in attempted trials; "
            "if pruning leaves too few completed trials, NopPruner completes only the "
            "missing quota. PRUNED and FAILED trials are retained for audit but never admitted."
        ),
        "parsimony_rule": (
            "fastest full-validation finalist within "
            f"{policy.parsimony_margin_score_points:.3f} score points of the best"
        ),
        "models": dict(models),
    }
    engine._write_json(_settings_path(config, dataset_key), payload)
    return payload


def _read_existing_settings(
    config: core.ExperimentConfig,
    dataset_key: str,
    policy: PerDatasetHPOPolicy,
) -> Dict[str, Any]:
    path = _settings_path(config, dataset_key)
    if not path.exists():
        return {}
    payload = engine._read_json(path)
    expected = _protocol_signature(policy)
    if payload.get("protocol_signature") != expected:
        raise RuntimeError(
            f"{path} belongs to a different HPO contract. Rename the output folder "
            "rather than mixing incompatible studies."
        )
    return payload


def _consolidate_dataset_hpo(
    config: core.ExperimentConfig,
    dataset_key: str,
    policy: PerDatasetHPOPolicy,
    payload: Mapping[str, Any],
) -> None:
    directory = _hpo_dir(config, dataset_key)

    def collect(suffix: str) -> pd.DataFrame:
        frames = []
        for path in sorted(directory.glob(f"*{suffix}")):
            if path.name.startswith("all_models_"):
                continue
            frame = pd.read_csv(path)
            if "dataset" not in frame:
                frame.insert(0, "dataset", dataset_key)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()

    trials = collect("_hpo_trials.csv")
    repeats = collect("_hpo_repeats.csv")
    confirmations = collect("_full_validation_confirmation.csv")
    if not trials.empty:
        trials.to_csv(directory / "all_models_hpo_trials.csv", index=False)
        convergence = trials.copy()
        convergence["value"] = pd.to_numeric(convergence.get("value"), errors="coerce")
        convergence["number"] = pd.to_numeric(
            convergence.get("number"), errors="coerce"
        )
        convergence = convergence.sort_values(
            ["dataset", "model", "hpo_seed", "number"]
        )
        convergence["best_score_so_far"] = convergence.groupby(
            ["dataset", "model", "hpo_seed"], dropna=False
        )["value"].cummin()
        convergence.to_csv(directory / "hpo_convergence.csv", index=False)
    else:
        convergence = pd.DataFrame()
    if not repeats.empty:
        repeats.to_csv(directory / "all_models_hpo_repeats.csv", index=False)
    if not confirmations.empty:
        confirmations.to_csv(
            directory / "all_models_full_validation_confirmation.csv", index=False
        )

    selected_rows = []
    for model, setting in payload["models"].items():
        selected_rows.append({
            "dataset": dataset_key,
            "model": model,
            "validation_score": setting.get("validation_score"),
            "selected_hpo_seed": setting.get("selected_hpo_seed"),
            "best_epoch": setting.get("best_epoch"),
            "gain": setting.get("gain"),
            "params_json": json.dumps(setting.get("params", {}), sort_keys=True),
            "selection_source": setting.get("selection_source"),
        })
    selected = pd.DataFrame(selected_rows).sort_values("model")
    selected.to_csv(directory / "selected_settings.csv", index=False)
    with pd.ExcelWriter(
        directory / f"{dataset_key}_local_hpo_report.xlsx", engine="openpyxl"
    ) as writer:
        selected.to_excel(writer, sheet_name="Selected_Settings", index=False)
        repeats.to_excel(writer, sheet_name="HPO_Repeats", index=False)
        confirmations.to_excel(writer, sheet_name="Full_Val_Confirmation", index=False)
        convergence.to_excel(writer, sheet_name="Convergence", index=False)
        search_space_manifest().to_excel(writer, sheet_name="Search_Spaces", index=False)
        protocol_manifest(policy, (dataset_key,)).to_excel(
            writer, sheet_name="Protocol", index=False
        )


def build_or_load_dataset_settings(
    config: core.ExperimentConfig,
    dataset_key: str,
    policy: PerDatasetHPOPolicy = PerDatasetHPOPolicy(),
) -> Dict[str, Any]:
    """Run or resume identical model-specific HPO on one dataset."""
    existing = _read_existing_settings(config, dataset_key, policy)
    models: Dict[str, Any] = dict(existing.get("models", {}))
    if existing.get("status") == "complete" and REQUIRED_SETTINGS.issubset(models):
        directory = _hpo_dir(config, dataset_key)
        if not all((directory / name).exists() for name in (
            "selected_settings.csv", "hpo_convergence.csv",
            f"{dataset_key}_local_hpo_report.xlsx",
        )):
            _consolidate_dataset_hpo(config, dataset_key, policy, existing)
        print(f"[RESUME] {dataset_key}: local HPO settings already complete.", flush=True)
        return existing

    _, bundles = core.build_dataset_bundles(config, dataset_key)
    train, validation = bundles["train"], bundles["val"]
    val_tune, _ = core.split_validation_bundle(validation)
    directory = _hpo_dir(config, dataset_key)

    if "RIDGE" not in models:
        with engine.progress_stage(config, "HPO_RIDGE", dataset=dataset_key, model="RIDGE"):
            ridge, ridge_table = core.tune_ridge(train, val_tune)
            ridge_table.to_csv(directory / "ridge_search.csv", index=False)
            models["RIDGE"] = {
                "params": {"alpha": float(ridge["alpha"])},
                "gain": float(ridge["gain"]),
                "best_epoch": None,
                "validation_score": float(ridge_table.iloc[0]["selection_score"]),
                "selected_hpo_seed": 0,
                "selection_source": "dataset-local chronological validation grid",
            }
            _checkpoint_settings(config, dataset_key, policy, models, "in_progress")

    hpo_order = (
        "LASH_SEQ", "RF", "GBM", "XGBOOST", "LIGHTGBM", "CATBOOST",
        "MLP", "LSTM", "GRU", "CNN_LSTM", "TCN", "TRANSFORMER",
    )
    for model_name in hpo_order:
        if model_name in models:
            print(f"[RESUME] {dataset_key} HPO {model_name}", flush=True)
            continue
        with engine.progress_stage(
            config, f"HPO_{model_name}", dataset=dataset_key, model=model_name
        ):
            if model_name in TREE_MODELS:
                setting, trials, repeats, confirmations = engine._tune_tree_development(
                    config, policy, model_name, train, val_tune, dataset_key=dataset_key
                )
            else:
                setting, trials, repeats, confirmations = engine._tune_neural_development(
                    config, policy, model_name, train, val_tune, dataset_key=dataset_key
                )
            setting["selection_source"] = (
                "two-seed dataset-local HPO plus complete validation-tune confirmation"
            )
            trials.insert(0, "dataset", dataset_key)
            repeats.insert(0, "dataset", dataset_key)
            confirmations.insert(0, "dataset", dataset_key)
            trials.to_csv(directory / f"{model_name.lower()}_hpo_trials.csv", index=False)
            repeats.to_csv(directory / f"{model_name.lower()}_hpo_repeats.csv", index=False)
            confirmations.to_csv(
                directory / f"{model_name.lower()}_full_validation_confirmation.csv",
                index=False,
            )
            models[model_name] = setting
            _checkpoint_settings(config, dataset_key, policy, models, "in_progress")

    payload = _checkpoint_settings(config, dataset_key, policy, models, "complete")
    _consolidate_dataset_hpo(config, dataset_key, policy, payload)
    return payload


def aggregate_hpo_reports(
    config: core.ExperimentConfig,
    policy: PerDatasetHPOPolicy = PerDatasetHPOPolicy(),
) -> Dict[str, pd.DataFrame]:
    selected_parts: List[pd.DataFrame] = []
    repeat_parts: List[pd.DataFrame] = []
    confirmation_parts: List[pd.DataFrame] = []
    convergence_parts: List[pd.DataFrame] = []
    for dataset_key in config.dataset_keys:
        directory = _hpo_dir(config, dataset_key)
        for filename, destination in (
            ("selected_settings.csv", selected_parts),
            ("all_models_hpo_repeats.csv", repeat_parts),
            ("all_models_full_validation_confirmation.csv", confirmation_parts),
            ("hpo_convergence.csv", convergence_parts),
        ):
            path = directory / filename
            if path.exists():
                destination.append(pd.read_csv(path))
    selected = pd.concat(selected_parts, ignore_index=True, sort=False)
    repeats = pd.concat(repeat_parts, ignore_index=True, sort=False)
    confirmations = pd.concat(confirmation_parts, ignore_index=True, sort=False)
    convergence = pd.concat(convergence_parts, ignore_index=True, sort=False)
    output = _root(config) / "01_all_datasets_local_hpo_summary.xlsx"
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        selected.to_excel(writer, sheet_name="Selected_Settings", index=False)
        repeats.to_excel(writer, sheet_name="HPO_Repeats", index=False)
        confirmations.to_excel(writer, sheet_name="Full_Val_Confirmation", index=False)
        convergence.to_excel(writer, sheet_name="Convergence", index=False)
        protocol_manifest(policy, config.dataset_keys).to_excel(
            writer, sheet_name="Protocol", index=False
        )
        search_space_manifest().to_excel(writer, sheet_name="Search_Spaces", index=False)
    return {
        "selected": selected,
        "repeats": repeats,
        "confirmations": confirmations,
        "convergence": convergence,
    }


def run_per_dataset_hpo_benchmark(
    config: core.ExperimentConfig,
    policy: PerDatasetHPOPolicy = PerDatasetHPOPolicy(),
) -> Dict[str, Dict[str, pd.DataFrame]]:
    results: Dict[str, Dict[str, pd.DataFrame]] = {}
    for dataset_key in EXECUTION_ORDER:
        if dataset_key not in config.dataset_keys:
            continue
        print("\n" + "=" * 84)
        print(f"LOCAL HPO + FULL BENCHMARK: {dataset_key}")
        print("=" * 84, flush=True)
        settings = build_or_load_dataset_settings(config, dataset_key, policy)
        results[dataset_key] = engine.run_dataset_with_selected_settings(
            config,
            dataset_key,
            settings,
            policy=policy,
        )
    aggregate_hpo_reports(config, policy)
    return results


def run_per_dataset_controlled_mechanisms(
    config: core.ExperimentConfig,
    policy: PerDatasetHPOPolicy = PerDatasetHPOPolicy(),
    dataset_keys: Sequence[str] = ("CLUSTER_1", "CLUSTER_2"),
) -> Dict[str, pd.DataFrame]:
    summary_parts: List[pd.DataFrame] = []
    for dataset_key in dataset_keys:
        settings = _read_existing_settings(config, dataset_key, policy)
        if settings.get("status") != "complete":
            raise FileNotFoundError(f"Complete local HPO settings missing for {dataset_key}")
        benchmark_dir = core.dataset_output_dir(config, dataset_key)
        reference_rows = []
        for seed in policy.mechanism_seeds:
            for component, prediction_model in (
                ("SEQUENTIAL_ONLY_PRIMARY", "LASH_SEQ_COMPONENT"),
                ("REROUTED_PIPELINE_SECONDARY", "LASH"),
            ):
                payload = core.load_prediction(benchmark_dir, prediction_model, seed)
                reference_rows.append({
                    "dataset": dataset_key,
                    "variant": "FULL_MODEL_REFERENCE",
                    "component": component,
                    "seed": seed,
                    **core.regression_metrics(payload["actual"], payload["prediction"]),
                })
        summary_parts.append(pd.DataFrame(reference_rows))

        for name, flags in ablation.STRUCTURAL_VARIANTS.items():
            with engine.progress_stage(
                config, "CONTROLLED_ABLATION", dataset=dataset_key, variant=name
            ):
                result = engine.run_one_frozen_variant(
                    config, settings, dataset_key, name, policy, ablation_flags=flags
                )
            summary_parts.append(result["summary"])
        for name, options in ablation.FEATURE_VARIANTS.items():
            with engine.progress_stage(
                config, "CONTROLLED_ABLATION", dataset=dataset_key, variant=name
            ):
                result = engine.run_one_frozen_variant(
                    config,
                    settings,
                    dataset_key,
                    name,
                    policy,
                    include_nonlinear_weather=options.get("include_nonlinear_weather", True),
                    include_phase_shift_calendar=options.get(
                        "include_phase_shift_calendar", True
                    ),
                )
            summary_parts.append(result["summary"])
        for name, components in ablation.YOY_VARIANTS.items():
            with engine.progress_stage(
                config, "CONTROLLED_YOY", dataset=dataset_key, variant=name
            ):
                result = engine.run_one_frozen_variant(
                    config,
                    settings,
                    dataset_key,
                    name,
                    policy,
                    yoy_components=components,
                )
            summary_parts.append(result["summary"])

    summary = pd.concat(summary_parts, ignore_index=True, sort=False)
    reference = summary.loc[
        summary["variant"].eq("FULL_MODEL_REFERENCE"),
        ["dataset", "component", "seed", "selection_score"],
    ].rename(columns={"selection_score": "full_model_selection_score"})
    summary = summary.merge(reference, on=["dataset", "component", "seed"], how="left")
    summary["delta_selection_score_vs_full"] = (
        summary["selection_score"] - summary["full_model_selection_score"]
    )
    output = _root(config) / "02_per_dataset_controlled_mechanism_tests.xlsx"
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Controlled_Effects", index=False)
        pd.DataFrame([{
            "optimization_control": (
                "each dataset uses its own full-model HPO setting; every local variant "
                "freezes that setting"
            ),
            "datasets": str(tuple(dataset_keys)),
            "primary_estimand": "SEQUENTIAL_ONLY_PRIMARY",
            "secondary_estimand": "REROUTED_PIPELINE_SECONDARY",
            "mechanism_seeds": str(policy.mechanism_seeds),
            "test_used_for_selection": False,
        }]).to_excel(writer, sheet_name="Protocol", index=False)
    return {
        key: group.reset_index(drop=True)
        for key, group in summary.groupby("dataset")
    }


def run_priority_statistical_tests(
    config: core.ExperimentConfig,
    policy: PerDatasetHPOPolicy = PerDatasetHPOPolicy(),
) -> Dict[str, pd.DataFrame]:
    return engine.run_priority_statistical_tests(config, policy)


def write_reviewer_evidence_map(
    config: core.ExperimentConfig,
    policy: PerDatasetHPOPolicy = PerDatasetHPOPolicy(),
) -> pd.DataFrame:
    rows = [
        {
            "reviewer_comment": "R1-C4 / R3-C2 external generalizability",
            "protocol_response": (
                "CLUSTER_1, CLUSTER_2, BDG_EDU, and BDG_DORM are evaluated with "
                "independent chronological local HPO; no test target enters selection."
            ),
            "primary_outputs": "benchmark/*; hpo_by_dataset/*",
        },
        {
            "reviewer_comment": "R3-C3 operational weather availability",
            "protocol_response": "Historical-only weather is enforced for every model and dataset.",
            "primary_outputs": "benchmark/*/configs/data_and_protocol.json",
        },
        {
            "reviewer_comment": "R3-C5 omitted tree ensembles",
            "protocol_response": (
                "RF, GBM, XGBoost, LightGBM, and CatBoost each receive local HPO and "
                "three causally valid 24 horizon-specific direct refits."
            ),
            "primary_outputs": "benchmark/*/*_benchmark_results.xlsx",
        },
        {
            "reviewer_comment": "R3-C6 tuning fairness and computation",
            "protocol_response": (
                "Every dataset and every stochastic model uses the same two HPO seeds, "
                "dimension-adaptive 3-5 completed-trial rule, systematic chronology-wide "
                "screening, auditable pruning/completion handling, and complete "
                "validation-tune finalist confirmation. RF has no exception."
            ),
            "primary_outputs": (
                "01_all_datasets_local_hpo_summary.xlsx; "
                "hpo_by_dataset/*/*_local_hpo_report.xlsx"
            ),
        },
        {
            "reviewer_comment": "R3-C7 router justification",
            "protocol_response": (
                "Router grid, smoothing, shrinkage, threshold, temporal-block, and NNLS "
                "sensitivity use locally trained frozen experts."
            ),
            "primary_outputs": "router_sensitivity/*; benchmark/*/calibration",
        },
        {
            "reviewer_comment": "R3-C8 matched ablation",
            "protocol_response": (
                "Each CLUSTER dataset's variants freeze that dataset's full-model HPO "
                "setting and use the same three seeds; sequential-only effects are primary "
                "and rerouted effects secondary."
            ),
            "primary_outputs": "02_per_dataset_controlled_mechanism_tests.xlsx",
        },
        {
            "reviewer_comment": "R3-C9 YoY components",
            "protocol_response": (
                "Annual lag, holiday-matched proxy, annual scaling, annual anchor, and their "
                "combination are introduced separately under locally matched settings."
            ),
            "primary_outputs": "controlled_mechanism/*/YOY_*",
        },
        {
            "reviewer_comment": "R3-C10 dependence and seed variability",
            "protocol_response": (
                "HAC/CBB sensitivity covers 24, 72, 168, and 336 hours; hierarchical "
                "block bootstrap resamples three matched stochastic refits and time blocks."
            ),
            "primary_outputs": "benchmark/*/tables/dependence_aware_tests.csv",
        },
        {
            "reviewer_comment": "R1-C5 engineering relevance",
            "protocol_response": (
                "Peak timing/magnitude, high-load recall, and demand-response sensitivity "
                "are computed from all saved forecast-origin trajectories."
            ),
            "primary_outputs": "benchmark/*/tables/operational_peak_metrics.csv",
        },
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(_root(config) / "reviewer_evidence_map.csv", index=False)
    return frame


def write_manuscript_protocol_text(
    config: core.ExperimentConfig,
    hardware: Mapping[str, Any],
    policy: PerDatasetHPOPolicy = PerDatasetHPOPolicy(),
) -> Path:
    text = f"""METHODS - Dataset-specific hyperparameter optimization
For each of the four datasets (CLUSTER_1, CLUSTER_2, BDG_EDU, and BDG_DORM), hyperparameter optimization was conducted independently using only that dataset's chronological training and validation-tune partitions. Random Forest, GBM, XGBoost, LightGBM, CatBoost, MLP, LSTM, GRU, CNN-LSTM, TCN, Transformer, and the LASH sequential expert all followed the same prespecified selection contract. Two independent TPE searches (seeds {policy.hpo_seeds[0]} and {policy.hpo_seeds[1]}) used dimension-adaptive budgets of three to five finite completed trials per repeat. Neural median pruning was permitted for at most twice the declared budget in attempted trials; if this left the completed-trial quota unfilled, a deterministic no-pruning completion fallback trained only the missing quota. Pruned and failed attempts remained in the Optuna audit trail but were never eligible for finalist selection. The screening sets deterministically covered the complete local chronology, with at most {policy.hpo_train_origins:,} training origins and {policy.hpo_validation_origins:,} validation origins. The best candidate from each repeat was then refit on the screening training set and evaluated on the complete validation-tune segment. A prespecified parsimony rule selected the fastest finalist whose composite validation score was within {policy.parsimony_margin_score_points:.2f} percentage points of the best. Ridge used the same eight-value chronological validation grid in every dataset. No prior Optuna trial, hyperparameter, fitted model, or prediction was imported, and no test target was used for tuning, calibration, early stopping, or model selection. After selection, the local setting was frozen and the model was refit on all eligible hourly train-plus-validation origins before one-time test evaluation.

METHODS - Seed allocation and inference
Every stochastic tree ensemble and neural model, including LASH, was independently refit with seeds 42, 142, and 242 in every dataset. Ridge and the seasonal/anchor references are deterministic and were therefore fitted or evaluated once. Dependence-aware inference formed seed-matched paired loss differences and hierarchically resampled the three stochastic refits and circular time blocks. Sensitivity was reported for HAC lags and circular-block lengths of 24, 72, 168, and 336 hours. Inferential contrasts were prespecified for Seasonal-24, Ridge, all five requested tree ensembles, and Transformer; all remaining models were retained in the descriptive, horizon-wise, and operational tables.

METHODS - Controlled mechanism tests
Controlled structural, feature, and year-over-year experiments were conducted for CLUSTER_1 and CLUSTER_2. Each variant retained that dataset's locally selected full-model hyperparameters, epoch count, residual gain, data splits, and the same three seeds (42, 142, and 242). The sequential-expert result was the primary estimand because it isolates the altered mechanism before routing. The separately labeled secondary result recalibrated the router only on the held-out validation-calibration segment. A Ridge-only routing outcome was therefore treated as a deployment decision rather than direct evidence against a neural mechanism.

COMPUTATIONAL ENVIRONMENT
Experiments were executed on {hardware.get('cpu', 'the reported CPU')} with {hardware.get('ram_gb', 'reported')} GB RAM and {hardware.get('gpu', 'the reported GPU')} ({hardware.get('gpu_vram_gb', 'reported')} GB VRAM. Neural models used CUDA automatic mixed precision; XGBoost and CatBoost used validated CUDA backends, whereas RF, GBM, LightGBM, Ridge, and deterministic baselines used the CPU. The same hardware and software environment was used for all four dataset-specific HPO studies. Runtime was recorded descriptively and did not affect any test result.
"""
    path = _root(config) / "manuscript_protocol_per_dataset_hpo.txt"
    path.write_text(text, encoding="utf-8")
    return path
