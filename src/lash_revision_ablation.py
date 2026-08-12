"""Matched ablation, router sensitivity, and component-wise YoY experiments for LASH.

This module is deliberately separate from the primary benchmark so that the main
benchmark remains frozen and readable.  Every structural / feature-changing
ablation retunes the LASH sequential expert on the same chronological
validation-tuning partition.  Router calibration always uses the held-out,
purged validation-calibration partition.  Test data are evaluated only after
all choices are frozen.
"""
from __future__ import annotations

import json
import gc
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from lash_revision_core import (
    ExperimentConfig, HORIZON,
    build_dataset_bundles, split_validation_bundle, concat_bundles,
    tune_ridge, fit_ridge, predict_ridge_raw,
    tune_neural_repeated, fit_neural, predict_neural_raw,
    apply_residual_gain, select_router, blend_predictions, nnls_router,
    regression_metrics, per_horizon_metrics, origin_loss_frame,
    dataset_output_dir, save_json, load_prediction,
    router_sensitivity_table, router_temporal_stability,
)

STRUCTURAL_VARIANTS: Dict[str, Dict[str, Any]] = {
    "NO_FEATURE_GATES": {"use_feature_gates": False},
    "NO_CAUSAL_TCN": {"use_tcn": False},
    "MEAN_POOLING": {"learned_pooling": False},
    "NO_GRB": {"use_grb": False},
    "NO_HORIZON_EMBEDDING": {"use_horizon_embedding": False},
}

FEATURE_VARIANTS: Dict[str, Dict[str, Any]] = {
    "NO_NONLINEAR_WEATHER": {"include_nonlinear_weather": False},
    "NO_PHASE_SHIFT_CALENDAR": {"include_phase_shift_calendar": False},
}

YOY_VARIANTS: Dict[str, Tuple[str, ...]] = {
    "YOY_ANNUAL_LAG_ONLY": ("annual_lag",),
    "YOY_HOLIDAY_MATCHED_ONLY": ("holiday_matched_proxy",),
    "YOY_ANNUAL_SCALING_ONLY": ("annual_scaling",),
    "YOY_ANNUAL_ANCHOR_ONLY": ("annual_anchor",),
    "YOY_ALL_COMPONENTS": (
        "annual_lag", "holiday_matched_proxy", "annual_scaling", "annual_anchor"
    ),
}


def _variant_dir(config: ExperimentConfig, dataset_key: str, variant: str) -> Path:
    d = config.output_root / "ablation" / dataset_key / variant
    for sub in ("tables", "predictions", "configs"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


def _save_variant_prediction(path: Path, bundle, pred: np.ndarray, seed: int,
                             component: str) -> None:
    np.savez_compressed(
        path,
        actual=bundle.y.astype(np.float32),
        prediction=np.asarray(pred, dtype=np.float32),
        forecast_origin=bundle.forecast_origin,
        target_time=bundle.target_time,
        seed=np.asarray([seed]),
        component=np.asarray([component]),
    )


def run_one_matched_variant(
    config: ExperimentConfig,
    dataset_key: str,
    variant_name: str,
    *,
    ablation_flags: Optional[Mapping[str, Any]] = None,
    include_nonlinear_weather: bool = True,
    include_phase_shift_calendar: bool = True,
    yoy_components: Sequence[str] = (),
) -> Dict[str, pd.DataFrame]:
    """Retune and evaluate one LASH variant under matched optimization.

    Both the sequential expert and Ridge are selected using only validation-tune.
    The held-out validation-calibration segment selects the router.  Final experts
    are then refit on train + complete validation with all settings frozen.
    """
    variant_dir = _variant_dir(config, dataset_key, variant_name)
    _, bundles = build_dataset_bundles(
        config, dataset_key,
        yoy_components=yoy_components,
        include_nonlinear_weather=include_nonlinear_weather,
        include_phase_shift_calendar=include_phase_shift_calendar,
    )
    train, val, test = bundles["train"], bundles["val"], bundles["test"]
    val_tune, val_cal = split_validation_bundle(val)
    pre_cal = concat_bundles(train, val_tune, split="train_plus_val_tune")
    pretest = concat_bundles(train, val, split="train_plus_validation")

    # Ridge is cheap, so it is re-selected even for structural variants.  This
    # removes any ambiguity about whether one variant received a stale linear
    # expert configuration.
    ridge_sel, ridge_search = tune_ridge(train, val_tune)

    suffix = "__" + variant_name.lower()
    seq_sel, seq_trials, seq_repeats = tune_neural_repeated(
        config, dataset_key, "LASH_SEQ", train, val_tune,
        ablation=dict(ablation_flags or {}), study_suffix=suffix,
    )

    # Held-out calibration predictions; no calibration target is used for expert fitting.
    cal_seq_fit = fit_neural(
        "LASH_SEQ", pre_cal, seq_sel["params"], config,
        validation_bundle=None, fixed_epochs=seq_sel["best_epoch"], seed=42,
        ablation=dict(ablation_flags or {}),
    )
    cal_seq_raw = predict_neural_raw(cal_seq_fit, val_cal, seq_sel["params"], config)
    cal_seq = apply_residual_gain(val_cal, cal_seq_raw, seq_sel["gain"])

    cal_ridge_fit = fit_ridge(pre_cal, ridge_sel["alpha"])
    cal_ridge_raw = predict_ridge_raw(cal_ridge_fit, val_cal)
    cal_ridge = apply_residual_gain(val_cal, cal_ridge_raw, ridge_sel["gain"])
    router = select_router(val_cal, cal_seq, cal_ridge)

    np.savez_compressed(
        variant_dir / "predictions" / "calibration_experts.npz",
        actual=val_cal.y.astype(np.float32),
        sequential=cal_seq.astype(np.float32),
        ridge=cal_ridge.astype(np.float32),
        forecast_origin=val_cal.forecast_origin,
        target_time=val_cal.target_time,
    )

    del cal_seq_fit, cal_ridge_fit
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Ridge is deterministic under fixed inputs and alpha, so one final fit is enough.
    final_ridge = fit_ridge(pretest, ridge_sel["alpha"])
    ridge_test_raw = predict_ridge_raw(final_ridge, test)
    ridge_test = apply_residual_gain(test, ridge_test_raw, ridge_sel["gain"])

    metric_rows = []
    horizon_parts = []
    origin_parts = []
    for seed in config.final_refit_seeds:
        seq_fit = fit_neural(
            "LASH_SEQ", pretest, seq_sel["params"], config,
            validation_bundle=None, fixed_epochs=seq_sel["best_epoch"], seed=seed,
            ablation=dict(ablation_flags or {}),
        )
        seq_raw = predict_neural_raw(seq_fit, test, seq_sel["params"], config)
        seq_test = apply_residual_gain(test, seq_raw, seq_sel["gain"])
        routed_test = blend_predictions(seq_test, ridge_test, router.sequential_weights)

        for component, prediction in (
            ("SEQUENTIAL_ONLY", seq_test),
            ("ROUTED_PIPELINE", routed_test),
        ):
            metric_rows.append({
                "dataset": dataset_key,
                "variant": variant_name,
                "component": component,
                "seed": seed,
                **regression_metrics(test.y, prediction),
            })
            h = per_horizon_metrics(test.y, prediction)
            h.insert(0, "seed", seed)
            h.insert(0, "component", component)
            h.insert(0, "variant", variant_name)
            horizon_parts.append(h)
            o = origin_loss_frame(test, prediction, f"{variant_name}::{component}", seed)
            o.insert(0, "variant", variant_name)
            o.insert(1, "component", component)
            origin_parts.append(o)
            _save_variant_prediction(
                variant_dir / "predictions" / f"{component.lower()}__seed{seed}.npz",
                test, prediction, seed, component,
            )

        del seq_fit
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics = pd.DataFrame(metric_rows)
    horizons = pd.concat(horizon_parts, ignore_index=True)
    origins = pd.concat(origin_parts, ignore_index=True)

    summary = metrics.groupby(["variant", "component"], as_index=False)[
        ["MAPE", "CVRMSE", "NMAE", "selection_score"]
    ].agg(["mean", "std"])
    summary.columns = ["variant", "component"] + [
        f"{a}_{b}" for a, b in summary.columns.tolist()[2:]
    ]

    selected = pd.DataFrame([
        {
            "dataset": dataset_key,
            "variant": variant_name,
            "ridge_alpha": ridge_sel["alpha"],
            "ridge_gain": ridge_sel["gain"],
            "sequential_params_json": json.dumps(seq_sel["params"], sort_keys=True),
            "sequential_best_epoch": seq_sel["best_epoch"],
            "sequential_gain": seq_sel["gain"],
            "selected_hpo_seed": seq_sel["selected_hpo_seed"],
            "router_mode": router.mode,
            "router_weights_json": json.dumps(router.sequential_weights.tolist()),
            "ablation_flags_json": json.dumps(dict(ablation_flags or {}), sort_keys=True),
            "yoy_components_json": json.dumps(list(yoy_components)),
            "include_nonlinear_weather": include_nonlinear_weather,
            "include_phase_shift_calendar": include_phase_shift_calendar,
        }
    ])

    ridge_search.to_csv(variant_dir / "tables" / "ridge_search.csv", index=False)
    seq_trials.to_csv(variant_dir / "tables" / "sequential_hpo_trials.csv", index=False)
    seq_repeats.to_csv(variant_dir / "tables" / "sequential_hpo_repeats.csv", index=False)
    router.table.to_csv(variant_dir / "tables" / "router_calibration.csv", index=False)
    metrics.to_csv(variant_dir / "tables" / "metrics_by_seed.csv", index=False)
    horizons.to_csv(variant_dir / "tables" / "horizon_metrics.csv", index=False)
    origins.to_csv(variant_dir / "tables" / "origin_losses.csv.gz", index=False, compression="gzip")
    selected.to_csv(variant_dir / "tables" / "selected_settings.csv", index=False)

    save_json(variant_dir / "configs" / "variant_protocol.json", {
        "dataset": asdict(config.specs[dataset_key]),
        "variant": variant_name,
        "ablation_flags": dict(ablation_flags or {}),
        "yoy_components": list(yoy_components),
        "include_nonlinear_weather": include_nonlinear_weather,
        "include_phase_shift_calendar": include_phase_shift_calendar,
        "router": {
            "mode": router.mode,
            "weights": router.sequential_weights.tolist(),
        },
        "test_used_for_selection": False,
    })

    return {
        "summary": summary,
        "metrics": metrics,
        "horizons": horizons,
        "origins": origins,
        "selected": selected,
        "hpo_repeats": seq_repeats,
        "router": router.table,
    }


def run_matched_ablation_suite(
    config: ExperimentConfig,
    dataset_keys: Sequence[str] = ("CLUSTER_1", "CLUSTER_2"),
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Run structural and feature ablations on the two long primary datasets."""
    out: Dict[str, Dict[str, pd.DataFrame]] = {}
    for dataset_key in dataset_keys:
        parts = []
        for name, flags in STRUCTURAL_VARIANTS.items():
            parts.append(run_one_matched_variant(
                config, dataset_key, name, ablation_flags=flags
            ))
        for name, opts in FEATURE_VARIANTS.items():
            parts.append(run_one_matched_variant(
                config, dataset_key, name,
                include_nonlinear_weather=opts.get("include_nonlinear_weather", True),
                include_phase_shift_calendar=opts.get("include_phase_shift_calendar", True),
            ))

        merged = {
            key: pd.concat([p[key] for p in parts], ignore_index=True)
            for key in ("summary", "metrics", "horizons", "origins", "selected", "hpo_repeats", "router")
        }

        # Import the already-frozen full-model control from the primary benchmark.
        # This avoids rerunning an identical HPO search while placing the full
        # sequential expert and the routed LASH pipeline beside the retuned ablations.
        bench_book = config.output_root / "benchmark" / dataset_key / f"{dataset_key}_benchmark_results.xlsx"
        if bench_book.exists():
            main_metrics = pd.read_excel(bench_book, sheet_name="Metrics_By_Seed")
            comp_metrics = pd.read_excel(bench_book, sheet_name="LASH_Components")
            full_parts = []
            seq = comp_metrics.copy()
            seq["dataset"] = dataset_key; seq["variant"] = "FULL_LASH"; seq["component"] = "SEQUENTIAL_ONLY"
            full_parts.append(seq[["dataset","variant","component","seed","MAPE","CVRMSE","NMAE","selection_score"]])
            routed = main_metrics.loc[main_metrics["model"].eq("LASH")].copy()
            routed["dataset"] = dataset_key; routed["variant"] = "FULL_LASH"; routed["component"] = "ROUTED_PIPELINE"
            full_parts.append(routed[["dataset","variant","component","seed","MAPE","CVRMSE","NMAE","selection_score"]])
            full_metrics = pd.concat(full_parts, ignore_index=True)
            full_summary = full_metrics.groupby(["variant","component"], as_index=False)[
                ["MAPE","CVRMSE","NMAE","selection_score"]
            ].agg(["mean","std"])
            full_summary.columns = ["variant","component"] + [f"{a}_{b}" for a,b in full_summary.columns.tolist()[2:]]
            merged["metrics"] = pd.concat([full_metrics, merged["metrics"]], ignore_index=True)
            merged["summary"] = pd.concat([full_summary, merged["summary"]], ignore_index=True)

        d = config.output_root / "ablation" / dataset_key
        workbook = d / f"{dataset_key}_matched_ablation_results.xlsx"
        with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
            merged["summary"].to_excel(writer, sheet_name="Summary", index=False)
            merged["metrics"].to_excel(writer, sheet_name="Metrics_By_Seed", index=False)
            merged["horizons"].to_excel(writer, sheet_name="Horizon_Metrics", index=False)
            merged["selected"].to_excel(writer, sheet_name="Selected_Settings", index=False)
            merged["hpo_repeats"].to_excel(writer, sheet_name="HPO_Repeats", index=False)
            merged["router"].to_excel(writer, sheet_name="Router_Calibration", index=False)
        out[dataset_key] = merged
    return out


def run_yoy_component_suite(
    config: ExperimentConfig,
    dataset_keys: Sequence[str] = ("CLUSTER_1", "CLUSTER_2"),
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Introduce annual-information components separately under matched validation.

    BDG is intentionally excluded because one year of history does not support a
    meaningful prior-year representation across its evaluation period.
    """
    out: Dict[str, Dict[str, pd.DataFrame]] = {}
    for dataset_key in dataset_keys:
        parts = []
        for name, components in YOY_VARIANTS.items():
            parts.append(run_one_matched_variant(
                config, dataset_key, name, yoy_components=components
            ))
        merged = {
            key: pd.concat([p[key] for p in parts], ignore_index=True)
            for key in ("summary", "metrics", "horizons", "origins", "selected", "hpo_repeats", "router")
        }
        d = config.output_root / "yoy" / dataset_key
        d.mkdir(parents=True, exist_ok=True)
        workbook = d / f"{dataset_key}_yoy_component_results.xlsx"
        with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
            merged["summary"].to_excel(writer, sheet_name="Summary", index=False)
            merged["metrics"].to_excel(writer, sheet_name="Metrics_By_Seed", index=False)
            merged["horizons"].to_excel(writer, sheet_name="Horizon_Metrics", index=False)
            merged["selected"].to_excel(writer, sheet_name="Selected_Settings", index=False)
            merged["hpo_repeats"].to_excel(writer, sheet_name="HPO_Repeats", index=False)
        out[dataset_key] = merged
    return out


def run_router_sensitivity_suite(
    config: ExperimentConfig,
    dataset_keys: Optional[Sequence[str]] = None,
    weight_steps: Sequence[float] = (0.01, 0.05, 0.10),
    smooth_windows: Sequence[int] = (1, 3, 5),
    shrinkages: Sequence[float] = (0.0, 0.5, 1.0),
    thresholds: Sequence[float] = (0.0, 0.0025, 0.005, 0.01),
) -> Dict[str, pd.DataFrame]:
    """Evaluate router hyperparameters and NNLS using frozen expert predictions.

    Router choices are made exclusively on the calibration partition.  Alternative
    routers are then applied to the already-frozen test expert predictions.
    """
    keys = tuple(dataset_keys or config.dataset_keys)
    outputs: Dict[str, pd.DataFrame] = {}
    for dataset_key in keys:
        d = dataset_output_dir(config, dataset_key)
        cal_npz = np.load(d / "calibration" / "expert_predictions.npz", allow_pickle=True)
        # Reconstruct a minimal bundle-like object by rebuilding the dataset bundle.
        _, bundles = build_dataset_bundles(config, dataset_key)
        _, cal_bundle = split_validation_bundle(bundles["val"])
        cal_seq = np.asarray(cal_npz["seq"], float)
        cal_ridge = np.asarray(cal_npz["ridge"], float)

        sensitivity = router_sensitivity_table(
            cal_bundle, cal_seq, cal_ridge,
            steps=weight_steps,
            smooth_windows=smooth_windows,
            shrinkages=shrinkages,
            thresholds=thresholds,
        )
        stability = router_temporal_stability(cal_bundle, cal_seq, cal_ridge, n_blocks=3)

        candidate_rows = []
        for label, weights in (
            ("NNLS_GLOBAL", nnls_router(cal_bundle, cal_seq, cal_ridge, horizon_specific=False)),
            ("NNLS_HORIZON", nnls_router(cal_bundle, cal_seq, cal_ridge, horizon_specific=True)),
        ):
            p = blend_predictions(cal_seq, cal_ridge, weights)
            candidate_rows.append({
                "router": label,
                "calibration_score": regression_metrics(cal_bundle.y, p)["selection_score"],
                "weights_json": json.dumps(weights.tolist()),
            })
        nnls_table = pd.DataFrame(candidate_rows)

        # Apply every sensitivity-selected router to frozen test expert predictions.
        test_rows = []
        for seed in config.final_refit_seeds:
            seq_test = load_prediction(d, "LASH_SEQ_COMPONENT", seed)["prediction"]
            ridge_test = load_prediction(d, "RIDGE", 0)["prediction"]
            test_bundle = bundles["test"]

            for _, row in sensitivity.iterrows():
                weights = np.asarray(json.loads(row["weights_json"]), float)
                pred = blend_predictions(seq_test, ridge_test, weights)
                test_rows.append({
                    "seed": seed,
                    "router_family": "GRID_SENSITIVITY",
                    "weight_step": row["weight_step"],
                    "smooth_window": row["smooth_window"],
                    "shrinkage": row["shrinkage_to_global"],
                    "threshold": row["min_improvement"],
                    "selected_mode": row["mode"],
                    **regression_metrics(test_bundle.y, pred),
                })

            for label, weights in (
                ("NNLS_GLOBAL", nnls_router(cal_bundle, cal_seq, cal_ridge, False)),
                ("NNLS_HORIZON", nnls_router(cal_bundle, cal_seq, cal_ridge, True)),
            ):
                pred = blend_predictions(seq_test, ridge_test, weights)
                test_rows.append({
                    "seed": seed,
                    "router_family": label,
                    "weight_step": np.nan,
                    "smooth_window": np.nan,
                    "shrinkage": np.nan,
                    "threshold": np.nan,
                    "selected_mode": label,
                    **regression_metrics(test_bundle.y, pred),
                })

        test_eval = pd.DataFrame(test_rows)
        out_dir = config.output_root / "router_sensitivity" / dataset_key
        out_dir.mkdir(parents=True, exist_ok=True)
        workbook = out_dir / f"{dataset_key}_router_sensitivity.xlsx"
        with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
            sensitivity.to_excel(writer, sheet_name="Calibration_Grid", index=False)
            stability.to_excel(writer, sheet_name="Temporal_Stability", index=False)
            nnls_table.to_excel(writer, sheet_name="NNLS", index=False)
            test_eval.to_excel(writer, sheet_name="Frozen_Test_Eval", index=False)
        outputs[dataset_key] = test_eval
    return outputs
