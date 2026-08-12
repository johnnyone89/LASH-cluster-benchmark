"""Post-training analysis utilities for the LASH revision package."""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from scipy import stats
from statsmodels.api import OLS, add_constant
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import acf

from lash_revision_core import (
    HORIZON, LOOKBACK, ExperimentConfig, WindowBundle,
    apply_residual_gain, blend_predictions, build_dataset_bundles,
    concat_bundles, dataset_output_dir, fit_neural, fit_ridge,
    high_load_recall, load_neural_artifact, load_prediction, load_ridge_artifact,
    load_selected_params, load_tree_artifact, make_tabular_origin_design,
    nnls_router, origin_loss_frame, per_horizon_metrics, predict_neural_raw,
    predict_ridge_raw, regression_metrics, router_sensitivity_table,
    router_temporal_stability, select_residual_gain, select_router,
    split_validation_bundle, take_bundle, tune_neural_repeated, tune_ridge,
    daily_peak_metrics, save_prediction_matrix,
)


# -----------------------------------------------------------------------------
# Excel/result discovery
# -----------------------------------------------------------------------------
def scan_result_workbooks(output_root: Path) -> Dict[str, Path]:
    output_root = Path(output_root)
    return {p.parent.name: p for p in sorted(output_root.glob("benchmark/*/*_benchmark_results.xlsx"))}


def load_excel_results(output_root: Path) -> Dict[str, Dict[str, pd.DataFrame]]:
    results: Dict[str, Dict[str, pd.DataFrame]] = {}
    for key, path in scan_result_workbooks(output_root).items():
        xls = pd.ExcelFile(path)
        results[key] = {s: pd.read_excel(path, sheet_name=s) for s in xls.sheet_names}
    return results


# -----------------------------------------------------------------------------
# Paired dependence-aware inference
# -----------------------------------------------------------------------------
def newey_west_mean_test(diff: np.ndarray, maxlags: int) -> Dict[str, float]:
    d = np.asarray(diff, float)
    d = d[np.isfinite(d)]
    if len(d) < maxlags + 5:
        return {"mean_diff": np.nan, "se_hac": np.nan, "t_hac": np.nan, "p_hac": np.nan, "n": len(d)}
    X = np.ones((len(d), 1))
    model = OLS(d, X).fit(cov_type="HAC", cov_kwds={"maxlags": int(maxlags)})
    return {
        "mean_diff": float(model.params[0]),
        "se_hac": float(model.bse[0]),
        "t_hac": float(model.tvalues[0]),
        "p_hac": float(model.pvalues[0]),
        "n": int(len(d)),
    }


def circular_block_bootstrap_mean(diff: np.ndarray, block_length: int = 168,
                                  reps: int = 3000, seed: int = 2026) -> Dict[str, float]:
    d = np.asarray(diff, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n == 0:
        return {"boot_mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_boot": np.nan}
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / block_length))
    stats_rep = np.empty(reps, float)
    base = np.arange(block_length)
    for r in range(reps):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([(s + base) % n for s in starts])[:n]
        stats_rep[r] = d[idx].mean()
    low, high = np.quantile(stats_rep, [0.025, 0.975])
    p = 2 * min((stats_rep <= 0).mean(), (stats_rep >= 0).mean())
    return {
        "boot_mean": float(stats_rep.mean()),
        "ci_low": float(low), "ci_high": float(high), "p_boot": float(min(p, 1.0)),
    }


def hierarchical_seed_block_bootstrap(seed_diffs: Mapping[int, np.ndarray], block_length: int = 168,
                                      reps: int = 3000, seed: int = 2026) -> Dict[str, float]:
    """Resample training seeds first, then circular time blocks within each sampled seed."""
    keys = sorted(seed_diffs)
    if not keys:
        return {"hier_mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_hier": np.nan}
    arrays = {k: np.asarray(seed_diffs[k], float) for k in keys}
    rng = np.random.default_rng(seed)
    rep_stats = np.empty(reps, float)
    for r in range(reps):
        sampled_keys = rng.choice(keys, size=len(keys), replace=True)
        seed_means = []
        for k in sampled_keys:
            d = arrays[int(k)]
            n = len(d); nb = int(math.ceil(n / block_length))
            starts = rng.integers(0, n, size=nb)
            idx = np.concatenate([((s + np.arange(block_length)) % n) for s in starts])[:n]
            seed_means.append(d[idx].mean())
        rep_stats[r] = np.mean(seed_means)
    low, high = np.quantile(rep_stats, [0.025, 0.975])
    p = 2 * min((rep_stats <= 0).mean(), (rep_stats >= 0).mean())
    return {
        "hier_mean": float(rep_stats.mean()), "ci_low": float(low), "ci_high": float(high),
        "p_hier": float(min(p, 1.0)),
    }


def paired_loss_acf(diff: np.ndarray, nlags: int = 336) -> pd.DataFrame:
    d = np.asarray(diff, float)
    vals = acf(d, nlags=min(nlags, len(d) - 1), fft=True, missing="drop")
    threshold = 1.96 / math.sqrt(max(len(d), 1))
    return pd.DataFrame({"lag": np.arange(len(vals)), "acf": vals, "approx_95_threshold": threshold})


def _seed_pair_arrays(origin_df: pd.DataFrame, comparator: str, loss_col: str) -> Dict[int, np.ndarray]:
    lash = origin_df[origin_df.model.eq("LASH")]
    comp = origin_df[origin_df.model.eq(comparator)]
    lash_seeds = sorted(lash.seed.unique())
    comp_seeds = sorted(comp.seed.unique())
    out = {}
    for seed in lash_seeds:
        l = lash[lash.seed.eq(seed)][["forecast_origin", loss_col]].rename(columns={loss_col: "lash"})
        if seed in comp_seeds:
            c = comp[comp.seed.eq(seed)][["forecast_origin", loss_col]].rename(columns={loss_col: "comp"})
        elif len(comp_seeds) == 1:
            c = comp[comp.seed.eq(comp_seeds[0])][["forecast_origin", loss_col]].rename(columns={loss_col: "comp"})
        else:
            # If stochastic comparator seeds differ, match by rank as a deterministic fallback.
            cseed = comp_seeds[min(lash_seeds.index(seed), len(comp_seeds)-1)]
            c = comp[comp.seed.eq(cseed)][["forecast_origin", loss_col]].rename(columns={loss_col: "comp"})
        merged = l.merge(c, on="forecast_origin", how="inner").sort_values("forecast_origin")
        out[int(seed)] = (merged.lash - merged.comp).to_numpy(float)  # negative favors LASH
    return out


def dependence_aware_comparison(origin_df: pd.DataFrame, comparator: str,
                                hac_lags=(24, 72, 168, 336), block_lengths=(24, 72, 168, 336),
                                bootstrap_reps: int = 3000, loss_col: str = "origin_NMAE_pct") -> Tuple[pd.DataFrame, pd.DataFrame]:
    seed_diffs = _seed_pair_arrays(origin_df, comparator, loss_col)
    rows = []
    for seed, d in seed_diffs.items():
        for lag in hac_lags:
            rows.append({"comparator": comparator, "seed": seed, "method": "HAC", "setting": lag,
                         **newey_west_mean_test(d, lag)})
        for bl in block_lengths:
            rows.append({"comparator": comparator, "seed": seed, "method": "CBB", "setting": bl,
                         **circular_block_bootstrap_mean(d, bl, bootstrap_reps, seed=2026+seed+bl)})
    for bl in block_lengths:
        rows.append({"comparator": comparator, "seed": "hierarchical", "method": "HIER_CBB", "setting": bl,
                     **hierarchical_seed_block_bootstrap(seed_diffs, bl, bootstrap_reps, seed=9100+bl)})
    acf_df = paired_loss_acf(np.mean(np.vstack([seed_diffs[k] for k in sorted(seed_diffs)]), axis=0))
    return pd.DataFrame(rows), acf_df


def run_all_statistical_tests(config: ExperimentConfig, bootstrap_reps: int = 3000) -> Dict[str, pd.DataFrame]:
    outputs = {}
    for key in config.dataset_keys:
        ddir = dataset_output_dir(config, key)
        origin_path = ddir / "tables" / "origin_losses.csv.gz"
        if not origin_path.exists():
            continue
        origin_df = pd.read_csv(origin_path, parse_dates=["forecast_origin"])
        comparators = [m for m in sorted(origin_df.model.unique()) if m != "LASH"]
        all_rows = []; acfs = []
        for comp in comparators:
            table, acf_df = dependence_aware_comparison(origin_df, comp, bootstrap_reps=bootstrap_reps)
            all_rows.append(table)
            acf_df.insert(0, "comparator", comp); acfs.append(acf_df)
        tests = pd.concat(all_rows, ignore_index=True)
        # Holm adjustment within each inferential family / setting / seed.
        tests["p_raw"] = tests["p_hac"].where(tests.method.eq("HAC"),
                            tests["p_boot"].where(tests.method.eq("CBB"), tests["p_hier"]))
        tests["p_holm"] = np.nan
        for _, idx in tests.groupby(["method", "setting", "seed"], dropna=False).groups.items():
            idx = list(idx)
            p = tests.loc[idx, "p_raw"].fillna(1.0).to_numpy(float)
            tests.loc[idx, "p_holm"] = multipletests(p, method="holm")[1]
        acf_all = pd.concat(acfs, ignore_index=True)
        tests.to_csv(ddir / "tables" / "dependence_aware_tests.csv", index=False)
        acf_all.to_csv(ddir / "tables" / "paired_loss_acf.csv", index=False)
        outputs[key] = tests
    return outputs


# -----------------------------------------------------------------------------
# Horizon-specific paired tests from saved prediction matrices
# -----------------------------------------------------------------------------
def horizon_paired_tests(config: ExperimentConfig, dataset_key: str, comparator: str,
                         hac_lag: int = 168) -> pd.DataFrame:
    ddir = dataset_output_dir(config, dataset_key)
    lash_seeds = config.final_refit_seeds
    rows = []
    for seed in lash_seeds:
        lash = load_prediction(ddir, "LASH", seed)
        try:
            comp = load_prediction(ddir, comparator, seed)
        except FileNotFoundError:
            comp = load_prediction(ddir, comparator, 0)
        for h in range(HORIZON):
            d = np.abs(lash["actual"][:, h] - lash["prediction"][:, h]) - np.abs(comp["actual"][:, h] - comp["prediction"][:, h])
            res = newey_west_mean_test(d, hac_lag)
            rows.append({"seed": seed, "comparator": comparator, "horizon": h+1, **res})
    df = pd.DataFrame(rows)
    df["p_holm_within_seed"] = np.nan
    for seed, idx in df.groupby("seed").groups.items():
        p = df.loc[idx, "p_hac"].fillna(1.0).to_numpy()
        df.loc[idx, "p_holm_within_seed"] = multipletests(p, method="holm")[1]
    return df


# -----------------------------------------------------------------------------
# Temporal robustness and BEMS-oriented peak sensitivity
# -----------------------------------------------------------------------------
def temporal_robustness_frame(pred_payload: Mapping[str, np.ndarray]) -> pd.DataFrame:
    actual = pred_payload["actual"].astype(float)
    pred = pred_payload["prediction"].astype(float)
    origins = pd.to_datetime(pred_payload["forecast_origin"])
    mae = np.abs(actual - pred).mean(axis=1)
    nmae = mae / np.mean(actual) * 100
    df = pd.DataFrame({"forecast_origin": origins, "origin_NMAE_pct": nmae})
    df["month"] = df.forecast_origin.dt.to_period("M").astype(str)
    df["quarter"] = df.forecast_origin.dt.to_period("Q").astype(str)
    df["rolling_30d_NMAE"] = df.origin_NMAE_pct.rolling(24*30, min_periods=24*7).mean()
    return df


def operational_metrics_from_prediction(payload: Mapping[str, np.ndarray]) -> Dict[str, float]:
    out = daily_peak_metrics(payload["target_time"], payload["actual"], payload["prediction"])
    out["high_load_recall_q90"] = high_load_recall(payload["actual"], payload["prediction"], 0.90)
    out["high_load_recall_q95"] = high_load_recall(payload["actual"], payload["prediction"], 0.95)
    return out


def demand_response_sensitivity(payload: Mapping[str, np.ndarray], top_k_values=(1,2,4), curtailments=(0.05,0.10)) -> pd.DataFrame:
    # Use each forecast origin as one decision opportunity over the next 24 hours.
    actual = payload["actual"].astype(float); pred = payload["prediction"].astype(float)
    rows = []
    for k in top_k_values:
        selected = np.argpartition(pred, -k, axis=1)[:, -k:]
        for c in curtailments:
            after = actual.copy()
            for i in range(len(after)):
                after[i, selected[i]] *= (1.0 - c)
            peak_before = actual.max(axis=1); peak_after = after.max(axis=1)
            oracle = actual.copy()
            oracle_sel = np.argpartition(actual, -k, axis=1)[:, -k:]
            for i in range(len(oracle)):
                oracle[i, oracle_sel[i]] *= (1.0 - c)
            oracle_red = peak_before - oracle.max(axis=1)
            realized_red = peak_before - peak_after
            capture = np.divide(realized_red, oracle_red, out=np.zeros_like(realized_red), where=oracle_red>1e-12)
            rows.append({
                "top_k_hours": k, "curtailment_fraction": c,
                "mean_peak_reduction_pct": float(np.mean(realized_red / np.maximum(peak_before,1e-12))*100),
                "oracle_relative_peak_capture": float(np.mean(np.clip(capture,0,1))),
            })
    return pd.DataFrame(rows)


def run_operational_analysis(config: ExperimentConfig, models: Optional[Sequence[str]] = None) -> Dict[str, pd.DataFrame]:
    outputs = {}
    for key in config.dataset_keys:
        ddir = dataset_output_dir(config, key)
        metrics_path = ddir / "tables" / "metrics_by_seed.csv"
        if not metrics_path.exists(): continue
        metric_df = pd.read_csv(metrics_path)
        model_list = list(models) if models else sorted(metric_df.model.unique())
        rows = []; dr_rows = []; temporal_parts = []
        for model in model_list:
            seeds = sorted(metric_df.loc[metric_df.model.eq(model), "seed"].unique())
            for seed in seeds:
                try:
                    payload = load_prediction(ddir, model, int(seed))
                except FileNotFoundError:
                    continue
                rows.append({"model": model, "seed": int(seed), **operational_metrics_from_prediction(payload)})
                dr = demand_response_sensitivity(payload); dr.insert(0,"seed",int(seed)); dr.insert(0,"model",model); dr_rows.append(dr)
                tr = temporal_robustness_frame(payload); tr.insert(0,"seed",int(seed)); tr.insert(0,"model",model); temporal_parts.append(tr)
        op = pd.DataFrame(rows); dr = pd.concat(dr_rows,ignore_index=True); temporal = pd.concat(temporal_parts,ignore_index=True)
        op.to_csv(ddir / "tables" / "operational_peak_metrics.csv", index=False)
        dr.to_csv(ddir / "tables" / "demand_response_sensitivity.csv", index=False)
        temporal.to_csv(ddir / "tables" / "temporal_robustness.csv.gz", index=False, compression="gzip")
        outputs[key] = op
    return outputs


# -----------------------------------------------------------------------------
# LASH XAI: gates, temporal pooling, Integrated Gradients, grouped permutation
# -----------------------------------------------------------------------------
def lash_internal_diagnostics(config: ExperimentConfig, dataset_key: str, seed: int = 42,
                              sample_origins: int = 256) -> Dict[str, pd.DataFrame]:
    _, bundles = build_dataset_bundles(config, dataset_key)
    test = bundles["test"]
    idx = np.linspace(0, len(test)-1, min(sample_origins, len(test)), dtype=int)
    sample = take_bundle(test, idx, split="xai_sample")
    fit, params, gain = load_neural_artifact(config, dataset_key, "LASH_SEQ", seed, test)
    _, aux = predict_neural_raw(fit, sample, params, config, return_explanation=True)

    past = pd.DataFrame({
        "feature": sample.past_cols,
        "mean_gate": aux["past_feature_weights"].mean(axis=0),
        "sd_gate": aux["past_feature_weights"].std(axis=0),
    }).sort_values("mean_gate", ascending=False)
    temporal = pd.DataFrame({
        "lag_hour": np.arange(-LOOKBACK, 0),
        "mean_weight": aux["temporal_weights"].mean(axis=0),
        "sd_weight": aux["temporal_weights"].std(axis=0),
    })
    fw = aux["future_feature_weights"]
    future_rows = []
    for h in range(HORIZON):
        for j, feature in enumerate(sample.future_cols):
            future_rows.append({"horizon":h+1,"feature":feature,
                                "mean_gate":float(fw[:,h,j].mean()),"sd_gate":float(fw[:,h,j].std())})
    future = pd.DataFrame(future_rows)
    return {"past_gates": past, "temporal_pooling": temporal, "future_gates": future}


def integrated_gradients_lash(config: ExperimentConfig, dataset_key: str, seed: int = 42,
                              horizons=(1,12,24), sample_origins: int = 128, n_steps: int = 32) -> pd.DataFrame:
    try:
        from captum.attr import IntegratedGradients
    except ImportError as e:
        raise ImportError("Install captum to run Integrated Gradients: pip install captum") from e
    _, bundles = build_dataset_bundles(config, dataset_key); test=bundles["test"]
    idx=np.linspace(0,len(test)-1,min(sample_origins,len(test)),dtype=int); sample=take_bundle(test,idx,"ig")
    fit,params,gain=load_neural_artifact(config,dataset_key,"LASH_SEQ",seed,test)
    p,f = fit.scaler.past_scaler.transform(sample.past.reshape(-1,sample.past.shape[-1])).reshape(len(sample),LOOKBACK,-1), \
          fit.scaler.future_scaler.transform(sample.future.reshape(-1,sample.future.shape[-1])).reshape(len(sample),HORIZON,-1)
    p=torch.tensor(p,dtype=torch.float32,device=config.device); f=torch.tensor(f,dtype=torch.float32,device=config.device)
    rows=[]
    for horizon in horizons:
        h=horizon-1
        def forward_func(past, future):
            return fit.model(past,future)[:,h]
        ig=IntegratedGradients(forward_func)
        attr_p,attr_f=ig.attribute((p,f), baselines=(torch.zeros_like(p),torch.zeros_like(f)), n_steps=n_steps)
        ap=attr_p.detach().abs().cpu().numpy().mean(axis=(0,1))
        af=attr_f.detach().abs().cpu().numpy().mean(axis=(0,1))
        for feat,val in zip(sample.past_cols,ap): rows.append({"horizon":horizon,"input":"past","feature":feat,"mean_abs_IG":float(val)})
        for feat,val in zip(sample.future_cols,af): rows.append({"horizon":horizon,"input":"future","feature":feat,"mean_abs_IG":float(val)})
    return pd.DataFrame(rows)


def grouped_permutation_lash(config: ExperimentConfig, dataset_key: str, seed: int = 42,
                             sample_origins: int = 256, repetitions: int = 5) -> pd.DataFrame:
    _, bundles=build_dataset_bundles(config,dataset_key); test=bundles["test"]
    idx=np.linspace(0,len(test)-1,min(sample_origins,len(test)),dtype=int); sample=take_bundle(test,idx,"perm")
    fit,params,gain=load_neural_artifact(config,dataset_key,"LASH_SEQ",seed,test)
    ridge_payload=load_ridge_artifact(config,dataset_key); ridge_fit=ridge_payload["fit"]; ridge_gain=ridge_payload["tuning"]["gain"]
    router_json=json.load(open(dataset_output_dir(config,dataset_key)/"configs"/"router_default.json",encoding="utf-8"))
    weights=np.asarray(router_json["weights"],float)
    seq=apply_residual_gain(sample,predict_neural_raw(fit,sample,params,config),gain)
    ridge=apply_residual_gain(sample,predict_ridge_raw(ridge_fit,sample),ridge_gain)
    base=blend_predictions(seq,ridge,weights); base_score=regression_metrics(sample.y,base)["selection_score"]

    groups={
        "historical_consumption": ([sample.past_cols.index("Consumption")], []),
        "calendar": ([i for i,c in enumerate(sample.past_cols) if any(k in c for k in ["hour_","dow_","month_","doy_","Holi","Weekend"])],
                     [i for i,c in enumerate(sample.future_cols) if any(k in c for k in ["hour_","dow_","month_","doy_","Holi","Weekend"])]),
        "weather": ([i for i,c in enumerate(sample.past_cols) if c in {"Temp","Humi","WS","THI","WCT","HDD18","CDD18","HDD18_sq","CDD18_sq","Temp_sq","Temp_x_Humi"}],
                    [i for i,c in enumerate(sample.future_cols) if any(w in c for w in ["Temp","Humi","WS","THI","WCT","HDD18","CDD18"])]),
        "safe_demand_context": ([], [i for i,c in enumerate(sample.future_cols) if any(k in c for k in ["cons_lag","Cons_avg","recent_"]) ]),
        "anchor_pathway": ([], [i for i,c in enumerate(sample.future_cols) if c=="anchor"]),
    }
    rng=np.random.default_rng(2026); rows=[]
    for group,(pi,fi) in groups.items():
        increases=[]
        for _ in range(repetitions):
            perm=rng.permutation(len(sample)); pert=take_bundle(sample,np.arange(len(sample)),"pert")
            pert.past=pert.past.copy(); pert.future=pert.future.copy(); pert.anchor=pert.anchor.copy()
            if pi: pert.past[:,:,pi]=pert.past[perm][:,:,pi]
            if fi: pert.future[:,:,fi]=pert.future[perm][:,:,fi]
            if group=="anchor_pathway": pert.anchor=pert.anchor[perm]
            s=apply_residual_gain(pert,predict_neural_raw(fit,pert,params,config),gain)
            r=apply_residual_gain(pert,predict_ridge_raw(ridge_fit,pert),ridge_gain)
            pp=blend_predictions(s,r,weights)
            increases.append(regression_metrics(sample.y,pp)["selection_score"]-base_score)
        rows.append({"group":group,"mean_score_increase":float(np.mean(increases)),"sd_score_increase":float(np.std(increases,ddof=1))})
    return pd.DataFrame(rows).sort_values("mean_score_increase",ascending=False)


def tree_shap_lightgbm(config: ExperimentConfig, dataset_key: str, seed: int = 42,
                       horizons=(1,12,24), sample_origins: int = 256) -> pd.DataFrame:
    import shap
    _,bundles=build_dataset_bundles(config,dataset_key); test=bundles["test"]
    idx=np.linspace(0,len(test)-1,min(sample_origins,len(test)),dtype=int); sample=take_bundle(test,idx,"shap")
    payload=load_tree_artifact(config,dataset_key,"LIGHTGBM",seed); fit=payload["fit"]
    X=make_tabular_origin_design(sample)
    names=[]
    # Derive names locally to avoid a circular import of helper details.
    names += [f"past_consumption_lag_{LOOKBACK-i}" for i in range(LOOKBACK)]
    for prefix in ("last","mean168","std168"):
        names += [f"{prefix}__{c}" for c in sample.past_cols]
    for h in range(HORIZON): names += [f"h{h+1:02d}__{c}" for c in sample.future_cols]
    names += [f"anchor_h{h+1:02d}" for h in range(HORIZON)]
    rows=[]
    for horizon in horizons:
        model=fit.models[horizon-1]
        explainer=shap.TreeExplainer(model)
        values=np.asarray(explainer.shap_values(X))
        imp=np.mean(np.abs(values),axis=0)
        top=np.argsort(imp)[::-1][:40]
        for j in top: rows.append({"horizon":horizon,"feature":names[j],"mean_abs_SHAP":float(imp[j])})
    return pd.DataFrame(rows)

