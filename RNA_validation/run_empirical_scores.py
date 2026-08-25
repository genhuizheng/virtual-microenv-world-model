#!/usr/bin/env python3
"""Run COMPASS baseline immune scores on aligned expression and evaluate response."""

from __future__ import annotations

import argparse
import importlib
import inspect
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from RNA_validation.common import require_empty_output, write_json


RESPONSE_HIGH = {
    "PD1", "PDL1", "CTLA4", "NETBIO", "PGM", "GENEBIO", "CIS", "TEFF", "NRS",
    "IFNG", "CTL", "CKS", "IS", "ICA", "CD8", "MIAS", "GEP", "IMPRES",
}
RESISTANCE_HIGH = {"TAM", "TEXH", "CAF", "TIDE"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-kind", choices=["paired", "tcga"], required=True)
    p.add_argument("--prepared-dir", type=Path, required=True)
    p.add_argument("--prediction-dir", type=Path, default=None)
    p.add_argument("--compass-repo", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--layers", default="pre", help="paired: pre,observed_post,predicted_post; tcga: pre,predicted_post")
    p.add_argument("--methods", default="all", help="comma-separated names or all")
    p.add_argument("--default-drug-target", default="PD1")
    p.add_argument("--response-subset", choices=["ici", "all"], default="ici")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def expression_frame(x: np.ndarray, genes: pd.DataFrame) -> pd.DataFrame:
    symbols = genes["gene_symbol"].astype(str).str.strip().str.upper()
    keep = ~symbols.isin(["", "NA", "NAN", "NONE"])
    # COMPASS IFNG uses qnorm, whose result is float64. Supplying float64 here
    # avoids pandas emitting one incompatible-dtype warning per gene column.
    # This changes only the in-memory dtype, not the raw-count values.
    frame = pd.DataFrame(
        x[:, keep.to_numpy()].astype(np.float64, copy=False),
        columns=symbols[keep].tolist(),
    )
    if frame.columns.duplicated().any():
        frame = frame.T.groupby(level=0, sort=False).mean().T
    return frame


def load_inputs(args) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    genes = pd.read_csv(args.prepared_dir / "model_genes.tsv", sep="\t", dtype=str, keep_default_na=False)
    layers = [x.strip() for x in args.layers.split(",") if x.strip()]
    result = []
    if args.input_kind == "paired":
        arrays = np.load(args.prepared_dir / "paired_expression.npz")
        meta = pd.read_csv(args.prepared_dir / "patients.tsv", sep="\t", dtype=str, keep_default_na=False)
        predicted = None
        if "predicted_post" in layers:
            if args.prediction_dir is None:
                raise ValueError("predicted_post requires --prediction-dir")
            predicted = np.load(args.prediction_dir / "cross_validated_predictions.npz")["pred_world_finetuned"]
        values = {"pre": arrays["x_pre"], "observed_post": arrays["y_post"], "predicted_post": predicted}
        for layer in layers:
            if layer not in values or values[layer] is None:
                raise ValueError(f"Unsupported/unavailable paired layer: {layer}")
            result.append((layer, expression_frame(values[layer], genes), meta.copy()))
    else:
        if args.prediction_dir is None:
            raise ValueError("TCGA scoring requires --prediction-dir from infer_tcga.py")
        manifest = pd.read_csv(args.prediction_dir / "manifest.tsv", sep="\t")
        for row in manifest.itertuples(index=False):
            arrays = np.load(args.prediction_dir / row.prediction_npz)
            meta = pd.read_csv(args.prediction_dir / row.metadata_tsv, sep="\t", dtype=str, keep_default_na=False)
            meta["cancer_code"] = str(row.dataset_id).removeprefix("tcga_").upper()
            meta["drug_target"] = args.default_drug_target
            values = {"pre": arrays["x_pre"], "predicted_post": arrays["pred_post"]}
            for layer in layers:
                if layer not in values:
                    raise ValueError(f"Unsupported TCGA layer: {layer}")
                result.append((layer, expression_frame(values[layer], genes), meta.copy()))
    return result


def load_methods(compass_repo: Path):
    sys.path.insert(0, str(compass_repo.resolve()))
    # COMPASS publishes this package and registry with the original
    # "immnue" spelling. Match the upstream repository exactly.
    module = importlib.import_module("baseline.immnue_score")
    methods = getattr(module, "immnue_score_methods", None)
    if not isinstance(methods, dict):
        raise RuntimeError(
            "COMPASS baseline.immnue_score.immnue_score_methods was not found"
        )
    return methods


def make_scorer(factory, cancer: str, target: str):
    try:
        signature = inspect.signature(factory)
        kwargs = {}
        if "cancer_type" in signature.parameters:
            kwargs["cancer_type"] = cancer
        if "drug_target" in signature.parameters:
            kwargs["drug_target"] = target
        return factory(**kwargs)
    except (TypeError, ValueError):
        return factory(cancer_type=cancer, drug_target=target)


def coerce_scores(value, index: pd.Index, method: str) -> pd.DataFrame:
    if isinstance(value, pd.Series):
        return value.reindex(index).to_frame(method)
    if isinstance(value, pd.DataFrame):
        value = value.reindex(index)
        if len(value.columns) == 1:
            value.columns = [method]
        else:
            value.columns = [f"{method}__{c}" for c in value.columns]
        return value
    array = np.asarray(value)
    if array.ndim == 1:
        return pd.DataFrame({method: array}, index=index)
    return pd.DataFrame(array, index=index, columns=[f"{method}__{i}" for i in range(array.shape[1])])


def score_one_layer(expr: pd.DataFrame, meta: pd.DataFrame, methods: dict, selected: list[str], default_target: str):
    scores = pd.DataFrame(index=expr.index)
    errors = []
    cancer = meta.get("cancer_code", pd.Series("PAN", index=meta.index)).replace("NA", "PAN")
    target = meta.get("drug_target", pd.Series(default_target, index=meta.index)).replace("NA", default_target)
    groups = pd.DataFrame({"cancer": cancer, "target": target}).groupby(["cancer", "target"], dropna=False).groups
    for name in selected:
        factory = methods[name]
        pieces = []
        for (cancer_code, drug_target), idx in groups.items():
            try:
                scorer = make_scorer(factory, str(cancer_code), str(drug_target))
                value = scorer(expr.loc[idx])
                pieces.append(coerce_scores(value, expr.loc[idx].index, name))
            except Exception as exc:
                errors.append({"method": name, "cancer_code": cancer_code, "drug_target": drug_target, "error": repr(exc)})
        if pieces:
            method_scores = pd.concat(pieces).sort_index()
            scores = scores.join(method_scores, how="left")
    return scores, errors


def direction_for(method_column: str) -> int:
    base = method_column.split("__", 1)[0].upper()
    if base in RESISTANCE_HIGH:
        return -1
    return 1


def metric_pair(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(score)
    y, score = y[valid], score[valid]
    if len(y) < 3 or len(np.unique(y)) < 2:
        return math.nan, math.nan
    return float(roc_auc_score(y, score)), float(average_precision_score(y, score))


def bootstrap_metrics(y, score, n_bootstrap, rng):
    values = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        values.append(metric_pair(y[idx], score[idx]))
    if not values:
        return (math.nan,) * 4
    array = np.asarray(values)
    return tuple(np.nanpercentile(array[:, j], q) for j in range(2) for q in (2.5, 97.5))


def evaluate_response(score_table: pd.DataFrame, meta: pd.DataFrame, layer: str, n_bootstrap: int, seed: int, subset: str):
    labels = pd.to_numeric(meta["response_binary"], errors="coerce").to_numpy()
    labeled = np.isfinite(labels)
    if subset == "ici" and "drug_target" in meta:
        labeled &= meta["drug_target"].isin(["PD1", "PDL1", "CTLA4"]).to_numpy()
    rng = np.random.default_rng(seed)
    pooled, cohorts = [], []
    for column in score_table.columns:
        direction = direction_for(column)
        score = pd.to_numeric(score_table[column], errors="coerce").to_numpy() * direction
        valid = labeled & np.isfinite(score)
        auroc, auprc = metric_pair(labels[valid], score[valid])
        lo_roc, hi_roc, lo_pr, hi_pr = bootstrap_metrics(labels[valid], score[valid], n_bootstrap, rng) if valid.sum() else (math.nan,) * 4
        pooled.append({"layer": layer, "score": column, "direction": direction, "n": int(valid.sum()), "auroc": auroc,
                       "auroc_ci_low": lo_roc, "auroc_ci_high": hi_roc, "auprc": auprc,
                       "auprc_ci_low": lo_pr, "auprc_ci_high": hi_pr})
        for cohort, idx in meta.groupby("source_dataset_id").groups.items():
            idx = np.asarray(list(idx), dtype=int)
            ok = np.isfinite(labels[idx]) & np.isfinite(score[idx])
            c_roc, c_pr = metric_pair(labels[idx][ok], score[idx][ok])
            if np.isfinite(c_roc):
                cohorts.append({"layer": layer, "score": column, "source_dataset_id": cohort, "n": int(ok.sum()), "auroc": c_roc, "auprc": c_pr})
    return pd.DataFrame(pooled), pd.DataFrame(cohorts)


def main() -> None:
    args = parse_args()
    out = require_empty_output(args.output_dir, args.overwrite)
    # The upstream IFNG implementation warns even when the missing-marker list
    # is empty. Suppress only that exact no-op message; real missing markers
    # remain visible.
    warnings.filterwarnings(
        "ignore",
        message=r"Markers of \[\] are missed and not used\.",
        category=UserWarning,
    )
    available = load_methods(args.compass_repo)
    if args.methods == "all":
        selected = list(available)
    else:
        requested = [x.strip() for x in args.methods.split(",") if x.strip()]
        lookup = {x.upper(): x for x in available}
        missing = [x for x in requested if x.upper() not in lookup]
        if missing:
            raise ValueError(f"Unknown methods {missing}; available={list(available)}")
        selected = [lookup[x.upper()] for x in requested]
    all_scores, all_errors, pooled_tables, cohort_tables = [], [], [], []
    for layer, expr, meta in load_inputs(args):
        scores, errors = score_one_layer(expr, meta, available, selected, args.default_drug_target)
        identifiers = meta[[c for c in ("patient_id", "biological_patient_id", "dataset_id", "source_dataset_id", "response_binary", "cancer_code", "drug_target") if c in meta.columns]].copy()
        identifiers.insert(0, "layer", layer)
        all_scores.append(pd.concat([identifiers.reset_index(drop=True), scores.reset_index(drop=True)], axis=1))
        for error in errors:
            all_errors.append({"layer": layer, **error})
        if args.input_kind == "paired" and "response_binary" in meta:
            pooled, cohort = evaluate_response(scores, meta, layer, args.bootstrap, args.seed, args.response_subset)
            pooled_tables.append(pooled)
            cohort_tables.append(cohort)
    pd.concat(all_scores, ignore_index=True).to_csv(out / "empirical_scores.tsv.gz", sep="\t", index=False, compression="gzip")
    pd.DataFrame(all_errors).to_csv(out / "score_errors.tsv", sep="\t", index=False)
    if pooled_tables:
        pd.concat(pooled_tables, ignore_index=True).to_csv(out / "response_performance_pooled.tsv", sep="\t", index=False)
        cohorts = pd.concat(cohort_tables, ignore_index=True) if cohort_tables else pd.DataFrame()
        cohorts.to_csv(out / "response_performance_by_cohort.tsv", sep="\t", index=False)
        if not cohorts.empty:
            mean = cohorts.groupby(["layer", "score"])[["auroc", "auprc"]].agg(["mean", "sem", "count"])
            mean.columns = ["_".join(c) for c in mean.columns]
            mean = mean.reset_index()
            mean["auroc_ci_low"] = mean["auroc_mean"] - 1.96 * mean["auroc_sem"]
            mean["auroc_ci_high"] = mean["auroc_mean"] + 1.96 * mean["auroc_sem"]
            mean["auprc_ci_low"] = mean["auprc_mean"] - 1.96 * mean["auprc_sem"]
            mean["auprc_ci_high"] = mean["auprc_mean"] + 1.96 * mean["auprc_sem"]
            mean.to_csv(out / "mean_cohort_performance_95ci.tsv", sep="\t", index=False)
    write_json(out / "run_summary.json", {
        "input_kind": args.input_kind,
        "layers": args.layers,
        "compass_repo": str(args.compass_repo),
        "methods_requested": selected,
        "response_subset": args.response_subset,
        "n_errors": len(all_errors),
        "direction_policy": {"response_high": sorted(RESPONSE_HIGH), "resistance_high": sorted(RESISTANCE_HIGH)},
        "note": "Scores are run from the public COMPASS baseline implementation; raw score scales are not compared by MSE.",
    })


if __name__ == "__main__":
    main()
