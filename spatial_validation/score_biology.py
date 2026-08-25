#!/usr/bin/env python3
"""Score pretreatment and predicted post-treatment biology with COMPASS signatures."""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from RNA_validation.common import require_empty_output, write_json
from RNA_validation.run_empirical_scores import (
    direction_for,
    expression_frame,
    load_methods,
    score_one_layer,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--prediction-dir", type=Path, required=True)
    p.add_argument("--compass-repo", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--methods", default="all")
    p.add_argument("--default-drug-target", default="PD1")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def response_binary(value: object) -> float:
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"response", "partial_response", "disease_control", "complete_response"}:
        return 1.0
    if text in {"non_response", "no_disease_control", "non_pr"}:
        return 0.0
    return math.nan


def add_bh_fdr(table: pd.DataFrame, p_column: str) -> pd.DataFrame:
    if p_column not in table.columns:
        table["bh_fdr"] = np.nan
        return table
    p = pd.to_numeric(table[p_column], errors="coerce").to_numpy(float)
    valid = np.flatnonzero(np.isfinite(p))
    q = np.full(len(table), np.nan)
    if len(valid):
        order = valid[np.argsort(p[valid])]
        ranked = p[order] * len(valid) / np.arange(1, len(valid) + 1)
        q[order] = np.minimum.accumulate(ranked[::-1])[::-1].clip(0, 1)
    table["bh_fdr"] = q
    return table


def main() -> None:
    args = parse_args()
    out = require_empty_output(args.output_dir, args.overwrite)
    warnings.filterwarnings("ignore", message=r"Markers of \[\] are missed and not used\.", category=UserWarning)
    available = load_methods(args.compass_repo)
    if args.methods == "all":
        selected = list(available)
    else:
        lookup = {name.upper(): name for name in available}
        requested = [x.strip().upper() for x in args.methods.split(",") if x.strip()]
        missing = [x for x in requested if x not in lookup]
        if missing:
            raise ValueError(f"Unknown COMPASS methods: {missing}")
        selected = [lookup[x] for x in requested]

    manifest = pd.read_csv(args.prediction_dir / "prediction_manifest.tsv", sep="\t", dtype=str, keep_default_na=False)
    genes = pd.read_csv(args.prediction_dir / "model_genes.tsv", sep="\t", dtype=str, keep_default_na=False)
    pre_rows, pred_rows, measured_rows = [], [], []
    for row in manifest.itertuples(index=False):
        table = pd.read_csv(args.prediction_dir / row.gene_table, sep="\t")
        pre_rows.append(table["pretreatment_mean"].to_numpy(np.float64))
        pred_rows.append(table["predicted_posttreatment_mean"].to_numpy(np.float64))
        measured_rows.append(table["measured_pre"].astype(str).str.lower().eq("true").to_numpy())
    common_measured = np.logical_and.reduce(measured_rows)
    if common_measured.sum() < 100:
        raise ValueError(f"Only {common_measured.sum()} genes are measured in every baseline sample")
    scoring_genes = genes.loc[common_measured].reset_index(drop=True)
    pre_expr = expression_frame(np.vstack(pre_rows)[:, common_measured], scoring_genes)
    pred_expr = expression_frame(np.vstack(pred_rows)[:, common_measured], scoring_genes)
    meta = manifest[["sample_id", "gse_id", "patient_key", "patient_id", "platform", "cancer_type", "response_group"]].copy()
    meta["cancer_code"] = "PAN"
    meta["drug_target"] = args.default_drug_target
    meta["response_binary"] = meta["response_group"].map(response_binary)

    # Score both layers together so methods that normalize across samples use a
    # shared scale. Scoring each layer independently can create artificial
    # pre/post differences for quantile- or cohort-normalized signatures.
    combined_expression = pd.concat([pre_expr, pred_expr], ignore_index=True)
    combined_meta = pd.concat([meta, meta], ignore_index=True)
    combined_scores, errors = score_one_layer(
        combined_expression, combined_meta, available, selected, args.default_drug_target
    )
    n = len(meta)
    scores_by_layer = {
        "pretreatment": combined_scores.iloc[:n].reset_index(drop=True),
        "predicted_posttreatment": combined_scores.iloc[n:].reset_index(drop=True),
    }
    errors = [{"layer": "joint_pretreatment_and_predicted", **error} for error in errors]
    for layer, scores in scores_by_layer.items():
        table = pd.concat([meta.reset_index(drop=True), scores], axis=1)
        table.insert(0, "layer", layer)
        table.to_csv(out / f"{layer}_empirical_scores.tsv.gz", sep="\t", index=False, compression="gzip")

    common_scores = sorted(set(scores_by_layer["pretreatment"].columns) & set(scores_by_layer["predicted_posttreatment"].columns))
    delta = scores_by_layer["predicted_posttreatment"][common_scores] - scores_by_layer["pretreatment"][common_scores]
    delta.columns = [f"{column}__delta" for column in delta.columns]
    delta_table = pd.concat([meta.reset_index(drop=True), delta.reset_index(drop=True)], axis=1)
    delta_table.to_csv(out / "empirical_score_deltas.tsv.gz", sep="\t", index=False, compression="gzip")

    response_rows = []
    labels = meta["response_binary"].to_numpy(float)
    for column in common_scores:
        values = pd.to_numeric(delta[f"{column}__delta"], errors="coerce").to_numpy(float)
        adjusted = values * direction_for(column)
        responder = adjusted[labels == 1]
        nonresponder = adjusted[labels == 0]
        if len(responder) >= 2 and len(nonresponder) >= 2:
            statistic, pvalue = mannwhitneyu(responder, nonresponder, alternative="two-sided")
        else:
            statistic, pvalue = math.nan, math.nan
        response_rows.append({
            "score": column,
            "direction": direction_for(column),
            "responders": len(responder),
            "nonresponders": len(nonresponder),
            "mean_adjusted_delta_responders": float(np.mean(responder)) if len(responder) else math.nan,
            "mean_adjusted_delta_nonresponders": float(np.mean(nonresponder)) if len(nonresponder) else math.nan,
            "responder_minus_nonresponder": float(np.mean(responder) - np.mean(nonresponder)) if len(responder) and len(nonresponder) else math.nan,
            "mannwhitney_statistic": statistic,
            "mannwhitney_pvalue": pvalue,
        })
    add_bh_fdr(pd.DataFrame(response_rows), "mannwhitney_pvalue").to_csv(
        out / "empirical_score_response_associations.tsv", sep="\t", index=False
    )
    pd.DataFrame(errors).to_csv(out / "score_errors.tsv", sep="\t", index=False)
    write_json(out / "biology_score_summary.json", {
        "samples": len(meta),
        "methods_requested": selected,
        "scores_produced": common_scores,
        "genes_measured_in_every_sample": int(common_measured.sum()),
        "comparison": "model-predicted posttreatment minus pretreatment",
        "measured_posttreatment_used": False,
        "compass_repo": str(args.compass_repo),
        "score_errors": len(errors),
    })
    print(f"biology_scores={len(common_scores)} samples={len(meta)} output={out}")


if __name__ == "__main__":
    main()
